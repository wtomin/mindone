# reference to https://github.com/mlfoundations/open_clip
import numbers
from collections import OrderedDict
from typing import Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.common.initializer import initializer


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, has_bias=False, pad_mode="valid")
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, has_bias=False, pad_mode="valid")
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU()

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, has_bias=False, pad_mode="valid")
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU()

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.SequentialCell(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, has_bias=False, pad_mode="valid"),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def construct(self, x):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out


class AttentionPool2d(nn.Cell):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, dtype: mstype = ms.float32
    ):
        super().__init__()
        self.dtype = dtype
        self.positional_embedding = Parameter(
            ms.numpy.randn(spacial_dim**2 + 1, embed_dim, dtype=self.dtype) / embed_dim**0.5
        )
        self.k_proj = nn.Dense(embed_dim, embed_dim).to_float(dtype)
        self.q_proj = nn.Dense(embed_dim, embed_dim).to_float(dtype)
        self.v_proj = nn.Dense(embed_dim, embed_dim).to_float(dtype)
        self.c_proj = nn.Dense(embed_dim, output_dim or embed_dim).to_float(dtype)
        self.num_heads = num_heads

    def construct(self, x: Tensor):
        x = x.flatten(start_dim=2)
        x = ops.transpose(x, (2, 0, 1))  # NCHW -> (HW)NC
        x = ops.concat([x.mean(axis=0, keep_dims=True), x], axis=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :]  # (HW+1)NC
        x, _ = ops.nn_func.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=ops.concat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Cell):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, has_bias=False, pad_mode="valid")
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, has_bias=False, pad_mode="valid")
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, has_bias=False, pad_mode="valid")
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.SequentialCell(layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features**-0.5
            for weight in (
                self.attnpool.q_proj.weight,
                self.attnpool.k_proj.weight,
                self.attnpool.v_proj.weight,
                self.attnpool.c_proj.weight,
            ):
                weight.set_data(initializer.initializer(initializer.Normal(sigma=std), weight.shape, weight.dtype))

        for resnet_block in (self.layer1, self.layer2, self.layer3, self.layer4):
            for name, weight in resnet_block.parameters_and_names():
                if name.endswith("bn3.gamma"):
                    weight.set_data(initializer.initializer(initializer.Zero(), weight.shape, weight.dtype))

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def construct(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.Cell):
    r"""Applies Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `ops.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = ops.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = ops.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = LayerNorm([C, H, W])
        >>> output = layer_norm(input)
    """

    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine: bool = True, bias=True, dtype=ms.float32):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        _weight = np.ones(normalized_shape, dtype=ms.dtype_to_nptype(dtype))
        _bias = np.zeros(normalized_shape, dtype=ms.dtype_to_nptype(dtype))
        if self.elementwise_affine:
            self.weight = Parameter(ms.Tensor.from_numpy(_weight), name="weight")
            if bias:
                self.bias = Parameter(ms.Tensor.from_numpy(_bias), name="bias")
            else:
                self.bias = ms.Tensor.from_numpy(_bias)
        else:
            self.weight = ms.Tensor.from_numpy(_weight)
            self.bias = ms.Tensor.from_numpy(_bias)
        # TODO: In fact, we need -len(normalized_shape) instead of -1, but LayerNorm doesn't allow it.
        #  For positive axis, the ndim of input is needed. Put it in construct?
        self.layer_norm = ops.LayerNorm(-1, -1, epsilon=eps)

    def construct(self, x: ms.Tensor):
        x, _, _ = self.layer_norm(x, self.weight, self.bias)
        return x


class QuickGELU(nn.Cell):
    def construct(self, x: Tensor):
        return x * ops.sigmoid(1.702 * x)


class MultiheadAttention(nn.Cell):
    def __init__(self, d_model: int, n_head: int, dtype: mstype):
        super(MultiheadAttention, self).__init__()

        self.num_heads = n_head
        self.head_dim = d_model // n_head

        self.scaling = self.head_dim**-0.5
        self.in_proj = nn.Dense(d_model, 3 * d_model).to_float(dtype)
        self.out_proj = nn.Dense(d_model, d_model).to_float(dtype)

    def construct(self, query: ms.Tensor, attn_mask: Optional[ms.Tensor] = None):
        r"""Construct

        Args:
            query (ms.Tensor): query of attention.
            attn_mask (Optional[ms.Tensor]): attention mask.

        Returns:
            attn_output (ms.Tensor): attention output.
        """
        len_tgt, batch_size, width = query.shape
        qkv = self.in_proj(query)
        qkv = ops.reshape(qkv, (len_tgt, batch_size, 3, width)).transpose((2, 0, 1, 3))

        att_q = qkv[0:1]
        att_q = ops.Squeeze(0)(att_q)
        att_q = att_q * self.scaling
        att_q = att_q.view(len_tgt, batch_size * self.num_heads, self.head_dim).transpose((1, 0, 2))

        att_k = qkv[1:2]
        att_k = ops.Squeeze(0)(att_k)
        att_k = att_k.view(-1, batch_size * self.num_heads, self.head_dim).transpose((1, 0, 2))

        att_v = qkv[2:3]
        att_v = ops.Squeeze(0)(att_v)
        att_v = att_v.view(-1, batch_size * self.num_heads, self.head_dim).transpose((1, 0, 2))

        if attn_mask is not None:
            attn_output_weights = attn_mask + ops.matmul(att_q, att_k.transpose((0, 2, 1)))
        else:
            attn_output_weights = ops.matmul(att_q, att_k.transpose((0, 2, 1)))
        attn_output_weights = ops.softmax(attn_output_weights, axis=-1)
        attn_output = ops.matmul(attn_output_weights, att_v)
        attn_output = ops.transpose(attn_output, (1, 0, 2))
        attn_output = attn_output.view(len_tgt, batch_size, width)
        attn_output = self.out_proj(attn_output)
        return attn_output


class ResidualAttentionBlock(nn.Cell):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: Tensor = None,
        epsilon: float = 1e-5,
        use_quick_gelu: bool = False,
        dtype: mstype = ms.float32,
    ):
        super().__init__()

        self.dtype = dtype
        self.attn = MultiheadAttention(d_model, n_head, dtype)
        self.ln_1 = LayerNorm((d_model,), eps=epsilon)
        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    ("c_fc", nn.Dense(d_model, d_model * 4).to_float(self.dtype)),
                    ("gelu", QuickGELU().to_float(self.dtype) if use_quick_gelu else nn.GELU().to_float(self.dtype)),
                    ("c_proj", nn.Dense(d_model * 4, d_model).to_float(self.dtype)),
                ]
            )
        )
        self.ln_2 = LayerNorm((d_model,), eps=epsilon)
        self.attn_mask = attn_mask

    def attention(self, x: Tensor):
        return self.attn(x, attn_mask=self.attn_mask)

    def construct(self, x: Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Cell):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: Tensor = None,
        epsilon: float = 1e-5,
        use_quick_gelu: bool = False,
        dtype: mstype = ms.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell(
            *[
                ResidualAttentionBlock(
                    width, heads, attn_mask, epsilon=epsilon, use_quick_gelu=use_quick_gelu, dtype=self.dtype
                )
                for _ in range(layers)
            ]
        )

    def construct(self, x: Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Cell):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        epsilon: float = 1e-5,
        use_quick_gelu: bool = False,
        dtype: mstype = ms.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, pad_mode="pad", has_bias=False
        ).to_float(self.dtype)

        scale = width**-0.5
        self.class_embedding = Parameter(scale * ms.numpy.randn(width, dtype=self.dtype))
        self.positional_embedding = Parameter(
            scale * ms.numpy.randn((input_resolution // patch_size) ** 2 + 1, width, dtype=self.dtype)
        )
        self.ln_pre = LayerNorm((width,), eps=epsilon)

        self.transformer = Transformer(
            width, layers, heads, epsilon=epsilon, use_quick_gelu=use_quick_gelu, dtype=self.dtype
        )

        self.ln_post = LayerNorm((width,), eps=epsilon)
        self.proj = Parameter(scale * ms.numpy.randn(width, output_dim, dtype=self.dtype))

    def construct(self, x: Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = ops.concat(
            [self.class_embedding + ops.zeros((x.shape[0], 1, x.shape[-1]), dtype=self.dtype), x], axis=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Cell):
    def __init__(
        self,
        embed_dim: int,  # 512
        # vision
        image_resolution: int,  # 224
        vision_layers: Union[Tuple[int, int, int, int], int],  # 12
        vision_width: int,  # 768
        vision_patch_size: int,  # 16
        # text
        context_length: int,  # 77
        vocab_size: int,  # 49408
        transformer_width: int,  # 512
        transformer_heads: int,  # 8
        transformer_layers: int,  # 12
    ):
        super().__init__()
        # pdb.set_trace()
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(ops.randn((self.context_length, transformer_width)))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(ops.randn((transformer_width, embed_dim)))
        self.logit_scale = nn.Parameter(ops.ones([]) * ms.numpy.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = ops.ones((self.context_length, self.context_length)) * float("-inf")
        mask = ops.triu(mask, diagonal=1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(
        self, image, return_side_out=False, return_all_tokens=False, return_all_final_tokens=False, **kwargs
    ):
        return self.visual(image.to(self.dtype), return_all_tokens, return_all_final_tokens, **kwargs)

    def encode_text(self, text, return_all_tokens=False, return_patch_tokens=False):
        x = self.token_embedding(text).to(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).to(self.dtype)

        if return_patch_tokens:
            return x
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if return_all_tokens:
            # pdb.set_trace()
            x = x @ self.text_projection
        else:
            x = x[ops.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def construct(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)  # convert applicable model parameters to fp16
    pnl, nnl = ms.load_param_into_net(model, state_dict)
    print("pnl", pnl)
    print("nnl", nnl)
    model.set_train(False)
    for param in model.trainable_params():
        param.requires_grad = False
    return model
