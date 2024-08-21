import numbers
import os
from functools import partial
from typing import Tuple

import numpy as np
from utils import load_torch_state_dict_to_ms_ckpt

import mindspore as ms
from mindspore import Parameter, nn, ops


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


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.swapaxes(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.query = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.key = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.value = nn.Dense(dim, dim, has_bias=qkv_bias)

        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x, codebook_embeds, codebook_mask):
        B, N, C = codebook_embeds.shape
        _, N_x, _ = x.shape

        q = self.query(x).reshape(B, N_x, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(codebook_embeds).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(codebook_embeds).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.swapaxes(-2, -1)) * self.scale

        extended_mask = codebook_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0
        attn = attn + extended_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).swapaxes(1, 2).reshape(B, N_x, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
    ):
        super().__init__()
        self.norm0 = norm_layer(dim)
        self.self_attn = SelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x, codebook_embeds, codebook_mask):
        x = x + self.self_attn(self.norm0(x))
        x = x + self.cross_attn(self.norm1(x), codebook_embeds, codebook_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionPool2d(nn.Cell):
    def __init__(self, seq_len: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = ms.Parameter(ops.randn(seq_len + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Dense(embed_dim, embed_dim)
        self.q_proj = nn.Dense(embed_dim, embed_dim)
        self.v_proj = nn.Dense(embed_dim, embed_dim)
        self.c_proj = nn.Dense(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def construct(self, x, return_all_tokens=False):
        # x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = x.permute(1, 0, 2)  # (N(HW)C) => (HW)NC
        x = ops.cat([x.mean(dim=0, keepdim=True), x], axis=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = ops.nn_func.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=ops.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
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
        if return_all_tokens:
            return x
        else:
            return x[0]


class VQDecoder(nn.Cell):
    def __init__(
        self,
        img_size=224,
        patch_size=14,
        in_chans=32,
        embed_dim=1408,
        depth=12,
        num_heads=16,
        mlp_ratio=4.3637,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=LayerNorm,
        **kwargs,
    ):
        super().__init__()

        self.in_proj = nn.Dense(in_chans, embed_dim)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.num_patches = num_patches

        self.pos_embed = ms.Parameter(ops.zeros(1, num_patches, embed_dim))  # The postion embedding for the latent code

        self.query_embed = ms.Parameter(ops.zeros(1, num_patches, embed_dim))  # The query embedding for reconstruction

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.CellList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # The decoder task layer
        self.decoder_out_dim = 1408
        self.decode_task_layer = nn.SequentialCell(
            nn.Dense(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Dense(embed_dim, self.decoder_out_dim),
        )

        self.unet_proj = nn.Dense(self.decoder_out_dim, 768)

    def get_num_layers(self):
        return len(self.blocks)

    def construct(self, x, token_num):
        # codebook_fea
        # B, nc, w, h = codebook_fea.shape
        x = self.in_proj(x)
        B = len(token_num)
        num_tokens, C = x.shape

        x_list = ops.split(x, token_num.tolist(), dim=0)
        max_token_num = int(token_num.max())
        x_pad = ops.zeros(B, max_token_num, C, dtype=x.dtype)
        mask = ops.zeros(B, max_token_num, dtype=x.dtype)

        for i, x_tensor in enumerate(x_list):
            x_pad[i][: len(x_tensor)] = x_tensor
            mask[i][: len(x_tensor)] = 1

        x_pad = x_pad + self.pos_embed[:, :max_token_num]
        x_pad = self.pos_drop(x_pad)

        query_embeds = self.query_embed.broadcast_to((B, -1, -1))

        for blk in self.blocks:
            query_embeds = blk(query_embeds, codebook_embeds=x_pad, codebook_mask=mask)

        query_embeds = self.norm(query_embeds)  # To align with the raw vit features

        visual_rec = self.decode_task_layer(query_embeds)

        visual_rec = self.unet_proj(visual_rec)

        return visual_rec


class HighresVQDecoder(nn.Cell):
    def __init__(
        self,
        img_size=224,
        patch_size=14,
        in_chans=32,
        embed_dim=1408,
        depth=12,
        num_heads=16,
        mlp_ratio=4.3637,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=partial(LayerNorm, eps=1e-5),
        **kwargs,
    ):
        super().__init__()

        self.in_proj = nn.Dense(in_chans, embed_dim)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.num_patches = num_patches

        self.pos_embed = ms.Parameter(ops.zeros(1, num_patches, embed_dim))  # The postion embedding for the latent code

        self.query_embed = ms.Parameter(ops.zeros(1, num_patches, embed_dim))  # The query embedding for reconstruction

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.CellList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # The decoder task layer
        self.decoder_out_dim = 1408
        self.decode_task_layer = nn.Sequential(
            nn.Dense(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Dense(embed_dim, self.decoder_out_dim),
        )

        # Convert the decoded features to Unet Condition
        self.unet_proj_1 = nn.Dense(self.decoder_out_dim, 768)
        self.unet_proj_2 = nn.Dense(self.decoder_out_dim, 1280)
        self.unet_attnpool = AttentionPool2d(num_patches, self.decoder_out_dim, num_heads, 1280)

    def get_num_layers(self):
        return len(self.blocks)

    def construct(self, x, token_num):
        # codebook_fea
        # B, nc, w, h = codebook_fea.shape
        x = self.in_proj(x)
        B = len(token_num)
        num_tokens, C = x.shape

        x_list = ops.split(x, token_num.tolist(), axis=0)
        max_token_num = token_num.max().item()
        x_pad = ops.zeros(B, max_token_num, C, dtype=x.dtype)
        mask = ops.zeros(B, max_token_num, dtype=x.dtype)

        for i, x_tensor in enumerate(x_list):
            x_pad[i][: len(x_tensor)] = x_tensor
            mask[i][: len(x_tensor)] = 1

        x_pad = x_pad + self.pos_embed[:, :max_token_num]
        x_pad = self.pos_drop(x_pad)

        query_embeds = self.query_embed.broadcast_to((B, -1, -1))

        for blk in self.blocks:
            query_embeds = blk(query_embeds, codebook_embeds=x_pad, codebook_mask=mask)

        query_embeds = self.norm(query_embeds)  # To align with the raw vit features

        visual_rec = self.decode_task_layer(query_embeds)

        encoder_hidden_1 = self.unet_proj_1(visual_rec)  # [bs, 256, 768]
        encoder_hidden_2 = self.unet_proj_2(visual_rec)  # [bs, 256, 1280]
        prompt_embeds = ops.cat([encoder_hidden_1, encoder_hidden_2], axis=-1)  # [bs, 256, 2048]
        pooled_prompt_embeds = self.unet_attnpool(visual_rec)  # [bs, 1280]

        return prompt_embeds, pooled_prompt_embeds


def build_tokenizer_decoder(model_path="", pixel_decoding="highres"):
    if pixel_decoding == "lowres":
        model = VQDecoder(depth=12)
        weight_path = os.path.join(model_path, "visual_tokenizer", "tokenizer_decoder.bin")
    else:
        model = HighresVQDecoder(depth=12)
        weight_path = os.path.join(model_path, "visual_tokenizer", "highres_tokenizer_decoder.bin")

    print(f"Load visual tokenizer decoder weight from {weight_path}")
    state_dict = load_torch_state_dict_to_ms_ckpt(weight_path)
    param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict)
    print(f"param_not_load:{param_not_load}")
    print(f"uckpt_not_load: {ckpt_not_load}")
    return model
