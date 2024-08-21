import numbers
import os
from functools import partial
from typing import Tuple

import numpy as np
from models.modeling_visual_encoder import build_eva_clip
from utils import load_torch_state_dict_to_ms_ckpt

import mindspore as ms
from mindspore import Parameter, mint, nn, ops

from mindone.models.utils import constant_, trunc_normal_


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
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
        drop=0.0,
        subln=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.ffn_ln(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


def l2norm(t, eps=1e-8):
    out = t / (ops.norm(t, dim=-1, keepdim=True) + eps)
    return out


class CodebookEmbedding(nn.Cell):
    def __init__(self, num_tokens, codebook_dim):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        weight = ops.randn(num_tokens, codebook_dim)
        weight = l2norm(weight)
        self.weight = Parameter(weight)

    def construct(self, embed_id):
        return mint.nn.functional.embedding(embed_id, self.weight)


class VectorQuantizer(nn.Cell):
    def __init__(self, n_embed, embedding_dim):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.embedding = CodebookEmbedding(self.num_tokens, self.codebook_dim)

    def tokenize(self, z):
        z = l2norm(z)
        z_flattened = z.reshape(-1, self.codebook_dim)

        d = (
            z_flattened.pow(2).sum(axis=1, keepdims=True)
            + self.embedding.weight.pow(2).sum(axis=1)
            - 2 * z_flattened @ self.embedding.weight.swapaxes(0, 1)
        )  # 'n d -> d n'

        encoding_indices = ops.argmin(d, axis=1)

        z_q = self.embedding(encoding_indices)  # [np, d]

        # encodings = ops.one_hot(encoding_indices, self.num_tokens).to(z.dtype)   # [np, 16384]

        return z_q, encoding_indices

    def get_quantize_from_id(self, encoding_indices):
        z_q = self.embedding(encoding_indices)  # [np, d]
        return z_q


class TokenCrossAttention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.query = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.key = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.value = nn.Dense(dim, dim, has_bias=qkv_bias)

        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N = policy.shape
        B, H, N, N = attn.shape
        fuse_policy = 1 - policy  # Each token only attend to the dropped tokens
        attn_policy = fuse_policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        attn_policy = mint.broadcast_to(attn_policy, (B, 1, N, N))
        max_att = ops.max(attn, axis=-1, keepdims=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(ms.float32).exp() * attn_policy.to(ms.float32)
        attn = (attn + eps / N) / (attn.sum(axis=-1, keepdims=True) + eps)

        return attn.type_as(max_att)

    def construct(self, x, x_origin, decisions):
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(x_origin).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(x_origin).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.swapaxes(-2, -1)) * self.scale
        attn = self.softmax_with_policy(attn, decisions)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TokenCausalAttention(nn.Cell):
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

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N = policy.shape

        assert attn.shape[-1] == attn.shape[-2]
        assert attn.shape[-2] == N
        B, H, N, N = attn.shape

        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = ops.eye(N, dtype=attn_policy.dtype).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye

        # Use the causal attention
        seq_ids = ops.arange(N)
        causal_mask = seq_ids[None, None, :].repeat(B, axis=0).repeat(N, axis=1) <= seq_ids[None, :, None]
        causal_mask = causal_mask[:, None, :, :].to(attn_policy.dtype)
        attn_policy = attn_policy * causal_mask

        max_att = ops.max(attn, axis=-1, keepdims=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(ms.float32).exp() * attn_policy.to(ms.float32)
        attn = (attn + eps / N) / (attn.sum(axis=-1, keepdims=True) + eps)
        return attn.type_as(max_att)

    def construct(self, x, decisions):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.swapaxes(-2, -1)) * self.scale

        if decisions is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, decisions)

        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CausalFuserBlock(nn.Cell):
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
        norm_layer=partial(LayerNorm, eps=1e-5),
    ):
        super().__init__()

        self.norm0 = norm_layer(dim)
        self.token_causal_attn = TokenCausalAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm1 = norm_layer(dim)
        self.token_cross_attn = TokenCrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x, x_origin, decisions):
        x = x + self.token_causal_attn(self.norm0(x), decisions)
        x = x + self.token_cross_attn(self.norm1(x), x_origin, decisions)
        x = x + self.mlp(self.norm2(x))
        return x


class TokenMerger(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads,
        depth=1,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
    ):
        super().__init__()
        self.blocks = nn.CellList(
            [
                CausalFuserBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.ln_vision = norm_layer(dim)

        self.norm = norm_layer(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Dense) and m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            constant_(m.bias, 0)
            constant_(m.weight, 1.0)

    def construct(self, x, decisions):
        x_origin = self.ln_vision(x)  # the  raw vit features needs layer normalization

        for blk in self.blocks:
            x = blk(x, x_origin, decisions)

        x = self.norm(x)  # the post norm, for next stage use

        return x


class TokenPredictor(nn.Cell):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.SequentialCell(LayerNorm(embed_dim, eps=1e-5), nn.Dense(embed_dim, embed_dim), nn.GELU())

        self.out_conv = nn.SequentialCell(
            nn.Dense(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dense(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dense(embed_dim // 4, 2),
            nn.LogSoftmax(axis=-1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Dense) and m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            constant_(m.bias, 0)
            constant_(m.weight, 1.0)

    def construct(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.shape
        local_x = x[:, :, : C // 2]
        global_x = (x[:, :, C // 2 :] * policy).sum(axis=1, keepdims=True) / ops.sum(policy, dim=1, keepdim=True)
        x = ops.cat([local_x, mint.broadcast_to(global_x, (B, N, C // 2))], axis=-1)
        return self.out_conv(x)


class DynamicVisualTokenizer(nn.Cell):
    def __init__(
        self,
        img_size=224,
        patch_size=14,
        width=1408,
        layers=12,
        heads=16,
        n_code=16384,
        code_dim=32,
        model_path="",
    ):
        """
        The dynamic visual tokenizer in LaVIT, it has 12 transformer blocks
        """
        super().__init__()

        self.encoder = build_eva_clip(model_path=model_path)
        self.encoder.set_train(False)
        # Freeze the vit encoder
        for param in self.encoder.get_parameters():
            param.requires_grad = False  # fix encoder model

        encoder_config = dict(
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
        )

        encoder_config["img_size"] = img_size
        encoder_config["patch_size"] = patch_size
        encoder_config["embed_dim"] = width
        encoder_config["depth"] = layers
        encoder_config["num_heads"] = heads

        # The token predictor
        self.token_predictor = TokenPredictor(encoder_config["embed_dim"])

        # The token merger
        self.causal_encoder = TokenMerger(
            encoder_config["embed_dim"],
            num_heads=encoder_config["num_heads"],
            depth=encoder_config["depth"],
            mlp_ratio=encoder_config["mlp_ratio"],
            qkv_bias=encoder_config["qkv_bias"],
            qk_scale=encoder_config["qk_scale"],
            drop=encoder_config["drop_rate"],
            attn_drop=encoder_config["attn_drop_rate"],
        )

        # The code book embeddings
        self.quantize = VectorQuantizer(n_embed=n_code, embedding_dim=code_dim)

        # encoder task layer, map the feature to the codebook's dimension
        self.encode_task_layer = nn.SequentialCell(
            nn.Dense(encoder_config["embed_dim"], encoder_config["embed_dim"]),
            nn.Tanh(),
            nn.Dense(encoder_config["embed_dim"], code_dim),  # for quantize
        )

        # The vit projection, map the visual feature to LLM's input space
        llm_embed_dim = 4096  # LLaMA 7B's embedding dimension: 4096
        self.vit_proj = nn.Dense(width, llm_embed_dim)

    def encode_features(self, x):
        """
        x: B, 3, H, W
        Usage: Given the input image, encode the visual features for the LLM, without quantization,
            Used for Understanding
        """

        encoder_features = self.encoder(x, return_all_features=True)  # N, 257, D
        encoder_features = encoder_features[:, 1:, :]

        B, num_patches, _ = encoder_features.shape
        mask = ops.ones(B, num_patches, 1, dtype=encoder_features.dtype)

        # To evalaute the score
        pred_score = self.token_predictor(encoder_features.to(ms.float32), mask).reshape(B, -1, 2)
        # Sample from the score distribution
        hard_keep_decision = ops.gumbel_softmax(pred_score, hard=True)[:, :, 0]  # [N, num_patches]

        # Update the existed features from dropped tokens (To remain the information flow)
        updated_features = self.causal_encoder(encoder_features, hard_keep_decision)
        updated_features = self.vit_proj(updated_features)  # [bs, 256, 4096]

        B, N, C = updated_features.shape
        index_select = hard_keep_decision.long()

        token_num = index_select.sum(axis=-1)
        index_select = index_select.bool()

        remained_token = ops.masked_select(updated_features, index_select[:, :, None])
        remained_token = remained_token.reshape(-1, C)  # [Num Patch]
        remained_token_list = ops.split(remained_token, token_num.tolist())  # [bs]
        remained_token_list = list(remained_token_list)

        return remained_token_list

    def tokenize_image(self, x_tensor, add_special=False, used_for_llm=True):
        # x_tensor: [bs, 3, h, w]
        feature_targets = self.encoder(x_tensor, return_all_features=True)  # N, 257, D
        encoder_features = feature_targets[:, 1:, :]

        B, num_patches, _ = encoder_features.shape
        mask = ops.ones(B, num_patches, 1, dtype=encoder_features.dtype)

        pred_score = self.token_predictor(encoder_features.to(ms.float32), mask).reshape(B, -1, 2)
        # Sample from the score distribution
        hard_keep_decision = ops.gumbel_softmax(pred_score, hard=True)[:, :, 0]  # [N, num_patches]

        # Update the existed features from dropped tokens (To remain the information flow)
        updated_features = self.causal_encoder(encoder_features, hard_keep_decision)

        B, N, C = updated_features.shape
        index_select = hard_keep_decision.long()
        token_nums = index_select.sum(axis=-1)
        index_select = index_select.bool()
        remained_token = ops.masked_select(updated_features, index_select[:, :, None]).reshape(-1, C)  # [Num Patch]

        to_quantizer_features = self.encode_task_layer(remained_token.type_as(self.encode_task_layer[-1].weight))
        quantize, embed_ind = self.quantize.tokenize(to_quantizer_features)

        if not used_for_llm:
            return quantize, token_nums

        embed_ind = embed_ind + 32002
        embed_ind_list = ops.split(embed_ind, token_nums.tolist(), dim=0)

        if add_special:
            # If pad the special image start and end tokens, default is False
            output_embed_ind = []
            image_special = ms.Tensor([32000, 32001], dtype=ms.int32)
            for ele in embed_ind_list:
                output_embed_ind.append(ops.cat([image_special[:1], ele, image_special[1:]]))
            return output_embed_ind

        return embed_ind_list


def build_dynamic_tokenizer(model_path="", for_understanding=False, model_sub_dir="language_model"):
    model = DynamicVisualTokenizer(model_path=model_path)
    weight_path = os.path.join(model_path, "visual_tokenizer", "tokenizer_encoder.bin")
    print(f"Load visual tokenizer encoder weight from {weight_path}")
    state_dict = load_torch_state_dict_to_ms_ckpt(weight_path)
    param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict)
    print(f"param_not_load: {param_not_load}")
    print(f"ckpt_not_load: {ckpt_not_load}")

    if for_understanding:
        # For Understanding, the LaVIT use the continuous visual features,
        # so needs to load the token merger weight trained with LLM
        visual_weight_path = os.path.join(model_path, model_sub_dir, "visual_weight.bin")
        print(f"For multi-modal understanding, Load visual tokenizer weight from {visual_weight_path}")
        state_dict = load_torch_state_dict_to_ms_ckpt(visual_weight_path)
        param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict)
        print(f"param_not_load: {param_not_load}")
        print(f"ckpt_not_load: {ckpt_not_load}")

    return model
