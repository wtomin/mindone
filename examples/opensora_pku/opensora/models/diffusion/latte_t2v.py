import json
import logging
import numbers
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import initializer

from mindone.diffusers.configuration_utils import ConfigMixin, register_to_config
from mindone.diffusers.models.activations import GEGLU, GELU, ApproximateGELU

# from mindone.diffusers.utils import USE_PEFT_BACKEND
from mindone.diffusers.models.embeddings import (
    ImagePositionalEmbeddings,
    PatchEmbed,
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
    get_1d_sincos_pos_embed_from_grid,
)
from mindone.diffusers.models.modeling_utils import ModelMixin
from mindone.diffusers.models.normalization import AdaLayerNorm, AdaLayerNormZero
from mindone.models.modules.flash_attention import FLASH_IS_AVAILABLE, MSFlashAttention

from .activations import GELU as SP_GELU
from .activations import ApproximateGELU as SP_ApproximateGELU
from .flash_attention import FlashAttentionSP

# from mindone.diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear

logger = logging.getLogger(__name__)


def get_1d_sincos_pos_embed(embed_dim, length, interpolation_scale=1.0, base_size=16):
    pos = np.arange(0, length)[:, None] / interpolation_scale
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    return pos_embed


class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine: bool = True, dtype=ms.float32):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gamma = Parameter(initializer("ones", normalized_shape, dtype=dtype))
            self.beta = Parameter(initializer("zeros", normalized_shape, dtype=dtype))
        else:
            self.gamma = ops.ones(normalized_shape, dtype=dtype)
            self.beta = ops.zeros(normalized_shape, dtype=dtype)
        self.layer_norm = ops.LayerNorm(-1, -1, epsilon=eps)

    def construct(self, x: ms.Tensor):
        oridtype = x.dtype
        x, _, _ = self.layer_norm(x.to(ms.float32), self.gamma.to(ms.float32), self.beta.to(ms.float32))
        return x.to(oridtype)


class Attention(nn.Cell):
    def __init__(self, dim_head, attn_drop=0.0, upcast_attention=False, upcast_softmax=True):
        super().__init__()
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.scale = dim_head**-0.5
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

    def construct(self, q, k, v, mask=None):
        if self.upcast_attention:
            q, k, v = [x.astype(ms.float32) for x in (q, k, v)]
        sim = ops.matmul(q, self.transpose(k, (0, 2, 1))) * self.scale
        if self.upcast_softmax:
            sim = sim.astype(ms.float32)
        if mask is not None:
            sim += mask

        # use fp32 for exponential inside
        attn = self.softmax(sim).astype(v.dtype)
        attn = self.attn_drop(attn)

        out = ops.matmul(attn, v)

        return out


class SeqParallelAttention(nn.Cell):
    def __init__(
        self,
        num_heads: int,
        dim_head: int,
        attn_drop: float = 0.0,
        has_mask: bool = False,
        parallel_config: Dict[str, Any] = {},
        upcast_attention=False,
        upcast_softmax=True,
    ) -> None:
        super().__init__()
        self.scale = ms.Tensor(dim_head**-0.5, dtype=ms.float32)
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.has_mask = has_mask
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.bmm = ops.BatchMatMul()
        self.mul = ops.Mul()
        self.softmax = ops.Softmax(axis=-1)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.matmul = ops.BatchMatMul()
        self.transpose = ops.Transpose()
        self.transpose_a2a = ops.Transpose()

        self.one = ms.Tensor(1, dtype=ms.float32)

        if self.has_mask:
            self.sub = ops.Sub()
            self.mul_mask = ops.Mul()
            self.add = ops.Add()

        self.minus_inf = Tensor(np.finfo(np.float32).min, dtype=ms.float32)

        self.parallel_config = parallel_config
        self.shard()

    def _merge_head(self, x: Tensor) -> Tensor:
        x = self.transpose(x, (0, 3, 1, 2, 4))  # (b, n, h/mp, mp, d)
        x = self.transpose_a2a(x, (0, 1, 3, 2, 4))
        x = ops.reshape(x, (-1, self.num_heads * self.dim_head))
        return x

    def construct(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # mask: (b 1 1 1 n_k), 1 - keep, 0 indicates discard.
        if self.upcast_attention:
            q, k, v = [x.astype(ms.float32) for x in (q, k, v)]
        sim = self.bmm(q, k)
        sim = self.mul(sim, self.scale)
        if self.upcast_softmax:
            sim = sim.astype(ms.float32)

        if mask is not None:
            assert self.has_mask
            mask = self.sub(self.one, mask.to(ms.float32))
            mask = self.mul_mask(mask, self.minus_inf)
            sim = self.add(mask, sim)

        attn = self.softmax(sim).astype(v.dtype)
        attn = self.attn_drop(attn)
        out = self.matmul(attn, v)
        out = self._merge_head(out)
        return out

    def shard(self):
        self.dp = self.parallel_config.get("data_parallel", 1)
        self.mp = self.parallel_config.get("model_parallel", 1)
        self.sp = self.parallel_config.get("sequence_parallel", 1)

        if self.sp > self.num_heads // self.mp:
            self.sp_ds = self.num_heads // self.mp
            self.sp_co = self.sp // self.sp_ds
        else:
            self.sp_ds = self.sp
            self.sp_co = 1

        self.bmm.shard(((self.dp, self.sp_ds, self.mp, self.sp_co, 1), (self.dp, self.sp_ds, self.mp, 1, 1)))
        self.bmm.add_prim_attr(
            "layout",
            {
                "dev_matrix": (self.dp, self.sp_co, self.sp_ds, self.mp, 1),
                "input_tensor_map": ((4, 2, 1, 3, 0), (4, 2, 1, -1, 0)),
            },
        )

        self.mul.shard(((self.dp, self.sp_ds, self.mp, self.sp_co, 1), ()))
        self.mul.add_prim_attr(
            "layout",
            {"dev_matrix": (self.dp, self.sp_co, self.sp_ds, self.mp, 1), "input_tensor_map": ((4, 2, 1, 3, 0), ())},
        )

        self.softmax.shard(((self.dp, self.sp_ds, self.mp, self.sp_co, 1),))
        self.softmax.add_prim_attr(
            "layout",
            {"dev_matrix": (self.dp, self.sp_co, self.sp_ds, self.mp, 1), "input_tensor_map": ((4, 2, 1, 3, 0),)},
        )

        self.attn_drop.dropout.shard(((self.dp, self.sp_ds, self.mp, self.sp_co, 1),))
        self.attn_drop.dropout.add_prim_attr(
            "layout",
            {"dev_matrix": (self.dp, self.sp_co, self.sp_ds, self.mp, 1), "input_tensor_map": ((4, 2, 1, 3, 0),)},
        )

        self.matmul.shard(((self.dp, self.sp_ds, self.mp, self.sp_co, 1), (self.dp, self.sp_ds, self.mp, 1, 1)))
        self.matmul.add_prim_attr(
            "layout",
            {
                "dev_matrix": (self.dp, self.sp_co, self.sp_ds, self.mp, 1),
                "input_tensor_map": ((4, 2, 1, 3, 0), (4, 2, 1, -1, 0)),
            },
        )

        self.transpose.shard(((self.dp, self.sp_ds, self.mp, self.sp_co, 1),))
        self.transpose.add_prim_attr(
            "layout",
            {"dev_matrix": (self.dp, self.sp_co, self.sp_ds, self.mp, 1), "input_tensor_map": ((4, 2, 1, 3, 0),)},
        )

        self.transpose_a2a.shard(((self.dp, self.sp, 1, self.mp, 1),))

        if self.has_mask:
            self.sub.shard(((), (self.dp, 1, 1, self.sp_co, 1)))

            self.mul_mask.shard(((self.dp, 1, 1, self.sp_co, 1), ()))

            self.add.shard(((self.dp, 1, 1, self.sp_co, 1), (self.dp, self.sp_ds, self.mp, self.sp_co, 1)))
            self.add.add_prim_attr(
                "layout",
                {
                    "dev_matrix": (self.dp, self.sp_co, self.sp_ds, self.mp, 1),
                    "input_tensor_map": ((4, -1, -1, 3, 0), (4, 2, 1, 3, 0)),
                },
            )


class SpatialNorm(nn.Cell):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def construct(self, f: ms.Tensor, zq: ms.Tensor) -> ms.Tensor:
        f_size = f.shape[-2:]
        zq = ops.ResizeNearestNeighbor(size=f_size)(zq)
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


class MultiHeadAttention(nn.Cell):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to True):
            Set to `True` to upcast the softmax computation to `float32`.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = True,
        out_bias: bool = True,
        only_cross_attention: bool = False,
        dtype=ms.float32,
        enable_flash_attention=False,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.dropout = dropout
        self.heads = heads
        self.dtype = dtype

        self.only_cross_attention = only_cross_attention

        self.to_q = nn.Dense(query_dim, self.inner_dim, has_bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Dense(self.cross_attention_dim, self.inner_dim, has_bias=bias)
            self.to_v = nn.Dense(self.cross_attention_dim, self.inner_dim, has_bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.to_out = nn.SequentialCell(nn.Dense(self.inner_dim, query_dim, has_bias=out_bias), nn.Dropout(p=dropout))

        self.enable_flash_attention = (
            enable_flash_attention and FLASH_IS_AVAILABLE and (ms.context.get_context("device_target") == "Ascend")
        )

        if self.enable_flash_attention:
            self.flash_attention = MSFlashAttention(
                head_dim=dim_head, head_num=heads, fix_head_dims=[72], attention_dropout=attn_drop
            )
        else:
            self.attention = Attention(
                dim_head=dim_head, attn_drop=attn_drop, upcast_attention=upcast_attention, upcast_softmax=upcast_softmax
            )

    @staticmethod
    def _rearange_in(x, h):
        # (b, n, h*d) -> (b*h, n, d)
        b, n, d = x.shape
        d = d // h

        x = ops.reshape(x, (b, n, h, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b * h, n, d))
        return x

    @staticmethod
    def _rearange_out(x, h):
        # (b*h, n, d) -> (b, n, h*d)
        b, n, d = x.shape
        b = b // h

        x = ops.reshape(x, (b, h, n, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, n, h * d))
        return x

    def construct(
        self,
        x,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
    ):
        x_dtype = x.dtype
        h = self.heads
        mask = attention_mask

        q = self.to_q(x)
        context = encoder_hidden_states if encoder_hidden_states is not None else x

        k = self.to_k(context)
        v = self.to_v(context)
        q_b, q_n, _ = q.shape  # (b n h*d)
        k_b, k_n, _ = k.shape
        v_b, v_n, _ = v.shape

        # # convert sequence mask to attention mask: (b, q_n) to (b, q_n, k_n)
        # if mask is not None:
        #     mask = self.reshape(mask, (mask.shape[0], -1))
        #     attn_mask = ops.zeros((q_b, q_n, k_n), self.dtype)
        #     mask = ops.expand_dims(mask, axis=1)  # (q_b, 1, k_n)
        #     attn_mask = attn_mask.masked_fill(~mask, -ms.numpy.inf)
        #     mask = attn_mask

        if self.enable_flash_attention:
            # reshape qkv shape ((b n h*d) -> (b h n d))and mask dtype for FA input format
            q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
            k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
            v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)
            if mask is not None and mask.dim() == 3:
                # (b, 1, k_n) - > (b, q_n, k_n), manual broadcast
                if mask.shape[-2] == 1:
                    mask = mask.repeat(q_n, axis=-2)
                mask = ops.expand_dims(mask, axis=1)  # (q_b, 1, q_n, k_n)
            out = self.flash_attention(q, k, v, mask)
            b, h, n, d = out.shape
            # reshape FA output to original attn input format, (b h n d) -> (b n h*d)
            out = out.transpose(0, 2, 1, 3).view(b, n, -1)
        else:
            # (b, n, h*d) -> (b*h, n, d)
            q = self._rearange_in(q, h)
            k = self._rearange_in(k, h)
            v = self._rearange_in(v, h)
            if mask is not None and mask.shape[0] != q.shape[0]:
                mask = mask.repeat(h, axis=0)

            out = self.attention(q, k, v, mask)
            # (b*h, n, d) -> (b, n, h*d)
            out = self._rearange_out(out, h)

        return self.to_out(out).to(x_dtype)


class SeqParallelMultiHeadAttention(nn.Cell):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = True,
        out_bias: bool = True,
        only_cross_attention: bool = False,
        dtype=ms.float32,
        enable_flash_attention=False,
        parallel_config: Dict[str, Any] = {},
    ):
        super().__init__()
        assert query_dim % heads == 0, "query dim must be divisible by num_heads"

        self.inner_dim = dim_head * heads
        self.head_dim = dim_head
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.dropout = dropout
        self.heads = heads
        self.dtype = dtype

        self.only_cross_attention = only_cross_attention
        self.parallel_config = parallel_config
        self.enable_flash_attention = enable_flash_attention
        self.has_bias = bias

        self.to_q = nn.Dense(query_dim, self.inner_dim, has_bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Dense(self.cross_attention_dim, self.inner_dim, has_bias=bias)
            self.to_v = nn.Dense(self.cross_attention_dim, self.inner_dim, has_bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.to_out = nn.SequentialCell(nn.Dense(self.inner_dim, query_dim, has_bias=out_bias), nn.Dropout(p=dropout))

        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.transpose_a2a = ops.Transpose()
        self.merge_head_transpose_a2a = ops.Transpose()
        self.tile = ops.Tile()
        self.tile_fa = ops.Tile()

        self.pad = ops.Pad(((0, 0), (0, 0), (0, 0), (0, 8)))
        # FIXME: stride_slice does not support non-zero mask in semi-parallel mode? Remove it once FA supports dim=72.
        self.stride_slice = ops.StridedSlice(15, 7, 0, 0, 0)  # for head_dim=72 only
        self.shard()
        self.enable_flash_attention = (
            enable_flash_attention and FLASH_IS_AVAILABLE and (ms.context.get_context("device_target") == "Ascend")
        )
        if self.enable_flash_attention:
            self.attention = FlashAttentionSP(
                head_num=dim_head,
                keep_prob=1 - attn_drop,
                scale_value=dim_head**-0.5,
                input_layout="BSH",
                use_attention_mask=True,
                dp=self.dp,
                mp=self.sp_ds * self.mp,
                sp=self.sp_co,
            )
        else:
            self.attention = SeqParallelAttention(
                self.heads,
                dim_head,
                attn_drop=attn_drop,
                has_mask=True,
                parallel_config=parallel_config,
                upcast_attention=upcast_attention,
                upcast_softmax=upcast_softmax,
            )

    def _rearange_in(self, x, b, n, h, transpose=False):
        # (b*n, h*d) -> (b, h/mp, mp, n, d)
        x = ops.reshape(x, (b, n, self.mp, h // self.mp, -1))
        x = self.transpose_a2a(x, (0, 1, 3, 2, 4))
        if not transpose:
            x = self.transpose(x, (0, 2, 3, 1, 4))
        else:
            x = self.transpose(x, (0, 2, 3, 4, 1))
        return x

    def _rearange_in_fa(self, x, b, n, h):
        # (b*n, h*d) -> (b, n, h*d)
        if self.sp_ds > 1:
            # (b*n, h*d) -> (b, n, h/mp, mp, d)
            x = ops.reshape(x, (b, n, self.mp, h // self.mp, -1))
            x = self.transpose_a2a(x, (0, 1, 3, 2, 4))
            x = self.transpose(x, (0, 1, 2, 3, 4))
        x = ops.reshape(x, (b, n, h, -1))
        # x = self.pad(x, (0, 8), 0)
        x = self.pad(x)
        x = ops.reshape(x, (b, n, -1))
        return x

    def _rearange_out_fa(self, x, b, n, h):
        # (b, n, d) -> (b*n, h*d)
        if self.sp_ds > 1:
            x = ops.reshape(x, (b, n, h // self.mp, self.mp, -1))
            x = self.transpose(x, (0, 1, 2, 3, 4))
            x = self.merge_head_transpose_a2a(x, (0, 1, 3, 2, 4))
        x = ops.reshape(x, (b, n, h, -1))
        x = self.stride_slice(x, (0, 0, 0, 0), (0, 0, 0, self.head_dim), (1, 1, 1, 1))
        x = ops.reshape(x, (b * n, -1))
        return x

    def construct(
        self,
        x,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
    ):
        """
        Inputs:
            x: (B, N, C), N=seq_len=h*w*t, C = hidden_size = head_dim * num_heads
            encoder_hidden_states: (1, B*N_tokens, C) (B, N_tokens, C)
            attention_mask : (B, N_tokens), 1 - valid tokens, 0 - padding tokens
        Return:
            (B, N, C)
        """
        x_dtype = x.dtype
        h = self.heads
        mask = attention_mask

        b, n, d = x.shape
        context = encoder_hidden_states if encoder_hidden_states is not None else x
        n_c = context.shape[1]

        x = ops.reshape(x, (-1, x.shape[-1]))  # (B*N, C)
        context = ops.reshape(context, (-1, context.shape[-1]))  # (B*N, C)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        if not self.enable_flash_attention:
            q = self._rearange_in(q, b, n, h)
            k = self._rearange_in(k, b, n_c, h, transpose=True)
            v = self._rearange_in(v, b, n_c, h)
            if mask is not None:
                mask = ops.reshape(mask, (b, 1, 1, 1, n_c))
                mask = self.tile(mask, (1, 1, 1, n, 1))
            out = self.attention(q, k, v, mask)
        else:
            q = self._rearange_in_fa(q, b, n, h).to(ms.float16)
            k = self._rearange_in_fa(k, b, n_c, h).to(ms.float16)
            v = self._rearange_in_fa(v, b, n_c, h).to(ms.float16)
            if mask is not None:
                mask = ops.reshape(mask, (b, 1, 1, n_c))
                mask = self.tile_fa(mask, (1, 1, n, 1)).to(ms.uint8)
                mask = ops.stop_gradient(mask)
            out = self.attention(q, k, v, mask)
            out = self._rearange_out_fa(out, b, n, h).to(x.dtype)

        return self.to_out(out).to(x_dtype)

    def shard(self):
        self.dp = self.parallel_config.get("data_parallel", 1)
        self.mp = self.parallel_config.get("model_parallel", 1)
        self.sp = self.parallel_config.get("sequence_parallel", 1)

        if self.sp > self.heads // self.mp:
            self.sp_ds = self.heads // self.mp
            self.sp_co = self.sp // self.sp_ds
        else:
            self.sp_ds = self.sp
            self.sp_co = 1

        self.to_q.matmul.shard(((self.dp * self.sp, 1), (self.mp, 1)))
        self.to_k.matmul.shard(((self.dp * self.sp, 1), (self.mp, 1)))
        self.to_v.matmul.shard(((self.dp * self.sp, 1), (self.mp, 1)))
        if self.has_bias:
            self.to_q.bias_add.shard(((self.dp * self.sp, self.mp), (self.mp,)))
            self.to_k.bias_add.shard(((self.dp * self.sp, self.mp), (self.mp,)))
            self.to_v.bias_add.shard(((self.dp * self.sp, self.mp), (self.mp,)))

        self.transpose_a2a.shard(((self.dp, self.sp, self.mp, 1, 1),))
        self.transpose.shard(((self.dp, self.sp_co, self.sp_ds, self.mp, 1),))
        self.merge_head_transpose_a2a.shard(((self.dp, self.sp, 1, self.mp, 1),))

        self.tile.shard(((self.dp, 1, 1, self.sp_co, 1),))
        self.tile_fa.shard(((self.dp, 1, self.sp_co, 1),))

        self.to_out[0].matmul.shard(((self.dp * self.sp, self.mp), (1, self.mp)))
        self.to_out[0].bias_add.shard(((self.dp * self.sp, 1), (1,)))

        self.to_out[1].dropout.shard(((self.dp * self.sp, 1),))

        self.pad.shard(((self.dp, self.sp_co, self.sp_ds * self.mp, 1),))
        self.stride_slice.shard(((self.dp, self.sp, self.mp, 1),))


class CaptionProjection(nn.Cell):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, num_tokens=120):
        super().__init__()
        self.linear_1 = nn.Dense(in_features, hidden_size)
        self.act_1 = nn.GELU(True)
        self.linear_2 = nn.Dense(hidden_size, hidden_size)
        self.y_embedding = Parameter(ops.randn(num_tokens, in_features) / in_features**0.5, requires_grad=False)

    def construct(self, caption, force_drop_ids=None):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CombinedTimestepSizeEmbeddings(nn.Cell):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.use_additional_conditions = True
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)

    def apply_condition(self, size: ms.Tensor, batch_size: int, embedder: nn.Cell):
        if size.ndim == 1:
            size = size[:, None]

        if size.shape[0] != batch_size:
            size = size.repeat_interleave(batch_size // size.shape[0], 1)
            if size.shape[0] != batch_size:
                raise ValueError(f"`batch_size` should be {size.shape[0]} but found {batch_size}.")

        current_batch_size, dims = size.shape[0], size.shape[1]
        size = size.reshape(-1)
        size_freq = self.additional_condition_proj(size).to(size.dtype)

        size_emb = embedder(size_freq)
        size_emb = size_emb.reshape(current_batch_size, dims * self.outdim)
        return size_emb

    def construct(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            resolution = self.apply_condition(resolution, batch_size=batch_size, embedder=self.resolution_embedder)
            aspect_ratio = self.apply_condition(
                aspect_ratio, batch_size=batch_size, embedder=self.aspect_ratio_embedder
            )
            conditioning = timesteps_emb + ops.cat([resolution, aspect_ratio], axis=1)
        else:
            conditioning = timesteps_emb

        return conditioning


class AdaLayerNormSingle(nn.Cell):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        super().__init__()

        self.emb = CombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        self.linear = nn.Dense(embedding_dim, 6 * embedding_dim)

    def construct(
        self,
        timestep: ms.Tensor,
        added_cond_kwargs: Dict[str, ms.Tensor] = None,
        batch_size: int = None,
        hidden_dtype=None,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        # No modulation happening here.
        embedded_timestep = self.emb(
            timestep, batch_size=batch_size, hidden_dtype=hidden_dtype, resolution=None, aspect_ratio=None
        )
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class GatedSelfAttentionDense(nn.Cell):
    r"""
    A gated self-attention dense layer that combines visual features and object features.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        context_dim (`int`): The number of channels in the context.
        n_heads (`int`): The number of heads to use for attention.
        d_head (`int`): The number of channels in each head.
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Dense(context_dim, query_dim)

        self.attn = MultiHeadAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = LayerNorm(query_dim)
        self.norm2 = LayerNorm(query_dim)

        self.alpha_attn = ms.Tensor(0.0)
        self.alpha_dense = ms.Tensor(0.0)

        self.enabled = True

    def construct(self, x: ms.Tensor, objs: ms.Tensor) -> ms.Tensor:
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.alpha_attn.tanh() * self.attn(self.norm1(ops.cat([x, objs], dim=1)))[:, :n_visual, :]
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x


class FeedForward(nn.Cell):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = nn.Dense

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)

        self.net = nn.CellList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(p=dropout))
        # project out
        self.net.append(linear_cls(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(p=dropout))

    def construct(self, hidden_states: ms.Tensor, scale: float = 1.0) -> ms.Tensor:
        compatible_cls = GEGLU
        for module in self.net:
            if isinstance(module, compatible_cls):
                hidden_states = module(hidden_states, scale)
            else:
                hidden_states = module(hidden_states)
        return hidden_states


class SeqParallelFeedForward(nn.Cell):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        parallel_config: Dict[str, Any] = {},
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = nn.Dense
        self.final_dropout = final_dropout
        self.activation_fn = activation_fn

        if activation_fn == "gelu":
            raise NotImplementedError
        if activation_fn == "gelu-approximate":
            act_fn = SP_GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            raise NotImplementedError
        elif activation_fn == "geglu-approximate":
            act_fn = SP_ApproximateGELU(dim, inner_dim)

        self.net = nn.CellList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(p=dropout))
        # project out
        self.net.append(linear_cls(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if self.final_dropout:
            self.net.append(nn.Dropout(p=dropout))

        self.parallel_config = parallel_config
        self.shard()

    def construct(self, hidden_states: ms.Tensor, scale: float = 1.0) -> ms.Tensor:
        compatible_cls = GEGLU
        for module in self.net:
            if isinstance(module, compatible_cls):
                hidden_states = module(hidden_states, scale)
            else:
                hidden_states = module(hidden_states)
        return hidden_states

    def shard(self):
        self.dp = self.parallel_config.get("data_parallel", 1)
        self.mp = self.parallel_config.get("model_parallel", 1)
        self.sp = self.parallel_config.get("sequence_parallel", 1)
        if self.activation_fn == "gelu-approximate":
            self.net[0].proj.matmul.shard(((self.dp * self.sp, self.mp), (1, self.mp)))
            self.net[0].proj.bias_add.shard(((self.dp * self.sp, 1), (1,)))
            self.net[0].gelu.shard(((self.dp * self.sp, self.mp),))
        elif self.activation_fn == "geglu-approximate":
            self.net[0].proj.matmul.shard(((self.dp * self.sp, self.mp), (1, self.mp)))
            self.net[0].proj.bias_add.shard(((self.dp * self.sp, 1), (1,)))
            self.net[0].sigmoid.shard(((self.dp * self.sp, self.mp),))
        else:
            raise NotImplementedError

        self.net[1].dropout.shard(((self.dp * self.sp, 1),))
        self.net[2].matmul.shard(((self.dp * self.sp, self.mp), (1, self.mp)))
        self.net[2].bias_add.shard(((self.dp * self.sp, 1), (1,)))
        if self.final_dropout:
            self.net[-1].dropout.shard(((self.dp * self.sp, 1),))


class BasicTransformerBlock_(nn.Cell):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        enable_flash_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1_ada = AdaLayerNorm(dim, num_embeds_ada_norm)
            self.norm1_ada.norm = LayerNorm(dim, elementwise_affine=False)
        elif self.use_ada_layer_norm_zero:
            self.norm1_ada_zero = AdaLayerNormZero(dim, num_embeds_ada_norm)
            self.norm1_ada_zero.norm = LayerNorm(dim, elementwise_affine=False)
        else:
            self.norm1_ln = LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = MultiHeadAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            enable_flash_attention=enable_flash_attention,
        )

        self.norm3 = LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = ms.Parameter(ops.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        timestep: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1_ada(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1_ada_zero(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1_ln(hidden_states)
        elif self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, axis=1)
            norm_hidden_states = self.norm1_ln(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            # norm_hidden_states = norm_hidden_states.squeeze(1)  # error message
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        if "gligen" in cross_attention_kwargs:
            gligen_kwargs = cross_attention_kwargs["gligen"]
            # del cross_attention_kwargs["gligen"]
        else:
            gligen_kwargs = None
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.use_ada_layer_norm_single:
            # norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = self.norm3(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]}"
                    f"has to be divisible by chunk size: {self._chunk_size}. Make sure to set an"
                    f"appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = ops.cat(
                [
                    self.ff(hid_slice, scale=lora_scale)
                    for hid_slice in norm_hidden_states.chunk(num_chunks, axis=self._chunk_dim)
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class SeqParallelBasicTransformerBlock_(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        enable_flash_attention: bool = False,
        parallel_config={},
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1_ada = AdaLayerNorm(dim, num_embeds_ada_norm)
            self.norm1_ada.norm = LayerNorm(dim, elementwise_affine=False)
        elif self.use_ada_layer_norm_zero:
            self.norm1_ada_zero = AdaLayerNormZero(dim, num_embeds_ada_norm)
            self.norm1_ada_zero.norm = LayerNorm(dim, elementwise_affine=False)
        else:
            self.norm1_ln = LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = SeqParallelMultiHeadAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            enable_flash_attention=enable_flash_attention,
            parallel_config=parallel_config,
        )

        self.norm3 = LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = SeqParallelFeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            parallel_config=parallel_config,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = ms.Parameter(ops.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0
        self.add = ops.Add()
        self.mult = ops.Mul()
        self.split = ops.Split(axis=1, output_num=6)
        self.parallel_config = parallel_config
        self.shard()

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        timestep: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1_ada(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1_ada_zero(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1_ln(hidden_states)
        elif self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.split(
                self.add(self.scale_shift_table[None], timestep.reshape(batch_size, 6, -1))
            )
            norm_hidden_states = self.norm1_ln(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            # norm_hidden_states = norm_hidden_states.squeeze(1)  # error message
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        if "gligen" in cross_attention_kwargs:
            gligen_kwargs = cross_attention_kwargs["gligen"]
            # del cross_attention_kwargs["gligen"]
        else:
            gligen_kwargs = None
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = self.mult(gate_msa.unsqueeze(1), attn_output)
        elif self.use_ada_layer_norm_single:
            attn_output = self.mult(gate_msa, attn_output)

        hidden_states = self.add(attn_output, hidden_states)
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.use_ada_layer_norm_single:
            # norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = self.norm3(hidden_states)
            norm_hidden_states = self.add(self.mult(norm_hidden_states, (1 + scale_mlp)), shift_mlp)

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]}"
                    f"has to be divisible by chunk size: {self._chunk_size}. Make sure to set an"
                    f"appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = ops.cat(
                [
                    self.ff(hid_slice, scale=lora_scale)
                    for hid_slice in norm_hidden_states.chunk(num_chunks, axis=self._chunk_dim)
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = self.mult(gate_mlp.unsqueeze(1), ff_output)
        elif self.use_ada_layer_norm_single:
            ff_output = self.mult(gate_mlp, ff_output)

        hidden_states = self.add(ff_output, hidden_states)
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

    def shard(self):
        self.dp = self.parallel_config.get("data_parallel", 1)
        self.mp = self.parallel_config.get("model_parallel", 1)
        self.sp = self.parallel_config.get("sequence_parallel", 1)

        self.add.shard(((self.dp, self.sp, 1), (self.dp, self.sp, 1)))
        self.mult.shard(((self.dp, self.sp, 1), (self.dp, 1, 1)))

        self.split.shard(((self.dp, 1, 1),))

        if self.use_ada_layer_norm_single:
            # current only implement this layernorm
            self.norm1_ln.layer_norm.shard(((self.dp, self.sp, 1), (1,), (1,)))
            self.norm3.layer_norm.shard(((self.dp, self.sp, 1), (1,), (1,)))
        else:
            raise NotImplementedError


class BasicTransformerBlock(nn.Cell):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        enable_flash_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1_ada = AdaLayerNorm(dim, num_embeds_ada_norm)
            self.norm1_ada.norm = LayerNorm(dim, elementwise_affine=False)
        elif self.use_ada_layer_norm_zero:
            self.norm1_ada_zero = AdaLayerNormZero(dim, num_embeds_ada_norm)
            self.norm1_ada_zero.norm = LayerNorm(dim, elementwise_affine=False)
        else:
            self.norm1_ln = LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = MultiHeadAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            enable_flash_attention=enable_flash_attention,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if self.use_ada_layer_norm:
                self.norm2_ada = AdaLayerNorm(dim, num_embeds_ada_norm)
                self.norm2_ada.norm = LayerNorm(dim, elementwise_affine=False)
            else:
                self.norm2_ln = LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

            self.attn2 = MultiHeadAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                enable_flash_attention=enable_flash_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if not self.use_ada_layer_norm_single:
            self.norm3 = LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = ms.Parameter(ops.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        timestep: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]
        gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1_ada(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1_ada_zero(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1_ln(hidden_states)
        elif self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, axis=1)
            norm_hidden_states = self.norm1_ln(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            # norm_hidden_states = norm_hidden_states.squeeze(1)  # error message
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        if "gligen" in cross_attention_kwargs:
            gligen_kwargs = cross_attention_kwargs["gligen"]
            # del cross_attention_kwargs["gligen"]
        else:
            gligen_kwargs = None
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2_ada(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                norm_hidden_states = self.norm2_ln(hidden_states)
            elif self.use_ada_layer_norm_single:
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if not self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm2_ln(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class SeqParallelBasicTransformerBlock(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        enable_flash_attention: bool = False,
        parallel_config={},
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1_ada = AdaLayerNorm(dim, num_embeds_ada_norm)
            self.norm1_ada.norm = LayerNorm(dim, elementwise_affine=False)
        elif self.use_ada_layer_norm_zero:
            self.norm1_ada_zero = AdaLayerNormZero(dim, num_embeds_ada_norm)
            self.norm1_ada_zero.norm = LayerNorm(dim, elementwise_affine=False)
        else:
            self.norm1_ln = LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = SeqParallelMultiHeadAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            enable_flash_attention=enable_flash_attention,
            parallel_config=parallel_config,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if self.use_ada_layer_norm:
                self.norm2_ada = AdaLayerNorm(dim, num_embeds_ada_norm)
                self.norm2_ada.norm = LayerNorm(dim, elementwise_affine=False)
            else:
                self.norm2_ln = LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

            self.attn2 = SeqParallelMultiHeadAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                enable_flash_attention=enable_flash_attention,
                parallel_config=parallel_config,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if not self.use_ada_layer_norm_single:
            self.norm3 = LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = SeqParallelFeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            parallel_config=parallel_config,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = ms.Parameter(ops.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0
        self.add = ops.Add()
        self.mult = ops.Mul()
        self.split = ops.Split(axis=1, output_num=6)

        self.parallel_config = parallel_config
        self.shard()

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        timestep: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]
        gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1_ada(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1_ada_zero(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1_ln(hidden_states)
        elif self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.split(
                self.add(self.scale_shift_table[None], timestep.reshape(batch_size, 6, -1))
            )
            norm_hidden_states = self.norm1_ln(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            # norm_hidden_states = norm_hidden_states.squeeze(1)  # error message
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        if "gligen" in cross_attention_kwargs:
            gligen_kwargs = cross_attention_kwargs["gligen"]
            # del cross_attention_kwargs["gligen"]
        else:
            gligen_kwargs = None
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = self.add(gate_msa.unsqueeze(1), attn_output)
        elif self.use_ada_layer_norm_single:
            attn_output = self.add(gate_msa, attn_output)

        hidden_states = self.add(attn_output, hidden_states)
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2_ada(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                norm_hidden_states = self.norm2_ln(hidden_states)
            elif self.use_ada_layer_norm_single:
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = self.add(attn_output, hidden_states)

        # 4. Feed-forward
        if not self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = self.add(self.mult(norm_hidden_states, (1 + scale_mlp[:, None])), shift_mlp[:, None])

        if self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm2_ln(hidden_states)
            norm_hidden_states = self.add(self.mult(norm_hidden_states, (1 + scale_mlp)), shift_mlp)

        ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = self.mult(gate_mlp.unsqueeze(1), ff_output)
        elif self.use_ada_layer_norm_single:
            ff_output = self.mult(gate_mlp, ff_output)

        hidden_states = self.add(ff_output, hidden_states)
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

    def shard(self):
        self.dp = self.parallel_config.get("data_parallel", 1)
        self.mp = self.parallel_config.get("model_parallel", 1)
        self.sp = self.parallel_config.get("sequence_parallel", 1)

        self.add.shard(((self.dp, self.sp, 1), (self.dp, self.sp, 1)))
        self.mult.shard(((self.dp, self.sp, 1), (self.dp, 1, 1)))

        self.split.shard(((self.dp, 1, 1),))

        if self.use_ada_layer_norm_single:
            # current only implement this layernorm
            self.norm1_ln.layer_norm.shard(((self.dp, self.sp, 1), (1,), (1,)))
            self.norm3.layer_norm.shard(((self.dp, self.sp, 1), (1,), (1,)))
        else:
            raise NotImplementedError


class Latte(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        patch_size_t: int = 1,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        video_length: int = 16,
        enable_flash_attention: bool = False,
        dtype=ms.float32,
        use_recompute=False,
    ):
        super().__init__()
        # self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.video_length = video_length
        self.norm_type = norm_type
        self.use_recompute = use_recompute

        conv_cls = nn.Conv2d  # if USE_PEFT_BACKEND else LoRACompatibleConv
        linear_cls = nn.Dense  # if USE_PEFT_BACKEND else LoRACompatibleLinear

        # 1. Transformer2DModel can process both standard continuous images of shape \
        # `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            logger.warning("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif self.is_input_vectorized and self.is_input_patches:
            raise ValueError(
                f"Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make"
                " sure that either `num_vector_embeds` or `num_patches` is None."
            )
        elif not self.is_input_continuous and not self.is_input_vectorized and not self.is_input_patches:
            raise ValueError(
                f"Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size:"
                f" {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None."
            )

        # 2. Define input layers
        if self.is_input_continuous:
            self.in_channels = in_channels

            self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            if use_linear_projection:
                self.proj_in = linear_cls(in_channels, inner_dim)
            else:
                self.proj_in = conv_cls(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            assert sample_size is not None, "Transformer2DModel over discrete input must provide sample_size"
            assert num_vector_embeds is not None, "Transformer2DModel over discrete input must provide num_embed"

            self.height = sample_size[0]
            self.width = sample_size[1]
            self.num_vector_embeds = num_vector_embeds
            self.num_latent_pixels = self.height * self.width

            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=num_vector_embeds, embed_dim=inner_dim, height=self.height, width=self.width
            )
        elif self.is_input_patches:
            assert sample_size is not None, "Transformer2DModel over patched input must provide sample_size"

            self.height = sample_size[0]
            self.width = sample_size[1]

            self.patch_size = patch_size
            interpolation_scale = self.config.sample_size[0] // 64  # => 64 (= 512 pixart) has interpolation scale 1
            interpolation_scale = max(interpolation_scale, 1)
            self.pos_embed = PatchEmbed(
                height=sample_size[0],
                width=sample_size[1],
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                interpolation_scale=interpolation_scale,
            )

        # 3. Define transformers blocks
        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock_(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=None,  # unconditon do not need cross attn
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=False,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    enable_flash_attention=enable_flash_attention,
                )
                for d in range(num_layers)
            ]
        )

        # Define temporal transformers blocks
        self.temporal_transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock_(  # one attention
                    inner_dim,
                    num_attention_heads,  # num_attention_heads
                    attention_head_dim,  # attention_head_dim 72
                    dropout=dropout,
                    cross_attention_dim=None,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=False,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    enable_flash_attention=enable_flash_attention,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.is_input_continuous:
            # TODO: should use out_channels for continuous projections
            if use_linear_projection:
                self.proj_out = linear_cls(inner_dim, in_channels)
            else:
                self.proj_out = conv_cls(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            self.norm_out = LayerNorm(inner_dim)
            self.out = nn.Dense(inner_dim, self.num_vector_embeds - 1)
        elif self.is_input_patches and norm_type != "ada_norm_single":
            self.norm_out = LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Dense(inner_dim, 2 * inner_dim)
            self.proj_out_2 = nn.Dense(inner_dim, patch_size * patch_size * self.out_channels)
        elif self.is_input_patches and norm_type == "ada_norm_single":
            self.norm_out = LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = ms.Parameter(ops.randn(2, inner_dim) / inner_dim**0.5)
            self.proj_out = nn.Dense(inner_dim, patch_size * patch_size * self.out_channels)

        # 5. PixArt-Alpha blocks.
        self.adaln_single = None
        self.use_additional_conditions = False
        if norm_type == "ada_norm_single":
            # self.use_additional_conditions = self.config.sample_size[0] == 128  # False, 128 -> 1024
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=self.use_additional_conditions)

        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = CaptionProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.gradient_checkpointing = False

        interpolation_scale = self.config.video_length // 5  # => 5 (= 5 our causalvideovae) has interpolation scale 1
        interpolation_scale = max(interpolation_scale, 1)
        temp_pos_embed = get_1d_sincos_pos_embed(
            inner_dim, video_length, interpolation_scale=interpolation_scale
        )  # 1152 hidden size
        self.temp_pos_embed = Parameter(ms.Tensor(temp_pos_embed).float().unsqueeze(0), requires_grad=False)

        if self.use_recompute:
            for block in self.transformer_blocks:
                self.recompute(block)

            for block in self.temporal_transformer_blocks:
                self.recompute(block)

    def construct(
        self,
        hidden_states: ms.Tensor,
        timestep: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        added_cond_kwargs: Dict[str, ms.Tensor] = None,
        class_labels: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        use_image_num: int = 0,
        enable_temporal_attentions: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`ms.Tensor` of shape `(batch size, num latent pixels)` if discrete, `ms.Tensor` of shape \
                `(batch size, frame, channel, height, width)` if continuous): Input `hidden_states`.
            encoder_hidden_states ( `ms.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `ms.Tensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `ms.Tensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `ms.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `ms.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
        """
        input_batch_size, c, frame, h, w = hidden_states.shape
        frame = frame - use_image_num
        # b c f h w -> (b f) c h w
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).view(-1, c, h, w)

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -ms.numpy.inf)
            attention_mask = attention_mask.unsqueeze(1)  # (b, 1, key_len)
            attention_mask = ops.zeros(attention_mask.shape).masked_fill(~attention_mask, -ms.numpy.inf)

        # # Retrieve lora scale.
        # lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        assert self.is_input_patches, "Currently only support input patches!"
        # if self.is_input_patches:  # here
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        num_patches = height * width

        hidden_states = self.pos_embed(hidden_states.to(self.dtype))  # alrady add positional embeddings

        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                )
            # batch_size = hidden_states.shape[0]
            batch_size = input_batch_size
            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
        else:
            embedded_timestep = None

        # prepare timesteps for spatial and temporal block
        # b d -> (b f) d
        timestep_spatial = timestep.repeat_interleave(frame + use_image_num, dim=0)
        # b d -> (b p) d
        timestep_temp = timestep.repeat_interleave(num_patches, dim=0)

        for i, (spatial_block, temp_block) in enumerate(zip(self.transformer_blocks, self.temporal_transformer_blocks)):
            hidden_states = spatial_block(
                hidden_states,
                attention_mask,
                None,  # encoder_hidden_states_spatial
                None,  # encoder_attention_mask
                timestep_spatial,
                cross_attention_kwargs,
                class_labels,
            )

            if enable_temporal_attentions:
                # (b f) t d -> (b t) f d
                hidden_states = hidden_states.view(
                    input_batch_size, frame + use_image_num, hidden_states.shape[1], hidden_states.shape[2]
                )
                hidden_states = hidden_states.permute(0, 2, 1, 3).view(
                    -1, frame + use_image_num, hidden_states.shape[-1]
                )

                if use_image_num != 0 and self.training:
                    hidden_states_video = hidden_states[:, :frame, ...]
                    hidden_states_image = hidden_states[:, frame:, ...]

                    hidden_states_video = temp_block(
                        hidden_states_video,
                        None,  # attention_mask
                        None,  # encoder_hidden_states
                        None,  # encoder_attention_mask
                        timestep_temp,
                        cross_attention_kwargs,
                        class_labels,
                    )

                    hidden_states = ops.cat([hidden_states_video, hidden_states_image], axis=1)
                    # (b t) f d -> (b f) t d
                    hidden_states = hidden_states.view(
                        input_batch_size, -1, hidden_states.shape[1], hidden_states.shape[2]
                    )
                    hidden_states = hidden_states.permute(0, 2, 1, 3).view(
                        input_batch_size * (frame + use_image_num), -1, hidden_states.shape[-1]
                    )

                else:
                    if i == 0:
                        hidden_states = hidden_states + self.temp_pos_embed

                    hidden_states = temp_block(
                        hidden_states,
                        None,  # attention_mask
                        None,  # encoder_hidden_states
                        None,  # encoder_attention_mask
                        timestep_temp,
                        cross_attention_kwargs,
                        class_labels,
                    )

                    # (b t) f d -> (b f) t d
                    hidden_states = hidden_states.view(
                        input_batch_size, -1, hidden_states.shape[1], hidden_states.shape[2]
                    )
                    hidden_states = hidden_states.permute(0, 2, 1, 3).view(
                        input_batch_size * (frame + use_image_num), -1, hidden_states.shape[-1]
                    )

        # if self.is_input_patches:
        if self.norm_type != "ada_norm_single":
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(ops.silu(conditioning)).chunk(2, axis=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)
        elif self.norm_type == "ada_norm_single":
            # b d -> (b f) d
            assert embedded_timestep is not None, "embedded_timestep should be not None"
            embedded_timestep = embedded_timestep.repeat_interleave(frame + use_image_num, dim=0)
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, axis=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        # nhwpqc->nchpwq
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size))
        # (b f) c h w -> b c f h w
        output = output.view(input_batch_size, -1, output.shape[-3], output.shape[-2], output.shape[-1])
        output = output.permute(0, 2, 1, 3, 4)

        return output

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, **kwargs):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, "config.json")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        model = cls.from_config(config, **kwargs)
        return model

    def construct_with_cfg(self, x, timestep, class_labels=None, cfg_scale=7.0, attention_mask=None):
        """
        Forward pass of Latte, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # x shape : b f c h w
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        model_out = self.construct(combined, timestep, class_labels=class_labels, attention_mask=attention_mask)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :, : self.in_channels], model_out[:, :, self.in_channels :]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=2)

    def load_params_from_ckpt(self, ckpt):
        # load param from a ckpt file path or a parameter dictionary
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f"{ckpt} does not exist!"
            logger.info(f"Loading {ckpt} params into Latte_T2V_SD model...")
            param_dict = ms.load_checkpoint(ckpt)
        elif isinstance(ckpt, dict):
            param_dict = ckpt
        else:
            raise ValueError("Expect to receive a ckpt path or parameter dictionary as input!")
        _, ckpt_not_load = ms.load_param_into_net(
            self,
            param_dict,
        )
        if len(ckpt_not_load):
            print(f"{ckpt_not_load} not load")

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute()
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        else:
            b.add_flags(output_no_recompute=True)


class LatteT2VBlock(nn.Cell):
    def __init__(self, block_id, temp_pos_embed, spatial_block, temp_block):
        super().__init__()
        self.spatial_block = spatial_block
        self.temp_block = temp_block
        self.is_first_block = block_id == 0
        self.temp_pos_embed = temp_pos_embed

    def construct(
        self,
        hidden_states: ms.Tensor,
        class_labels: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states_spatial: Optional[ms.Tensor] = None,
        timestep_spatial: Optional[ms.Tensor] = None,
        timestep_temp: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        use_image_num: int = 0,
        input_batch_size: int = 0,
        frame: int = 0,
        enable_temporal_attentions: bool = True,
    ):
        hidden_states = self.spatial_block(
            hidden_states,
            attention_mask,
            encoder_hidden_states_spatial,
            encoder_attention_mask,
            timestep_spatial,
            cross_attention_kwargs,
            class_labels,
        )

        if enable_temporal_attentions:
            # b c f h w, f = 16 + 4
            # (b f) t d -> (b t) f d
            hidden_states = hidden_states.view(input_batch_size, frame + use_image_num, -1, hidden_states.shape[-1])
            hidden_states = hidden_states.permute(0, 2, 1, 3).view(-1, frame + use_image_num, hidden_states.shape[-1])

            if use_image_num != 0 and self.training:
                hidden_states_video = hidden_states[:, :frame, ...]
                hidden_states_image = hidden_states[:, frame:, ...]
                if self.is_first_block:
                    hidden_states_video = hidden_states_video + self.temp_pos_embed

                hidden_states_video = self.temp_block(
                    hidden_states_video,
                    None,  # attention_mask
                    None,  # encoder_hidden_states
                    None,  # encoder_attention_mask
                    timestep_temp,
                    cross_attention_kwargs,
                    class_labels,
                )

                hidden_states = ops.cat([hidden_states_video, hidden_states_image], axis=1)
                # (b t) f d -> (b f) t d
                hidden_states = hidden_states.view(input_batch_size, -1, frame + use_image_num, hidden_states.shape[-1])
                hidden_states = hidden_states.permute(0, 2, 1, 3).view(
                    input_batch_size * (frame + use_image_num), -1, hidden_states.shape[-1]
                )

            else:
                if self.is_first_block:
                    hidden_states = hidden_states + self.temp_pos_embed

                hidden_states = self.temp_block(
                    hidden_states,
                    None,  # attention_mask
                    None,  # encoder_hidden_states
                    None,  # encoder_attention_mask
                    timestep_temp,
                    cross_attention_kwargs,
                    class_labels,
                )
                # (b t) f d -> (b f) t d
                hidden_states = hidden_states.view(input_batch_size, -1, frame + use_image_num, hidden_states.shape[-1])
                hidden_states = hidden_states.permute(0, 2, 1, 3).view(
                    input_batch_size * (frame + use_image_num), -1, hidden_states.shape[-1]
                )
        return hidden_states


class LatteT2V(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        patch_size_t: int = 1,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        video_length: int = 16,
        enable_flash_attention: bool = False,
        use_recompute=False,
        enable_sequence_parallelism=False,
        temporal_parallel_config={},
        spatial_parallel_config={},
    ):
        super().__init__()
        # self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.video_length = video_length
        self.norm_type = norm_type
        self.use_recompute = use_recompute
        self.enable_sequence_parallelism = enable_sequence_parallelism
        conv_cls = nn.Conv2d  # if USE_PEFT_BACKEND else LoRACompatibleConv
        linear_cls = nn.Dense  # if USE_PEFT_BACKEND else LoRACompatibleLinear

        # 1. Transformer2DModel can process both standard continuous images of shape \
        # `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape \
        # `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            logger.warning("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif self.is_input_vectorized and self.is_input_patches:
            raise ValueError(
                f"Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make"
                " sure that either `num_vector_embeds` or `num_patches` is None."
            )
        elif not self.is_input_continuous and not self.is_input_vectorized and not self.is_input_patches:
            raise ValueError(
                f"Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size:"
                f" {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None."
            )

        # 2. Define input layers
        if self.is_input_continuous:
            self.in_channels = in_channels

            self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            if use_linear_projection:
                self.proj_in = linear_cls(in_channels, inner_dim)
            else:
                self.proj_in = conv_cls(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            assert sample_size is not None, "Transformer2DModel over discrete input must provide sample_size"
            assert num_vector_embeds is not None, "Transformer2DModel over discrete input must provide num_embed"

            self.height = sample_size[0] if isinstance(sample_size, (tuple, list)) else sample_size
            self.width = sample_size[1] if isinstance(sample_size, (tuple, list)) else sample_size
            self.num_vector_embeds = num_vector_embeds
            self.num_latent_pixels = self.height * self.width

            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=num_vector_embeds, embed_dim=inner_dim, height=self.height, width=self.width
            )
        elif self.is_input_patches:
            assert sample_size is not None, "Transformer2DModel over patched input must provide sample_size"

            self.height = sample_size[0] if isinstance(sample_size, (tuple, list)) else sample_size
            self.width = sample_size[1] if isinstance(sample_size, (tuple, list)) else sample_size

            self.patch_size = patch_size
            if isinstance(self.config.sample_size, (tuple, list)):
                interpolation_scale = self.config.sample_size[0] // 64  # => 64 (= 512 pixart) has interpolation scale 1
            else:
                interpolation_scale = self.config.sample_size // 64
            interpolation_scale = max(interpolation_scale, 1)
            self.pos_embed = PatchEmbed(
                height=sample_size[0] if isinstance(sample_size, (tuple, list)) else sample_size,
                width=sample_size[1] if isinstance(sample_size, (tuple, list)) else sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                interpolation_scale=interpolation_scale,
            )

        # 3. Define transformers blocks, spatial attention
        if not enable_sequence_parallelism:
            self.transformer_blocks = nn.CellList(
                [
                    BasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=double_self_attention,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                        enable_flash_attention=enable_flash_attention,
                    )
                    for d in range(num_layers)
                ]
            )
        else:
            self.transformer_blocks = nn.CellList(
                [
                    SeqParallelBasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=double_self_attention,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                        enable_flash_attention=enable_flash_attention,
                        parallel_config=spatial_parallel_config,
                    )
                    for d in range(num_layers)
                ]
            )
        # Define temporal transformers blocks
        if not enable_sequence_parallelism:
            self.temporal_transformer_blocks = nn.CellList(
                [
                    BasicTransformerBlock_(  # one attention
                        inner_dim,
                        num_attention_heads,  # num_attention_heads
                        attention_head_dim,  # attention_head_dim 72
                        dropout=dropout,
                        cross_attention_dim=None,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=False,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                        enable_flash_attention=enable_flash_attention,
                    )
                    for d in range(num_layers)
                ]
            )
        else:
            self.temporal_transformer_blocks = nn.CellList(
                [
                    SeqParallelBasicTransformerBlock_(  # one attention
                        inner_dim,
                        num_attention_heads,  # num_attention_heads
                        attention_head_dim,  # attention_head_dim 72
                        dropout=dropout,
                        cross_attention_dim=None,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=False,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        attention_type=attention_type,
                        enable_flash_attention=enable_flash_attention,
                        parallel_config=temporal_parallel_config,
                    )
                    for d in range(num_layers)
                ]
            )
        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.is_input_continuous:
            # TODO: should use out_channels for continuous projections
            if use_linear_projection:
                self.proj_out = linear_cls(inner_dim, in_channels)
            else:
                self.proj_out = conv_cls(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            self.norm_out = LayerNorm(inner_dim)
            self.out = nn.Dense(inner_dim, self.num_vector_embeds - 1)
        elif self.is_input_patches and norm_type != "ada_norm_single":
            self.norm_out = LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Dense(inner_dim, 2 * inner_dim)
            self.proj_out_2 = nn.Dense(inner_dim, patch_size * patch_size * self.out_channels)
        elif self.is_input_patches and norm_type == "ada_norm_single":
            self.norm_out = LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = ms.Parameter(ops.randn(2, inner_dim) / inner_dim**0.5)
            self.proj_out = nn.Dense(inner_dim, patch_size * patch_size * self.out_channels)

        # 5. PixArt-Alpha blocks.
        self.adaln_single = None
        self.use_additional_conditions = False
        if norm_type == "ada_norm_single":
            # self.use_additional_conditions = self.config.sample_size[0] == 128  # False, 128 -> 1024
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=self.use_additional_conditions)

        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = CaptionProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.gradient_checkpointing = False

        # define temporal positional embedding
        # temp_pos_embed = self.get_1d_sincos_temp_embed(inner_dim, video_length)  # 1152 hidden size

        interpolation_scale = self.config.video_length // 5  # => 5 (= 5 our causalvideovae) has interpolation scale 1
        interpolation_scale = max(interpolation_scale, 1)
        temp_pos_embed = get_1d_sincos_pos_embed(
            inner_dim, video_length, interpolation_scale=interpolation_scale
        )  # 1152 hidden size

        self.temp_pos_embed = ms.Tensor(temp_pos_embed).float().unsqueeze(0)

        self.blocks = nn.CellList(
            [
                LatteT2VBlock(d, self.temp_pos_embed, self.transformer_blocks[d], self.temporal_transformer_blocks[d])
                for d in range(num_layers)
            ]
        )

        if self.use_recompute:
            for block in self.blocks:
                self.recompute(block)

    def construct(
        self,
        hidden_states: ms.Tensor,
        timestep: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        added_cond_kwargs: Dict[str, ms.Tensor] = None,
        class_labels: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        use_image_num: int = 0,
        enable_temporal_attentions: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`ms.Tensor` of shape `(batch size, num latent pixels)` if discrete, \
                `ms.Tensor` of shape `(batch size, frame, channel, height, width)` if continuous): Input `hidden_states`.
            encoder_hidden_states ( `ms.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `ms.Tensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `ms.Tensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `ms.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `ms.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.

        """
        input_batch_size, c, frame, h, w = hidden_states.shape
        # print(hidden_states.shape, input_batch_size, c, frame, h, w, use_image_num)
        # print(timestep)
        # print(encoder_hidden_states.shape)
        # print(encoder_attention_mask.shape)
        frame = frame - use_image_num  # 20-4=16
        # b c f h w -> (b f) c h w
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            input_batch_size * (frame + use_image_num), c, h, w
        )
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -ms.numpy.inf)
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = ops.zeros(attention_mask.shape).masked_fill(~attention_mask, -ms.numpy.inf)
            attention_mask = attention_mask.to(self.dtype)
        # 1 + 4, 1 -> video condition, 4 -> image condition
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:  # ndim == 2 means no image joint
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            encoder_attention_mask = ops.zeros(encoder_attention_mask.shape).masked_fill(
                ~encoder_attention_mask, -ms.numpy.inf
            )
            # b 1 l -> (b f) 1 l
            encoder_attention_mask = encoder_attention_mask.repeat_interleave(frame, dim=0)
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)
        elif encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  # ndim == 3 means image joint
            encoder_attention_mask = ops.zeros(encoder_attention_mask.shape).masked_fill(
                ~encoder_attention_mask, -ms.numpy.inf
            )
            encoder_attention_mask_video = encoder_attention_mask[:, :1, ...]
            encoder_attention_mask_video = encoder_attention_mask_video.repeat_interleave(frame, dim=1)
            encoder_attention_mask_image = encoder_attention_mask[:, 1:, ...]
            encoder_attention_mask = ops.cat([encoder_attention_mask_video, encoder_attention_mask_image], axis=1)
            # b n l -> (b n) l
            encoder_attention_mask = encoder_attention_mask.view(-1, encoder_attention_mask.shape[-1]).unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)

        # # Retrieve lora scale.
        # lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        assert self.is_input_patches, "Only support input patches now!"
        # if self.is_input_patches:  # here
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        num_patches = height * width

        hidden_states = self.pos_embed(hidden_states.to(self.dtype))  # alrady add positional embeddings

        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                )
            # batch_size = hidden_states.shape[0]
            batch_size = input_batch_size
            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
        else:
            embedded_timestep = None

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states.to(self.dtype))  # 3 120 1152

            if use_image_num != 0 and self.training:
                encoder_hidden_states_video = encoder_hidden_states[:, :1, ...]
                # b 1 t d -> b (1 f) t d
                encoder_hidden_states_video = encoder_hidden_states_video.repeat_interleave(frame, dim=1)
                encoder_hidden_states_image = encoder_hidden_states[:, 1:, ...]
                encoder_hidden_states = ops.cat([encoder_hidden_states_video, encoder_hidden_states_image], axis=1)
                # b f t d -> (b f) t d
                encoder_hidden_states_spatial = encoder_hidden_states.view(
                    -1, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1]
                )
            else:
                # b t d -> (b f) t d
                encoder_hidden_states_spatial = encoder_hidden_states.repeat_interleave(frame, dim=0)
        else:
            encoder_hidden_states_spatial = encoder_hidden_states.repeat_interleave(frame, dim=0)  # for graph mode

        # prepare timesteps for spatial and temporal block
        # b d -> (b f) d
        timestep_spatial = timestep.repeat_interleave(frame + use_image_num, dim=0)
        # b d -> (b p) d
        timestep_temp = timestep.repeat_interleave(num_patches, dim=0)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                class_labels,
                cross_attention_kwargs,
                attention_mask,
                encoder_hidden_states_spatial,
                timestep_spatial,
                timestep_temp,
                encoder_attention_mask,
                use_image_num,
                input_batch_size,
                frame,
                enable_temporal_attentions,
            )

        # if self.is_input_patches:
        if self.norm_type != "ada_norm_single":
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(ops.silu(conditioning)).chunk(2, axis=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)
        elif self.norm_type == "ada_norm_single":
            # b d -> (b f) d
            assert embedded_timestep is not None, "embedded_timestep is expected to be not None"
            embedded_timestep = embedded_timestep.repeat_interleave(frame + use_image_num, dim=0)
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, axis=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)

        hidden_states = hidden_states.reshape(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        # nhwpqc->nchpwq
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        # (b f) c h w -> b c f h w
        output = output.view(
            input_batch_size, frame + use_image_num, output.shape[-3], output.shape[-2], output.shape[-1]
        )
        output = output.permute(0, 2, 1, 3, 4)
        return output

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, **kwargs):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, "config.json")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        model = cls.from_config(config, **kwargs)
        return model

    def construct_with_cfg(self, x, timestep, class_labels=None, cfg_scale=7.0, attention_mask=None):
        """
        Forward pass of Latte, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        model_out = self.construct(combined, timestep, class_labels=class_labels, attention_mask=attention_mask)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :, : self.in_channels], model_out[:, :, self.in_channels :]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=2)

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute(parallel_optimizer_comm_recompute=True)
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        else:
            b.add_flags(output_no_recompute=True)

    def load_from_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {ckpt_path} not found. No checkpoint loaded!!")
        else:
            sd = ms.load_checkpoint(ckpt_path)
            # filter 'network.' prefix
            rm_prefix = ["network."]
            all_pnames = list(sd.keys())
            for pname in all_pnames:
                for pre in rm_prefix:
                    if pname.startswith(pre):
                        new_pname = pname.replace(pre, "")
                        sd[new_pname] = sd.pop(pname)

            m, u = ms.load_param_into_net(self, sd)
            print("net param not load: ", m, len(m))
            print("ckpt param not load: ", u, len(u))


# depth = num_layers * 2
def Latte_XL_122(**kwargs):
    return Latte(
        num_layers=28,
        attention_head_dim=72,
        num_attention_heads=16,
        patch_size_t=1,
        patch_size=2,
        norm_type="ada_norm_single",
        **kwargs,
    )


def LatteClass_XL_122(**kwargs):
    return Latte(
        num_layers=28,
        attention_head_dim=72,
        num_attention_heads=16,
        patch_size_t=1,
        patch_size=2,
        norm_type="ada_norm_zero",
        **kwargs,
    )


def LatteT2V_XL_122(**kwargs):
    return LatteT2V(
        num_layers=28,
        attention_head_dim=72,
        num_attention_heads=16,
        patch_size_t=1,
        patch_size=2,
        norm_type="ada_norm_single",
        caption_channels=4096,
        cross_attention_dim=1152,
        **kwargs,
    )


Latte_models = {
    "Latte-XL/122": Latte_XL_122,
    "LatteClass-XL/122": LatteClass_XL_122,
    "LatteT2V-XL/122": LatteT2V_XL_122,
}