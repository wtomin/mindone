import mindspore as ms
import mindspore.mint as mint
from mindspore import Tensor, ops

from mindone.transformers.mindspore_adapter.utils import _DTYPE_2_MIN


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, dtype=None, training=True
):
    # force dtype(fp16 or bf16) precision calculation
    ori_dtype = query.dtype
    if dtype is not None:
        query, key, value = query.astype(dtype), key.astype(dtype), value.astype(dtype)

    if attn_mask is not None:
        if attn_mask.dtype == ms.bool_:
            attn_mask = attn_mask.to(ms.float32)
            attn_mask = attn_mask.masked_fill((1 - attn_mask).to(ms.bool_), _DTYPE_2_MIN[ms.float16])
        attn_mask = attn_mask.to(query.dtype)

        attn_weight = mint.nn.functional.softmax(
            mint.matmul(query, mint.transpose(key, -2, -1)) / (query.shape[-1] ** 0.5) + attn_mask,
            dim=-1,
            dtype=ms.float32,
        ).astype(query.dtype)
    else:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = mint.zeros((L, S), dtype=query.dtype)
        if is_causal:
            # assert attn_mask is None
            temp_mask = mint.ones((L, S), dtype=ms.bool_).tril(diagonal=0)
            attn_bias = ops.masked_fill(attn_bias, mint.logical_not(temp_mask), _DTYPE_2_MIN[ms.float16])
            attn_bias = attn_bias.to(query.dtype)

        attn_weight = mint.nn.functional.softmax(
            mint.matmul(query, mint.transpose(key, -2, -1)) / (query.shape[-1] ** 0.5) + attn_bias,
            dim=-1,
            dtype=ms.float32,
        ).astype(query.dtype)

    attn_weight = mint.nn.functional.dropout(attn_weight, p=dropout_p, training=training)

    out = mint.matmul(attn_weight, value)
    out = out.astype(ori_dtype)

    return out


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = scaled_dot_product_attention(q, k, v)
    B, H, L, D = x.shape
    x = x.permute(0, 2, 1, 3).reshape(B, L, H * D)

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = (
        mint.arange(
            0,
            dim,
            2,
            dtype=ms.float32,
        )
        / dim
    )
    omega = 1.0 / (theta**scale)
    out = mint.einsum("...n,d->...nd", pos, omega)
    out = mint.stack([mint.cos(out), -mint.sin(out), mint.sin(out), mint.cos(out)], dim=-1)
    b, n, d, ij = out.shape
    out = out.reshape(b, n, d, 2, 2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
