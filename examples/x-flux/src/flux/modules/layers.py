import mindspore as ms
from mindspore import mint, nn, ops
import math
from dataclasses import dataclass
import numpy as np

import mindspore as ms
# from einops import rearrange  # einops not supported in MindSpore
from mindspore import Tensor, nn

from ..math import attention, rope
from src.flux.math import scaled_dot_product_attention


class EmbedND(ms.nn.Cell):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def construct(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = mint.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = mint.exp(-math.log(max_period) * mint.arange(start=0, end=half, dtype=ms.float32) / half)

    args = t[:, None].float() * freqs[None]
    embedding = mint.cat([mint.cos(args), mint.sin(args)], dim=-1)
    if dim % 2:
        embedding = mint.cat([embedding, mint.zeros_like(embedding[:, :1])], dim=-1)
    if t.dtype in [ms.float16, ms.float32, ms.float64, ms.bfloat16]:
        embedding = embedding.to(t.dtype)
    return embedding


class MLPEmbedder(ms.nn.Cell):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = mint.nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = mint.nn.SiLU()
        self.out_layer = mint.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def construct(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(ms.nn.Cell):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = ms.Parameter(mint.ones(dim))

    def construct(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = mint.rsqrt(mint.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(ms.nn.Cell):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def construct(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v.dtype), k.to(v.dtype)

class LoRALinearLayer(ms.nn.Cell):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()

        self.down = mint.nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.up = mint.nn.Linear(rank, out_features, bias=False, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def construct(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

class FLuxSelfAttnProcessor:
    def __call__(self, attn, x, pe, **attention_kwargs):
        print('2' * 30)

        qkv = attn.qkv(x)
        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        B, L, _ = qkv.shape
        qkv_reshaped = qkv.reshape(B, L, 3, self.num_heads, -1)
        q, k, v = qkv_reshaped.permute(2, 0, 3, 1, 4)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x)
        return x

class LoraFluxAttnProcessor(ms.nn.Cell):

    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight


    def __call__(self, attn, x, pe, **attention_kwargs):
        qkv = attn.qkv(x) + self.qkv_lora(x) * self.lora_weight
        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        B, L, _ = qkv.shape
        qkv_reshaped = qkv.reshape(B, L, 3, self.num_heads, -1)
        q, k, v = qkv_reshaped.permute(2, 0, 3, 1, 4)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x) + self.proj_lora(x) * self.lora_weight
        print('1' * 30)
        print(x.norm(), (self.proj_lora(x) * self.lora_weight).norm(), 'norm')
        return x

class SelfAttention(ms.nn.Cell):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = mint.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = mint.nn.Linear(dim, dim)
    def construct(self, x: Tensor):
        # a dummy construct function to avoid error
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(ms.nn.Cell):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = mint.nn.Linear(dim, self.multiplier * dim, bias=True)

    def construct(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(mint.nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

class DoubleStreamBlockLoraProcessor(ms.nn.Cell):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def construct(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        # img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        B, L, _ = img_qkv.shape
        img_qkv_reshaped = img_qkv.reshape(B, L, 3, attn.num_heads, -1)
        img_q, img_k, img_v = img_qkv_reshaped.permute(2, 0, 3, 1, 4)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        # txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        B, L, _ = txt_qkv.shape
        txt_qkv_reshaped = txt_qkv.reshape(B, L, 3, attn.num_heads, -1)
        txt_q, txt_k, txt_v = txt_qkv_reshaped.permute(2, 0, 3, 1, 4)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = mint.cat((txt_q, img_q), dim=2)
        k = mint.cat((txt_k, img_k), dim=2)
        v = mint.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class IPDoubleStreamBlockProcessor(ms.nn.Cell):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim):
        super().__init__()

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_double_stream_k_proj = mint.nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = mint.nn.Linear(context_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

    def __call__(self, attn, img, txt, vec, pe, image_proj, ip_scale=1.0, **attention_kwargs):

        # Prepare image for attention
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        # img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        B, L, _ = img_qkv.shape
        img_qkv_reshaped = img_qkv.reshape(B, L, 3, attn.num_heads, attn.head_dim)
        img_q, img_k, img_v = img_qkv_reshaped.permute(2, 0, 3, 1, 4)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        # txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        B, L, _ = txt_qkv.shape
        txt_qkv_reshaped = txt_qkv.reshape(B, L, 3, attn.num_heads, attn.head_dim)
        txt_q, txt_k, txt_v = txt_qkv_reshaped.permute(2, 0, 3, 1, 4)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = mint.cat((txt_q, img_q), dim=2)
        k = mint.cat((txt_k, img_k), dim=2)
        v = mint.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]

        # print(f"txt_attn shape: {txt_attn.size()}")
        # print(f"img_attn shape: {img_attn.size()}")

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)


        # IP-adapter processing
        ip_query = img_q  # latent sample query
        ip_key = self.ip_adapter_double_stream_k_proj(image_proj)
        ip_value = self.ip_adapter_double_stream_v_proj(image_proj)

        # Reshape projections for multi-head attention
        # ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        # ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        B, L, _ = ip_key.shape
        ip_key = ip_key.reshape(B, L, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3)
        ip_value = ip_value.reshape(B, L, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3)

        # Compute attention between IP projections and the latent query
        ip_attention = scaled_dot_product_attention(
            ip_query,
            ip_key,
            ip_value,
            dropout_p=0.0,
            is_causal=False
        )
        # ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
        ip_attention = ip_attention.permute(0, 2, 1, 3).reshape(B, L, -1)

        img = img + ip_scale * ip_attention

        return img, txt

class DoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        # img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        B, L, _ = img_qkv.shape
        img_qkv_reshaped = img_qkv.reshape(B, L, 3, attn.num_heads, attn.head_dim)
        img_q, img_k, img_v = img_qkv_reshaped.permute(2, 0, 3, 1, 4)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        # txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        B, L, _ = txt_qkv.shape
        txt_qkv_reshaped = txt_qkv.reshape(B, L, 3, attn.num_heads, attn.head_dim)
        txt_q, txt_k, txt_v = txt_qkv_reshaped.permute(2, 0, 3, 1, 4)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = mint.cat((txt_q, img_q), dim=2)
        k = mint.cat((txt_k, img_k), dim=2)
        v = mint.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class DoubleStreamBlock(ms.nn.Cell):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = ms.nn.SequentialCell(
            mint.nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            mint.nn.GELU(approximate="tanh"),
            mint.nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = ms.nn.SequentialCell(
            mint.nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            mint.nn.GELU(approximate="tanh"),
            mint.nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        processor = DoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def construct(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor = None,
        ip_scale: float =1.0,
    ) -> tuple[Tensor, Tensor]:
        if image_proj is None:
            return self.processor(self, img, txt, vec, pe)
        else:
            return self.processor(self, img, txt, vec, pe, image_proj, ip_scale)

class IPSingleStreamBlockProcessor(ms.nn.Cell):
    """Attention processor for handling IP-adapter with single stream block."""
    def __init__(self, context_dim, hidden_dim):
        super().__init__()

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_single_stream_k_proj = mint.nn.Linear(context_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj = mint.nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)

    def __call__(
        self,
        attn: ms.nn.Cell,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0
    ) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = mint.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        B, L, _ = qkv.shape
        qkv_reshaped = qkv.reshape(B, L, 3, attn.num_heads, attn.head_dim)
        q, k, v = qkv_reshaped.permute(2, 0, 3, 1, 4)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # IP-adapter processing
        ip_query = q
        ip_key = self.ip_adapter_single_stream_k_proj(image_proj)
        ip_value = self.ip_adapter_single_stream_v_proj(image_proj)

        # Reshape projections for multi-head attention
        # ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        # ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        B, L, _ = ip_key.shape
        ip_key = ip_key.reshape(B, L, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3)
        ip_value = ip_value.reshape(B, L, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3)


        # Compute attention between IP projections and the latent query
        ip_attention = scaled_dot_product_attention(
            ip_query,
            ip_key,
            ip_value
        )
        # ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")
        ip_attention = ip_attention.permute(0, 2, 1, 3).reshape(B, L, -1)

        attn_out = attn_1 + ip_scale * ip_attention

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(mint.cat((attn_out, attn.mlp_act(mlp)), 2))
        out = x + mod.gate * output

        return out


class SingleStreamBlockLoraProcessor(ms.nn.Cell):
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def construct(self, attn: ms.nn.Cell, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = mint.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        qkv = qkv + self.qkv_lora(x_mod) * self.lora_weight

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        B, L, _ = qkv.shape
        qkv_reshaped = qkv.reshape(B, L, 3, attn.num_heads, -1)
        q, k, v = qkv_reshaped.permute(2, 0, 3, 1, 4)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(mint.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = output + self.proj_lora(mint.cat((attn_1, attn.mlp_act(mlp)), 2)) * self.lora_weight
        output = x + mod.gate * output
        return output


class SingleStreamBlockProcessor:
    def __call__(self, attn: ms.nn.Cell, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = mint.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        B, L, _ = qkv.shape
        qkv_reshaped = qkv.reshape(B, L, 3, attn.num_heads, -1)
        q, k, v = qkv_reshaped.permute(2, 0, 3, 1, 4)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(mint.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = x + mod.gate * output
        return output

class SingleStreamBlock(ms.nn.Cell):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = mint.nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = mint.nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = mint.nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        processor = SingleStreamBlockProcessor()
        self.set_processor(processor)


    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def construct(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0
    ) -> Tensor:
        if image_proj is None:
            return self.processor(self, x, vec, pe)
        else:
            return self.processor(self, x, vec, pe, image_proj, ip_scale)



class LastLayer(ms.nn.Cell):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = mint.nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = ms.nn.SequentialCell(mint.nn.SiLU(), mint.nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def construct(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x

class ImageProjModel(ms.nn.Cell):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = mint.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = mint.nn.LayerNorm(cross_attention_dim)

    def construct(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

