# Copyright 2025 The Genmo team and The HuggingFace Team.
# All rights reserved.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union

import mindspore as ms
from mindspore import mint, nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import logging
from ..attention import FeedForward
from ..attention_processor import Attention
from ..embeddings import PixArtAlphaTextProjection
from ..layers_compat import unflatten
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormSingle, RMSNorm

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LTXVideoAttentionProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the LTX model. It applies a normalization layer and rotary embedding on the query and key vector.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = unflatten(query, 2, (attn.heads, -1)).swapaxes(1, 2)
        key = unflatten(key, 2, (attn.heads, -1)).swapaxes(1, 2)
        value = unflatten(value, 2, (attn.heads, -1)).swapaxes(1, 2)

        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.swapaxes(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LTXVideoRotaryPosEmbed(nn.Cell):
    def __init__(
        self,
        dim: int,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        patch_size: int = 1,
        patch_size_t: int = 1,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_num_frames = base_num_frames
        self.base_height = base_height
        self.base_width = base_width
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.theta = theta

    def _prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        rope_interpolation_scale: Tuple[ms.Tensor, float, float],
    ) -> ms.Tensor:
        # Always compute rope in fp32
        grid_h = mint.arange(height, dtype=ms.float32)
        grid_w = mint.arange(width, dtype=ms.float32)
        grid_f = mint.arange(num_frames, dtype=ms.float32)
        grid = mint.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = mint.stack(grid, dim=0)
        grid = grid.unsqueeze(0).tile((batch_size, 1, 1, 1, 1))

        if rope_interpolation_scale is not None:
            grid[:, 0:1] = grid[:, 0:1] * rope_interpolation_scale[0] * self.patch_size_t / self.base_num_frames
            grid[:, 1:2] = grid[:, 1:2] * rope_interpolation_scale[1] * self.patch_size / self.base_height
            grid[:, 2:3] = grid[:, 2:3] * rope_interpolation_scale[2] * self.patch_size / self.base_width

        grid = grid.flatten(2, 4).swapaxes(1, 2)

        return grid

    def construct(
        self,
        hidden_states: ms.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Tuple[ms.Tensor, float, float]] = None,
        video_coords: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        batch_size = hidden_states.shape[0]

        if video_coords is None:
            grid = self._prepare_video_coords(
                batch_size,
                num_frames,
                height,
                width,
                rope_interpolation_scale=rope_interpolation_scale,
            )
        else:
            grid = mint.stack(
                [
                    video_coords[:, 0] / self.base_num_frames,
                    video_coords[:, 1] / self.base_height,
                    video_coords[:, 2] / self.base_width,
                ],
                dim=-1,
            )

        start = 1.0
        end = self.theta
        freqs = self.theta ** mint.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 6,
            dtype=ms.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid.unsqueeze(-1) * 2 - 1)
        freqs = freqs.swapaxes(-1, -2).flatten(2)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 6 != 0:
            cos_padding = mint.ones_like(cos_freqs[:, :, : self.dim % 6])
            sin_padding = mint.zeros_like(cos_freqs[:, :, : self.dim % 6])
            cos_freqs = mint.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = mint.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs


class LTXVideoTransformerBlock(nn.Cell):
    r"""
    Transformer block used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=LTXVideoAttentionProcessor2_0(),
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=LTXVideoAttentionProcessor2_0(),
        )

        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.scale_shift_table = ms.Parameter(mint.randn(6, dim) / dim**0.5, name="scale_shift_table")

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        temb: ms.Tensor,
        image_rotary_emb: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        batch_size = hidden_states.shape[0]
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.shape[1], num_ada_params, -1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class LTXVideoTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    r"""
    A Transformer model for video-like data used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, defaults to `128`):
            The number of channels in the output.
        patch_size (`int`, defaults to `1`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        cross_attention_dim (`int`, defaults to `2048 `):
            The number of channels for cross attention heads.
        num_layers (`int`, defaults to `28`):
            The number of layers of Transformer blocks to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        qk_norm (`str`, defaults to `"rms_norm_across_heads"`):
            The normalization layer to use.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        num_layers: int = 28,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        self.proj_in = mint.nn.Linear(in_channels, inner_dim)

        self.scale_shift_table = ms.Parameter(mint.randn(2, inner_dim) / inner_dim**0.5, name="scale_shift_table")
        self.time_embed = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.rope = LTXVideoRotaryPosEmbed(
            dim=inner_dim,
            base_num_frames=20,
            base_height=2048,
            base_width=2048,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            theta=10000.0,
        )

        self.transformer_blocks = nn.CellList(
            [
                LTXVideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = mint.nn.LayerNorm(inner_dim, eps=1e-6)
        self.proj_out = mint.nn.Linear(inner_dim, out_channels)

        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        timestep: ms.Tensor,
        encoder_attention_mask: ms.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Union[Tuple[float, float, float], ms.Tensor]] = None,
        video_coords: Optional[ms.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> ms.Tensor:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()

        image_rotary_emb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale, video_coords)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.shape[0]
        hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        temb = temb.view(batch_size, -1, temb.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
            )

        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def apply_rotary_emb(x, freqs):
    cos, sin = freqs
    x_real, x_imag = unflatten(x, 2, (-1, 2)).unbind(-1)  # [B, S, C // 2]
    x_rotated = mint.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out
