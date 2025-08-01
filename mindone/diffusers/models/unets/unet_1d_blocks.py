# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple, Union

import mindspore as ms
from mindspore import mint, nn, ops

from ..activations import get_activation
from ..layers_compat import conv_transpose1d, pad
from ..resnet import Downsample1D, ResidualTemporalBlock1D, Upsample1D, rearrange_dims


class DownResnetBlock1D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        conv_shortcut: bool = False,
        temb_channels: int = 32,
        groups: int = 32,
        groups_out: Optional[int] = None,
        non_linearity: Optional[str] = None,
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.non_linearity = non_linearity
        self.time_embedding_norm = time_embedding_norm
        self.add_downsample = add_downsample
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=temb_channels)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels))

        self.resnets = nn.CellList(resnets)

        if self.non_linearity is not None:
            self.nonlinearity = get_activation(non_linearity)

        if self.add_downsample:
            self.downsample = Downsample1D(out_channels, use_conv=True, padding=1)

    def construct(self, hidden_states: ms.Tensor, temb: Optional[ms.Tensor] = None) -> ms.Tensor:
        output_states = ()

        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        output_states += (hidden_states,)

        if self.non_linearity is not None:
            hidden_states = self.nonlinearity(hidden_states)

        if self.add_downsample:
            hidden_states = self.downsample(hidden_states)

        return hidden_states, output_states


class UpResnetBlock1D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        temb_channels: int = 32,
        groups: int = 32,
        groups_out: Optional[int] = None,
        non_linearity: Optional[str] = None,
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.non_linearity = non_linearity
        self.time_embedding_norm = time_embedding_norm
        self.add_upsample = add_upsample
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(2 * in_channels, out_channels, embed_dim=temb_channels)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels))

        self.resnets = nn.CellList(resnets)

        if self.non_linearity is not None:
            self.nonlinearity = get_activation(non_linearity)

        if self.add_upsample:
            self.upsample = Upsample1D(out_channels, use_conv_transpose=True)

    def construct(
        self,
        hidden_states: ms.Tensor,
        res_hidden_states_tuple: Optional[Tuple[ms.Tensor, ...]] = None,
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        if res_hidden_states_tuple is not None:
            res_hidden_states = res_hidden_states_tuple[-1]
            hidden_states = mint.cat((hidden_states, res_hidden_states), dim=1)

        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        if self.non_linearity is not None:
            hidden_states = self.nonlinearity(hidden_states)

        if self.add_upsample:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


class ValueFunctionMidBlock1D(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        self.res1 = ResidualTemporalBlock1D(in_channels, in_channels // 2, embed_dim=embed_dim)
        self.down1 = Downsample1D(out_channels // 2, use_conv=True)
        self.res2 = ResidualTemporalBlock1D(in_channels // 2, in_channels // 4, embed_dim=embed_dim)
        self.down2 = Downsample1D(out_channels // 4, use_conv=True)

    def construct(self, x: ms.Tensor, temb: Optional[ms.Tensor] = None) -> ms.Tensor:
        x = self.res1(x, temb)
        x = self.down1(x)
        x = self.res2(x, temb)
        x = self.down2(x)
        return x


class MidResTemporalBlock1D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_layers: int = 1,
        add_downsample: bool = False,
        add_upsample: bool = False,
        non_linearity: Optional[str] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_downsample = add_downsample
        self.add_upsample = add_upsample
        self.non_linearity = non_linearity

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=embed_dim)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=embed_dim))

        self.resnets = nn.CellList(resnets)

        if self.non_linearity is not None:
            self.nonlinearity = get_activation(non_linearity)

        if self.add_upsample:
            self.upsample = Upsample1D(out_channels, use_conv=True)

        if self.add_downsample:
            self.downsample = Downsample1D(out_channels, use_conv=True)

        if self.add_upsample and self.add_downsample:
            raise ValueError("Block cannot downsample and upsample")

    def construct(self, hidden_states: ms.Tensor, temb: ms.Tensor) -> ms.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        if self.add_upsample:
            hidden_states = self.upsample(hidden_states)
        if self.add_downsample:
            hidden_states = self.downsample(hidden_states)

        return hidden_states


class OutConv1DBlock(nn.Cell):
    def __init__(self, num_groups_out: int, out_channels: int, embed_dim: int, act_fn: str):
        super().__init__()
        # todo: unavailable mint interface
        self.final_conv1d_1 = nn.Conv1d(embed_dim, embed_dim, 5, padding=2, has_bias=True, pad_mode="pad")
        self.final_conv1d_gn = mint.nn.GroupNorm(num_groups_out, embed_dim)
        self.final_conv1d_act = get_activation(act_fn)
        # todo: unavailable mint interface
        self.final_conv1d_2 = nn.Conv1d(embed_dim, out_channels, 1, has_bias=True, pad_mode="valid")

    def construct(self, hidden_states: ms.Tensor, temb: Optional[ms.Tensor] = None) -> ms.Tensor:
        hidden_states = self.final_conv1d_1(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_gn(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_act(hidden_states)
        hidden_states = self.final_conv1d_2(hidden_states)
        return hidden_states


class OutValueFunctionBlock(nn.Cell):
    def __init__(self, fc_dim: int, embed_dim: int, act_fn: str = "mish"):
        super().__init__()
        self.final_block = nn.CellList(
            [
                mint.nn.Linear(fc_dim + embed_dim, fc_dim // 2),
                get_activation(act_fn),
                mint.nn.Linear(fc_dim // 2, 1),
            ]
        )

    def construct(self, hidden_states: ms.Tensor, temb: ms.Tensor) -> ms.Tensor:
        hidden_states = hidden_states.view(hidden_states.shape[0], -1)
        hidden_states = mint.cat((hidden_states, temb), dim=-1)
        for layer in self.final_block:
            hidden_states = layer(hidden_states)

        return hidden_states


_kernels = {
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "cubic": [-0.01171875, -0.03515625, 0.11328125, 0.43359375, 0.43359375, 0.11328125, -0.03515625, -0.01171875],
    "lanczos3": [
        0.003689131001010537,
        0.015056144446134567,
        -0.03399861603975296,
        -0.066637322306633,
        0.13550527393817902,
        0.44638532400131226,
        0.44638532400131226,
        0.13550527393817902,
        -0.066637322306633,
        -0.03399861603975296,
        0.015056144446134567,
        0.003689131001010537,
    ],
}


class Downsample1d(nn.Cell):
    def __init__(self, kernel: str = "linear", pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = ms.tensor(_kernels[kernel])
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.kernel = ms.Parameter(kernel_1d, name="kernel")

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        dtype = hidden_states.dtype
        hidden_states = pad(hidden_states, (self.pad,) * 2, self.pad_mode)
        weight = hidden_states.new_zeros(
            [hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]], dtype=hidden_states.dtype
        )
        indices = mint.arange(hidden_states.shape[1])
        kernel = self.kernel.to(weight.dtype)[None, :].broadcast_to((hidden_states.shape[1], -1))
        weight[indices, indices] = kernel
        # todo: unavailabel mint interface
        return ops.conv1d(hidden_states.float(), weight.float(), stride=2).to(dtype)


class Upsample1d(nn.Cell):
    def __init__(self, kernel: str = "linear", pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = ms.tensor(_kernels[kernel]) * 2
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.kernel = ms.Parameter(kernel_1d, name="kernel")

    def construct(self, hidden_states: ms.Tensor, temb: Optional[ms.Tensor] = None) -> ms.Tensor:
        hidden_states = pad(hidden_states, ((self.pad + 1) // 2,) * 2, self.pad_mode)
        weight = hidden_states.new_zeros(
            [hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]], dtype=hidden_states.dtype
        )
        indices = mint.arange(hidden_states.shape[1])
        kernel = self.kernel.to(weight.dtype)[None, :].broadcast_to((hidden_states.shape[1], -1))
        weight[indices, indices] = kernel
        return conv_transpose1d(hidden_states, weight, stride=2, padding=self.pad * 2 + 1)


class SelfAttention1d(nn.Cell):
    def __init__(self, in_channels: int, n_head: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        self.channels = in_channels
        self.group_norm = mint.nn.GroupNorm(1, num_channels=in_channels)
        self.num_heads = n_head

        self.query = mint.nn.Linear(self.channels, self.channels)
        self.key = mint.nn.Linear(self.channels, self.channels)
        self.value = mint.nn.Linear(self.channels, self.channels)

        self.proj_attn = mint.nn.Linear(self.channels, self.channels, bias=True)

        self.dropout = mint.nn.Dropout(p=dropout_rate)

    def transpose_for_scores(self, projection: ms.Tensor) -> ms.Tensor:
        new_projection_shape = projection.shape[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        residual = hidden_states
        batch, channel_dim, seq = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.swapaxes(1, 2)

        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        scale = float(1 / mint.sqrt(mint.sqrt(ms.tensor(key_states.shape[-1], dtype=ms.float64))))

        attention_scores = mint.matmul(query_states * scale, mint.transpose(key_states, -1, -2) * scale)
        attention_probs = mint.nn.functional.softmax(attention_scores, dim=-1)

        # compute attention output
        hidden_states = mint.matmul(attention_probs, value_states)

        hidden_states = hidden_states.permute(0, 2, 1, 3)
        new_hidden_states_shape = hidden_states.shape[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.swapaxes(1, 2)
        hidden_states = self.dropout(hidden_states)

        output = hidden_states + residual

        return output


class ResConvBlock(nn.Cell):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, is_last: bool = False):
        super().__init__()
        self.is_last = is_last
        self.has_conv_skip = in_channels != out_channels

        if self.has_conv_skip:
            # todo: unavailable mint interface
            self.conv_skip = nn.Conv1d(in_channels, out_channels, 1, pad_mode="valid")

        # todo: unavailable mint interface
        self.conv_1 = nn.Conv1d(in_channels, mid_channels, 5, padding=2, has_bias=True, pad_mode="pad")
        self.group_norm_1 = mint.nn.GroupNorm(1, mid_channels)
        self.gelu_1 = mint.nn.GELU()
        # todo: unavailable mint interface
        self.conv_2 = nn.Conv1d(mid_channels, out_channels, 5, padding=2, has_bias=True, pad_mode="pad")

        if not self.is_last:
            self.group_norm_2 = mint.nn.GroupNorm(1, out_channels)
            self.gelu_2 = mint.nn.GELU()

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        residual = self.conv_skip(hidden_states) if self.has_conv_skip else hidden_states

        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.group_norm_1(hidden_states)
        hidden_states = self.gelu_1(hidden_states)
        hidden_states = self.conv_2(hidden_states)

        if not self.is_last:
            hidden_states = self.group_norm_2(hidden_states)
            hidden_states = self.gelu_2(hidden_states)

        output = hidden_states + residual
        return output


class UNetMidBlock1D(nn.Cell):
    def __init__(self, mid_channels: int, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        # there is always at least one resnet
        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]
        self.up = Upsample1d(kernel="cubic")

        self.attentions = nn.CellList(attentions)
        self.resnets = nn.CellList(resnets)

    def construct(self, hidden_states: ms.Tensor, temb: Optional[ms.Tensor] = None) -> ms.Tensor:
        hidden_states = self.down(hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class AttnDownBlock1D(nn.Cell):
    def __init__(self, out_channels: int, in_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]

        self.attentions = nn.CellList(attentions)
        self.resnets = nn.CellList(resnets)

    def construct(self, hidden_states: ms.Tensor, temb: Optional[ms.Tensor] = None) -> ms.Tensor:
        hidden_states = self.down(hidden_states)

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        return hidden_states, (hidden_states,)


class DownBlock1D(nn.Cell):
    def __init__(self, out_channels: int, in_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        self.resnets = nn.CellList(resnets)

    def construct(self, hidden_states: ms.Tensor, temb: Optional[ms.Tensor] = None) -> ms.Tensor:
        hidden_states = self.down(hidden_states)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states, (hidden_states,)


class DownBlock1DNoSkip(nn.Cell):
    def __init__(self, out_channels: int, in_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        self.resnets = nn.CellList(resnets)

    def construct(self, hidden_states: ms.Tensor, temb: Optional[ms.Tensor] = None) -> ms.Tensor:
        hidden_states = mint.cat([hidden_states, temb], dim=1)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states, (hidden_states,)


class AttnUpBlock1D(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]

        self.attentions = nn.CellList(attentions)
        self.resnets = nn.CellList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def construct(
        self,
        hidden_states: ms.Tensor,
        res_hidden_states_tuple: Tuple[ms.Tensor, ...],
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = mint.cat([hidden_states, res_hidden_states], dim=1)

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class UpBlock1D(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        self.resnets = nn.CellList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def construct(
        self,
        hidden_states: ms.Tensor,
        res_hidden_states_tuple: Tuple[ms.Tensor, ...],
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = mint.cat([hidden_states, res_hidden_states], dim=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class UpBlock1DNoSkip(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels, is_last=True),
        ]

        self.resnets = nn.CellList(resnets)

    def construct(
        self,
        hidden_states: ms.Tensor,
        res_hidden_states_tuple: Tuple[ms.Tensor, ...],
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = mint.cat([hidden_states, res_hidden_states], dim=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states


DownBlockType = Union[DownResnetBlock1D, DownBlock1D, AttnDownBlock1D, DownBlock1DNoSkip]
MidBlockType = Union[MidResTemporalBlock1D, ValueFunctionMidBlock1D, UNetMidBlock1D]
OutBlockType = Union[OutConv1DBlock, OutValueFunctionBlock]
UpBlockType = Union[UpResnetBlock1D, UpBlock1D, AttnUpBlock1D, UpBlock1DNoSkip]


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
) -> DownBlockType:
    if down_block_type == "DownResnetBlock1D":
        return DownResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
        )
    elif down_block_type == "DownBlock1D":
        return DownBlock1D(out_channels=out_channels, in_channels=in_channels)
    elif down_block_type == "AttnDownBlock1D":
        return AttnDownBlock1D(out_channels=out_channels, in_channels=in_channels)
    elif down_block_type == "DownBlock1DNoSkip":
        return DownBlock1DNoSkip(out_channels=out_channels, in_channels=in_channels)
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type: str, num_layers: int, in_channels: int, out_channels: int, temb_channels: int, add_upsample: bool
) -> UpBlockType:
    if up_block_type == "UpResnetBlock1D":
        return UpResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
        )
    elif up_block_type == "UpBlock1D":
        return UpBlock1D(in_channels=in_channels, out_channels=out_channels)
    elif up_block_type == "AttnUpBlock1D":
        return AttnUpBlock1D(in_channels=in_channels, out_channels=out_channels)
    elif up_block_type == "UpBlock1DNoSkip":
        return UpBlock1DNoSkip(in_channels=in_channels, out_channels=out_channels)
    raise ValueError(f"{up_block_type} does not exist.")


def get_mid_block(
    mid_block_type: str,
    num_layers: int,
    in_channels: int,
    mid_channels: int,
    out_channels: int,
    embed_dim: int,
    add_downsample: bool,
) -> MidBlockType:
    if mid_block_type == "MidResTemporalBlock1D":
        return MidResTemporalBlock1D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            add_downsample=add_downsample,
        )
    elif mid_block_type == "ValueFunctionMidBlock1D":
        return ValueFunctionMidBlock1D(in_channels=in_channels, out_channels=out_channels, embed_dim=embed_dim)
    elif mid_block_type == "UNetMidBlock1D":
        return UNetMidBlock1D(in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels)
    raise ValueError(f"{mid_block_type} does not exist.")


def get_out_block(
    *, out_block_type: str, num_groups_out: int, embed_dim: int, out_channels: int, act_fn: str, fc_dim: int
) -> Optional[OutBlockType]:
    if out_block_type == "OutConv1DBlock":
        return OutConv1DBlock(num_groups_out, out_channels, embed_dim, act_fn)
    elif out_block_type == "ValueFunction":
        return OutValueFunctionBlock(fc_dim, embed_dim, act_fn)
    return None
