# Copyright (c) 2023 Dominic Rampas MIT License
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

import math
from typing import Dict, Union

from mindspore import mint, nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from ...models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from ...models.modeling_utils import ModelMixin
from .modeling_wuerstchen_common import AttnBlock, ResBlock, TimestepBlock, WuerstchenLayerNorm


class WuerstchenPrior(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin):
    unet_name = "prior"
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, c_in=16, c=1280, c_cond=1024, c_r=64, depth=16, nhead=16, dropout=0.1):
        super().__init__()

        self.c_r = c_r
        self.projection = mint.nn.Conv2d(c_in, c, kernel_size=1)
        self.cond_mapper = nn.SequentialCell(
            mint.nn.Linear(c_cond, c),
            # todo: unavailable mint interface
            nn.LeakyReLU(0.2),
            mint.nn.Linear(c, c),
        )

        blocks = []
        for _ in range(depth):
            blocks.append(ResBlock(c, dropout=dropout))
            blocks.append(TimestepBlock(c, c_r))
            blocks.append(AttnBlock(c, c, nhead, self_attn=True, dropout=dropout))
        self.blocks = nn.CellList(blocks)
        self.out = nn.SequentialCell(
            WuerstchenLayerNorm(c, elementwise_affine=False, eps=1e-6),
            mint.nn.Conv2d(c, c_in * 2, kernel_size=1),
        )

        self._gradient_checkpointing = False
        self.set_default_attn_processor()

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: nn.Cell, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.name_cells().items():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.name_cells().items():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: nn.Cell, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.name_cells().items():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.name_cells().items():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = mint.arange(half_dim).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = mint.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = mint.nn.functional.pad(emb, (0, 1), mode="constant")
        return emb.to(dtype=r.dtype)

    def construct(self, x, r, c):
        x_in = x
        x = self.projection(x)
        c_embed = self.cond_mapper(c)
        r_embed = self.gen_r_embedding(r)

        for block in self.blocks:
            if isinstance(block, AttnBlock):
                x = block(x, c_embed)
            elif isinstance(block, TimestepBlock):
                x = block(x, r_embed)
            else:
                x = block(x)
        a, b = self.out(x).chunk(2, dim=1)
        return (x_in - a) / ((1 - b).abs() + 1e-5)
