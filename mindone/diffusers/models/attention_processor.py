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
from typing import Callable, List, Optional, Tuple, Union

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import mint, nn, ops

from ..image_processor import IPAdapterMaskProcessor
from ..utils import deprecate, is_mindspore_version, logging
from ..utils.mindspore_utils import dtype_to_min
from .layers_compat import unflatten

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Attention(nn.Cell):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        kv_heads (`int`,  *optional*, defaults to `None`):
            The number of key and value heads to use for multi-head attention. Defaults to `heads`. If
            `kv_heads=heads`, the model will use Multi Head Attention (MHA), if `kv_heads=1` the model will use Multi
            Query Attention (MQA) otherwise GQA is used.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()

        # To prevent circular import.
        from .normalization import FP32LayerNorm, LpNorm, RMSNorm

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        # use `scale_sqrt` for ops.baddbmm to get same outputs with torch.baddbmm in fp16
        self.scale_sqrt = float(math.sqrt(self.scale))

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."  # noqa: E501
            )

        if norm_num_groups is not None:
            self.group_norm = mint.nn.GroupNorm(
                num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True
            )
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = mint.nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = mint.nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "fp32_layer_norm":
            self.norm_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            self.norm_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
        elif qk_norm == "layer_norm_across_heads":
            # Lumina applies qk norm across all heads
            self.norm_q = mint.nn.LayerNorm(dim_head * heads, eps=eps)
            self.norm_k = mint.nn.LayerNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim_head * heads, eps=eps)
            self.norm_k = RMSNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "l2":
            self.norm_q = LpNorm(p=2, dim=-1, eps=eps)
            self.norm_k = LpNorm(p=2, dim=-1, eps=eps)
        else:
            raise ValueError(
                f"unknown qk_norm: {qk_norm}. Should be one of None, 'layer_norm', 'fp32_layer_norm', 'layer_norm_across_heads', 'rms_norm', 'rms_norm_across_heads', 'l2'."  # noqa
            )

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = mint.nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = mint.nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.to_q = mint.nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = mint.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
            self.to_v = mint.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = mint.nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            self.add_v_proj = mint.nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            if self.context_pre_only is not None:
                self.add_q_proj = mint.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
        else:
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None

        if not self.pre_only:
            self.to_out = nn.CellList(
                [mint.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias), mint.nn.Dropout(p=dropout)]
            )
        else:
            self.to_out = None

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = mint.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)
        else:
            self.to_add_out = None

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "layer_norm":
                self.norm_added_q = mint.nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
                self.norm_added_k = mint.nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            elif qk_norm == "fp32_layer_norm":
                self.norm_added_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
                self.norm_added_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            elif qk_norm == "rms_norm":
                self.norm_added_q = RMSNorm(dim_head, eps=eps)
                self.norm_added_k = RMSNorm(dim_head, eps=eps)
            elif qk_norm == "rms_norm_across_heads":
                # Wan applies qk norm across all heads
                # Wan also doesn't apply a q norm
                self.norm_added_q = None
                self.norm_added_k = RMSNorm(dim_head * kv_heads, eps=eps)
            else:
                raise ValueError(
                    f"unknown qk_norm: {qk_norm}. Should be one of `None,'layer_norm','fp32_layer_norm','rms_norm'`"
                )
        else:
            self.norm_added_q = None
            self.norm_added_k = None

        # MindSpore flash attention settings
        # flash attention only supports fp16 and bf 16, force cast to fp16 defaultly
        self.fa_op_available = (
            is_mindspore_version(">=", "2.3.0") and ms.get_context("device_target").lower() == "ascend"
        )
        self._enable_flash_sdp = True
        self.set_flash_attention_force_cast_dtype(ms.float16)

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None:
            processor = AttnProcessor2_0() if self.scale_qk else AttnProcessor()
        self.processor = processor

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ) -> None:
        r"""
        Set whether to use memory efficient attention from `xformers` or not.

        Args:
            use_memory_efficient_attention_xformers (`bool`):
                Whether to use memory efficient attention from `xformers` or not.
            attention_op (`Callable`, *optional*):
                Not supported for now.
        """
        if use_memory_efficient_attention_xformers:
            if not hasattr(ops.operations.nn_ops, "FlashAttentionScore"):
                raise ModuleNotFoundError(
                    f"Memory efficient attention on mindspore uses flash attention under the hoods. "
                    f"The implementation of flash attention is `FlashAttentionScore`, "
                    f"which should be available in `mindspore.ops.operations.nn_ops`. "
                    f"However, we cannot find it in current environment(mindspore version: {ms.__version__})."
                )
            elif ms.get_context("device_target") != "Ascend":
                raise ValueError(
                    f"Memory efficient attention is only available for Ascend, "
                    f"but got current device: {ms.get_context('device_target')}"
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    flash_attn = ops.operations.nn_ops.FlashAttentionScore(1, input_layout="BSH")
                    q = mint.randn(1, 16, 64, dtype=ms.float16)
                    _ = flash_attn(q, q, q)
                except Exception as e:
                    raise e

            # The following lines is a patch for flash attn, which calculates implicit padding on head_dim.
            # TODO: Remove it if flash attention has better supports.
            import bisect

            self.flash_attn_valid_head_dims = [64, 80, 96, 120, 128, 256]
            self.head_dim = self.inner_dim // self.heads
            if self.head_dim in self.flash_attn_valid_head_dims:
                self.head_dim_padding = 0
            else:
                minimum_larger_index = bisect.bisect_right(self.flash_attn_valid_head_dims, self.head_dim)
                if minimum_larger_index >= len(self.flash_attn_valid_head_dims):
                    self.head_dim_padding = -1  # head_dim is bigger than the largest one, we cannot do padding
                else:
                    self.head_dim_padding = self.flash_attn_valid_head_dims[minimum_larger_index] - self.head_dim

            if self.head_dim_padding == 0:
                logger.info(
                    f"The head dimension of '{self.to_q.weight.name[:-12]}' is {self.head_dim}. "
                    f"Successfully set to use the flash attention."
                )
                processor = XFormersAttnProcessor(attention_op=attention_op)
            elif self.head_dim_padding > 0:
                logger.warning(
                    f"Flash attention requires that the head dimension must be one of "
                    f"{self.flash_attn_valid_head_dims}, but got {self.head_dim} in '{self.to_q.weight.name[:-12]}'. "
                    f"We will implicitly pad the head dimension to {self.head_dim + self.head_dim_padding}."
                )
                processor = XFormersAttnProcessor(attention_op=attention_op)
            else:
                logger.warning(
                    f"Flash attention requires that the head dimension must be one of "
                    f"{self.flash_attn_valid_head_dims}, but got {self.head_dim} in '{self.to_q.weight.name[:-12]}'. "
                    f"Fallback to the vanilla implementation of attention."
                )
                processor = AttnProcessor()

            # # The following lines is a patch for flash attn, which fallbacks to vanilla attn if head_dim is invalid.
            # # TODO: Remove it if flash attention has better supports.
            # self.flash_attn_valid_head_dims = [64, 80, 96, 120, 128, 256]
            # self.head_dim = self.inner_dim // self.heads
            # self.head_dim_padding = 0
            # if self.head_dim in self.flash_attn_valid_head_dims:
            #     logger.info(
            #         f"The head dimension of '{self.to_q.weight.name[:-12]}' is {self.head_dim}. "
            #         f"Successfully set to use the flash attention."
            #     )
            #     processor = XFormersAttnProcessor(attention_op=attention_op)
            # else:
            #     logger.warning(
            #         f"Flash attention requires that the head dimension must be one of "
            #         f"{self.flash_attn_valid_head_dims}, but got {self.head_dim} in '{self.to_q.weight.name[:-12]}'. "
            #         f"Fallback to the vanilla implementation of attention."
            #     )
            #     processor = AttnProcessor()
        else:
            # set attention processor
            # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
            # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
            processor = AttnProcessor()

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if hasattr(self, "processor") and isinstance(self.processor, nn.Cell) and not isinstance(processor, nn.Cell):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._cells.pop("processor")

        self.processor = processor

    def get_processor(self) -> "AttentionProcessor":
        r"""
        Get the attention processor in use.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        return self.processor

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        **cross_attention_kwargs,
    ) -> ms.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`ms.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`ms.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`ms.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `ms.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: ms.Tensor) -> ms.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`ms.Tensor`): The tensor to reshape.

        Returns:
            `ms.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor: ms.Tensor, out_dim: int = 3) -> ms.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`ms.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `ms.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

    def get_attention_scores(
        self, query: ms.Tensor, key: ms.Tensor, attention_mask: Optional[ms.Tensor] = None
    ) -> ms.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`ms.Tensor`): The query tensor.
            key (`ms.Tensor`): The key tensor.
            attention_mask (`ms.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `ms.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            attention_scores = mint.bmm(
                query * self.scale_sqrt,
                key.swapaxes(-1, -2) * self.scale_sqrt,
            )
        else:
            attention_scores = mint.baddbmm(
                attention_mask.to(query.dtype),
                query * self.scale_sqrt,
                key.swapaxes(-1, -2) * self.scale_sqrt,
                beta=1,
                alpha=1,
            )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = mint.softmax(attention_scores, dim=-1)

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(
        self, attention_mask: ms.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> Optional[ms.Tensor]:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`ms.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `ms.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length = attention_mask.shape[-1]
        if current_length != target_length:
            # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
            #       we want to instead pad by (0, remaining_length), where remaining_length is:
            #       remaining_length: int = target_length - current_length
            # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
            attention_mask = mint.nn.functional.pad(attention_mask, (0, target_length), value=0.0)

        # bool input dtype is not supported in tensor.repeat_interleave
        attention_mask_dtype = attention_mask.dtype
        if attention_mask_dtype == ms.bool_:
            attention_mask = attention_mask.float()

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(
                    head_size, dim=0, output_size=attention_mask.shape[0] * head_size
                )
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(
                head_size, dim=1, output_size=attention_mask.shape[1] * head_size
            )

        if attention_mask_dtype == ms.bool_:
            attention_mask = attention_mask.bool()

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states: ms.Tensor) -> ms.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`ms.Tensor`): Hidden states of the encoder.

        Returns:
            `ms.Tensor`: The normalized encoder hidden states.
        """
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, mint.nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, mint.nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.swapaxes(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.swapaxes(1, 2)
        else:
            assert False

        return encoder_hidden_states

    def fuse_projections(self, fuse=True):
        dtype = self.to_q.weight.dtype

        if not self.is_cross_attention:
            # fetch weight matrices.
            concatenated_weights = mint.cat([self.to_q.weight, self.to_k.weight, self.to_v.weight])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            # create a new single projection layer and copy over the weights.
            self.to_qkv = mint.nn.Linear(in_features, out_features, bias=self.use_bias, dtype=dtype)
            self.to_qkv.weight.set_data(concatenated_weights)
            if self.use_bias:
                concatenated_bias = mint.cat([self.to_q.bias, self.to_k.bias, self.to_v.bias])
                self.to_qkv.bias.set_data(concatenated_bias)

        else:
            concatenated_weights = mint.cat([self.to_k.weight, self.to_v.weight])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_kv = mint.nn.Linear(in_features, out_features, bias=self.use_bias, dtype=dtype)
            self.to_kv.weight.set_data(concatenated_weights)
            if self.use_bias:
                concatenated_bias = mint.cat([self.to_k.bias, self.to_v.bias])
                self.to_kv.bias.set_data(concatenated_bias)

        # handle added projections for SD3 and others.
        if (
            getattr(self, "add_q_proj", None) is not None
            and getattr(self, "add_k_proj", None) is not None
            and getattr(self, "add_v_proj", None) is not None
        ):
            concatenated_weights = mint.cat([self.add_q_proj.weight, self.add_k_proj.weight, self.add_v_proj.weight])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_added_qkv = mint.nn.Linear(in_features, out_features, bias=self.added_proj_bias, dtype=dtype)
            self.to_added_qkv.weight.set_data(concatenated_weights)
            if self.added_proj_bias:
                concatenated_bias = mint.cat([self.add_q_proj.bias, self.add_k_proj.bias, self.add_v_proj.bias])
                self.to_added_qkv.bias.set_data(concatenated_bias)

        self.fused_projections = fuse

    def enable_flash_sdp(self, enabled: bool):
        r"""
        .. warning:: This flag is beta and subject to change.

        Enables or disables flash scaled dot product attention.
        """
        self._enable_flash_sdp = enabled

    def set_flash_attention_force_cast_dtype(self, force_cast_dtype: Optional[ms.Type]):
        r"""
        Since the flash-attention operator in MindSpore only supports float16 and bfloat16 data types,
        we need to manually set whether to force data type conversion.

        When the attention interface encounters data of an unsupported data type, if `force_cast_dtype`
        is not None, the function will forcibly convert the data to `force_cast_dtype` for computation
        and then restore it to the original data type afterward. If `force_cast_dtype` is None, it will
        fall back to the original attention calculation using mathematical formulas.

        Parameters:
            force_cast_dtype (Optional): The data type to which the input data should be forcibly converted.
                                         If None, no forced conversion is performed.
        """
        self.fa_force_dtype = force_cast_dtype

    def scaled_dot_product_attention(
        self,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        attn_mask: Optional[ms.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ):
        r"""
        Perform scaled dot-product attention using either the flash attention operator or the mathematical
        formula-based attention, depending on the availability of the flash attention operator and the
        data-types of the inputs.

        Parameters:
            query (ms.Tensor): The query tensor.
            key (ms.Tensor): The key tensor.
            value (ms.Tensor): The value tensor.
            attn_mask (Optional[ms.Tensor], optional): The attention mask tensor. Defaults to None.
            dropout_p (float, optional): The dropout probability. Defaults to 0.0.
            is_causal (bool): Un-used. Aligned with Torch
            scale (float, optional): scaled value

        Returns:
            ms.Tensor: The result of the scaled dot-product attention.

        Notes:
            - If the flash attention operator is not available (`self.fa_op_available` is False),
              the function falls back to the mathematical formula-based attention.
            - If the data types of `query`, `key`, and `value` are either `float16` or `bfloat16`, the
              flash-attention operator is used directly.
            - If `self.fa_force_dtype` is set to `float16` or `bfloat16`, the input tensors are cast to
              this data-type, the flash attention operator is applied, and the result is cast back to the
              original data type of `query`.
            - Otherwise, the function falls back to the mathematical formula-based attention.
        """
        head_dim = query.shape[-1]

        if not (self.fa_op_available and self._enable_flash_sdp):
            return self.math_attention_op(query, key, value, attn_mask)
        elif head_dim > 512:
            logger.warning("Flash attention requires that the head dimension must <= 512")
            return self.math_attention_op(query, key, value, attn_mask)
        elif query.dtype in (ms.float16, ms.bfloat16):
            return self.flash_attention_op(query, key, value, attn_mask, keep_prob=1 - dropout_p, scale=scale)
        elif self.fa_force_dtype in (ms.float16, ms.bfloat16):
            return self.flash_attention_op(
                query.to(self.fa_force_dtype),
                key.to(self.fa_force_dtype),
                value.to(self.fa_force_dtype),
                attn_mask,
                keep_prob=1 - dropout_p,
                scale=scale,
            ).to(query.dtype)
        else:
            return self.math_attention_op(query, key, value, attn_mask)

    def math_attention_op(
        self,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        attn_mask: Optional[ms.Tensor] = None,
    ):
        # Adapted from mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.construct
        if attn_mask is not None and attn_mask.dtype == ms.bool_:
            attn_mask = mint.logical_not(attn_mask) * dtype_to_min(query.dtype)

        has_extra_dims = query.ndim > 3
        # adapt to graph mode
        origin_query_shape = None
        if has_extra_dims:
            origin_query_shape = query.shape
            query = query.reshape(-1, query.shape[-2], query.shape[-1])
            key = key.reshape(-1, key.shape[-2], key.shape[-1])
            value = value.reshape(-1, value.shape[-2], value.shape[-1])
        else:
            origin_query_shape = None

        attention_probs = self.get_attention_scores(query, key, attn_mask)
        hidden_states = mint.bmm(attention_probs, value)

        if has_extra_dims:
            hidden_states = hidden_states.reshape(*origin_query_shape)

        return hidden_states

    def flash_attention_op(
        self,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        attn_mask: Optional[ms.Tensor] = None,
        keep_prob: float = 1.0,
        scale: Optional[float] = None,
    ):
        # For most scenarios, qkv has been processed into a BNSD layout before sdp
        input_layout = "BNSD"
        head_num = self.heads

        # In case qkv is 3-dim after `head_to_batch_dim`
        if query.ndim == 3:
            input_layout = "BSH"
            head_num = 1

        # process `attn_mask` as logic is different between PyTorch and Mindspore
        # In MindSpore, False indicates retention and True indicates discard, in PyTorch it is the opposite
        if attn_mask is not None:
            attn_mask = mint.logical_not(attn_mask) if attn_mask.dtype == ms.bool_ else attn_mask.bool()
            attn_mask = mint.broadcast_to(
                attn_mask, (attn_mask.shape[0], attn_mask.shape[1], query.shape[-2], key.shape[-2])
            )[:, :1, :, :]

        return ops.operations.nn_ops.FlashAttentionScore(
            head_num=head_num, keep_prob=keep_prob, scale_value=scale or self.scale, input_layout=input_layout
        )(query, key, value, None, None, None, attn_mask)[3]


class SanaMultiscaleAttentionProjection(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        kernel_size: int,
    ) -> None:
        super().__init__()

        channels = 3 * in_channels
        self.proj_in = mint.nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.proj_out = mint.nn.Conv2d(channels, channels, 1, 1, padding=0, groups=3 * num_attention_heads, bias=False)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states


class SanaMultiscaleLinearAttention(nn.Cell):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_attention_heads: Optional[int] = None,
        attention_head_dim: int = 8,
        mult: float = 1.0,
        norm_type: str = "batch_norm",
        kernel_sizes: Tuple[int, ...] = (5,),
        eps: float = 1e-15,
        residual_connection: bool = False,
    ):
        super().__init__()

        # To prevent circular import
        from .normalization import get_normalization

        self.eps = eps
        self.attention_head_dim = attention_head_dim
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        num_attention_heads = (
            int(in_channels // attention_head_dim * mult) if num_attention_heads is None else num_attention_heads
        )
        inner_dim = num_attention_heads * attention_head_dim

        self.to_q = mint.nn.Linear(in_channels, inner_dim, bias=False)
        self.to_k = mint.nn.Linear(in_channels, inner_dim, bias=False)
        self.to_v = mint.nn.Linear(in_channels, inner_dim, bias=False)

        to_qkv_multiscale = []
        for kernel_size in kernel_sizes:
            to_qkv_multiscale.append(SanaMultiscaleAttentionProjection(inner_dim, num_attention_heads, kernel_size))
        self.to_qkv_multiscale = nn.CellList(to_qkv_multiscale)

        self.nonlinearity = mint.nn.ReLU()
        self.to_out = mint.nn.Linear(inner_dim * (1 + len(kernel_sizes)), out_channels, bias=False)
        self.norm_out = get_normalization(norm_type, num_features=out_channels)

        self.processor = SanaMultiscaleAttnProcessor2_0()

    def apply_linear_attention(self, query: ms.Tensor, key: ms.Tensor, value: ms.Tensor) -> ms.Tensor:
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)  # Adds padding
        scores = mint.matmul(value, key.swapaxes(-1, -2))
        hidden_states = mint.matmul(scores, query)

        hidden_states = hidden_states.to(dtype=ms.float32)
        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)
        return hidden_states

    def apply_quadratic_attention(self, query: ms.Tensor, key: ms.Tensor, value: ms.Tensor) -> ms.Tensor:
        scores = mint.matmul(key.swapaxes(-1, -2), query)
        scores = scores.to(dtype=ms.float32)
        scores = scores / (mint.sum(scores, dim=2, keepdim=True) + self.eps)
        hidden_states = mint.matmul(value, scores.to(value.dtype))
        return hidden_states

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        return self.processor(self, hidden_states)


class MochiAttention(nn.Cell):
    def __init__(
        self,
        query_dim: int,
        added_kv_proj_dim: int,
        processor: "MochiAttnProcessor2_0",
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_proj_bias: bool = True,
        out_dim: Optional[int] = None,
        out_context_dim: Optional[int] = None,
        out_bias: bool = True,
        context_pre_only: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        from .normalization import MochiRMSNorm

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim else query_dim
        self.context_pre_only = context_pre_only

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.scale = dim_head**-0.5

        self.norm_q = MochiRMSNorm(dim_head, eps, True)
        self.norm_k = MochiRMSNorm(dim_head, eps, True)
        self.norm_added_q = MochiRMSNorm(dim_head, eps, True)
        self.norm_added_k = MochiRMSNorm(dim_head, eps, True)

        self.to_q = mint.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = mint.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = mint.nn.Linear(query_dim, self.inner_dim, bias=bias)

        self.add_k_proj = mint.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
        self.add_v_proj = mint.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
        if self.context_pre_only is not None:
            self.add_q_proj = mint.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)

        self.to_out = nn.CellList(
            [mint.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias), mint.nn.Dropout(p=dropout)]
        )

        if not self.context_pre_only:
            self.to_add_out = mint.nn.Linear(self.inner_dim, self.out_context_dim, bias=out_bias)

        self.processor = processor

    def scaled_dot_product_attention(
        self,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        attn_mask: Optional[ms.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ):
        if query.dtype in (ms.float16, ms.bfloat16):
            return self.flash_attention_op(query, key, value, attn_mask, keep_prob=1 - dropout_p, scale=scale)
        else:
            return self.flash_attention_op(
                query.to(ms.float16),
                key.to(ms.float16),
                value.to(ms.float16),
                attn_mask,
                keep_prob=1 - dropout_p,
                scale=scale,
            ).to(query.dtype)

    def flash_attention_op(
        self,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        attn_mask: Optional[ms.Tensor] = None,
        keep_prob: float = 1.0,
        scale: Optional[float] = None,
    ):
        # For most scenarios, qkv has been processed into a BNSD layout before sdp
        input_layout = "BNSD"
        head_num = self.heads

        # In case qkv is 3-dim after `head_to_batch_dim`
        if query.ndim == 3:
            input_layout = "BSH"
            head_num = 1

        # process `attn_mask` as logic is different between PyTorch and Mindspore
        # In MindSpore, False indicates retention and True indicates discard, in PyTorch it is the opposite
        if attn_mask is not None:
            attn_mask = mint.logical_not(attn_mask) if attn_mask.dtype == ms.bool_ else attn_mask.bool()
            attn_mask = mint.broadcast_to(
                attn_mask, (attn_mask.shape[0], attn_mask.shape[1], query.shape[-2], key.shape[-2])
            )[:, :1, :, :]

        return ops.operations.nn_ops.FlashAttentionScore(
            head_num=head_num, keep_prob=keep_prob, scale_value=scale or self.scale, input_layout=input_layout
        )(query, key, value, None, None, None, attn_mask)[3]

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        **kwargs,
    ):
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )


@ms.jit_class
class MochiAttnProcessor2_0:
    """Attention processor used in Mochi."""

    def __init__(self):
        pass

    def apply_rotary_emb(self, x, freqs_cos, freqs_sin):
        x_even = x[..., 0::2].float()
        x_odd = x[..., 1::2].float()

        cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
        sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

        return mint.stack([cos, sin], dim=-1).flatten(start_dim=-2)

    def __call__(
        self,
        attn: "MochiAttention",
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
        image_rotary_emb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = unflatten(query, 2, (attn.heads, -1))
        key = unflatten(key, 2, (attn.heads, -1))
        value = unflatten(value, 2, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = unflatten(encoder_query, 2, (attn.heads, -1))
        encoder_key = unflatten(encoder_key, 2, (attn.heads, -1))
        encoder_value = unflatten(encoder_value, 2, (attn.heads, -1))

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        if image_rotary_emb is not None:
            query = self.apply_rotary_emb(query, *image_rotary_emb)
            key = self.apply_rotary_emb(key, *image_rotary_emb)

        query, key, value = query.swapaxes(1, 2), key.swapaxes(1, 2), value.swapaxes(1, 2)
        encoder_query, encoder_key, encoder_value = (
            encoder_query.swapaxes(1, 2),
            encoder_key.swapaxes(1, 2),
            encoder_value.swapaxes(1, 2),
        )

        sequence_length = query.shape[2]
        encoder_sequence_length = encoder_query.shape[2]
        total_length = sequence_length + encoder_sequence_length

        batch_size, heads, _, dim = query.shape
        attn_outputs = []
        for idx in range(batch_size):
            mask = attention_mask[idx][None, :]
            valid_prompt_token_indices = mint.nonzero(mask.flatten(), as_tuple=False).flatten()

            valid_encoder_query = encoder_query[idx : idx + 1, :, valid_prompt_token_indices, :]
            valid_encoder_key = encoder_key[idx : idx + 1, :, valid_prompt_token_indices, :]
            valid_encoder_value = encoder_value[idx : idx + 1, :, valid_prompt_token_indices, :]

            valid_query = mint.cat([query[idx : idx + 1], valid_encoder_query], dim=2)
            valid_key = mint.cat([key[idx : idx + 1], valid_encoder_key], dim=2)
            valid_value = mint.cat([value[idx : idx + 1], valid_encoder_value], dim=2)

            attn_output = attn.scaled_dot_product_attention(
                valid_query, valid_key, valid_value, dropout_p=0.0, is_causal=False
            )
            valid_sequence_length = attn_output.shape[2]
            attn_output = F.pad(attn_output, (0, 0, 0, total_length - valid_sequence_length))
            attn_outputs.append(attn_output)

        hidden_states = mint.cat(attn_outputs, dim=0)
        hidden_states = hidden_states.swapaxes(1, 2).flatten(start_dim=2, end_dim=3)

        # hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
        #     (sequence_length, encoder_sequence_length), dim=1
        # )
        hidden_states, encoder_hidden_states = (
            hidden_states[:, :sequence_length, :],
            hidden_states[:, sequence_length:, :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


@ms.jit_class
class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = mint.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CustomDiffusionAttnProcessor(nn.Cell):
    r"""
    Processor for implementing attention for the Custom Diffusion method.

    Args:
        train_kv (`bool`, defaults to `True`):
            Whether to newly train the key and value matrices corresponding to the text features.
        train_q_out (`bool`, defaults to `True`):
            Whether to newly train query matrices corresponding to the latent image features.
        hidden_size (`int`, *optional*, defaults to `None`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        out_bias (`bool`, defaults to `True`):
            Whether to include the bias parameter in `train_q_out`.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
    """

    def __init__(
        self,
        train_kv: bool = True,
        train_q_out: bool = True,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.train_kv = train_kv
        self.train_q_out = train_q_out

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        # `_custom_diffusion` id for easy serialization and loading.
        if self.train_kv:
            self.to_k_custom_diffusion = mint.nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_custom_diffusion = mint.nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if self.train_q_out:
            self.to_q_custom_diffusion = mint.nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_out_custom_diffusion = []
            self.to_out_custom_diffusion.append(mint.nn.Linear(hidden_size, hidden_size, bias=out_bias))
            self.to_out_custom_diffusion.append(mint.nn.Dropout(p=dropout))
            self.to_out_custom_diffusion = nn.CellList(self.to_out_custom_diffusion)

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states).to(attn.to_q.weight.dtype)
        else:
            query = attn.to_q(hidden_states.to(attn.to_q.weight.dtype))

        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.train_kv:
            key = self.to_k_custom_diffusion(encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype))
            value = self.to_v_custom_diffusion(encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype))
            key = key.to(attn.to_q.weight.dtype)
            value = value.to(attn.to_q.weight.dtype)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if crossattn:
            detach = mint.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = mint.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if self.train_q_out:
            # linear proj
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            # dropout
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


@ms.jit_class
class AttnAddedKVProcessor:
    r"""
    Processor for performing attention-related computations with extra learnable key and value matrices for the text
    encoder.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).swapaxes(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = mint.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = mint.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = mint.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.swapaxes(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


@ms.jit_class
class JointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = mint.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = mint.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = mint.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = attn.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


@ms.jit_class
class PAGJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
    ) -> ms.Tensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        batch_size, channel, height, width = None, None, None, None
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)

        # store the length of image patch sequences to create a mask that prevents interaction between patches
        # similar to making the self-attention map an identity matrix
        identity_block_size = hidden_states.shape[1]

        # chunk
        hidden_states_org, hidden_states_ptb = hidden_states.chunk(2)
        encoder_hidden_states_org, encoder_hidden_states_ptb = encoder_hidden_states.chunk(2)

        # original path
        batch_size = encoder_hidden_states_org.shape[0]

        # `sample` projections.
        query_org = attn.to_q(hidden_states_org)
        key_org = attn.to_k(hidden_states_org)
        value_org = attn.to_v(hidden_states_org)

        # `context` projections.
        encoder_hidden_states_org_query_proj = attn.add_q_proj(encoder_hidden_states_org)
        encoder_hidden_states_org_key_proj = attn.add_k_proj(encoder_hidden_states_org)
        encoder_hidden_states_org_value_proj = attn.add_v_proj(encoder_hidden_states_org)

        # attention
        query_org = mint.cat([query_org, encoder_hidden_states_org_query_proj], dim=1)
        key_org = mint.cat([key_org, encoder_hidden_states_org_key_proj], dim=1)
        value_org = mint.cat([value_org, encoder_hidden_states_org_value_proj], dim=1)

        inner_dim = key_org.shape[-1]
        head_dim = inner_dim // attn.heads
        query_org = query_org.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key_org = key_org.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value_org = value_org.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        hidden_states_org = attn.scaled_dot_product_attention(
            query_org, key_org, value_org, dropout_p=0.0, is_causal=False
        )
        hidden_states_org = hidden_states_org.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query_org.dtype)

        # Split the attention outputs.
        hidden_states_org, encoder_hidden_states_org = (
            hidden_states_org[:, : residual.shape[1]],
            hidden_states_org[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)
        if not attn.context_pre_only:
            encoder_hidden_states_org = attn.to_add_out(encoder_hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.swapaxes(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_org = encoder_hidden_states_org.swapaxes(-1, -2).reshape(
                batch_size, channel, height, width
            )

        # perturbed path

        batch_size = encoder_hidden_states_ptb.shape[0]

        # `sample` projections.
        query_ptb = attn.to_q(hidden_states_ptb)
        key_ptb = attn.to_k(hidden_states_ptb)
        value_ptb = attn.to_v(hidden_states_ptb)

        # `context` projections.
        encoder_hidden_states_ptb_query_proj = attn.add_q_proj(encoder_hidden_states_ptb)
        encoder_hidden_states_ptb_key_proj = attn.add_k_proj(encoder_hidden_states_ptb)
        encoder_hidden_states_ptb_value_proj = attn.add_v_proj(encoder_hidden_states_ptb)

        # attention
        query_ptb = mint.cat([query_ptb, encoder_hidden_states_ptb_query_proj], dim=1)
        key_ptb = mint.cat([key_ptb, encoder_hidden_states_ptb_key_proj], dim=1)
        value_ptb = mint.cat([value_ptb, encoder_hidden_states_ptb_value_proj], dim=1)

        inner_dim = key_ptb.shape[-1]
        head_dim = inner_dim // attn.heads
        query_ptb = query_ptb.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key_ptb = key_ptb.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value_ptb = value_ptb.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        # create a full mask with all entries set to 0
        seq_len = query_ptb.shape[2]
        full_mask = mint.zeros((seq_len, seq_len), dtype=query_ptb.dtype)

        # set the attention value between image patches to -inf
        full_mask[:identity_block_size, :identity_block_size] = float("-inf")

        # set the diagonal of the attention value between image patches to 0
        full_mask[:identity_block_size, :identity_block_size] = full_mask[
            :identity_block_size, :identity_block_size
        ].fill_diagonal(0.0)

        # expand the mask to match the attention weights shape
        full_mask = full_mask.unsqueeze(0).unsqueeze(0)  # Add batch and num_heads dimensions

        hidden_states_ptb = attn.scaled_dot_product_attention(
            query_ptb, key_ptb, value_ptb, attn_mask=full_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states_ptb = hidden_states_ptb.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_ptb = hidden_states_ptb.to(query_ptb.dtype)

        # split the attention outputs.
        hidden_states_ptb, encoder_hidden_states_ptb = (
            hidden_states_ptb[:, : residual.shape[1]],
            hidden_states_ptb[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)
        if not attn.context_pre_only:
            encoder_hidden_states_ptb = attn.to_add_out(encoder_hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.swapaxes(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_ptb = encoder_hidden_states_ptb.swapaxes(-1, -2).reshape(
                batch_size, channel, height, width
            )

        # concat
        hidden_states = mint.cat([hidden_states_org, hidden_states_ptb])
        encoder_hidden_states = mint.cat([encoder_hidden_states_org, encoder_hidden_states_ptb])

        return hidden_states, encoder_hidden_states


@ms.jit_class
class PAGCFGJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        batch_size, channel, height, width = None, None, None, None
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)

        identity_block_size = hidden_states.shape[
            1
        ]  # patch embeddings width * height (correspond to self-attention map width or height)

        # chunk
        hidden_states_uncond, hidden_states_org, hidden_states_ptb = hidden_states.chunk(3)
        hidden_states_org = mint.cat([hidden_states_uncond, hidden_states_org])

        (
            encoder_hidden_states_uncond,
            encoder_hidden_states_org,
            encoder_hidden_states_ptb,
        ) = encoder_hidden_states.chunk(3)
        encoder_hidden_states_org = mint.cat([encoder_hidden_states_uncond, encoder_hidden_states_org])

        # original path
        batch_size = encoder_hidden_states_org.shape[0]

        # `sample` projections.
        query_org = attn.to_q(hidden_states_org)
        key_org = attn.to_k(hidden_states_org)
        value_org = attn.to_v(hidden_states_org)

        # `context` projections.
        encoder_hidden_states_org_query_proj = attn.add_q_proj(encoder_hidden_states_org)
        encoder_hidden_states_org_key_proj = attn.add_k_proj(encoder_hidden_states_org)
        encoder_hidden_states_org_value_proj = attn.add_v_proj(encoder_hidden_states_org)

        # attention
        query_org = mint.cat([query_org, encoder_hidden_states_org_query_proj], dim=1)
        key_org = mint.cat([key_org, encoder_hidden_states_org_key_proj], dim=1)
        value_org = mint.cat([value_org, encoder_hidden_states_org_value_proj], dim=1)

        inner_dim = key_org.shape[-1]
        head_dim = inner_dim // attn.heads
        query_org = query_org.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key_org = key_org.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value_org = value_org.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        hidden_states_org = attn.scaled_dot_product_attention(
            query_org, key_org, value_org, dropout_p=0.0, is_causal=False
        )
        hidden_states_org = hidden_states_org.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query_org.dtype)

        # Split the attention outputs.
        hidden_states_org, encoder_hidden_states_org = (
            hidden_states_org[:, : residual.shape[1]],
            hidden_states_org[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)
        if not attn.context_pre_only:
            encoder_hidden_states_org = attn.to_add_out(encoder_hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.swapaxes(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_org = encoder_hidden_states_org.swapaxes(-1, -2).reshape(
                batch_size, channel, height, width
            )

        # perturbed path

        batch_size = encoder_hidden_states_ptb.shape[0]

        # `sample` projections.
        query_ptb = attn.to_q(hidden_states_ptb)
        key_ptb = attn.to_k(hidden_states_ptb)
        value_ptb = attn.to_v(hidden_states_ptb)

        # `context` projections.
        encoder_hidden_states_ptb_query_proj = attn.add_q_proj(encoder_hidden_states_ptb)
        encoder_hidden_states_ptb_key_proj = attn.add_k_proj(encoder_hidden_states_ptb)
        encoder_hidden_states_ptb_value_proj = attn.add_v_proj(encoder_hidden_states_ptb)

        # attention
        query_ptb = mint.cat([query_ptb, encoder_hidden_states_ptb_query_proj], dim=1)
        key_ptb = mint.cat([key_ptb, encoder_hidden_states_ptb_key_proj], dim=1)
        value_ptb = mint.cat([value_ptb, encoder_hidden_states_ptb_value_proj], dim=1)

        inner_dim = key_ptb.shape[-1]
        head_dim = inner_dim // attn.heads
        query_ptb = query_ptb.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key_ptb = key_ptb.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value_ptb = value_ptb.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        # create a full mask with all entries set to 0
        seq_len = query_ptb.shape[2]
        full_mask = mint.zeros((seq_len, seq_len), dtype=query_ptb.dtype)

        # set the attention value between image patches to -inf
        full_mask[:identity_block_size, :identity_block_size] = float("-inf")

        # set the diagonal of the attention value between image patches to 0
        full_mask[:identity_block_size, :identity_block_size] = full_mask[
            :identity_block_size, :identity_block_size
        ].fill_diagonal(0.0)

        # expand the mask to match the attention weights shape
        full_mask = full_mask.unsqueeze(0).unsqueeze(0)  # Add batch and num_heads dimensions

        hidden_states_ptb = attn.scaled_dot_product_attention(
            query_ptb, key_ptb, value_ptb, attn_mask=full_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states_ptb = hidden_states_ptb.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_ptb = hidden_states_ptb.to(query_ptb.dtype)

        # split the attention outputs.
        hidden_states_ptb, encoder_hidden_states_ptb = (
            hidden_states_ptb[:, : residual.shape[1]],
            hidden_states_ptb[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)
        if not attn.context_pre_only:
            encoder_hidden_states_ptb = attn.to_add_out(encoder_hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.swapaxes(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_ptb = encoder_hidden_states_ptb.swapaxes(-1, -2).reshape(
                batch_size, channel, height, width
            )

        # concat
        hidden_states = mint.cat([hidden_states_org, hidden_states_ptb])
        encoder_hidden_states = mint.cat([encoder_hidden_states_org, encoder_hidden_states_ptb])

        return hidden_states, encoder_hidden_states


@ms.jit_class
class FusedJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        batch_size, channel, height, width = (None,) * 4
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = mint.split(qkv, split_size, dim=-1)

        # `context` projections.
        encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
        split_size = encoder_qkv.shape[-1] // 3
        (
            encoder_hidden_states_query_proj,
            encoder_hidden_states_key_proj,
            encoder_hidden_states_value_proj,
        ) = mint.split(encoder_qkv, split_size, dim=-1)

        # attention
        query = mint.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = mint.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = mint.cat([value, encoder_hidden_states_value_proj], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        hidden_states = attn.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states


@ms.jit_class
class AllegroAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Allegro model. It applies a normalization layer and rotary embedding on the query and key vector.
    """

    def __init__(self):
        from .embeddings import apply_rotary_emb_allegro

        self.apply_rotary_emb_allegro = apply_rotary_emb_allegro

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        # Apply RoPE if needed
        if image_rotary_emb is not None and not attn.is_cross_attention:
            query = self.apply_rotary_emb_allegro(query, image_rotary_emb[0], image_rotary_emb[1])
            key = self.apply_rotary_emb_allegro(key, image_rotary_emb[0], image_rotary_emb[1])

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@ms.jit_class
class AuraFlowAttnProcessor2_0:
    """Attention processor used typically in processing Aura Flow."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        else:
            encoder_hidden_states_query_proj = None
            encoder_hidden_states_key_proj = None
            encoder_hidden_states_value_proj = None

        # Reshape.
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # Apply QK norm.
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Concatenate the projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            )
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            )

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_q(encoder_hidden_states_key_proj)

            query = mint.cat([encoder_hidden_states_query_proj, query], dim=1)
            key = mint.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = mint.cat([encoder_hidden_states_value_proj, value], dim=1)

        query = query.swapaxes(1, 2)
        key = key.swapaxes(1, 2)
        value = value.swapaxes(1, 2)

        # Attention.
        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, scale=attn.scale, is_causal=False
        )
        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, encoder_hidden_states.shape[1] :],
                hidden_states[:, : encoder_hidden_states.shape[1]],
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


@ms.jit_class
class FusedAuraFlowAttnProcessor2_0:
    """Attention processor used typically in processing Aura Flow with fused projections."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
    ) -> ms.Tensor:
        batch_size = hidden_states.shape[0]

        # `sample` projections.
        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = mint.split(qkv, split_size, dim=-1)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
            split_size = encoder_qkv.shape[-1] // 3
            (
                encoder_hidden_states_query_proj,
                encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj,
            ) = mint.split(encoder_qkv, split_size, dim=-1)

        # Reshape.
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # Apply QK norm.
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Concatenate the projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            )
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            )

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_q(encoder_hidden_states_key_proj)

            query = mint.cat([encoder_hidden_states_query_proj, query], dim=1)
            key = mint.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = mint.cat([encoder_hidden_states_value_proj, value], dim=1)

        query = query.swapaxes(1, 2)
        key = key.swapaxes(1, 2)
        value = value.swapaxes(1, 2)

        # Attention.
        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, scale=attn.scale, is_causal=False
        )
        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, encoder_hidden_states.shape[1] :],
                hidden_states[:, : encoder_hidden_states.shape[1]],
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


@ms.jit_class
class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        # move importing from __call__ to __init__ as it is not supported in construct()
        from .embeddings import apply_rotary_emb

        self.apply_rotary_emb = apply_rotary_emb

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = mint.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = mint.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = mint.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = self.apply_rotary_emb(query, image_rotary_emb)
            key = self.apply_rotary_emb(key, image_rotary_emb)

        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            """
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            """
            encoder_hidden_states, hidden_states = mint.split(
                hidden_states,
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


@ms.jit_class
class FusedFluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        # move importing from __call__ to __init__ as it is not supported in construct()
        from .embeddings import apply_rotary_emb

        self.apply_rotary_emb = apply_rotary_emb

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = mint.split(qkv, split_size, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
            split_size = encoder_qkv.shape[-1] // 3
            (
                encoder_hidden_states_query_proj,
                encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj,
            ) = mint.split(encoder_qkv, split_size, dim=-1)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = mint.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = mint.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = mint.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = self.apply_rotary_emb(query, image_rotary_emb)
            key = self.apply_rotary_emb(key, image_rotary_emb)

        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FluxIPAdapterJointAttnProcessor2_0(nn.Cell):
    """Flux Attention processor for IP-Adapter."""

    def __init__(self, hidden_size: int, cross_attention_dim: int, num_tokens=(4,), scale=1.0, dtype=None):
        super().__init__()
        from .embeddings import apply_rotary_emb

        self.apply_rotary_emb = apply_rotary_emb

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.CellList(
            [mint.nn.Linear(cross_attention_dim, hidden_size, bias=True, dtype=dtype) for _ in range(len(num_tokens))]
        )
        self.to_v_ip = nn.CellList(
            [mint.nn.Linear(cross_attention_dim, hidden_size, bias=True, dtype=dtype) for _ in range(len(num_tokens))]
        )

    def construct(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
        ip_hidden_states: Optional[List[ms.Tensor]] = None,
        ip_adapter_masks: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        hidden_states_query_proj = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        hidden_states_query_proj = hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if attn.norm_q is not None:
            hidden_states_query_proj = attn.norm_q(hidden_states_query_proj)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = mint.cat([encoder_hidden_states_query_proj, hidden_states_query_proj], dim=2)
            key = mint.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = mint.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = self.apply_rotary_emb(query, image_rotary_emb)
            key = self.apply_rotary_emb(key, image_rotary_emb)

        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            # IP-adapter
            ip_query = hidden_states_query_proj
            ip_attn_output = mint.zeros_like(hidden_states)

            for current_ip_hidden_states, scale, to_k_ip, to_v_ip in zip(
                ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip
            ):
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)

                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                current_ip_hidden_states = attn.scaled_dot_product_attention(
                    ip_query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
                current_ip_hidden_states = current_ip_hidden_states.swapaxes(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )
                current_ip_hidden_states = current_ip_hidden_states.to(ip_query.dtype)
                ip_attn_output += scale * current_ip_hidden_states

            return hidden_states, encoder_hidden_states, ip_attn_output
        else:
            return hidden_states


@ms.jit_class
class CogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        # move importing from __call__ to __init__ as it is not supported in construct()
        from .embeddings import apply_rotary_emb

        self.apply_rotary_emb = apply_rotary_emb

    def apply_rotary_emb_for_image_part(
        self,
        hidden_state: ms.Tensor,
        image_rotary_emb: ms.Tensor,
        start_index: int,
        axis: int = 2,
    ) -> ms.Tensor:
        """
        Equivalence of expression(when axis=2):
            `hidden_state[:, :, start_index:] = self.apply_rotary_emb(hidden_state[:, :, start_index:], image_rotary_emb)`

        Rewrite it since implement above might call ops.ScatterNdUpdate which is super slow!
        """
        hidden_state_text, hidden_state_image = mint.split(
            hidden_state, (start_index, hidden_state.shape[axis] - start_index), dim=axis
        )
        hidden_state_image = self.apply_rotary_emb(hidden_state_image, image_rotary_emb)
        hidden_state = mint.cat([hidden_state_text, hidden_state_image], dim=axis)
        return hidden_state

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]

        hidden_states = mint.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        # rewrite the implement for performance, refer to `self.apply_rotary_emb_for_image_part`
        if image_rotary_emb is not None:
            query = self.apply_rotary_emb_for_image_part(query, image_rotary_emb, text_seq_length)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb_for_image_part(key, image_rotary_emb, text_seq_length)

        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = mint.split(
            hidden_states, [text_seq_length, hidden_states.shape[1] - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


@ms.jit_class
class FusedCogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        # move importing from __call__ to __init__ as it is not supported in construct()
        from .embeddings import apply_rotary_emb

        self.apply_rotary_emb = apply_rotary_emb

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]

        hidden_states = mint.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = mint.split(qkv, split_size, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = self.apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = self.apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.shape[1] - text_seq_length], axis=1
        )
        return hidden_states, encoder_hidden_states


@ms.jit_class
class XFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers-like interface.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        assert attention_op is None, (
            "Memory efficient attention on mindspore uses flash attention under the hoods. "
            "There is no other implementation for now. Please do not set `attention_op`."
        )
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.tile((1, query_tokens, 1))

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Memory efficient attention on mindspore uses flash attention under the hoods.
        # Flash attention implementation is called `FlashAttentionScore`
        # which is an experimental api with the following limitations:
        # 1. Sequence length of query must be divisible by 16 and in range of [1, 32768].
        # 2. Head dimensions must be one of [64, 80, 96, 120, 128, 256].
        # 3. The input dtype must be float16 or bfloat16.
        # Sequence length of query must be checked in runtime.
        _, query_tokens, _ = query.shape
        assert query_tokens % 16 == 0, f"Sequence length of query must be divisible by 16, but got {query_tokens=}."
        # Head dimension is checked in Attention.set_use_memory_efficient_attention_xformers. We maybe pad on head_dim.
        if attn.head_dim_padding > 0:
            query_padded = mint.nn.functional.pad(query, (0, attn.head_dim_padding), mode="constant", value=0.0)
            key_padded = mint.nn.functional.pad(key, (0, attn.head_dim_padding), mode="constant", value=0.0)
            value_padded = mint.nn.functional.pad(value, (0, attn.head_dim_padding), mode="constant", value=0.0)
        else:
            query_padded, key_padded, value_padded = query, key, value
        flash_attn = ops.operations.nn_ops.FlashAttentionScore(1, scale_value=attn.scale)
        hidden_states_padded = flash_attn(query_padded, key_padded, value_padded, None, None, None, attention_mask)[3]
        # If we did padding before calculate attention, undo it!
        if attn.head_dim_padding > 0:
            hidden_states = hidden_states_padded[..., : attn.head_dim]
        else:
            hidden_states = hidden_states_padded

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@ms.jit_class
class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@ms.jit_class
class MochiVaeAttnProcessor2_0:
    r"""
    Attention processor used in Mochi VAE.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states
        is_single_frame = hidden_states.shape[1] == 1

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if is_single_frame:
            hidden_states = attn.to_v(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=attn.is_causal
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@ms.jit_class
class HunyuanAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the HunyuanDiT model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self) -> None:
        # move importing from __call__ to __init__ as it is not supported in construct()
        from .embeddings import apply_rotary_emb

        self.apply_rotary_emb = apply_rotary_emb

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        batch_size, channel, height, width = (None,) * 4
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = self.apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb(key, image_rotary_emb)

        # # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@ms.jit_class
class StableAudioAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """

    def __init__(self) -> None:
        # move importing from __call__ to __init__ as it is not supported in construct()
        from .embeddings import apply_rotary_emb

        self.apply_rotary_emb = apply_rotary_emb

    def apply_partial_rotary_emb(
        self,
        x: ms.Tensor,
        freqs_cis: Tuple[ms.Tensor],
    ) -> ms.Tensor:
        rot_dim = freqs_cis[0].shape[-1]
        x_to_rotate, x_unrotated = x[..., :rot_dim], x[..., rot_dim:]

        x_rotated = self.apply_rotary_emb(x_to_rotate, freqs_cis, use_real=True, use_real_unbind_dim=-2)

        out = mint.cat((x_rotated, x_unrotated), dim=-1)
        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        rotary_emb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim

        batch_size, channel, height, width = (None,) * 4
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        key = key.view(batch_size, -1, kv_heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).swapaxes(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = mint.repeat_interleave(key, heads_per_kv_head, dim=1, output_size=key.shape[1] * heads_per_kv_head)
            value = mint.repeat_interleave(
                value, heads_per_kv_head, dim=1, output_size=value.shape[1] * heads_per_kv_head
            )

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if rotary_emb is not None:
            query_dtype = query.dtype
            key_dtype = key.dtype
            query = query.to(ms.float32)
            key = key.to(ms.float32)

            rot_dim = rotary_emb[0].shape[-1]
            query_to_rotate, query_unrotated = query[..., :rot_dim], query[..., rot_dim:]
            query_rotated = self.apply_rotary_emb(query_to_rotate, rotary_emb, use_real=True, use_real_unbind_dim=-2)

            query = mint.cat((query_rotated, query_unrotated), dim=-1)

            if not attn.is_cross_attention:
                key_to_rotate, key_unrotated = key[..., :rot_dim], key[..., rot_dim:]
                key_rotated = self.apply_rotary_emb(key_to_rotate, rotary_emb, use_real=True, use_real_unbind_dim=-2)

                key = mint.cat((key_rotated, key_unrotated), dim=-1)

            query = query.to(query_dtype)
            key = key.to(key_dtype)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # hidden_states = mint.bmm(attention_probs, value)

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SpatialNorm(nn.Cell):
    """
    Spatially conditioned normalization as defined in https://huggingface.co/papers/2209.09002.

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
        self.norm_layer = mint.nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = mint.nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_b = mint.nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def construct(self, f: ms.Tensor, zq: ms.Tensor) -> ms.Tensor:
        f_size = f.shape[-2:]
        zq = mint.nn.functional.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


class IPAdapterAttnProcessor(nn.Cell):
    r"""
    Attention processor for Multiple IP-Adapters.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
            The context length of the image features.
        scale (`float` or List[`float`], defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.CellList(
            [mint.nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )
        self.to_v_ip = nn.CellList(
            [mint.nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )

    def construct(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[ms.Tensor] = None,
    ):
        residual = hidden_states

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )
        else:
            ip_hidden_states = None

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = mint.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                    f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                    f"({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
                    if mask is None:
                        continue
                    if not isinstance(mask, ms.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape "
                            "[1, num_images_for_ip_adapter, height, width]."
                            " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.scale)

        # for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(scale, list):
                        scale = [scale] * mask.shape[1]

                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                        ip_key = attn.head_to_batch_dim(ip_key)
                        ip_value = attn.head_to_batch_dim(ip_value)

                        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
                        _current_ip_hidden_states = mint.bmm(ip_attention_probs, ip_value)
                        _current_ip_hidden_states = attn.batch_to_head_dim(_current_ip_hidden_states)

                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )

                        mask_downsample = mask_downsample.to(dtype=query.dtype)

                        hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
                else:
                    ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)

                    ip_key = attn.head_to_batch_dim(ip_key)
                    ip_value = attn.head_to_batch_dim(ip_value)

                    ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
                    current_ip_hidden_states = mint.bmm(ip_attention_probs, ip_value)
                    current_ip_hidden_states = attn.batch_to_head_dim(current_ip_hidden_states)

                    hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAdapterAttnProcessor2_0(nn.Cell):
    r"""
    Attention processor for IP-Adapter for PyTorch 2.0.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
            The context length of the image features.
        scale (`float` or `List[float]`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.CellList(
            [mint.nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )
        self.to_v_ip = nn.CellList(
            [mint.nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )

    def construct(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[ms.Tensor] = None,
    ):
        residual = hidden_states

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                )
                deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                    f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                    f"({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
                    if mask is None:
                        continue
                    if not isinstance(mask, ms.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape "
                            "[1, num_images_for_ip_adapter, height, width]."
                            " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.scale)

        # for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(scale, list):
                        scale = [scale] * mask.shape[1]

                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
                        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

                        # the output of sdp = (batch, num_heads, seq_len, head_dim)
                        # TODO: add support for attn.scale when we move to Torch 2.1
                        _current_ip_hidden_states = attn.scaled_dot_product_attention(
                            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )

                        _current_ip_hidden_states = _current_ip_hidden_states.swapaxes(1, 2).reshape(
                            batch_size, -1, attn.heads * head_dim
                        )
                        _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)

                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )

                        mask_downsample = mask_downsample.to(dtype=query.dtype)
                        hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
                else:
                    ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)

                    ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
                    ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

                    # the output of sdp = (batch, num_heads, seq_len, head_dim)
                    # TODO: add support for attn.scale when we move to Torch 2.1
                    current_ip_hidden_states = attn.scaled_dot_product_attention(
                        query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                    )

                    current_ip_hidden_states = current_ip_hidden_states.swapaxes(1, 2).reshape(
                        batch_size, -1, attn.heads * head_dim
                    )
                    current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

                    hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SD3IPAdapterJointAttnProcessor2_0(nn.Cell):
    """
    Attention processor for IP-Adapter used typically in processing the SD3-like self-attention projections, with
    additional image-based information and timestep embeddings.

    Args:
        hidden_size (`int`):
            The number of hidden channels.
        ip_hidden_states_dim (`int`):
            The image feature dimension.
        head_dim (`int`):
            The number of head channels.
        timesteps_emb_dim (`int`, defaults to 1280):
            The number of input channels for timestep embedding.
        scale (`float`, defaults to 0.5):
            IP-Adapter scale.
    """

    def __init__(
        self,
        hidden_size: int,
        ip_hidden_states_dim: int,
        head_dim: int,
        timesteps_emb_dim: int = 1280,
        scale: float = 0.5,
    ):
        super().__init__()

        # To prevent circular import
        from .normalization import AdaLayerNorm, RMSNorm

        self.norm_ip = AdaLayerNorm(timesteps_emb_dim, output_dim=ip_hidden_states_dim * 2, norm_eps=1e-6, chunk_dim=1)
        self.to_k_ip = mint.nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        self.to_v_ip = mint.nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        self.norm_q = RMSNorm(head_dim, 1e-6)
        self.norm_k = RMSNorm(head_dim, 1e-6)
        self.norm_ip_k = RMSNorm(head_dim, 1e-6)
        self.scale = scale

    def construct(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        ip_hidden_states: ms.Tensor = None,
        temb: ms.Tensor = None,
    ) -> ms.Tensor:
        """
        Perform the attention computation, integrating image features (if provided) and timestep embeddings.

        If `ip_hidden_states` is `None`, this is equivalent to using JointAttnProcessor2_0.

        Args:
            attn (`Attention`):
                Attention instance.
            hidden_states (`ms.Tensor`):
                Input `hidden_states`.
            encoder_hidden_states (`ms.Tensor`, *optional*):
                The encoder hidden states.
            attention_mask (`ms.Tensor`, *optional*):
                Attention mask.
            ip_hidden_states (`ms.Tensor`, *optional*):
                Image embeddings.
            temb (`ms.Tensor`, *optional*):
                Timestep embeddings.

        Returns:
            `ms.Tensor`: Output hidden states.
        """
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        img_query = query
        img_key = key
        img_value = value

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).swapaxes(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = mint.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = mint.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = mint.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = attn.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # IP Adapter
        if self.scale != 0 and ip_hidden_states is not None:
            # Norm image features
            norm_ip_hidden_states = self.norm_ip(ip_hidden_states, temb=temb)

            # To k and v
            ip_key = self.to_k_ip(norm_ip_hidden_states)
            ip_value = self.to_v_ip(norm_ip_hidden_states)

            # Reshape
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

            # Norm
            query = self.norm_q(img_query)
            img_key = self.norm_k(img_key)
            ip_key = self.norm_ip_k(ip_key)

            # cat img
            key = mint.cat([img_key, ip_key], dim=2)
            value = mint.cat([img_value, ip_value], dim=2)

            ip_hidden_states = attn.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            ip_hidden_states = ip_hidden_states.swapaxes(1, 2).view(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)

            hidden_states = hidden_states + ip_hidden_states * self.scale

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


@ms.jit_class
class PAGHunyuanAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the HunyuanDiT model. It applies a normalization layer and rotary embedding on query and key vector. This
    variant of the processor employs [Pertubed Attention Guidance](https://huggingface.co/papers/2403.17377).
    """

    def __init__(self):
        # move importing from __call__ to __init__ as it is not supported in construct()
        from .embeddings import apply_rotary_emb

        self.apply_rotary_emb = apply_rotary_emb

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        # chunk
        hidden_states_org, hidden_states_ptb = hidden_states.chunk(2)

        # 1. Original Path
        batch_size, sequence_length, _ = (
            hidden_states_org.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states_org = attn.group_norm(hidden_states_org.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states_org)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states_org
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = self.apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_org = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states_org = hidden_states_org.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query.dtype)

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        # 2. Perturbed Path
        if attn.group_norm is not None:
            hidden_states_ptb = attn.group_norm(hidden_states_ptb.swapaxes(1, 2)).swapaxes(1, 2)

        hidden_states_ptb = attn.to_v(hidden_states_ptb)
        hidden_states_ptb = hidden_states_ptb.to(query.dtype)

        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        # cat
        hidden_states = mint.cat([hidden_states_org, hidden_states_ptb])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@ms.jit_class
class PAGCFGHunyuanAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the HunyuanDiT model. It applies a normalization layer and rotary embedding on query and key vector. This
    variant of the processor employs [Pertubed Attention Guidance](https://huggingface.co/papers/2403.17377).
    """

    def __init__(self):
        # move importing from __call__ to __init__ as it is not supported in construct()
        from .embeddings import apply_rotary_emb

        self.apply_rotary_emb = apply_rotary_emb

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        # chunk
        hidden_states_uncond, hidden_states_org, hidden_states_ptb = hidden_states.chunk(3)
        hidden_states_org = mint.cat([hidden_states_uncond, hidden_states_org])

        # 1. Original Path
        batch_size, sequence_length, _ = (
            hidden_states_org.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states_org = attn.group_norm(hidden_states_org.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states_org)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states_org
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = self.apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_org = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states_org = hidden_states_org.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query.dtype)

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        # 2. Perturbed Path
        if attn.group_norm is not None:
            hidden_states_ptb = attn.group_norm(hidden_states_ptb.swapaxes(1, 2)).swapaxes(1, 2)

        hidden_states_ptb = attn.to_v(hidden_states_ptb)
        hidden_states_ptb = hidden_states_ptb.to(query.dtype)

        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        # cat
        hidden_states = mint.cat([hidden_states_org, hidden_states_ptb])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@ms.jit_class
class LuminaAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the LuminaNextDiT model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        from .embeddings import apply_rotary_emb

        self.apply_rotary_emb = apply_rotary_emb

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        query_rotary_emb: Optional[ms.Tensor] = None,
        key_rotary_emb: Optional[ms.Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> ms.Tensor:
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Apply Query-Key Norm if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.view(batch_size, -1, attn.heads, head_dim)

        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply RoPE if needed
        if query_rotary_emb is not None:
            query = self.apply_rotary_emb(query, query_rotary_emb, use_real=False)
        if key_rotary_emb is not None:
            key = self.apply_rotary_emb(key, key_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        softmax_scale = None
        # Apply proportional attention if true
        if key_rotary_emb is not None:
            if base_sequence_length is not None:
                softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            else:
                softmax_scale = attn.scale

        # perform Grouped-qurey Attention (GQA)
        n_rep = attn.heads // kv_heads
        if n_rep >= 1:
            key = key.unsqueeze(3).tile((1, 1, 1, n_rep, 1)).flatten(start_dim=2, end_dim=3)
            value = value.unsqueeze(3).tile((1, 1, 1, n_rep, 1)).flatten(start_dim=2, end_dim=3)

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        target_length = attention_mask.shape[-1]
        attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)
        attention_mask = attention_mask.broadcast_to((batch_size, attn.heads, sequence_length, target_length))

        query = query.swapaxes(1, 2)
        key = key.swapaxes(1, 2)
        value = value.swapaxes(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=softmax_scale
        )

        hidden_states = hidden_states.swapaxes(1, 2).to(dtype)

        return hidden_states


@ms.jit_class
class FusedAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). It uses
    fused projection layers. For self-attention modules, all projection matrices (i.e., query, key, value) are fused.
    For cross-attention modules, key and value projection matrices are fused.

    <Tip warning={true}>

    This API is currently 🧪 experimental in nature and can change in future.

    </Tip>
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        if encoder_hidden_states is None:
            qkv = attn.to_qkv(hidden_states)
            split_size = qkv.shape[-1] // 3
            query, key, value = mint.split(qkv, split_size, dim=-1)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            query = attn.to_q(hidden_states)

            kv = attn.to_kv(encoder_hidden_states)
            split_size = kv.shape[-1] // 2
            key, value = mint.split(kv, split_size, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@ms.jit_class
class PAGIdentitySelfAttnProcessor2_0:
    r"""
    Processor for implementing PAG using scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    PAG reference: https://huggingface.co/papers/2403.17377
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        # chunk
        hidden_states_org, hidden_states_ptb = hidden_states.chunk(2)

        # original path
        batch_size, sequence_length, _ = hidden_states_org.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states_org = attn.group_norm(hidden_states_org.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states_org)
        key = attn.to_k(hidden_states_org)
        value = attn.to_v(hidden_states_org)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_org = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states_org = hidden_states_org.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query.dtype)

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        # perturbed path (identity attention)
        batch_size, sequence_length, _ = hidden_states_ptb.shape

        if attn.group_norm is not None:
            hidden_states_ptb = attn.group_norm(hidden_states_ptb.swapaxes(1, 2)).swapaxes(1, 2)

        hidden_states_ptb = attn.to_v(hidden_states_ptb)
        hidden_states_ptb = hidden_states_ptb.to(query.dtype)

        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        # cat
        hidden_states = mint.cat([hidden_states_org, hidden_states_ptb])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@ms.jit_class
class PAGCFGIdentitySelfAttnProcessor2_0:
    r"""
    Processor for implementing PAG using scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    PAG reference: https://huggingface.co/papers/2403.17377
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        # chunk
        hidden_states_uncond, hidden_states_org, hidden_states_ptb = hidden_states.chunk(3)
        hidden_states_org = mint.cat([hidden_states_uncond, hidden_states_org])

        # original path
        batch_size, sequence_length, _ = hidden_states_org.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states_org = attn.group_norm(hidden_states_org.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states_org)
        key = attn.to_k(hidden_states_org)
        value = attn.to_v(hidden_states_org)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_org = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states_org = hidden_states_org.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query.dtype)

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        # perturbed path (identity attention)
        batch_size, sequence_length, _ = hidden_states_ptb.shape

        if attn.group_norm is not None:
            hidden_states_ptb = attn.group_norm(hidden_states_ptb.swapaxes(1, 2)).swapaxes(1, 2)

        value = attn.to_v(hidden_states_ptb)
        hidden_states_ptb = value
        hidden_states_ptb = hidden_states_ptb.to(query.dtype)

        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        # cat
        hidden_states = mint.cat([hidden_states_org, hidden_states_ptb])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@ms.jit_class
class SanaMultiscaleAttnProcessor2_0:
    r"""
    Processor for implementing multiscale quadratic attention.
    """

    def __call__(self, attn: SanaMultiscaleLinearAttention, hidden_states: ms.Tensor) -> ms.Tensor:
        height, width = hidden_states.shape[-2:]
        if height * width > attn.attention_head_dim:
            use_linear_attention = True
        else:
            use_linear_attention = False

        residual = hidden_states

        batch_size, _, height, width = list(hidden_states.shape)
        original_dtype = hidden_states.dtype

        hidden_states = hidden_states.movedim(1, -1)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        hidden_states = mint.cat([query, key, value], dim=3)
        hidden_states = hidden_states.movedim(-1, 1)

        multi_scale_qkv = [hidden_states]
        for block in attn.to_qkv_multiscale:
            multi_scale_qkv.append(block(hidden_states))

        hidden_states = mint.cat(multi_scale_qkv, dim=1)

        if use_linear_attention:
            # for linear attention upcast hidden_states to float32
            hidden_states = hidden_states.to(dtype=ms.float32)

        hidden_states = hidden_states.reshape(batch_size, -1, 3 * attn.attention_head_dim, height * width)

        query, key, value = hidden_states.chunk(3, axis=2)
        query = attn.nonlinearity(query)
        key = attn.nonlinearity(key)

        if use_linear_attention:
            hidden_states = attn.apply_linear_attention(query, key, value)
            hidden_states = hidden_states.to(dtype=original_dtype)
        else:
            hidden_states = attn.apply_quadratic_attention(query, key, value)

        hidden_states = mint.reshape(hidden_states, (batch_size, -1, height, width))
        hidden_states = attn.to_out(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if attn.norm_type == "rms_norm":
            hidden_states = attn.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        else:
            hidden_states = attn.norm_out(hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


@ms.jit_class
class FluxSingleAttnProcessor2_0(FluxAttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        deprecation_message = "`FluxSingleAttnProcessor2_0` is deprecated and will be removed in a future version. Please use `FluxAttnProcessor2_0` instead."
        deprecate("FluxSingleAttnProcessor2_0", "0.32.0", deprecation_message)
        super().__init__()


@ms.jit_class
class SanaLinearAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product linear attention.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        original_dtype = hidden_states.dtype

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # query = query.transpose(1, 2).unflatten(1, (attn.heads, -1))
        # key = key.transpose(1, 2).unflatten(1, (attn.heads, -1)).transpose(2, 3)
        # value = value.transpose(1, 2).unflatten(1, (attn.heads, -1))
        query = query.swapaxes(1, 2)
        query = query.reshape(query.shape[0], attn.heads, -1, *query.shape[2:])
        key = key.swapaxes(1, 2)
        key = key.reshape(key.shape[0], attn.heads, -1, *key.shape[2:]).swapaxes(2, 3)
        value = value.swapaxes(1, 2)
        value = value.reshape(value.shape[0], attn.heads, -1, *value.shape[2:])

        query = mint.nn.functional.relu(query)
        key = mint.nn.functional.relu(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
        scores = mint.matmul(value, key)
        hidden_states = mint.matmul(scores, query)

        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + 1e-15)
        hidden_states = hidden_states.flatten(start_dim=1, end_dim=2).swapaxes(1, 2)
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if original_dtype == ms.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


@ms.jit_class
class PAGCFGSanaLinearAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product linear attention.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        original_dtype = hidden_states.dtype

        hidden_states_uncond, hidden_states_org, hidden_states_ptb = hidden_states.chunk(3)
        hidden_states_org = mint.cat([hidden_states_uncond, hidden_states_org])

        query = attn.to_q(hidden_states_org)
        key = attn.to_k(hidden_states_org)
        value = attn.to_v(hidden_states_org)

        query = query.swapaxes(1, 2)
        query = query.reshape(query.shape[0], attn.heads, -1, *query.shape[2:])
        key = key.swapaxes(1, 2)
        key = key.reshape(key.shape[0], attn.heads, -1, *key.shape[2:]).swapaxes(2, 3)
        value = value.swapaxes(1, 2)
        value = value.reshape(value.shape[0], attn.heads, -1, *value.shape[2:])

        query = mint.nn.functional.relu(query)
        key = mint.nn.functional.relu(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
        scores = mint.matmul(value, key)
        hidden_states_org = mint.matmul(scores, query)

        hidden_states_org = hidden_states_org[:, :, :-1] / (hidden_states_org[:, :, -1:] + 1e-15)
        hidden_states_org = hidden_states_org.flatten(start_dim=1, end_dim=2).swapaxes(1, 2)
        hidden_states_org = hidden_states_org.to(original_dtype)

        hidden_states_org = attn.to_out[0](hidden_states_org)
        hidden_states_org = attn.to_out[1](hidden_states_org)

        # perturbed path (identity attention)
        hidden_states_ptb = attn.to_v(hidden_states_ptb).to(original_dtype)

        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        hidden_states = mint.cat([hidden_states_org, hidden_states_ptb])

        if original_dtype == ms.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


@ms.jit_class
class PAGIdentitySanaLinearAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product linear attention.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        original_dtype = hidden_states.dtype

        hidden_states_org, hidden_states_ptb = hidden_states.chunk(2)

        query = attn.to_q(hidden_states_org)
        key = attn.to_k(hidden_states_org)
        value = attn.to_v(hidden_states_org)

        query = query.swapaxes(1, 2)
        query = query.reshape(query.shape[0], attn.heads, -1, *query.shape[2:])
        key = key.swapaxes(1, 2)
        key = key.reshape(key.shape[0], attn.heads, -1, *key.shape[2:]).swapaxes(2, 3)
        value = value.swapaxes(1, 2)
        value = value.reshape(value.shape[0], attn.heads, -1, *value.shape[2:])

        query = mint.nn.functional.relu(query)
        key = mint.nn.functional.relu(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
        scores = mint.matmul(value, key)
        hidden_states_org = mint.matmul(scores, query)

        if hidden_states_org.dtype in [ms.float16, ms.bfloat16]:
            hidden_states_org = hidden_states_org.float()

        hidden_states_org = hidden_states_org[:, :, :-1] / (hidden_states_org[:, :, -1:] + 1e-15)
        hidden_states_org = hidden_states_org.flatten(start_dim=1, end_dim=2).swapaxes(1, 2)
        hidden_states_org = hidden_states_org.to(original_dtype)

        hidden_states_org = attn.to_out[0](hidden_states_org)
        hidden_states_org = attn.to_out[1](hidden_states_org)

        # perturbed path (identity attention)
        hidden_states_ptb = attn.to_v(hidden_states_ptb).to(original_dtype)

        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        hidden_states = mint.cat([hidden_states_org, hidden_states_ptb])

        if original_dtype == ms.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


ADDED_KV_ATTENTION_PROCESSORS = (AttnAddedKVProcessor,)

CROSS_ATTENTION_PROCESSORS = (
    AttnProcessor,
    AttnProcessor2_0,
    XFormersAttnProcessor,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
    FluxIPAdapterJointAttnProcessor2_0,
)

AttentionProcessor = Union[
    AttnProcessor,
    CustomDiffusionAttnProcessor,
    AttnAddedKVProcessor,
    JointAttnProcessor2_0,
    PAGJointAttnProcessor2_0,
    PAGCFGJointAttnProcessor2_0,
    FusedJointAttnProcessor2_0,
    AuraFlowAttnProcessor2_0,
    FusedAuraFlowAttnProcessor2_0,
    FluxAttnProcessor2_0,
    FusedFluxAttnProcessor2_0,
    CogVideoXAttnProcessor2_0,
    FusedCogVideoXAttnProcessor2_0,
    XFormersAttnProcessor,
    AttnProcessor2_0,
    MochiVaeAttnProcessor2_0,
    MochiAttnProcessor2_0,
    HunyuanAttnProcessor2_0,
    SanaLinearAttnProcessor2_0,
    PAGCFGSanaLinearAttnProcessor2_0,
    PAGIdentitySanaLinearAttnProcessor2_0,
    SanaMultiscaleLinearAttention,
    SanaMultiscaleAttnProcessor2_0,
    SanaMultiscaleAttentionProjection,
    PAGCFGIdentitySelfAttnProcessor2_0,
    PAGIdentitySelfAttnProcessor2_0,
    PAGCFGHunyuanAttnProcessor2_0,
    PAGHunyuanAttnProcessor2_0,
    PAGCFGHunyuanAttnProcessor2_0,
    LuminaAttnProcessor2_0,
    FusedAttnProcessor2_0,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
    SD3IPAdapterJointAttnProcessor2_0,
    PAGIdentitySelfAttnProcessor2_0,
    PAGCFGIdentitySelfAttnProcessor2_0,
]
