from typing import Callable, List, Optional, Tuple, Union

import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter, Tensor, mint
from mindspore.common.initializer import Normal, initializer

from mindone.transformers.mindspore_adapter.utils import _DTYPE_2_MIN

from ...activations import ACT2FN
from ...cache_utils import Cache, StaticCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_attn_mask_utils import AttentionMaskConverter

# from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import MSPreTrainedModel

# from ...processing_utils import Unpack
from ...utils import (  # LossKwargs,; add_start_docstrings,; add_start_docstrings_to_model_forward,; replace_return_docstrings,
    logging,
)

# from ...utils.deprecation import deprecate_kwarg
from .configuration_cohere2 import Cohere2Config

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "CohereConfig"


class Cohere2LayerNorm(nn.Cell):
    def __init__(self, hidden_size=None, eps=1e-5, bias=False):
        super().__init__()
        self.weight = Parameter(mint.ones(hidden_size, dtype=ms.float32))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(ms.float32)
        mean = hidden_states.mean(-1, keep_dims=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keep_dims=True)
        hidden_states = (hidden_states - mean) * mint.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(ms.float32) * hidden_states
        return hidden_states.to(input_dtype)


class Cohere2RotaryEmbedding(nn.Cell):
    def __init__(self, config: Cohere2Config):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config)
        self.inv_freq = inv_freq
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids):
        seq_len = mint.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, seq_len=seq_len)
            self.inv_freq = inv_freq
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.inv_freq = self.original_inv_freq
            self.max_seq_len_cached = self.original_max_seq_len

    def construct(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids)

        inv_freq_expanded = self.inv_freq[None, :, None].to(ms.float32).broadcast_to((position_ids.shape[0], -1, 1))
        position_ids_expanded = position_ids[:, None, :].to(ms.float32)

        freqs = (mint.matmul(inv_freq_expanded, position_ids_expanded)).swapaxes(1, 2)
        emb = mint.repeat_interleave(freqs, 2, dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Cohere2MLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = mint.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = mint.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = mint.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].broadcast_to((batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Cell,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = mint.matmul(query, key_states.swapaxes(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = mint.nn.functional.softmax(attn_weights, dim=-1).to(query.dtype)
    attn_output = mint.matmul(attn_weights, value_states)
    attn_output = attn_output.swapaxes(1, 2)

    return attn_output, attn_weights


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rot_x = mint.stack([-x2, x1], dim=-1).flatten(-2)
    return rot_x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    dtype = q.dtype
    q = q.to(ms.float32)
    k = k.to(ms.float32)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(dtype=dtype), k_embed.to(dtype=dtype)


class Cohere2Attention(nn.Cell):
    def __init__(self, config: Cohere2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        self.q_proj = mint.nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = mint.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = mint.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = mint.nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def construct(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = input_shape + (-1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).swapaxes(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).swapaxes(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).swapaxes(1, 2)

        cos, sin = position_embeddings
        if self.sliding_window is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                pass
                # attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        output_shape = input_shape + (-1,)
        attn_output = attn_output.reshape(output_shape)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Cohere2DecoderLayer(nn.Cell):
    def __init__(self, config: Cohere2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Cohere2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Cohere2MLP(config)
        self.input_layernorm = Cohere2LayerNorm(hidden_size=(config.hidden_size), eps=config.layer_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states_attention, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states_mlp = self.mlp(hidden_states)
        hidden_states = residual + hidden_states_attention + hidden_states_mlp

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Cohere2PreTrainedModel(MSPreTrainedModel):
    config_class = Cohere2Config
    base_model_prefix = "model"
    _no_split_modules = ["Cohere2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_flex_attn = False
    _supports_cache_class = True
    _supports_quantized_cache = False
    _supports_static_cache = True
    _supports_attention_backend = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, mint.nn.Linear):
            module.weight.set_data(initializer(Normal(std), module.weight.shape, module.weight.dtype))
            if module.bias is not None:
                module.bias.set_data(initializer("zeros", module.bias.shape, module.bias.dtype))
        elif isinstance(module, nn.Embedding):
            module.embedding_table.set_data(
                initializer(Normal(std), module.embedding_table.shape, module.embedding_table.dtype)
            )
            if module.padding_idx is not None:
                module.embedding_table.data[module.padding_idx] = 0
        elif isinstance(module, Cohere2LayerNorm):
            module.weight.set_data(initializer("ones", module.weight.shape, module.weight.dtype))


class Cohere2Model(Cohere2PreTrainedModel):
    def __init__(self, config: Cohere2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_seq_len_cached = config.max_position_embeddings

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.CellList(
            [Cohere2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Cohere2LayerNorm(hidden_size=(config.hidden_size), eps=config.layer_norm_eps)
        self.rotary_emb = Cohere2RotaryEmbedding(config=config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def prepare_static_cache(self, input_embeds, max_cache_len):
        bs = input_embeds.shape[0]
        max_batch_size, cache_dtype = (
            getattr(self.config, "num_beams", 1) * bs,
            self.dtype,
        )
        past_key_values = StaticCache(
            config=self.config, max_batch_size=max_batch_size, max_cache_len=max_cache_len, dtype=cache_dtype
        )
        return past_key_values

    def construct(
        self,
        input_ids: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Tensor] = None,
        **flash_attn_kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) or (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = self.prepare_static_cache(inputs_embeds, max_cache_len=self.max_seq_len_cached)

        if cache_position is None:
            past_seen_tokens = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
            cache_position = mint.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], dtype=ms.int32)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        causal_mask_mapping = attention_mask
        if not isinstance(causal_mask_mapping, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: Tensor,
        input_tensor: Tensor,
        cache_position: Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            # if isinstance(attention_mask, Tensor):
            #     attention_mask = make_flex_block_causal_mask(attention_mask)
            # if isinstance(attention_mask, BlockMask):
            #     return attention_mask
            raise NotImplementedError("flex_attention is not implemented for Cohere")

        past_seen_tokens = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if self.config._attn_implementation == "sdpa" and attention_mask is not None and not output_attentions:
            min_dtype = _DTYPE_2_MIN[ms.float32]
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: Tensor,
        sequence_length: int,
        target_length: int,
        dtype: ms.dtype,
        cache_position: Tensor,
        batch_size: int,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = _DTYPE_2_MIN[ms.float32]
            causal_mask = mint.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype)
            if sequence_length != 1:
                causal_mask = mint.triu(causal_mask, diagonal=1)
            causal_mask *= mint.arange(target_length) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].broadcast_to((batch_size, 1, -1, -1))
            if attention_mask is not None:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                # causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill( padding_mask, min_dtype)
                mask_temp = causal_mask[:, :, :, :mask_length].copy()
                mask_temp = mask_temp.masked_fill(padding_mask, min_dtype)
                causal_mask = mint.cat([mask_temp, causal_mask[:, :, :, mask_length:]], dim=-1)

        return causal_mask


class Cohere2ForCausalLM(Cohere2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Cohere2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = mint.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logit_scale = config.logit_scale
        self.tie_word_embeddings = config.tie_word_embeddings

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def construct(
        self,
        input_ids: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Union[Cache, List[Tensor]]] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Tensor] = None,
        logits_to_keep: Union[int, Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits * self.logit_scale

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["Cohere2ForCausalLM", "Cohere2Model", "Cohere2PreTrainedModel"]
