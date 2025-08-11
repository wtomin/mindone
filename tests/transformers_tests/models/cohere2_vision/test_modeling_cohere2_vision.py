"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/cohere2_vision/test_modeling_cohere2_vision.py."""

# This module follows the generic MS vs PT parity testing style used elsewhere:
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
# It creates a very small Cohere2Vision configuration (both vision and text), then forwards inputs through
# the PT and MS modules and compares outputs.

import inspect

import numpy as np
import pytest
import torch
from transformers import Cohere2Config, SiglipVisionConfig, Cohere2VisionConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-2, "fp16": 5e-2, "bf16": 5e-2}
MODES = [1]  # pynative only for now


class Cohere2VisionModelTester:
    def __init__(
        self,
        batch_size=1,
        seq_length=7,
        vocab_size=99,
        text_hidden_size=64,
        text_num_hidden_layers=2,
        text_num_attention_heads=4,
        text_intermediate_size=128,
        max_position_embeddings=64,
        vision_hidden_size=64,
        vision_intermediate_size=128,
        vision_image_size=128,
        vision_num_hidden_layers=2,
        vision_num_attention_heads=4,
        image_token_id=255036,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.text_hidden_size = text_hidden_size
        self.text_num_hidden_layers = text_num_hidden_layers
        self.text_num_attention_heads = text_num_attention_heads
        self.text_intermediate_size = text_intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.vision_hidden_size = vision_hidden_size
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_image_size = vision_image_size
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.image_token_id = image_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = ids_numpy([self.batch_size, self.seq_length], 2)

        text_config = Cohere2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.text_hidden_size,
            intermediate_size=self.text_intermediate_size,
            num_hidden_layers=self.text_num_hidden_layers,
            num_attention_heads=self.text_num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            use_cache=False,
            sliding_window=64,
        )
        vision_config = SiglipVisionConfig(
            hidden_size=self.vision_hidden_size,
            intermediate_size=self.vision_intermediate_size,
            image_size=self.vision_image_size,
            num_hidden_layers=self.vision_num_hidden_layers,
            num_attention_heads=self.vision_num_attention_heads,
        )

        config = Cohere2VisionConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=self.image_token_id,
            alignment_intermediate_size=self.vision_hidden_size * 4,
            downsample_factor=2,
        )

        # enable eager attention impl for both backends if applicable
        if hasattr(text_config, "_attn_implementation"):
            text_config._attn_implementation = "eager"
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"

        return (config, input_ids, attention_mask)


model_tester = Cohere2VisionModelTester()
(config, input_ids, attention_mask) = model_tester.prepare_config_and_inputs()


COHERE2_VISION_CASES = [
    [
        "Cohere2VisionForConditionalGeneration",
        "transformers.Cohere2VisionForConditionalGeneration",
        "mindone.transformers.Cohere2VisionForConditionalGeneration",
        (config,),
        {},
        (),
        {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": True},
        {
            "logits": "logits",
        },
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [
        case
        + [
            dtype,
        ]
        + [
            mode,
        ]
        for case in COHERE2_VISION_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
def test_named_modules(
    name,
    pt_module,
    ms_module,
    init_args,
    init_kwargs,
    inputs_args,
    inputs_kwargs,
    outputs_map,
    dtype,
    mode,
):
    ms.set_context(mode=mode)

    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    # set `hidden_dtype` if requiring
    if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
        pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
        ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = getattr(ms_outputs, ms_idx)
            if isinstance(pt_output, (list, tuple)):
                pt_outputs_n += list(pt_output)
                ms_outputs_n += list(ms_output)
            else:
                pt_outputs_n.append(pt_output)
                ms_outputs_n.append(ms_output)
        diffs = compute_diffs(pt_outputs_n, ms_outputs_n)
    else:
        diffs = compute_diffs(pt_outputs, ms_outputs)

    THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < THRESHOLD).all(), (
        f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
    )
