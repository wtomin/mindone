"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/colqwen2/test_modeling_colqwen2.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.

import inspect

import numpy as np
import pytest
import torch
from transformers import ColQwen2Config

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-3}
MODES = [1]  # 1: pynative mode (graph mode not supported yet)


class ColQwen2ModelTester:
    config_class = ColQwen2Config

    def __init__(
        self,
        batch_size=2,
        seq_length=7,
        image_size=224,
        num_channels=3,
        is_training=False,
        use_input_mask=True,
        use_labels=True,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        # VLM config (Qwen2VLConfig with reduced size)
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vision_config={
                "depth": 2,
                "in_chans": 3,
                "hidden_act": "silu",
                "intermediate_size": 32,
                "out_hidden_size": 128,
                "hidden_size": 128,
                "num_heads": 8,
                "patch_size": 14,
                "spatial_patch_size": 14,
                "spatial_merge_size": 1,
                "temporal_patch_size": 2,
        },
        # ColQwen2 specific
        embedding_dim=64,
        initializer_range=0.02,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.image_size = image_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices

        # Qwen2VL config parameters (reduced for testing)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vision_config = vision_config
        # ColQwen2 specific
        self.embedding_dim = embedding_dim
        self.initializer_range = initializer_range

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = self.vision_config["patch_size"]
        temporal_patch_size = self.vision_config["temporal_patch_size"]
        pixel_values = floats_numpy(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)

        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id
        input_ids[:, self.num_image_tokens - 1] = self.vision_start_token_id
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": np.array([[1, 1, 1]] * self.batch_size),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def get_config(self):
        from transformers.models.qwen2_vl import Qwen2VLConfig

        vlm_config = Qwen2VLConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            vision_config=self.vision_config,
            max_position_embeddings=512,
            initializer_range=self.initializer_range,
            use_cache=True,
        )

        return self.config_class(
            vlm_config=vlm_config,
            embedding_dim=self.embedding_dim,
            initializer_range=self.initializer_range,
        )


model_tester = ColQwen2ModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()

COLQWEN2_CASES = [
    [
        "ColQwen2ForRetrieval",
        "transformers.ColQwen2ForRetrieval",
        "mindone.transformers.ColQwen2ForRetrieval",
        (config,),
        {},
        (),
        inputs_dict,
        {"logits": "logits"},
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
        for case in COLQWEN2_CASES
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
    # set `hidden_dtype` if requiring, for some modules always compute in float
    # precision and require specific `hidden_dtype` to cast before return
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
            ms_output = ms_outputs[ms_idx]
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
