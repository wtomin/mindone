"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/mm_grounding_dino/test_modeling_mm_grounding_dino.py."""

# This module contains test cases that are defined in the .test_cases.py file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.

import inspect

import numpy as np
import pytest
import torch
from transformers import MMGroundingDinoConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-3}
MODES = [0, 1]  # 0: graph mode, 1: pynative mode


class MMGroundingDinoModelTester:
    config_class = MMGroundingDinoConfig

    def __init__(
        self,
        batch_size=2,
        num_channels=3,
        image_size=224,
        num_queries=10,  # Reduced from 900 for testing
        text_seq_length=16,  # Reduced text length for testing
        is_training=False,
        use_labels=True,
        # config parameters - reduced for testing
        d_model=32,  # Reduced from 256
        encoder_layers=2,  # Reduced from 6
        decoder_layers=2,  # Reduced from 6
        encoder_attention_heads=2,  # Reduced from 8
        decoder_attention_heads=2,  # Reduced from 8
        encoder_ffn_dim=64,  # Reduced from 2048
        decoder_ffn_dim=64,  # Reduced from 2048
        backbone_kwargs=None,
        **kwargs
    ):
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_queries = num_queries
        self.text_seq_length = text_seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim

        # Backbone config - use smaller swin
        self.backbone_config = {
            "window_size": 7,
            "image_size": image_size,
            "embed_dim": 48,  # Reduced from 96
            "depths": [1, 1, 2, 1],  # Reduced depths
            "num_heads": [2, 4, 8, 16],  # Reduced num_heads
            "out_indices": [1, 2, 3],
        }

        # Text config - use smaller bert
        self.text_config = {
            "vocab_size": 52,  # Small vocab for testing
            "hidden_size": 32,  # Small hidden size
            "num_hidden_layers": 2,  # Reduced layers
            "num_attention_heads": 2,  # Reduced attention heads
            "intermediate_size": 64,  # Small intermediate
            "max_position_embeddings": 32,  # Small max length
        }

    def prepare_config_and_inputs(self):
        # Image inputs
        pixel_values = np.random.normal(
            size=(self.batch_size, self.num_channels, self.image_size, self.image_size)
        ).astype(np.float32)

        # Text inputs - reduced length for testing
        input_ids = ids_numpy([self.batch_size, self.text_seq_length], self.text_config["vocab_size"])
        attention_mask = np.ones((self.batch_size, self.text_seq_length), dtype=np.int32)

        # For Grounding DINO, we need boolean attention mask
        text_token_mask = attention_mask.astype(bool)

        # Labels for training
        labels = None
        pixel_mask = None
        if self.use_labels:
            # Fake labels for testing
            labels = [{
                "class_labels": np.array([0, 1], dtype=np.int64),
                "boxes": np.array([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]], dtype=np.float32),
            }] * self.batch_size

        config = self.get_config()

        return config, pixel_values, input_ids, text_token_mask, labels, pixel_mask

    def get_config(self):
        return self.config_class(
            backbone_config={"model_type": "swin", **self.backbone_config},
            text_config={"model_type": "bert", **self.text_config},
            num_queries=self.num_queries,
            d_model=self.d_model,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            encoder_attention_heads=self.encoder_attention_heads,
            decoder_attention_heads=self.decoder_attention_heads,
            encoder_ffn_dim=self.encoder_ffn_dim,
            decoder_ffn_dim=self.decoder_ffn_dim,
            max_text_len=self.text_seq_length,
            use_timm_backbone=False,
            use_pretrained_backbone=False,
        )


model_tester = MMGroundingDinoModelTester()
(
    config,
    pixel_values,
    input_ids,
    text_token_mask,
    labels,
    pixel_mask,
) = model_tester.prepare_config_and_inputs()


MM_GROUNDING_DINO_CASES = [
    [
        "MMGroundingDinoModel",
        "transformers.MMGroundingDinoModel",
        "mindone.transformers.MMGroundingDinoModel",
        (config,),
        {},
        (pixel_values, input_ids),
        {
            "text_token_mask": text_token_mask,
            "pixel_mask": pixel_mask,
        },
        {
            "last_hidden_state": 0,
            "pred_boxes": 1,
            "encoder_last_hidden_state": 2,
        },
    ],
    [
        "MMGroundingDinoForObjectDetection",
        "transformers.MMGroundingDinoForObjectDetection",
        "mindone.transformers.MMGroundingDinoForObjectDetection",
        (config,),
        {},
        (pixel_values, input_ids),
        {
            "text_token_mask": text_token_mask,
            "pixel_mask": pixel_mask,
            "labels": labels,
        },
        {
            "pred_logits": 0,
            "pred_boxes": 1,
            "loss": 2,
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
        for case in MM_GROUNDING_DINO_CASES
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
    
    # Set text_token_mask to boolean type as required by model
    if "text_token_mask" in pt_inputs_kwargs:
        pt_inputs_kwargs["text_token_mask"] = torch.tensor(pt_inputs_kwargs["text_token_mask"], dtype=torch.bool)
        ms_inputs_kwargs["text_token_mask"] = ms.Tensor(ms_inputs_kwargs["text_token_mask"], dtype=ms.bool_)
    
    # Handle pixel_mask if provided
    if "pixel_mask" in pt_inputs_kwargs and pt_inputs_kwargs["pixel_mask"] is not None:
        pt_inputs_kwargs["pixel_mask"] = torch.tensor(pt_inputs_kwargs["pixel_mask"], dtype=torch.bool)
        ms_inputs_kwargs["pixel_mask"] = ms.Tensor(ms_inputs_kwargs["pixel_mask"], dtype=ms.bool_)
        
    # Convert labels if training mode
    if "labels" in pt_inputs_kwargs and pt_inputs_kwargs["labels"] is not None:
        labels = pt_inputs_kwargs["labels"]
        if labels is not None and len(labels) > 0:
            pt_labels = [{k: torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in label.items()} for label in labels]
            ms_labels = [{k: ms.Tensor(v) if isinstance(v, np.ndarray) else v for k, v in label.items()} for label in labels]
            pt_inputs_kwargs["labels"] = pt_labels
            ms_inputs_kwargs["labels"] = ms_labels
    
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