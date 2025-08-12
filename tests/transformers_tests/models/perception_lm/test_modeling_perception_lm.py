
import numpy as np
import pytest
import torch
from transformers import PerceptionLMConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]  # not support graph mode yet


class PerceptionLMModelTester:
    def __init__(
        self,
        # text config (tiny)
        text_config={
            "model_type": "llama",
            "vocab_size": 97,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "max_position_embeddings": 64,
            "initializer_range": 0.02,
            "pad_token_id": 1,
        },
        # vision config kept minimal; we won't feed images in this precision test
        vision_config=None,
        seq_length=7,
    ):
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.batch_size = 2

    def get_config(self):
        return PerceptionLMConfig(text_config=self.text_config, vision_config=self.vision_config)

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_numpy([self.batch_size, self.seq_length], config.text_config.vocab_size)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        return config, input_ids, attention_mask


model_tester = PerceptionLMModelTester()
config, input_ids, attention_mask = model_tester.prepare_config_and_inputs()


_CASES = [
    [
        "PerceptionLMModel",
        "transformers.PerceptionLMModel",
        "mindone.transformers.PerceptionLMModel",
        (config,),
        {},
        (),
        {"input_ids": input_ids, "attention_mask": attention_mask},
        {"last_hidden_state": "last_hidden_state"},
    ],
    [
        "PerceptionLMForConditionalGeneration",
        "transformers.PerceptionLMForConditionalGeneration",
        "mindone.transformers.PerceptionLMForConditionalGeneration",
        (config,),
        {},
        (),
        {"input_ids": input_ids, "attention_mask": attention_mask},
        {"logits": "logits"},
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [case + [dtype] + [mode] for case in _CASES for dtype in DTYPE_AND_THRESHOLDS.keys() for mode in MODES],
)
def test_named_modules(
    name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype, mode
):
    ms.set_context(mode=mode)

    (pt_model, ms_model, pt_dtype, ms_dtype) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_key in outputs_map.items():
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = getattr(ms_outputs, ms_key)
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
