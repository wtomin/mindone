from collections import OrderedDict
from typing import List

import mindspore as ms
from mindspore import nn, ops


def _get_pt2ms_mappings(m):
    mappings = {}  # pt_param_name: (ms_param_name, pt_param_to_ms_param_func)
    for name, cell in m.cells_and_names():
        if isinstance(cell, (nn.Conv1d, nn.Conv1dTranspose)):
            mappings[f"{name}.weight"] = f"{name}.weight", lambda x: ms.Parameter(
                ops.expand_dims(x, axis=-2), name=x.name
            )
        elif isinstance(cell, nn.Embedding):
            mappings[f"{name}.weight"] = f"{name}.embedding_table", lambda x: x
        elif isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            mappings[f"{name}.weight"] = f"{name}.gamma", lambda x: x
            mappings[f"{name}.bias"] = f"{name}.beta", lambda x: x
            if isinstance(cell, (nn.BatchNorm2d,)):
                mappings[f"{name}.running_mean"] = f"{name}.moving_mean", lambda x: x
                mappings[f"{name}.running_var"] = f"{name}.moving_variance", lambda x: x
                mappings[f"{name}.num_batches_tracked"] = None, lambda x: x
    return mappings


def _convert_state_dict(m, state_dict_pt):
    if not state_dict_pt:
        return state_dict_pt
    pt2ms_mappings = _get_pt2ms_mappings(m)
    state_dict_ms = {}
    while state_dict_pt:
        name_pt, data_pt = state_dict_pt.popitem()
        name_ms, data_mapping = pt2ms_mappings.get(name_pt, (name_pt, lambda x: x))
        data_ms = data_mapping(data_pt)
        if name_ms is not None:
            state_dict_ms[name_ms] = data_ms
    return state_dict_ms


def _load_state_dict_into_model(model_to_load, state_dict: OrderedDict, strict_load=False) -> List[str]:
    # TODO: error_msgs is always empty for now. Maybe we need to rewrite MindSpore's `load_param_into_net`.
    #  Error msgs should contain caught exception like size mismatch instead of missing/unexpected keys.
    # TODO: We should support loading float16 state_dict into float32 model, like PyTorch's behavior.
    error_msgs = []
    # TODO: State dict loading in mindspore does not cast dtype correctly. We do it manually. It's might unsafe.
    local_state = {k: v for k, v in model_to_load.parameters_and_names()}
    for k, v in state_dict.items():
        if k in local_state:
            v.set_dtype(local_state[k].dtype)
        else:
            pass  # unexpect key keeps origin dtype
    ms.load_param_into_net(model_to_load, state_dict, strict_load=strict_load)
    return error_msgs
