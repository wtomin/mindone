import logging
import os
import sys
from typing import Tuple

import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore.communication.management import get_group_size, get_rank, init

# TODO: remove in future when mindone is ready for install
mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(os.path.abspath("./"))
from opensora.models.diffusion.latte.modeling_latte import LatteT2V, LayerNorm
from opensora.models.diffusion.latte.modules import Attention
from opensora.utils.utils import get_precision, parse_env

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    parallel_mode: str = "data",
    enable_dvm: bool = False,
    precision_mode: str = None,
    global_bf16: bool = False,
) -> Tuple[int, int, int]:
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )
        if parallel_mode == "optim":
            print("use optim parallel")
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
                enable_parallel_optimizer=True,
            )
            init()
            device_num = get_group_size()
            rank_id = get_rank()
        else:
            init()
            device_num = get_group_size()
            rank_id = get_rank()
            logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
            ms.reset_auto_parallel_context()

            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
                device_num=device_num,
            )

        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        logger.info(dict(zip(var_info, var_value)))

    else:
        device_num = 1
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )

    if enable_dvm:
        print("enable dvm")
        ms.set_context(enable_graph_kernel=True, graph_kernel_flags="--disable_cluster_ops=Pow,Select")
    if precision_mode is not None and len(precision_mode) > 0:
        ms.set_context(ascend_config={"precision_mode": precision_mode})
    if global_bf16:
        print("Using global bf16")
        ms.set_context(
            ascend_config={"precision_mode": "allow_mix_precision_bf16"}
        )  # reset ascend precison mode globally
    return rank_id, device_num


def compare_torch_ms_npy(torch_data, ms_data, data_name=""):
    if isinstance(torch_data, str):
        torch_data = np.load(torch_data)
    if isinstance(ms_data, str):
        ms_data = np.load(ms_data)

    abs_diff = np.abs(torch_data - ms_data)
    rel_diff = (abs_diff / (np.abs(torch_data) + 1e-8)).mean()
    print(f"{data_name}: abs diff {abs_diff.mean()}, relative diff {rel_diff}")


if __name__ == "__main__":
    # 1. init env
    mode = 0
    FA = True
    precision = "bf16"
    parse_env("kbk")

    # 1. init
    rank_id, device_num = init_env(
        mode,
        enable_dvm=False,
        precision_mode=None,
    )

    # 4. latte model initiate and weight loading
    transformer_model = LatteT2V.from_pretrained(
        "LanguageBind/Open-Sora-Plan-v1.1.0/",
        subfolder="65x512x512",
        checkpoint_path=None,
        enable_flash_attention=FA,
    )
    transformer_model.force_images = False
    # mixed precision
    dtype = get_precision(precision)
    if precision in ["fp16", "bf16"]:
        amp_level = "O2"
        transformer_model = auto_mixed_precision(
            transformer_model,
            amp_level=amp_level,
            dtype=dtype,
            custom_fp32_cells=[LayerNorm, Attention, nn.SiLU, nn.GELU] if dtype == ms.float16 else [nn.MaxPool2d],
        )
        logger.info(f"Set mixed precision to O2 with dtype={precision}")

    elif precision == "fp32":
        amp_level = "O0"
    else:
        raise ValueError(f"Unsupported precision {precision}")

    transformer_model = transformer_model.set_train(False)
    for param in transformer_model.get_parameters():  # freeze transformer_model
        param.requires_grad = False

    torch_folder = "torch_npy_20240612/"
    latent_model_input = np.load(os.path.join(torch_folder, "latents.npy"))
    latent_model_input = ms.Tensor(latent_model_input, dtype=dtype)
    current_timestep = ms.Tensor([894], dtype=ms.int32)

    prompt_embeds = np.load(os.path.join(torch_folder, "prompt_embeds.npy"))
    prompt_embeds = ms.Tensor(prompt_embeds)
    prompt_embeds_mask = np.load(os.path.join(torch_folder, "prompt_embeds_mask.npy"))
    prompt_embeds_mask = ms.Tensor(prompt_embeds_mask)

    noise_pred = transformer_model(
        latent_model_input,  # (b c t h w)
        encoder_hidden_states=prompt_embeds,  # (b n c)
        timestep=current_timestep,  # (b)
        added_cond_kwargs={},
        enable_temporal_attentions=True,
        encoder_attention_mask=prompt_embeds_mask,  # (b n)
    )

    compare_torch_ms_npy(
        os.path.join(torch_folder, "noise_pred_torch.npy"), noise_pred.asnumpy(), "noise_pred (first step)"
    )
