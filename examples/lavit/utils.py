import logging
import os
from multiprocessing import Pool
from typing import Tuple

import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore.communication.management import get_group_size, get_rank, init

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    device_id: int = None,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    parallel_mode: str = "data",
    mempool_block_size: str = "9GB",
    global_bf16: bool = False,
    strategy_ckpt_save_file: str = "",
    optimizer_weight_shard_size: int = 8,
    sp_size: int = 1,
    jit_level: str = None,
    enable_parallel_fusion: bool = False,
    precision_mode: str = None,
) -> Tuple[int, int, int]:
    """
    Initialize MindSpore environment.

    Args:
        mode (int, default 0): MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed (int, default 42): The seed value for reproducibility. Default is 42.
        distributed (bool, False): Whether to enable distributed training. Default is False.
        device_id (int, default None): If distributed is False (single-device), and if device_id is provided, \
            use the device_id to set the device index; if not provided, wil get os.environ["DEVICE_ID"]
        max_device_memory (str, default None): the maximum available device memory, e.g., "30GB" for Ascend 910A or "59GB" for Ascend 910B.
        device_target (str, default "Ascend"): the target device type, supporting "Ascend" or "GPU"
        parallel_mode (str, default "data"): if `distributed` is True, `parallel_mode` will be one of ["data", "optim"]
        mempool_block_size (str, default "9GB"): Set the size of the memory pool block in PyNative mode for devices. \
            The format is “xxGB”. Default: “1GB”. Minimum size is “1G”.
        global_bf16 (bool, default False): Whether to use global_bf16 in GE mode (jit_level="O2").
        strategy_ckpt_save_file (str, default None): The path to strategy_ckpt when parallel_mode == "optim". \
            This strategy_ckpt is useful for merging multiple checkpoint shards.
        optimizer_weight_shard_size (int, default 8): Set the size of the communication domain split by the optimizer \
            weight when parallel_mode == "optim". The numerical range can be (0, device_num].
        sp_size (int, default 1): Set the sequence parallel size. Default is 1. The device_num should be >= sp_size \
            and device_num should be divisble by sp_size.
        jit_level (str, default None): If set, will set the compilation optimization level. Supports ["O0", "O1", "O2"]. \
            "O1" means KernelByKernel (KBK) mode, "O2" means DVM mode, and "O3" means GE mode.
        enable_parallel_fusion (bool, default None): If True, will enable optimizer parallel fusion for AdamW.
        precision_mode (str, default None): If provided, will set precision_mode to overwrite the default option "allow_fp32_to_fp16".
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)
    ms.set_context(mempool_block_size=mempool_block_size)

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)
    if enable_parallel_fusion:
        ms.set_context(graph_kernel_flags="--enable_parallel_fusion --enable_expand_ops=AdamApplyOneWithDecayAssign")

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
            ascend_config={"precision_mode": "allow_fp32_to_fp16"},
        )
        if parallel_mode == "optim":
            logger.info("use optim parallel")
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
                parallel_optimizer_config={"optimizer_weight_shard_size": optimizer_weight_shard_size},
                enable_parallel_optimizer=True,
                strategy_ckpt_config={
                    "save_file": strategy_ckpt_save_file,
                    "only_trainable_params": False,
                },
            )
            init()
            device_num = get_group_size()
            rank_id = get_rank()
        elif parallel_mode == "data":
            init()
            logger.info("use data parallel")
            device_num = get_group_size()
            rank_id = get_rank()
            logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
            ms.reset_auto_parallel_context()

            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
                device_num=device_num,
            )
        else:
            raise ValueError(f"{parallel_mode} not supported!")

        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        logger.info(dict(zip(var_info, var_value)))

    else:
        device_num = 1
        rank_id = device_id if device_id is not None else int(os.getenv("DEVICE_ID", 0))
        ms.set_context(
            mode=mode,
            device_target=device_target,
            ascend_config={"precision_mode": "allow_fp32_to_fp16"},
        )

    if jit_level is not None:
        if mode == 1:
            logger.info(f"Only graph mode supports jit_level! Will ignore jit_level {jit_level} in Pynative mode.")
        else:
            try:
                if jit_level in ["O0", "O1", "O2"]:
                    logger.info(f"Using jit_level: {jit_level}")
                    ms.context.set_context(jit_config={"jit_level": jit_level})  # O0: KBK, O1:DVM, O2: GE
                else:
                    logger.warning(
                        f"Unsupport jit_level: {jit_level}. The framework automatically selects the execution method"
                    )
            except Exception:
                logger.warning(
                    "The current jit_level is not suitable because current MindSpore version does not match,"
                    "please upgrade the MindSpore version."
                )
                raise Exception
    if precision_mode is not None and len(precision_mode) > 0:
        ms.set_context(ascend_config={"precision_mode": precision_mode})
    if global_bf16:
        logger.info("Using global bf16")
        assert jit_level is not None and jit_level == "O2", "global_bf16 is supported in GE mode only!"
        ms.set_context(ascend_config={"precision_mode": "allow_mix_precision_bf16"})

    assert device_num >= sp_size and device_num % sp_size == 0, (
        f"unable to use sequence parallelism, " f"device num: {device_num}, sp size: {sp_size}"
    )
    # initialize_sequence_parallel_state(sp_size)
    # if get_sequence_parallel_state():
    #     assert (
    #         parallel_mode == "data"
    #     ), f"only support seq parallelism with parallel mode `data`, but got `{parallel_mode}`"

    return rank_id, device_num


def get_precision(mixed_precision):
    if mixed_precision == "bf16":
        dtype = ms.bfloat16
    elif mixed_precision == "fp16":
        dtype = ms.float16
    else:
        dtype = ms.float32
    return dtype


def get_amp_model(model, dtype, amp_level, bf16_custom_fp32_cells=[], fp16_custom_fp32_cells=[]):
    if dtype in [ms.float16, ms.bfloat16]:
        model = auto_mixed_precision(
            model,
            amp_level=amp_level,
            dtype=dtype,
            custom_fp32_cells=fp16_custom_fp32_cells if dtype == ms.float16 else bf16_custom_fp32_cells,
        )
        logger.info(f"Set mixed precision to O2 with dtype={dtype}")

    elif dtype == ms.float32:
        amp_level = "O0"
    else:
        raise ValueError(f"Unsupported precision {dtype}")
    print(f"auto_mixed_precision level {amp_level}, dtype {dtype}")
    return model


def process_key(key_val):
    """Processes a single key-value pair from the source data."""
    k, val = key_val
    val = val.detach().float().numpy().astype(np.float32)
    return k, ms.Parameter(ms.Tensor(val, dtype=ms.float32))


def load_torch_state_dict_to_ms_ckpt(ckpt_file, num_workers=8, exclude_prefix=None, include_prefix=None):
    import torch

    source_data = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    if "state_dict" in source_data:
        source_data = source_data["state_dict"]
    if "ema" in source_data:
        source_data = source_data["ema"]

    if exclude_prefix is not None:
        if isinstance(exclude_prefix, str):
            exclude_prefix = [exclude_prefix]
        assert (
            isinstance(exclude_prefix, list)
            and len(exclude_prefix) > 0
            and isinstance(exclude_prefix[0], str)
            and len(exclude_prefix[0]) > 0
        )
    if exclude_prefix is not None and len(exclude_prefix) > 0:
        keys_to_remove = [key for key in source_data if any(key.startswith(prefix) for prefix in exclude_prefix)]
        for key in keys_to_remove:
            del source_data[key]

    if include_prefix is not None:
        if isinstance(include_prefix, str):
            include_prefix = [include_prefix]
        assert (
            isinstance(include_prefix, list)
            and len(include_prefix) > 0
            and isinstance(include_prefix[0], str)
            and len(include_prefix[0]) > 0
        )
    if include_prefix is not None and len(include_prefix) > 0:
        keys_to_retain = [key for key in source_data if any(key.startswith(prefix) for prefix in include_prefix)]
        for key in source_data.keys():
            if key not in keys_to_retain:
                del source_data[key]
    assert len(source_data.keys()), "state dict is empty!"
    # Use multiprocessing to process keys in parallel
    with Pool(processes=num_workers) as pool:
        target_data = dict(
            tqdm(pool.imap(process_key, source_data.items()), total=len(source_data), desc="Checkpoint Conversion")
        )

    return target_data
