import logging
import os
import sys

import numpy as np

import mindspore as ms
from mindspore import nn

from mindone.utils.amp import auto_mixed_precision

sys.path.append(".")
from opensora.models.ae import getae_wrapper
from opensora.models.ae.videobase.modules.updownsample import TrilinearInterpolate
from opensora.utils.utils import get_precision

logger = logging.getLogger(__name__)


def init_env(mode):
    # no parallel mode currently
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=mode,
        device_target="Ascend",
        device_id=device_id,
    )

    return device_id


def compare_torch_ms_npy(torch_data, ms_data, data_name=""):
    if isinstance(torch_data, str):
        torch_data = np.load(torch_data)
    if isinstance(ms_data, str):
        ms_data = np.load(ms_data)

    abs_diff = np.abs(torch_data - ms_data)
    rel_diff = (abs_diff / (np.abs(torch_data) + 1e-8)).mean()
    print(f"{data_name}: abs diff {abs_diff}, relative diff {rel_diff}")


def main():
    mode = 0
    precision = "bf16"
    print(f"mode {mode}, dtype {precision}")
    init_env(mode)

    vae = getae_wrapper("CausalVAEModel_4x8x8")("LanguageBind/Open-Sora-Plan-v1.1.0/vae")
    vae.vae.enable_tiling()
    vae.vae.tile_overlap_factor = 0.25
    vae.set_train(False)
    for param in vae.get_parameters():
        param.requires_grad = False

    if precision in ["fp16", "bf16"]:
        amp_level = "O2"
        dtype = get_precision(precision)
        custom_fp32_cells = [nn.GroupNorm] if dtype == ms.float16 else [nn.AvgPool2d, TrilinearInterpolate]
        vae = auto_mixed_precision(vae, amp_level, dtype, custom_fp32_cells=custom_fp32_cells)
        logger.info(f"Set mixed precision to O2 with dtype={precision}")
    elif precision == "fp32":
        amp_level = "O0"
    else:
        raise ValueError(f"Unsupported precision {precision}")
    torch_folder = "torch_npy_20260612/"
    x_vae = np.load(os.path.join(torch_folder, "x_vae.npy"))
    x_vae = ms.Tensor(x_vae, dtype)  # b c t h w

    latents = vae.encode(x_vae)
    compare_torch_ms_npy(os.path.join(torch_folder, "latents_torch.npy"), latents.asnumpy(), "encoder output")

    latents = latents.to(dtype)
    video_recon = vae.decode(latents)  # b t c h w
    compare_torch_ms_npy(os.path.join(torch_folder, "video_recon_torch.npy"), video_recon.asnumpy(), "decoder output")


if __name__ == "__main__":
    main()
