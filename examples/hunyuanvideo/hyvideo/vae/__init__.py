from pathlib import Path

import mindspore as ms

from mindone.safetensors.mindspore import load_file
from mindone.utils.amp import auto_mixed_precision

from ..constants import PRECISION_TO_TYPE, VAE_PATH
from .autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from .unet_causal_3d_blocks import GroupNorm, MSInterpolate, MSPad


def load_vae(
    vae_type: str = "884-16c-hy",
    vae_precision: str = None,
    sample_size: tuple = None,
    vae_path: str = None,
    logger=None,
    state_dict=None,
):
    """the fucntion to load the 3D VAE model

    Args:
        vae_type (str): the type of the 3D VAE model. Defaults to "884-16c-hy".
        vae_precision (str, optional): the precision to load vae. Defaults to None.
        sample_size (tuple, optional): the tiling size. Defaults to None.
        vae_path (str, optional): the path to vae. Defaults to None.
        logger (_type_, optional): logger. Defaults to None.
        state_dict (Dict, optional): existing state dictionary to be loaded.
    """
    if vae_path is None:
        vae_path = VAE_PATH[vae_type]

    if logger is not None:
        logger.info(f"Loading 3D VAE model ({vae_type}) from: {vae_path}")
    config = AutoencoderKLCausal3D.load_config(vae_path)
    if sample_size:
        vae = AutoencoderKLCausal3D.from_config(config, sample_size=sample_size)
    else:
        vae = AutoencoderKLCausal3D.from_config(config)
    if state_dict is None:
        vae_ckpt = Path(vae_path) / "model.safetensors"
        # assert vae_ckpt.exists(), f"VAE checkpoint not found: {vae_ckpt}"

        if vae_ckpt.exists():
            ckpt = load_file(vae_ckpt)
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            if any(k.startswith("vae.") for k in ckpt.keys()):
                ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
            vae.load_state_dict(ckpt)
        else:
            print("No vae ckpt is loaded")
    else:
        vae.load_state_dict(state_dict)

    spatial_compression_ratio = vae.config.spatial_compression_ratio
    time_compression_ratio = vae.config.time_compression_ratio

    # set mixed precision
    if vae_precision is not None:
        if vae_precision != "fp32":
            dtype = PRECISION_TO_TYPE[vae_precision]
            if dtype == ms.float16:
                custom_fp32_cells = [GroupNorm]
            elif dtype == ms.bfloat16:
                custom_fp32_cells = [MSPad, MSInterpolate]
            else:
                raise ValueError

            # TODO: try 'auto' in ms.amp.auto_mixed_precision
            amp_level = "O2"
            vae = auto_mixed_precision(vae, amp_level=amp_level, dtype=dtype, custom_fp32_cells=custom_fp32_cells)
            logger.info(
                f"Set mixed precision to {amp_level} with dtype={vae_precision}, custom fp32_cells {custom_fp32_cells}"
            )

    vae.set_train(False)
    for param in vae.trainable_params():
        param.requires_grad = False

    if logger is not None:
        logger.info(f"VAE param dtype: {vae.dtype}")

    return vae, vae_path, spatial_compression_ratio, time_compression_ratio
