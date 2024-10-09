import logging
import os
import sys

from omegaconf import OmegaConf

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
ad_lib_path = os.path.abspath(os.path.join(__dir__, "../"))
sys.path.append(ad_lib_path)
from mindone.utils.config import get_obj_from_str

num_timesteps = 1000

logger = logging.getLogger(__name__)


def build_model_from_config(config, unet_config_update=None, vae_use_fp16=None, snr_gamma=None):
    config = OmegaConf.load(config).model
    if unet_config_update is not None:
        # config["params"]["unet_config"]["params"]["enable_flash_attention"] = enable_flash_attention
        unet_args = config["params"]["unet_config"]["params"]
        for name, value in unet_config_update.items():
            if value is not None:
                logger.info("Arg `{}` updated: {} -> {}".format(name, unet_args[name], value))
                unet_args[name] = value

    if vae_use_fp16 is not None:
        config.params.first_stage_config.params.use_fp16 = vae_use_fp16

    if snr_gamma is not None:
        config.params.snr_gamma = snr_gamma

    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    config_params = config.get("params", dict())
    # config_params['cond_stage_trainable'] = cond_stage_trainable # TODO: easy config
    return get_obj_from_str(config["target"])(**config_params)


def run_inference(z, cond, conditioned_frames, visual_embedder, semantic_motion_predictor, model):
    num_frames = 16
    t = Tensor([12] * z.shape[0], dtype=mstype.int32)
    noisy_latents = ops.randn_like(z)
    # 4. get interpolated conditioned imagse embedding
    B, F, C, H, W = conditioned_frames.shape
    conditioned_frames = conditioned_frames.reshape((B * F, C, H, W))
    image_cond = visual_embedder(conditioned_frames)
    image_cond = image_cond.reshape((B, F, image_cond.shape[-1]))
    interpolated_image_cond = semantic_motion_predictor(
        image_cond, target_len=num_frames
    )  # (Bs, 77, num_frames, hidden_size)

    # 5.  unet forward, predict conditioned on conditions
    model_output = model(
        noisy_latents,
        t,
        append_to_context=interpolated_image_cond,  # append image embedding to text embedding for cross-attention
        **cond,
    )
    return model_output


bs = 1
h = w = 512
num_frames = 16
cond = {"c_crossattn": ops.randn(bs, 77, 768)}
z = ops.randn((bs, 4, num_frames, h // 8, w // 8))
conditioned_frames = ops.randn((bs, 2, 3, 224, 224))

# 2. build model
unet_config_update = dict(
    enable_flash_attention=True,
    use_recompute=False,
)
model_config = "configs/stable_diffusion/v1-train-mmv2-interpolation.yaml"
model = build_model_from_config(
    model_config,
    unet_config_update,
    vae_use_fp16=True,
    snr_gamma=5.0,
)


run_inference(z, cond, conditioned_frames, model.visual_embedder, model.semantic_motion_predictor, model.model)
