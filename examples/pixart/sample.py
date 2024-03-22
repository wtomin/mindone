import argparse
import datetime
import logging
import os
import sys

import numpy as np
import yaml
from modules.t5 import T5Embedder
from PIL import Image
from pipelines import InferPipeline
from tqdm import tqdm
from utils.data_utils import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST, get_chunks, prepare_prompt_ar
from utils.model_utils import _check_cfgs_in_parser, count_params, remove_pname_prefix, str2bool

import mindspore as ms
from mindspore import Tensor, ops

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)


from modules.autoencoder import SD_CONFIG, AutoencoderKL

from mindone.models.pixart import PixArt_XL_2, PixArtMS_XL_2
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.mode,
        device_target=args.device_target,
        device_id=device_id,
    )
    if args.precision_mode is not None:
        ms.set_context(ascend_config={"precision_mode": args.precision_mode})
    return device_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="image size in [512, 1024]",
    )
    parser.add_argument("--txt_file", default="asset/samples.txt", type=str)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size to generated videos",
    )
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="model checkpoint path. If specified, will load from it, otherwise, will use random initialization",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-ema.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument("--t5_checkpoint", default="models/t5-v1_1-xxl", type=str)
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )

    parser.add_argument("--sampling_steps", type=int, default=50, help="Diffusion Sampling Steps")
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="the scale for classifier-free guidance")
    # MS new args
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--enable_flash_attention",
        default=False,
        type=str2bool,
        help="whether to enable flash attention. Default is False",
    )
    parser.add_argument(
        "--use_model_dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for model. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    parser.add_argument(
        "--use_recompute",
        default=False,
        type=str2bool,
        help="whether use recompute.",
    )

    parser.add_argument("--ddim_sampling", type=str2bool, default=True, help="Whether to use DDIM for sampling")
    default_args = parser.parse_args()
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    return args


def inference_prompts(text_encoder, pipeline, items, batch_size, latent_size):
    for chunk in tqdm(list(get_chunks(items, batch_size)), unit="batch"):
        prompts = []
        if batch_size == 1:
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(
                chunk[0], base_ratios, show=False
            )  # ar for aspect ratio
            if args.image_size == 1024:
                latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
            else:
                hw = Tensor([[args.image_size, args.image_size]]).repeat(batch_size, 1)
                ar = Tensor([[1.0]]).repeat(batch_size, 1)
                latent_size_h, latent_size_w = latent_size, latent_size
            prompts.append(prompt_clean.strip())
        else:
            hw = Tensor([[args.image_size, args.image_size]]).repeat(batch_size, 1)
            ar = Tensor([[1.0]]).repeat(batch_size, 1)
            for prompt in chunk:
                prompts.append(prepare_prompt_ar(prompt, base_ratios, show=False)[0].strip())
            latent_size_h, latent_size_w = latent_size, latent_size
        # run inference for each batch
        n = len(prompts)

        # init inputs
        z = ops.randn((n, 4, latent_size_h, latent_size_w), dtype=ms.float32)
        tokens, mask = text_encoder.get_text_tokens_and_mask(prompts)

        inputs = {}
        inputs["noise"] = z
        inputs["tokens"] = tokens
        inputs["mask"] = mask
        inputs["scale"] = args.guidance_scale
        inputs["img_hw"] = hw
        inputs["aspect_ratio"] = ar
        # infer
        x_samples = pipeline(inputs)
        x_samples = x_samples.asnumpy()
        # save result
        for i, sample in enumerate(x_samples):
            save_fp = f"{save_dir}/{prompts[i][:100]}.jpg"
            img = Image.fromarray((sample * 255).astype(np.uint8))
            img.save(save_fp)
            logger.info(f"save to {save_fp}")


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{time_str}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init env
    args = parse_args()
    init_env(args)
    set_random_seed(args.seed)

    # 2. model initiate and weight loading
    # 2.1 model
    latent_size = args.image_size // 8
    lewei_scale = {512: 1, 1024: 2}  # trick for positional embedding interpolation

    # model setting
    if args.image_size == 512:
        model = PixArt_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size])
    else:
        model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size])
    # mixed precision
    if args.use_model_dtype == "fp16":
        model_dtype = ms.float16
        model = auto_mixed_precision(model, amp_level="O2", dtype=model_dtype)
    elif args.use_model_dtype == "bf16":
        model_dtype = ms.bfloat16
        model = auto_mixed_precision(model, amp_level="O2", dtype=model_dtype)
    else:
        model_dtype = ms.float32

    if len(args.checkpoint) > 0:
        param_dict = ms.load_checkpoint(args.checkpoint)
        logger.info(f"Loading ckpt {args.checkpoint} into model")
        # in case a save ckpt with "network." prefix, removing it before loading
        param_dict = remove_pname_prefix(param_dict, prefix="network.")
        del param_dict["pos_embed"]
        model.load_params_from_ckpt(param_dict)
    else:
        logger.warning("model uses random initialization!")

    model = model.set_train(False)
    for param in model.get_parameters():  # freeze model
        param.requires_grad = False

    if args.image_size == 512:
        base_ratios = ASPECT_RATIO_512_TEST
    elif args.image_size == 1024:
        base_ratios = ASPECT_RATIO_1024_TEST

    # 2.2 vae
    logger.info("vae init")
    vae = AutoencoderKL(
        SD_CONFIG,
        4,
        ckpt_path=args.vae_checkpoint,
        use_fp16=False,  # disable amp for vae
    )
    vae = vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False

    # t5
    logger.info("t5 init")
    text_encoder = T5Embedder(cache_dir=args.t5_checkpoint)

    # 3. build inference pipeline
    pipeline = InferPipeline(
        model,
        vae=vae,
        text_encoder=text_encoder,
        scale_factor=args.sd_scale_factor,
        num_inference_steps=args.sampling_steps,
        guidance_rescale=args.guidance_scale,
        ddim_sampling=args.ddim_sampling,
    )
    # prepare dataset
    with open(args.txt_file, "r") as f:
        items = [item.strip() for item in f.readlines()]
    # 4. print key info
    if vae:
        num_params_vae, num_params_vae_trainable = count_params(vae)
    else:
        num_params_vae, num_params_vae_trainable = 0, 0
    num_params_model, num_params_model_trainable = count_params(model)
    if text_encoder:
        num_params_text_encoder, num_params_text_encoder_trainable = count_params(
            text_encoder.model
        )  # only count embedding model
    else:
        num_params_text_encoder, num_params_text_encoder_trainable = 0, 0
    num_params = num_params_vae + num_params_model + num_params_text_encoder
    num_params_trainable = num_params_vae_trainable + num_params_model_trainable + num_params_text_encoder_trainable
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Num of samples: {len(items)}",
            f"Num params: {num_params:,} (model: {num_params_model:,}, vae: {num_params_vae:,}, text encoder : {num_params_text_encoder:,})",
            f"Num trainable params: {num_params_trainable:,}",
            f"Use model dtype: {model_dtype}",
            f"Sampling steps {args.sampling_steps}",
            f"DDIM sampling: {args.ddim_sampling}",
            f"CFG guidance scale: {args.guidance_scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    inference_prompts(text_encoder, pipeline, items, args.batch_size, latent_size)
