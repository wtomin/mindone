import argparse
import datetime
import logging
import os
import sys
import time

import numpy as np
import yaml

# from modules.text_encoders import get_text_encoder_and_tokenizer
# from pipelines import InferPipeline
from utils.model_utils import _check_cfgs_in_parser, count_params, str2bool

import mindspore as ms
from mindspore import nn

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from modules.latte_t2v import Attention, LatteT2V, LayerNorm
from pipelines.pipeline_videogen import VideoGenPipeline
from transformers import T5Tokenizer

from mindone.diffusers.models.autoencoders import AutoencoderKL, AutoencoderKLTemporalDecoder
from mindone.diffusers.schedulers import DDIMScheduler, DDPMScheduler
from mindone.transformers.models import T5EncoderModel
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

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
        default=256,
        help="image size in [256, 512]",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="number of frames",
    )
    parser.add_argument(
        "--enable_vae_temporal_decoder", type=str2bool, default=True, help="whether to use AutoencoderKLTemporalDecoder"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="number of videos to be generated unconditionally. If using text or class as conditions,"
        " the number of samples will be defined by the number of class labels or text captions",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="Latte-XL/2",
        help="Model name ",
    )
    parser.add_argument(
        "--condition",
        default=None,
        type=str,
        help="the condition types: `None` means using no conditions; `text` means using text embedding as conditions;"
        " `class` means using class labels as conditions.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/t2v.pt",
        help="latte t2v checkpoint path. If specified, will load from it, otherwise, will use random initialization",
    )
    parser.add_argument("--pretrained_model_path", type=str, default="", help="the directory to t2v required models.")
    parser.add_argument(
        "--text_encoder",
        default=None,
        type=str,
        choices=["clip", "t5"],
        help="text encoder for extract text embeddings: clip text encoder or t5-v1_1-xxl.",
    )
    parser.add_argument("--t5_cache_folder", default=None, type=str, help="the T5 cache folder path")
    parser.add_argument(
        "--clip_checkpoint",
        type=str,
        default=None,
        help="CLIP text encoder checkpoint (or sd checkpoint to only load the text encoder part.)",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-mse.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )

    parser.add_argument("--sampling_steps", type=int, default=50, help="Diffusion Sampling Steps")
    parser.add_argument("--guidance_scale", type=float, default=8.5, help="the scale for classifier-free guidance")
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
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
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
    parser.add_argument(
        "--patch_embedder",
        type=str,
        default="conv",
        choices=["conv", "linear"],
        help="Whether to use conv2d layer or dense (linear layer) as Patch Embedder.",
    )
    parser.add_argument(
        "--captions",
        type=str,
        nargs="+",
        help="A list of text captions to be generated with",
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
    # 2.1 latte
    logger.info(f"{args.model_name}-{args.image_size}x{args.image_size} init")
    latent_size = args.image_size // 8
    latte_model = LatteT2V.from_pretrained_2d(
        args.pretrained_model_path, subfolder="transformer", video_length=args.num_frames
    )
    # mixed precision
    if args.dtype == "fp32":
        model_dtype = ms.float32
    else:
        model_dtype = {"fp16": ms.float16, "bf16": ms.bfloat16}[args.dtype]
        latte_model = auto_mixed_precision(
            latte_model,
            amp_level=args.amp_level,
            dtype=model_dtype,
            custom_fp32_cells=[LayerNorm, Attention, nn.SiLU, nn.GELU],
        )

    if len(args.checkpoint) > 0:
        logger.info(f"Loading ckpt {args.checkpoint} into Latte")
        latte_model.load_from_checkpoint(args.checkpoint)
    else:
        logger.warning("Latte uses random initialization!")

    latte_model = latte_model.set_train(False)
    for param in latte_model.get_parameters():  # freeze latte_model
        param.requires_grad = False

    # 2.2 vae
    if args.enable_vae_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_path, subfolder="vae_temporal_decoder")
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")

    n = len(args.captions)
    assert n > 0, "No captions provided"
    if args.decode_latents:
        for i in range(n):
            for i_video in range(args.num_videos_per_prompt):
                save_fp = f"{args.input_latents_dir}/{i_video}-{args.captions[i].strip()[:100]}.npy"
                assert os.path.exists(
                    save_fp
                ), f"{save_fp} does not exist! Please check the `input_latents_dir` or check if you run `--save_latents` ahead."
                loaded_latent = np.load(save_fp)
                decode_data = vae.decode(ms.Tensor(loaded_latent) / args.sd_scale_factor)
                decode_data = ms.ops.clip_by_value(
                    (decode_data + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0
                ).asnumpy()
                save_fp = f"{save_dir}/{i_video}-{args.captions[i].strip()[:100]}.gif"
                save_video_data = decode_data.transpose(0, 2, 3, 4, 1)  # (b c t h w) -> (b t h w c)
                save_videos(save_video_data, save_fp, loop=0)
                logger.info(f"video save to {save_fp}")
        sys.exit()

    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")

    # 3. build inference pipeline
    if args.ddim_sampling:
        scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    else:
        scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    text_encoder = text_encoder.model
    pipeline = VideoGenPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=latte_model,
        vae_scale_factor=args.sd_scale_factor,
    )
    # 4. print key info
    num_params_vae, num_params_vae_trainable = count_params(vae)
    num_params_latte, num_params_latte_trainable = count_params(latte_model)
    num_params = num_params_vae + num_params_latte
    num_params_trainable = num_params_vae_trainable + num_params_latte_trainable
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Num of samples: {n}",
            f"Num params: {num_params:,} (latte: {num_params_latte:,}, vae: {num_params_vae:,})",
            f"Num trainable params: {num_params_trainable:,}",
            f"Use model dtype: {model_dtype}",
            f"Sampling steps {args.sampling_steps}",
            f"DDIM sampling: {args.ddim_sampling}",
            f"CFG guidance scale: {args.guidance_scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    logger.info(f"Sampling for {n} samples with condition {args.condition}")
    start_time = time.time()

    # infer
    video_grids = []
    if not isinstance(args.captions, list):
        args.captions = [args.captions]
    if len(args.captions) == 1 and args.captions[0].endswith("txt"):
        captions = open(args.captions[0], "r").readlines()
        args.captions = [i.strip() for i in captions]
    for prompt in args.captions:
        print("Processing the ({}) prompt".format(prompt))
        videos = pipeline(
            prompt,
            video_length=args.num_frames,
            height=args.image_size,
            width=args.image_size,
            num_inference_steps=args.sampling_steps,
            guidance_scale=args.guidance_scale,
            enable_temporal_attentions=True,
            num_videos_per_prompt=args.num_videos_per_prompt,
            mask_feature=False,
            output_type="latents" if args.save_latents else "pil",
        ).video.asnumpy()
        video_grids.append(videos)
    x_samples = np.stack(video_grids, axis=0)

    end_time = time.time()

    # save result
    for i in range(n):
        for i_video in range(args.num_videos_per_prompt):
            if args.save_latents:
                save_fp = f"{save_dir}/{i_video}-{args.captions[i].strip()[:100]}.npy"
                save_latent_data = x_samples[i : i + 1, i_video]
                np.save(save_fp, save_latent_data)
                logger.info(f"latent save to {save_fp}")
            else:
                save_fp = f"{save_dir}/{i_video}-{args.captions[i].strip()[:100]}.gif"
                save_video_data = x_samples[i : i + 1, i_video].transpose(0, 2, 3, 4, 1)  # (b c t h w) -> (b t h w c)
                save_videos(save_video_data, save_fp, loop=0)
                logger.info(f"video save to {save_fp}")
