import argparse
import logging
import os
import sys

import numpy as np
import yaml

import mindspore as ms
from mindspore import nn

# TODO: remove in future when mindone is ready for install
mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(os.path.abspath("./"))
from opensora.models.diffusion.opensora.modeling_opensora import LayerNorm, OpenSoraT2V
from opensora.models.diffusion.opensora.modules import Attention
from opensora.utils.ms_utils import init_env
from opensora.utils.utils import _check_cfgs_in_parser, get_precision

from mindone.diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument("--model_path", type=str, default="LanguageBind/Open-Sora-Plan-v1.2.0")
    parser.add_argument(
        "--ms_checkpoint",
        type=str,
        default=None,
        help="If not provided, will search for ckpt file under `model_path`"
        "If provided, will use this pretrained ckpt path.",
    )
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--ae", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--ae_path", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")

    parser.add_argument("--text_encoder_name", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")

    parser.add_argument("--guidance_scale", type=float, default=7.5, help="the scale for classifier-free guidance")
    parser.add_argument("--max_sequence_length", type=int, default=300, help="the maximum text tokens length")

    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50, help="Diffusion Sampling Steps")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--text_prompt",
        type=str,
        nargs="+",
        help="A list of text prompts to be generated with. Also allow input a txt file or csv file.",
    )
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)

    parser.add_argument("--enable_tiling", action="store_true", help="whether to use vae tiling to save memory")
    parser.add_argument("--model_3d", action="store_true")
    parser.add_argument("--udit", action="store_true")
    parser.add_argument("--save_memory", action="store_true")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size for dataloader")
    # MS new args
    parser.add_argument("--device", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--parallel_mode", default="data", type=str, choices=["data", "optim"], help="parallel mode: data, optim"
    )
    parser.add_argument("--jit_level", default="O0", help="Set jit level: # O0: KBK, O1:DVM, O2: GE")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")

    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--global_bf16", action="store_true", help="whether to enable gloabal bf16 for diffusion model training."
    )
    parser.add_argument(
        "--vae_precision",
        default="bf16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for vae. Default is `bf16`, which corresponds to ms.bfloat16",
    )
    parser.add_argument(
        "--vae_keep_gn_fp32",
        default=False,
        type=str2bool,
        help="whether keep GroupNorm in fp32. Defaults to False in inference mode. If training vae, better set it to True",
    )
    parser.add_argument(
        "--text_encoder_precision",
        default="bf16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for T5 text encoder. Default is `bf16`, which corresponds to ms.bfloat16",
    )
    parser.add_argument(
        "--amp_level", type=str, default="O2", help="Set the amp level for the transformer model. Defaults to O2."
    )
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    parser.add_argument(
        "--num_videos_per_prompt", type=int, default=1, help="the number of images to be generated for each prompt"
    )
    parser.add_argument(
        "--save_latents",
        action="store_true",
        help="Whether to save latents (before vae decoding) instead of video files.",
    )
    parser.add_argument(
        "--decode_latents",
        action="store_true",
        help="whether to load the existing latents saved in npy files and run vae decoding",
    )
    parser.add_argument("--model_type", type=str, default="dit", choices=["dit", "udit", "latte"])
    parser.add_argument("--cache_dir", type=str, default="./")
    parser.add_argument("--profile", default=False, type=str2bool, help="Profile or not")
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
    # 1. init env
    args = parse_args()
    save_dir = args.save_img_path
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)
    # 1. init
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        precision_mode=args.precision_mode,
        global_bf16=args.global_bf16,
        sp_size=args.sp_size,
        jit_level=args.jit_level,
    )

    # 4. latte model initiate and weight loading
    FA_dtype = get_precision(args.precision) if get_precision(args.precision) != ms.float32 else ms.bfloat16
    assert args.model_type == "dit", "Currently only suppport model_type as 'dit'@"
    if args.ms_checkpoint and os.path.exists(args.ms_checkpoint):
        logger.info(f"Initiate from MindSpore checkpoint file {args.ms_checkpoint}")
        skip_load_ckpt = True
    else:
        skip_load_ckpt = False
    kwargs = {"FA_dtype": FA_dtype}
    model_version = args.model_path.split("/")[-1]
    if int(model_version.split("x")[0]) != args.num_frames:
        logger.warning(
            f"Detect that the loaded model version is {model_version}, but found a mismatched number of frames {model_version.split('x')[0]}"
        )
    if int(model_version.split("x")[1][:-1]) != args.height:
        logger.warning(
            f"Detect that the loaded model version is {model_version}, but found a mismatched resolution {args.height}x{args.width}"
        )
    transformer_model = OpenSoraT2V.from_pretrained(
        args.model_path,
        model_file=args.ms_checkpoint,
        cache_dir=args.cache_dir,
        skip_load_ckpt=skip_load_ckpt,
        **kwargs,
    )
    if skip_load_ckpt:
        transformer_model.load_from_checkpoint(args.ms_checkpoint)
    # mixed precision
    dtype = get_precision(args.precision)
    if args.precision in ["fp16", "bf16"]:
        if not args.global_bf16:
            amp_level = args.amp_level
            transformer_model = auto_mixed_precision(
                transformer_model,
                amp_level=args.amp_level,
                dtype=dtype,
                custom_fp32_cells=[LayerNorm, Attention, nn.SiLU, nn.GELU, PixArtAlphaCombinedTimestepSizeEmbeddings]
                if dtype == ms.float16
                else [
                    nn.MaxPool2d,
                    nn.MaxPool3d,
                    LayerNorm,
                    nn.SiLU,
                    nn.GELU,
                    PixArtAlphaCombinedTimestepSizeEmbeddings,
                ],
            )
            logger.info(f"Set mixed precision to O2 with dtype={args.precision}")
        else:
            logger.info(f"Using global bf16 for latte t2v model. Force model dtype from {dtype} to ms.bfloat16")
            dtype = ms.bfloat16
    elif args.precision == "fp32":
        amp_level = "O0"
    else:
        raise ValueError(f"Unsupported precision {args.precision}")

    transformer_model = transformer_model.set_train(False)
    for param in transformer_model.get_parameters():  # freeze transformer_model
        param.requires_grad = False

    # intiate
    prompt_embeds = ms.Tensor(np.load("mT5-xxl-torch-res/prompt_embeds.npy"))
    prompt_attention_mask = ms.Tensor(np.load("mT5-xxl-torch-res/prompt_attention_mask.npy"))
    hidden_states = ms.Tensor(np.load("mT5-xxl-torch-res/noise.npy"))
    if prompt_embeds.ndim == 3:
        prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
    if prompt_attention_mask.ndim == 2:
        prompt_attention_mask = prompt_attention_mask.unsqueeze(1)  # b l -> b 1 l
    timestep = ms.Tensor([990])
    noise = transformer_model(
        hidden_states,
        timestep,
        encoder_hidden_states=prompt_embeds,
        attention_mask=ms.ops.ones_like(hidden_states)[:, 0],
        encoder_attention_mask=prompt_attention_mask,
    )