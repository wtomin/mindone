import argparse
import datetime
import logging
import os
import sys
import time

import yaml
from modules.text_encoders import get_text_encoder_and_tokenizer
from pipelines import InferPipeline
from utils.model_utils import _check_cfgs_in_parser, count_params, remove_pname_prefix, str2bool

import mindspore as ms
from mindspore import Tensor, ops

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)


from modules.autoencoder import SD_CONFIG, AutoencoderKL

from mindone.models.latte import Latte_models
from mindone.models.latte_t2v_sd import Latte_T2V_models
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
        "--num_classes",
        type=int,
        default=1000,
        help="number of classes, applies only when condition is `class`",
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
        default="",
        help="latte checkpoint path. If specified, will load from it, otherwise, will use random initialization",
    )
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
    MODELS_DICT = Latte_models if args.condition != "text" else Latte_T2V_models

    t2v_model_extra_kwargs = {}
    text_emb_dim = None
    if args.condition == "text":
        if args.text_encoder == "clip":
            text_emb_dim = 768
        elif args.text_encoder == "t5":
            text_emb_dim = 4096
        t2v_model_extra_kwargs["context_dim"] = text_emb_dim

    latte_model = MODELS_DICT[args.model_name](
        input_size=latent_size,
        num_classes=args.num_classes,
        block_kwargs={"enable_flash_attention": args.enable_flash_attention},
        condition=args.condition,
        num_frames=args.num_frames,
        use_recompute=args.use_recompute,
        patch_embedder=args.patch_embedder,
        **t2v_model_extra_kwargs,
    )
    if args.dtype == "fp16":
        model_dtype = ms.float16
        latte_model = auto_mixed_precision(latte_model, amp_level="O2", dtype=model_dtype)
    elif args.dtype == "bf16":
        model_dtype = ms.bfloat16
        latte_model = auto_mixed_precision(latte_model, amp_level="O2", dtype=model_dtype)
    else:
        model_dtype = ms.float32

    if len(args.checkpoint) > 0:
        param_dict = ms.load_checkpoint(args.checkpoint)
        logger.info(f"Loading ckpt {args.checkpoint} into Latte")
        # in case a save ckpt with "network." prefix, removing it before loading
        param_dict = remove_pname_prefix(param_dict, prefix="network.")
        latte_model.load_params_from_ckpt(param_dict)
    else:
        logger.warning("Latte uses random initialization!")

    latte_model = latte_model.set_train(False)
    for param in latte_model.get_parameters():  # freeze latte_model
        param.requires_grad = False
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

    if args.condition == "class":
        # Labels to condition the model with (feel free to change):
        class_labels = [1, 13, 100]
        n = len(class_labels)
        y = Tensor(class_labels)
        y_null = ops.ones_like(y) * args.num_classes
    elif args.condition == "text":
        if args.text_encoder == "t5":
            ckpt_path = args.t5_cache_folder
        elif args.text_encoder == "clip":
            ckpt_path = args.clip_checkpoint
        text_encoder, tokenizer = get_text_encoder_and_tokenizer(args.text_encoder, ckpt_path)
        # extracting text tokends and attention mask?
        n = len(args.captions)
        assert n > 0, "No captions provided"
        text_tokens, mask = text_encoder.get_text_tokens_and_mask(args.captions, return_tensor=True)
        y, y_null = None, None
    else:
        y, y_null = None, None
        n = args.num_samples

    z = ops.randn((n, args.num_frames, 4, latent_size, latent_size), dtype=ms.float32)
    # 3. build inference pipeline
    pipeline = InferPipeline(
        latte_model,
        vae,
        text_encoder=text_encoder,
        scale_factor=args.sd_scale_factor,
        num_inference_steps=args.sampling_steps,
        guidance_rescale=args.guidance_scale,
        ddim_sampling=args.ddim_sampling,
        condition=args.condition,
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

    # init inputs
    inputs = {}
    inputs["noise"] = z
    inputs["y"] = y  # None if condition is None; otherwise, a tensor with shape (n, )
    inputs["y_null"] = y_null
    inputs["scale"] = args.guidance_scale
    if args.condition == "text":
        inputs["text_tokens"] = text_tokens
        inputs["mask"] = mask

    logger.info(f"Sampling for {n} samples with condition {args.condition}")
    start_time = time.time()

    # infer
    x_samples = pipeline(inputs)
    x_samples = x_samples.asnumpy()

    end_time = time.time()

    # save result
    for i in range(n):
        save_fp = f"{save_dir}/{i}.gif"
        save_videos(x_samples[i : i + 1], save_fp, loop=0)
        logger.info(f"save to {save_fp}")