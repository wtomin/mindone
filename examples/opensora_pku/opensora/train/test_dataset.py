import logging
import os
import sys

import mindspore as ms

mindone_lib_path = os.path.abspath(os.path.abspath("../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.append("./")
from opensora.dataset import getdataset

# from mindcv.optim.adamw import AdamW
from opensora.dataset.t2v_datasets import create_dataloader
from opensora.models.ae import ae_stride_config
from opensora.models.text_encoder.t5 import T5Embedder
from opensora.train.commons import init_env, parse_args
from opensora.utils.dataset_utils import Collate

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"
os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"

logger = logging.getLogger(__name__)


def set_all_reduce_fusion(
    params,
    split_num: int = 7,
    distributed: bool = False,
    parallel_mode: str = "data",
) -> None:
    """Set allreduce fusion strategy by split_num."""

    if distributed and parallel_mode == "data":
        all_params_num = len(params)
        step = all_params_num // split_num
        split_list = [i * step for i in range(1, split_num)]
        split_list.append(all_params_num - 1)
        logger.info(f"Distribute config set: dall_params_num: {all_params_num}, set all_reduce_fusion: {split_list}")
        ms.set_auto_parallel_context(all_reduce_fusion_config=split_list)


def main(args):
    # 1. init
    save_src_strategy = args.use_parallel and args.parallel_mode != "data"
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        enable_dvm=args.enable_dvm,
        mempool_block_size=args.mempool_block_size,
        global_bf16=args.global_bf16,
        strategy_ckpt_save_file=os.path.join(args.output_dir, "src_strategy.ckpt") if save_src_strategy else "",
        optimizer_weight_shard_size=args.optimizer_weight_shard_size,
    )
    set_logger(output_dir=args.output_dir, rank=rank_id, log_level=eval(args.log_level))
    if args.use_deepspeed:
        raise NotImplementedError

    train_with_vae_latent = args.vae_latent_folder is not None and len(args.vae_latent_folder) > 0
    if train_with_vae_latent:
        assert os.path.exists(
            args.vae_latent_folder
        ), f"The provided vae latent folder {args.vae_latent_folder} is not existent!"
        logger.info("Train with vae latent cache.")

    else:
        logger.info("vae init")

        ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
        args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = ae_stride_t, ae_stride_h, ae_stride_w
        args.ae_stride = args.ae_stride_h
        patch_size = args.model[-3:]
        patch_size_t, patch_size_h, patch_size_w = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
        args.patch_size = patch_size_h
        args.patch_size_t, args.patch_size_h, args.patch_size_w = patch_size_t, patch_size_h, patch_size_w
        assert (
            ae_stride_h == ae_stride_w
        ), f"Support only ae_stride_h == ae_stride_w now, but found ae_stride_h ({ae_stride_h}), ae_stride_w ({ae_stride_w})"
        assert (
            patch_size_h == patch_size_w
        ), f"Support only patch_size_h == patch_size_w now, but found patch_size_h ({patch_size_h}), patch_size_w ({patch_size_w})"
        assert (
            args.max_image_size % ae_stride_h == 0
        ), f"Image size must be divisible by ae_stride_h, but found max_image_size ({args.max_image_size}), "
        " ae_stride_h ({ae_stride_h})."

        args.stride_t = ae_stride_t * patch_size_t
        args.stride = ae_stride_h * patch_size_h

    use_text_embed = args.text_embed_folder is not None and len(args.text_embed_folder) > 0
    if not use_text_embed:
        logger.info("T5 init")
        text_encoder = T5Embedder(
            dir_or_name=args.text_encoder_name,
            cache_dir="./",
            model_max_length=args.model_max_length,
        )
        # mixed precision
        text_encoder_dtype = ms.bfloat16  # using bf16 for text encoder and vae
        text_encoder = auto_mixed_precision(text_encoder, amp_level="O2", dtype=text_encoder_dtype)
        text_encoder.dtype = text_encoder_dtype
        logger.info(f"Use amp level O2 for text encoder T5 with dtype={text_encoder_dtype}")

        tokenizer = text_encoder.tokenizer
    else:
        assert os.path.exists(
            args.text_embed_folder
        ), f"The provided text_embed_folder {args.text_embed_folder} is not existent!"
        text_encoder = None
        tokenizer = None

    # 3. create dataset
    assert args.dataset == "t2v", "Support t2v dataset only."
    dataset = getdataset(args, tokenizer)
    batch = dataset[0]
    print(batch.keys())
    dataset = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        device_num=device_num,
        rank_id=rank_id,
        num_parallel_workers=args.num_parallel_workers,
        max_rowsize=args.max_rowsize,
        collate_fn=Collate(args),
        output_columns=["video", "text", "cond_mask", "attention_mask"],
    )
    dataset_size = dataset.get_dataset_size()
    assert dataset_size > 0, "Incorrect dataset size. Please check your dataset size and your global batch size"

    for batch in dataset:
        print(len(batch))


def parse_t2v_train_args(parser):
    parser.add_argument("--output_dir", default="outputs/", help="The directory where training results are saved.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--image_data", type=str, required=True)
    parser.add_argument("--video_data", type=str, required=True)
    parser.add_argument(
        "--text_embed_folder", type=str, default=None, help="the folder path to the t5 text embeddings and masks"
    )
    parser.add_argument("--vae_latent_folder", default=None, type=str, help="root dir for the vae latent data")
    parser.add_argument("--model", type=str, default="DiT-XL/122")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--ae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--ae_path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--sample_rate", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--max_image_size", type=int, default=512)
    parser.add_argument("--compress_kv", action="store_true")
    parser.add_argument("--compress_kv_factor", type=int, default=1)
    parser.add_argument("--use_rope", action="store_true")
    parser.add_argument("--attention_mode", type=str, choices=["xformers", "math", "flash"], default="math")
    parser.add_argument("--pretrained", type=str, default=None)

    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--enable_tiling", action="store_true")

    parser.add_argument("--video_folder", type=str, default="")
    parser.add_argument("--text_encoder_name", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--model_max_length", type=int, default=300)
    parser.add_argument("--multi_scale", action="store_true")

    # parser.add_argument("--enable_tracker", action="store_true")
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )
    parser.add_argument("--video_column", default="path", type=str, help="name of column for videos saved in csv file")
    parser.add_argument(
        "--caption_column", default="cap", type=str, help="name of column for captions saved in csv file"
    )
    parser.add_argument(
        "--enable_flip",
        action="store_true",
        help="enable random flip video (disable it to avoid motion direction and text mismatch)",
    )
    return parser


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args(additional_parse_args=parse_t2v_train_args)
    main(args)
