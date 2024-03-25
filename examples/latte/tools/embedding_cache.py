"""
Dataset Embedding Cache script
"""
import datetime
import logging
import os
import sys

import numpy as np
from embed_writers.mindrecord_writer import MindRecordEmbeddingCacheWriter
from embed_writers.numpy_writer import NumpyEmbeddingCacheWriter
from omegaconf import OmegaConf
from tqdm import tqdm

import mindspore as ms
from mindspore import ops

__dir__ = os.path.dirname(os.path.abspath(__file__))
latte_path = os.path.abspath(os.path.join(__dir__, "../"))
sys.path.insert(0, latte_path)

from args_train import parse_args
from data.dataset import get_dataset
from modules.autoencoder import SD_CONFIG, AutoencoderKL
from modules.text_encoders import get_text_encoder_and_tokenizer

mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.env import init_train_env

# load training modules
from mindone.utils.logger import set_logger

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)


def init_models(args):
    # 2.2 vae
    logger.info("vae init")
    vae = AutoencoderKL(
        SD_CONFIG,
        4,
        ckpt_path=args.vae_checkpoint,
        use_fp16=False,  # disable amp for vae . TODO: set by config file
    )
    vae = vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False

    if args.condition == "text":
        if args.text_encoder == "t5":
            ckpt_path = args.t5_cache_folder
            text_embed_shape = [1024]
        elif args.text_encoder == "clip":
            ckpt_path = args.clip_checkpoint
            text_embed_shape = [768]
        text_encoder, tokenizer = get_text_encoder_and_tokenizer(args.text_encoder, ckpt_path)
        text_embed_shape.insert(0, tokenizer.context_length)
    else:
        text_encoder, tokenizer, text_embed_shape = None, None, None
    return vae, text_encoder, tokenizer, text_embed_shape


def init_data(args, tokenizer, device_num, rank_id):
    data_config = OmegaConf.load(args.data_config_file).data_config
    # set some data params from argument parser
    data_config.sample_size = args.image_size
    data_config.sample_n_frames = args.num_frames
    data_config.batch_size = args.train_batch_size
    data_config.shuffle = False

    dataset, _ = get_dataset(
        args.dataset_name, data_config, tokenizer=tokenizer, device_num=device_num, rank_id=rank_id, return_dataset=True
    )
    return dataset


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args.output_path = os.path.join(args.output_path, time_str)

    # 1. init
    _, rank_id, device_num = init_train_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device_target,
        max_device_memory=args.max_device_memory,
        ascend_config=None if args.precision_mode is None else {"precision_mode": args.precision_mode},
    )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2. model initiate and weight loading
    vae, text_encoder, tokenizer, text_embed_shape = init_models(args)
    # select dataset
    dataset = init_data(args, tokenizer, device_num, rank_id)

    length = len(dataset)
    # parse train and save data types
    cache_file_type = args.cache_file_type
    if args.save_data_type == "float32":
        save_data_type = np.float32
    elif args.save_data_type == "float16":
        save_data_type = np.float16
    else:
        raise ValueError("Save data type {} is not supported!".format(args.save_data_type))

    # initiate cache folder and new dataset
    cache_folder = args.cache_folder
    assert len(cache_folder) > 0, "cache folder must be provided!"
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    # default save column names
    video_column = "video"  # video name/file path column
    latent_column = "video_latent"
    caption_column = "caption"
    text_emb_column = "text_emb"
    class_column = "class"
    mask_column = "token_mask"

    start_video_index = 0 if args.resume_cache_index is None else args.resume_cache_index
    logger.info(f"Start embedding cache from {start_video_index}th video...")

    if cache_file_type == "mindrecord":
        assert args.save_data_type == "float32"
        schema = {
            video_column: {"type": "string"},
            latent_column: {"type": args.save_data_type, "shape": [-1, 4, args.image_size // 8, args.image_size // 8]},
        }
        if args.condition == "class":
            schema[class_column] = {"type": "int"}
        elif args.condition == "text":
            schema[caption_column] = {"type": "string"}
            schema[text_emb_column] = {"type": args.save_data_type, "shape": text_embed_shape}

        dataset_name = args.data_config_file.split("/")[-1].split(".")[0]
        embed_cache_writer = MindRecordEmbeddingCacheWriter(
            cache_folder,
            dataset_name,
            schema,
            start_video_index=start_video_index,
            overwrite=False if args.resume_cache_index else True,
            max_page_size=args.max_page_size,
            dump_every_n_lines=min((length - start_video_index), args.dump_every_n_lines),
        )

    elif cache_file_type == "numpy":
        save_npz_keys = [video_column]
        if args.condition == "class":
            save_npz_keys += [class_column]
        if args.condition == "text":
            save_npz_keys += [caption_column, text_emb_column]
            if args.text_encoder == "t5":
                save_npz_keys += [mask_column]
        embed_cache_writer = NumpyEmbeddingCacheWriter(
            cache_folder,
            start_video_index,
            save_as_npy_keys=[latent_column],  # save video latent to npy file (frame-wise)
            save_as_npz_keys=save_npz_keys,
        )  # save others (class labels, captions, etc.) to npz file (video-wise)
    else:
        raise ValueError

    # print key info
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"cache_file_type: {args.cache_file_type}",
            f"save_data_type: {args.save_data_type}",
            f"cache_folder: {args.cache_folder}",
            f"Image size: {args.image_size}",
            f"Frames: {args.num_frames}",
            f"Enable flash attention: {args.enable_flash_attention}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    logger.info("Start dataset embedding cache...")
    data = []

    for video_index in tqdm(range(start_video_index, length), total=length - start_video_index):
        video_save_kwargs = {}
        video_latent = []
        all_frames_names = []
        for video_name, frames_paths, inputs in dataset.traverse_single_video_frames(video_index):
            clip_data = inputs["video"]
            clip_latent = ops.stop_gradient(vae.encode(ms.Tensor(clip_data, ms.float32))) * args.sd_scale_factor
            video_latent.append(clip_latent.asnumpy())
            all_frames_names.extend([os.path.basename(frame_path).split(".")[0] for frame_path in frames_paths])
        video_latent = np.concatenate(video_latent, axis=0)
        # 1. save video_latent (tensor) and video_name (string)
        video_save_kwargs[latent_column] = video_latent.copy().astype(save_data_type)
        video_save_kwargs[video_column] = video_name

        if args.condition == "text":
            # 2. save text_emb (tensor)
            assert len(inputs) > 1, "incorrect data return shape"
            token_ids = inputs["text"]
            mask = inputs.get("mask", None)
            if mask is not None:
                text_emb = ops.stop_gradient(
                    text_encoder(ms.Tensor(token_ids, ms.int32).unsqueeze(0), ms.Tensor(mask, ms.float32).unsqueeze(0))
                )[0].asnumpy()
                # 3. save mask if it needs
                video_save_kwargs[mask_column] = mask.copy().astype(save_data_type)
            else:
                text_emb = ops.stop_gradient(text_encoder(ms.Tensor(token_ids, ms.int32)).unsqueeze(0))[0].asnumpy()

            video_save_kwargs[text_emb_column] = text_emb.copy().astype(save_data_type)
            # 4. save caption (strings)
            caption = inputs["caption"]
            video_save_kwargs[caption_column] = caption
        elif args.condition == "class":
            # 2. save class_labels (tensor)
            assert len(inputs) > 1, "incorrect data return shape"
            class_labels = inputs["class"]
            video_save_kwargs[class_column] = class_labels.copy().astype(np.int32)

        if cache_file_type == "numpy":
            embed_cache_writer.save(video_name, all_frames_names, video_save_kwargs)
        elif cache_file_type == "mindrecord":
            data.append(video_save_kwargs)
            data = embed_cache_writer.save(data)
        else:
            raise ValueError("Train data type {} is not supported!".format(cache_file_type))

    if cache_file_type == "mindrecord":
        embed_cache_writer.save_data_and_close_writer(data)
    logger.info("Dataset embedding cache successfully saved in {}".format(cache_folder))
    embed_cache_writer.get_status()


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
