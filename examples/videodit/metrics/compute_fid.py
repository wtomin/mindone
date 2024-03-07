import sys

sys.path.append("../stable_diffusion_v2/tools/eval/")
import argparse
import glob
import os.path as osp

import albumentations
import cv2
import numpy as np
from albumentations import CenterCrop, SmallestMaxSize
from decord import VideoReader
from fid.fid import FrechetInceptionDistance
from PIL import Image, ImageSequence
from tqdm import tqdm

import mindspore as ms


def read_gif(gif_path, mode="RGB"):
    with Image.open(gif_path) as fp:
        frames = np.array([np.array(frame.convert(mode)) for frame in ImageSequence.Iterator(fp)])
    return frames


def get_pixel_transform(num_frames, h=299, w=299, interpolation="bicubic"):
    # NOTE: to ensure augment all frames in a video in the same way.
    targets = {"image{}".format(i): "image" for i in range(num_frames)}
    mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}

    pixel_transforms = albumentations.Compose(
        [
            SmallestMaxSize(max_size=h, interpolation=mapping[interpolation], always_apply=True),
            CenterCrop(h, w),
        ],
        additional_targets=targets,
    )
    return pixel_transforms


def read_all_files(real_dir, real_ext, gen_dir, gen_ext):
    real_files = sorted(
        glob.glob(osp.join(real_dir, "*" + real_ext)), key=lambda x: int(osp.basename(x).split(".")[0].split("_")[-1])
    )
    gen_files = sorted(
        glob.glob(osp.join(gen_dir, "*" + gen_ext)), key=lambda x: int(osp.basename(x).split(".")[0].split("-")[-1])
    )
    assert len(real_files) == len(gen_files)
    return real_files, gen_files


def video2tensors(video_path, sample_stride, sample_n_frames, pixel_transforms=None):
    if video_path.endswith(".gif"):
        video_reader = read_gif(video_path, mode="RGB")
    else:
        video_reader = VideoReader(video_path)
    video_length = len(video_reader)
    clip_length = min(video_length, (sample_n_frames - 1) * sample_stride + 1)
    # return all possible frame pixel values
    possible_frame_values = []
    for start_idx in range(video_length - clip_length + 1):
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, sample_n_frames, dtype=int)
        if video_path.endswith(".gif"):
            pixel_values = video_reader[batch_index]  # shape: (f, h, w, c)
        else:
            pixel_values = video_reader.get_batch(batch_index).asnumpy()  # shape: (f, h, w, c)

        if pixel_transforms is not None:
            inputs = {"image": pixel_values[0]}
            num_frames = len(pixel_values)
            for i in range(num_frames - 1):
                inputs[f"image{i}"] = pixel_values[i + 1]
            output = pixel_transforms(**inputs)
            pixel_values = np.stack(list(output.values()), axis=0)
        # (f h w c) -> (f c h w)
        pixel_values = np.transpose(pixel_values, (0, 3, 1, 2))
        possible_frame_values.append(pixel_values / 255.0)  # map to (0,1)
    del video_reader
    return possible_frame_values


def get_activations(pixel_values, model, batch_size=16, dims=2048):
    pred_arr = np.empty((len(pixel_values), dims))
    start_idx = 0
    for i_batch in range(len(pixel_values) // batch_size):
        batch = ms.Tensor(pixel_values[start_idx : start_idx + batch_size], ms.float32)
        pred = model(batch).asnumpy()
        pred_arr[start_idx : start_idx + batch_size] = pred
        start_idx = start_idx + pred.shape[0]

    if start_idx < len(pixel_values):
        batch = ms.Tensor(pixel_values[start_idx:])
        pred = model(batch).asnumpy()
        pred_arr[start_idx:] = pred
    return pred_arr


def compute_pairwise_fid_score(model, fid_scorer, real_pixel_values, gen_pixel_values, batch_size):
    gt_feats = get_activations(real_pixel_values, model, batch_size)
    gen_feats = get_activations(gen_pixel_values, model, batch_size)
    gen_mu, gen_sigma = fid_scorer.calculate_activation_stat(gen_feats)
    gt_mu, gt_sigma = fid_scorer.calculate_activation_stat(gt_feats)
    fid_value = fid_scorer.calculate_frechet_distance(gen_mu, gen_sigma, gt_mu, gt_sigma)

    return fid_value


def main(args):
    real_files, gen_files = read_all_files(args.real_dir, args.real_ext, args.gen_dir, args.gen_ext)
    # compute avg fid (mse)
    sample_stride = args.frames_stride
    sample_n_frames = args.num_frames
    fid_scorer = FrechetInceptionDistance(batch_size=sample_n_frames)
    pixel_transforms = get_pixel_transform(sample_n_frames, h=299, w=299)
    fid_scores_avg = []
    for real_file, gen_file in tqdm(zip(real_files, gen_files), total=len(real_files)):
        real_clips = video2tensors(real_file, sample_stride, sample_n_frames, pixel_transforms)
        gen_clips = video2tensors(gen_file, sample_stride, sample_n_frames, pixel_transforms)
        video_fid_scores = []
        if not args.exhaustive:
            real_clips = [real_clips[0]]
            gen_clips = [gen_clips[0]]
        for real_clip in real_clips:
            for gen_clip in gen_clips:
                fid_score = compute_pairwise_fid_score(
                    fid_scorer.model, fid_scorer, real_clip, gen_clip, batch_size=sample_n_frames
                )
                video_fid_scores.append(fid_score)
        video_fid_avg = np.mean(video_fid_scores)
        fid_scores_avg.append(video_fid_avg)
    print(f"The average FID score is {np.mean(fid_scores_avg)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real_dir",
        type=str,
        help="a folder path containing all real videos",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        help="a folder path containing all generated videos",
    )
    parser.add_argument(
        "--real_ext",
        type=str,
        default=".mp4",
        help="the extension of real videos",
    )
    parser.add_argument(
        "--gen_ext",
        type=str,
        default=".gif",
        help="the extension of generated videos",
    )
    parser.add_argument(
        "--frames_stride",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="whether to compute fid scores exhaustively for all possible clips from a single video",
    )
    default_args = parser.parse_args()
    return default_args


if __name__ == "__main__":
    args = parse_args()
    main(args)
