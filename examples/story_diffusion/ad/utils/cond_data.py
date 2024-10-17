"""Prepare conditional images: transform, normalization, and saving"""
import os

import numpy as np
from ad.data.dataset import _build_transform
from PIL import Image


def load_rgb_images(image_paths):
    assert isinstance(image_paths, list) and len(image_paths) > 0, "image paths must be a non-empty list of strings"
    return [Image.open(path).convert("RGB") for path in image_paths]


def transform_conditional_images(image_paths, save_dir=None):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    image_paths = list(image_paths)
    conditions = load_rgb_images(image_paths)
    clip_transforms = _build_transform()
    assert len(image_paths) == 2, "expect to have two images as the start and the ending frame"
    conditions = np.stack([clip_transforms(image=img)["image"] for img in conditions])  # (2 h w c) -> (2, 224, 224, 3)
    conditions = np.transpose(conditions, (0, 3, 1, 2))  # (2, 224, 224, 3) -> (2, 3, 224, 224)

    if save_dir is not None:
        assert os.path.exists(save_dir), f"save_dir {save_dir} does not exist!"
        os.makedirs(os.path.join(save_dir, "conditional_images"), exist_ok=True)
        my_save_dir = os.path.join(save_dir, "conditional_images")
        existing_files = [f for f in os.listdir(my_save_dir) if os.path.isfile(os.path.join(my_save_dir, f))]
        for i, image in enumerate(conditions, len(existing_files)):
            Image.fromarray((255.0 * (image.transpose(1, 2, 0))).astype(np.uint8)).save(
                f"{save_dir}/conditional_images/{i}.png"
            )

    conditions = np.expand_dims(conditions, axis=0)
    return conditions
