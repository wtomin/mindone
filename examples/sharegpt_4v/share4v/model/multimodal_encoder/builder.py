# Adapted from https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/share4v/model/multimodal_encoder/builder.py

import os

from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = vision_tower_cfg.get("mm_vision_tower", None)
    # vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    # Lin-Chen/ShareGPT4V-7B_Pretrained_vit-large336-l12
    is_absolute_path_exists = os.path.exists(vision_tower)
    if (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or vision_tower.startswith("Lin-Chen")
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
