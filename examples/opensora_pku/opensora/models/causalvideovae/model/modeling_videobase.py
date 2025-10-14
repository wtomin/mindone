# Adapted from
# https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/model/modeling_videobase.py

import glob
import logging
import os
from typing import Optional, Union

from huggingface_hub.utils import validate_hf_hub_args

import mindspore as ms

from mindone.diffusers import ModelMixin
from mindone.diffusers.configuration_utils import ConfigMixin

logger = logging.getLogger(__name__)


class VideoBaseAE(ModelMixin, ConfigMixin):
    config_name = "config.json"
    _supports_gradient_checkpointing = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def download_and_load_model(cls, model_name, cache_dir=None):
        pass

    def encode(self, x: ms.Tensor, *args, **kwargs):
        pass

    def decode(self, encoding: ms.Tensor, *args, **kwargs):
        pass

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        ckpt_files = glob.glob(os.path.join(pretrained_model_name_or_path, "*.ckpt"))
        if ckpt_files:
            # Adapt to checkpoint
            last_ckpt_file = ckpt_files[-1]
            config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            model = cls.from_config(config_file)
            model.init_from_ckpt(last_ckpt_file)
            return model
        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def init_from_ckpt(self, path, ignore_keys=list()):
        # TODO: support auto download pretrained checkpoints
        sd = ms.load_checkpoint(path)
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    logger.info("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        if "ema_state_dict" in sd and len(sd["ema_state_dict"]) > 0 and os.environ.get("NOT_USE_EMA_MODEL", 0) == 0:
            logger.info("Load from ema model!")
            sd = sd["ema_state_dict"]
            sd = {key.replace("module.", ""): value for key, value in sd.items()}
        elif "state_dict" in sd:
            logger.info("Load from normal model!")
            if "gen_model" in sd["state_dict"]:
                sd = sd["state_dict"]["gen_model"]
            else:
                sd = sd["state_dict"]

        sd = dict([k.replace("autoencoder.", "") if k.startswith("autoencoder.") else k, v] for k, v in sd.items())
        sd = dict([k.replace("_backbone.", "") if "_backbone." in k else k, v] for k, v in sd.items())

        ms.load_param_into_net(self, sd, strict_load=False)
        logger.info(f"Restored from {path}")
