import math
from typing import Callable

import numpy as np

import mindspore as ms
import mindspore.mint as mint
from mindspore import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    dtype: ms.common.dtype,
    seed: int,
):
    return mint.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=np.random.Generator().manual_seed(seed),
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    b, c, h_combined, w_combined = img.shape
    ph = 2
    pw = 2
    h = h_combined // ph
    w = w_combined // pw
    img = img.reshape(b, c, h, ph, w, pw).permute(0, 2, 4, 1, 3, 5).reshape(b, h * w, c * ph * pw)

    if img.shape[0] == 1 and bs > 1:
        img = img.broadcast_to((bs, *img.shape[1:]))

    img_ids = mint.zeros((h // 2, w // 2, 3))
    img_ids[..., 1] = img_ids[..., 1] + mint.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + mint.arange(w // 2)[None, :]

    img_ids = img_ids.unsqueeze(0).broadcast_to((bs, *img_ids.shape)).reshape(bs, -1, img_ids.shape[-1])

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)

    if txt.shape[0] == 1 and bs > 1:
        txt = txt.broadcast_to((bs, *txt.shape[1:]))
    txt_ids = mint.zeros((bs, txt.shape[1], 3))

    vec = clip(prompt)

    if vec.shape[0] == 1 and bs > 1:
        vec = vec.broadcast_to((bs, *vec.shape[1:]))

    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "vec": vec,
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = mint.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs=1,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor = None,
    neg_image_proj: Tensor = None,
    ip_scale: Tensor | float = 1.0,
    neg_ip_scale: Tensor | float = 1.0,
):
    i = 0
    # this is ignored for schnell
    guidance_vec = mint.full((img.shape[0],), guidance, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = mint.full(
            (img.shape[0],),
            t_curr,
            dtype=img.dtype,
        )
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg:
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale,
            )
            pred = neg_pred + true_gs * (pred - neg_pred)
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img


def denoise_controlnet(
    model: Flux,
    controlnet: None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs=1,
    controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor = None,
    neg_image_proj: Tensor = None,
    ip_scale: Tensor | float = 1,
    neg_ip_scale: Tensor | float = 1,
):
    # this is ignored for schnell
    i = 0
    guidance_vec = mint.full((img.shape[0],), guidance, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = mint.full(
            (img.shape[0],),
            t_curr,
            dtype=img.dtype,
        )
        block_res_samples = controlnet(
            img=img,
            img_ids=img_ids,
            controlnet_cond=controlnet_cond,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples],
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg:
            neg_block_res_samples = controlnet(
                img=img,
                img_ids=img_ids,
                controlnet_cond=controlnet_cond,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[i * controlnet_gs for i in neg_block_res_samples],
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale,
            )
            pred = neg_pred + true_gs * (pred - neg_pred)

        img = img + (t_prev - t_curr) * pred

        i += 1
    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    h = math.ceil(height / 16)
    w = math.ceil(width / 16)
    ph = 2
    pw = 2
    b, hw, cphpw = x.shape
    x = x.reshape(b, h, w, -1, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, -1, h * ph, w * pw)
    return x
