# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
from transformers import T5EncoderModel, T5TokenizerFast

import mindspore as ms
from mindspore import mint

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...loaders import FromSingleFileMixin, LTXVideoLoraLoaderMixin
from ...models.autoencoders import AutoencoderKLLTXVideo
from ...models.transformers import LTXVideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.mindspore_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import LTXPipelineOutput

XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import mindspore as ms
        >>> from mindone.diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
        >>> from mindone.diffusers.utils import export_to_video, load_video, load_image
        >>> import numpy as np

        >>> pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", mindspore_dtype=ms.bfloat16)

        >>> # Load input image and video
        >>> video = load_video(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
        ... )
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input.jpg"
        ... )

        >>> # Create conditioning objects
        >>> condition1 = LTXVideoCondition(
        ...     image=image,
        ...     frame_index=0,
        ... )
        >>> condition2 = LTXVideoCondition(
        ...     video=video,
        ...     frame_index=80,
        ... )

        >>> prompt = "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day. And then the camera switch to a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region." # noqa
        >>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

        >>> # Generate video
        >>> generator = np.random.Generator(np.random.PCG64(0))
        >>> # Text-only conditioning is also supported without the need to pass `conditions`
        >>> video = pipe(
        ...     conditions=[condition1, condition2],
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     width=768,
        ...     height=512,
        ...     num_frames=161,
        ...     num_inference_steps=40,
        ...     generator=generator,
        ... )[0][0]

        >>> export_to_video(video, "output.mp4", fps=24)
        ```
"""


@dataclass
class LTXVideoCondition:
    """
    Defines a single frame-conditioning item for LTX Video - a single frame or a sequence of frames.

    Attributes:
        image (`PIL.Image.Image`):
            The image to condition the video on.
        video (`List[PIL.Image.Image]`):
            The video to condition the video on.
        frame_index (`int`):
            The frame index at which the image or video will conditionally effect the video generation.
        strength (`float`, defaults to `1.0`):
            The strength of the conditioning effect. A value of `1.0` means the conditioning effect is fully applied.
    """

    image: Optional[PIL.Image.Image] = None
    video: Optional[List[PIL.Image.Image]] = None
    frame_index: int = 0
    strength: float = 1.0


# from LTX-Video/ltx_video/schedulers/rf.py
def linear_quadratic_schedule(num_steps, threshold_noise=0.025, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    if num_steps < 2:
        return ms.tensor([1.0])
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return ms.tensor(sigma_schedule[:-1])


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[ms.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    vae, encoder_output: ms.Tensor, generator: Optional[np.random.Generator] = None, sample_mode: str = "sample"
):
    if sample_mode == "sample":
        return vae.diag_gauss_dist.sample(encoder_output, generator=generator)
    elif sample_mode == "argmax":
        return vae.diag_gauss_dist.mode(encoder_output)
    # This branch is not needed because the encoder_output type is ms.Tensor as per AutoencoderKLOutput change
    # elif hasattr(encoder_output, "latents"):
    #     return encoder_output.latents
    else:
        return encoder_output


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`ms.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`ms.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`ms.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class LTXConditionPipeline(DiffusionPipeline, FromSingleFileMixin, LTXVideoLoraLoaderMixin):
    r"""
    Pipeline for text/image/video-to-video generation.

    Reference: https://github.com/Lightricks/LTX-Video

    Args:
        transformer ([`LTXVideoTransformer3DModel`]):
            Conditional Transformer architecture to denoise the encoded video latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLLTXVideo`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLLTXVideo,
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
        transformer: LTXVideoTransformer3DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 1
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if getattr(self, "transformer") is not None else 1
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if getattr(self, "tokenizer", None) is not None else 128
        )

        self.default_height = 512
        self.default_width = 704
        self.default_frames = 121

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        dtype: Optional[ms.Type] = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = ms.tensor(text_inputs.attention_mask)
        prompt_attention_mask = prompt_attention_mask.bool()

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="np").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not np.array_equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        text_input_ids = ms.tensor(text_input_ids)
        prompt_embeds = self.text_encoder(text_input_ids, attention_mask=prompt_attention_mask)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.tile((1, num_videos_per_prompt, 1))
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.tile((num_videos_per_prompt, 1))

        return prompt_embeds, prompt_attention_mask

    # Copied from diffusers.pipelines.mochi.pipeline_mochi.MochiPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        prompt_attention_mask: Optional[ms.Tensor] = None,
        negative_prompt_attention_mask: Optional[ms.Tensor] = None,
        max_sequence_length: int = 256,
        dtype: Optional[ms.Type] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt.
            prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            dtype: (`ms.Type`, *optional*):
                mindspore dtype
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def check_inputs(
        self,
        prompt,
        conditions,
        image,
        video,
        frame_index,
        strength,
        denoise_strength,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"  # noqa
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

        if conditions is not None and (image is not None or video is not None):
            raise ValueError("If `conditions` is provided, `image` and `video` must not be provided.")

        if conditions is None:
            if isinstance(image, list) and isinstance(frame_index, list) and len(image) != len(frame_index):
                raise ValueError(
                    "If `conditions` is not provided, `image` and `frame_index` must be of the same length."
                )
            elif isinstance(image, list) and isinstance(strength, list) and len(image) != len(strength):
                raise ValueError("If `conditions` is not provided, `image` and `strength` must be of the same length.")
            elif isinstance(video, list) and isinstance(frame_index, list) and len(video) != len(frame_index):
                raise ValueError(
                    "If `conditions` is not provided, `video` and `frame_index` must be of the same length."
                )
            elif isinstance(video, list) and isinstance(strength, list) and len(video) != len(strength):
                raise ValueError("If `conditions` is not provided, `video` and `strength` must be of the same length.")

        if denoise_strength < 0 or denoise_strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {denoise_strength}")

    @staticmethod
    def _prepare_video_ids(
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> ms.Tensor:
        latent_sample_coords = mint.meshgrid(
            mint.arange(0, num_frames, patch_size_t),
            mint.arange(0, height, patch_size),
            mint.arange(0, width, patch_size),
            indexing="ij",
        )
        latent_sample_coords = mint.stack(latent_sample_coords, dim=0)
        latent_coords = latent_sample_coords.unsqueeze(0).tile((batch_size, 1, 1, 1, 1))
        latent_coords = latent_coords.reshape(batch_size, -1, num_frames * height * width)

        return latent_coords

    @staticmethod
    def _scale_video_ids(
        video_ids: ms.Tensor,
        scale_factor: int = 32,
        scale_factor_t: int = 8,
        frame_index: int = 0,
    ) -> ms.Tensor:
        scaled_latent_coords = video_ids * ms.tensor([scale_factor_t, scale_factor, scale_factor])[None, :, None]
        scaled_latent_coords[:, 0] = (scaled_latent_coords[:, 0] + 1 - scale_factor_t).clamp(min=0)
        scaled_latent_coords[:, 0] += frame_index

        return scaled_latent_coords

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._pack_latents
    def _pack_latents(latents: ms.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> ms.Tensor:
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._unpack_latents
    def _unpack_latents(
        latents: ms.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> ms.Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.shape[0]
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._normalize_latents
    def _normalize_latents(
        latents: ms.Tensor, latents_mean: ms.Tensor, latents_std: ms.Tensor, scaling_factor: float = 1.0
    ) -> ms.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._denormalize_latents
    def _denormalize_latents(
        latents: ms.Tensor, latents_mean: ms.Tensor, latents_std: ms.Tensor, scaling_factor: float = 1.0
    ) -> ms.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    def trim_conditioning_sequence(self, start_frame: int, sequence_num_frames: int, target_num_frames: int):
        """
        Trim a conditioning sequence to the allowed number of frames.

        Args:
            start_frame (int): The target frame number of the first frame in the sequence.
            sequence_num_frames (int): The number of frames in the sequence.
            target_num_frames (int): The target number of frames in the generated video.
        Returns:
            int: updated sequence length
        """
        scale_factor = self.vae_temporal_compression_ratio
        num_frames = min(sequence_num_frames, target_num_frames - start_frame)
        # Trim down to a multiple of temporal_scale_factor frames plus 1
        num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
        return num_frames

    @staticmethod
    def add_noise_to_image_conditioning_latents(
        t: float,
        init_latents: ms.Tensor,
        latents: ms.Tensor,
        noise_scale: float,
        conditioning_mask: ms.Tensor,
        generator,
        eps=1e-6,
    ):
        """
        Add timestep-dependent noise to the hard-conditioning latents. This helps with motion continuity, especially
        when conditioned on a single frame.
        """
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            dtype=latents.dtype,
        )
        # Add noise only to hard-conditioning latents (conditioning_mask = 1.0)
        need_to_noise = (conditioning_mask > 1.0 - eps).unsqueeze(-1)
        noised_latents = init_latents + noise_scale * noise * (t**2)
        latents = mint.where(need_to_noise, noised_latents, latents)
        return latents

    def prepare_latents(
        self,
        conditions: Optional[List[ms.Tensor]] = None,
        condition_strength: Optional[List[float]] = None,
        condition_frame_index: Optional[List[int]] = None,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        num_prefix_latent_frames: int = 2,
        sigma: Optional[ms.Tensor] = None,
        latents: Optional[ms.Tensor] = None,
        generator: Optional[np.random.Generator] = None,
        dtype: Optional[ms.Type] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, int]:
        num_latent_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        noise = randn_tensor(shape, generator=generator, dtype=dtype)
        if latents is not None and sigma is not None:
            if latents.shape != shape:
                raise ValueError(
                    f"Latents shape {latents.shape} does not match expected shape {shape}. Please check the input."
                )
            latents = latents.to(dtype=dtype)
            sigma = sigma.to(dtype=dtype)
            latents = sigma * noise + (1 - sigma) * latents
        else:
            latents = noise

        if len(conditions) > 0:
            condition_latent_frames_mask = mint.zeros((batch_size, num_latent_frames), dtype=ms.float32)

            extra_conditioning_latents = []
            extra_conditioning_video_ids = []
            extra_conditioning_mask = []
            extra_conditioning_num_latents = 0
            for data, strength, frame_index in zip(conditions, condition_strength, condition_frame_index):
                condition_latents = retrieve_latents(self.vae, self.vae.encode(data)[0], generator=generator)
                condition_latents = self._normalize_latents(
                    condition_latents, self.vae.latents_mean, self.vae.latents_std
                ).to(dtype=dtype)

                num_data_frames = data.shape[2]
                num_cond_frames = condition_latents.shape[2]

                if frame_index == 0:
                    latents[:, :, :num_cond_frames] = mint.lerp(
                        latents[:, :, :num_cond_frames], condition_latents, strength
                    )
                    condition_latent_frames_mask[:, :num_cond_frames] = strength

                else:
                    if num_data_frames > 1:
                        if num_cond_frames < num_prefix_latent_frames:
                            raise ValueError(
                                f"Number of latent frames must be at least {num_prefix_latent_frames} but got {num_data_frames}."
                            )

                        if num_cond_frames > num_prefix_latent_frames:
                            start_frame = frame_index // self.vae_temporal_compression_ratio + num_prefix_latent_frames
                            end_frame = start_frame + num_cond_frames - num_prefix_latent_frames
                            latents[:, :, start_frame:end_frame] = mint.lerp(
                                latents[:, :, start_frame:end_frame],
                                condition_latents[:, :, num_prefix_latent_frames:],
                                strength,
                            )
                            condition_latent_frames_mask[:, start_frame:end_frame] = strength
                            condition_latents = condition_latents[:, :, :num_prefix_latent_frames]

                    noise = randn_tensor(condition_latents.shape, generator=generator, dtype=dtype)
                    condition_latents = mint.lerp(noise, condition_latents, strength)

                    condition_video_ids = self._prepare_video_ids(
                        batch_size,
                        condition_latents.shape[2],
                        latent_height,
                        latent_width,
                        patch_size=self.transformer_spatial_patch_size,
                        patch_size_t=self.transformer_temporal_patch_size,
                    )
                    condition_video_ids = self._scale_video_ids(
                        condition_video_ids,
                        scale_factor=self.vae_spatial_compression_ratio,
                        scale_factor_t=self.vae_temporal_compression_ratio,
                        frame_index=frame_index,
                    )
                    condition_latents = self._pack_latents(
                        condition_latents,
                        self.transformer_spatial_patch_size,
                        self.transformer_temporal_patch_size,
                    )
                    condition_conditioning_mask = mint.full(condition_latents.shape[:2], strength, dtype=dtype)

                    extra_conditioning_latents.append(condition_latents)
                    extra_conditioning_video_ids.append(condition_video_ids)
                    extra_conditioning_mask.append(condition_conditioning_mask)
                    extra_conditioning_num_latents += condition_latents.shape[1]

        video_ids = self._prepare_video_ids(
            batch_size,
            num_latent_frames,
            latent_height,
            latent_width,
            patch_size_t=self.transformer_temporal_patch_size,
            patch_size=self.transformer_spatial_patch_size,
        )
        if len(conditions) > 0:
            conditioning_mask = condition_latent_frames_mask.gather(1, video_ids[:, 0])
        else:
            conditioning_mask, extra_conditioning_num_latents = None, 0
        video_ids = self._scale_video_ids(
            video_ids,
            scale_factor=self.vae_spatial_compression_ratio,
            scale_factor_t=self.vae_temporal_compression_ratio,
            frame_index=0,
        )
        latents = self._pack_latents(latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)

        if len(conditions) > 0 and len(extra_conditioning_latents) > 0:
            latents = mint.cat([*extra_conditioning_latents, latents], dim=1)
            video_ids = mint.cat([*extra_conditioning_video_ids, video_ids], dim=2)
            conditioning_mask = mint.cat([*extra_conditioning_mask, conditioning_mask], dim=1)

        return latents, conditioning_mask, video_ids, extra_conditioning_num_latents

    def get_timesteps(self, sigmas, timesteps, num_inference_steps, strength):
        num_steps = min(int(num_inference_steps * strength), num_inference_steps)
        start_index = max(num_inference_steps - num_steps, 0)
        sigmas = sigmas[start_index:]
        timesteps = timesteps[start_index:]
        return sigmas, timesteps, num_inference_steps - start_index

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    def __call__(
        self,
        conditions: Union[LTXVideoCondition, List[LTXVideoCondition]] = None,
        image: Union[PipelineImageInput, List[PipelineImageInput]] = None,
        video: List[PipelineImageInput] = None,
        frame_index: Union[int, List[int]] = 0,
        strength: Union[float, List[float]] = 1.0,
        denoise_strength: float = 1.0,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        frame_rate: int = 25,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 3,
        guidance_rescale: float = 0.0,
        image_cond_noise_scale: float = 0.15,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        prompt_embeds: Optional[ms.Tensor] = None,
        prompt_attention_mask: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_attention_mask: Optional[ms.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            conditions (`List[LTXVideoCondition], *optional*`):
                The list of frame-conditioning items for the video generation.If not provided, conditions will be
                created using `image`, `video`, `frame_index` and `strength`.
            image (`PipelineImageInput` or `List[PipelineImageInput]`, *optional*):
                The image or images to condition the video generation. If not provided, one has to pass `video` or
                `conditions`.
            video (`List[PipelineImageInput]`, *optional*):
                The video to condition the video generation. If not provided, one has to pass `image` or `conditions`.
            frame_index (`int` or `List[int]`, *optional*):
                The frame index or frame indices at which the image or video will conditionally effect the video
                generation. If not provided, one has to pass `conditions`.
            strength (`float` or `List[float]`, *optional*):
                The strength or strengths of the conditioning effect. If not provided, one has to pass `conditions`.
            denoise_strength (`float`, defaults to `1.0`):
                The strength of the noise added to the latents for editing. Higher strength leads to more noise added
                to the latents, therefore leading to more differences between original video and generated video. This
                is useful for video-to-video editing.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `512`):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, defaults to `704`):
                The width in pixels of the generated image. This is set to 848 by default for the best results.
            num_frames (`int`, defaults to `161`):
                The number of video frames to generate
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, defaults to `3 `):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                One or a list of [np.random.Generator(s)](https://numpy.org/doc/stable/reference/random/generator.html)
                to make generation deterministic.
            latents (`ms.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`ms.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`ms.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            decode_timestep (`float`, defaults to `0.0`):
                The timestep at which generated video is decoded.
            decode_noise_scale (`float`, defaults to `None`):
                The interpolation factor between random noise and denoised latents at the decode timestep.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~pipelines.ltx.LTXPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to `128 `):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.ltx.LTXPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ltx.LTXPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            conditions=conditions,
            image=image,
            video=video,
            frame_index=frame_index,
            strength=strength,
            denoise_strength=denoise_strength,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if conditions is not None:
            if not isinstance(conditions, list):
                conditions = [conditions]

            strength = [condition.strength for condition in conditions]
            frame_index = [condition.frame_index for condition in conditions]
            image = [condition.image for condition in conditions]
            video = [condition.video for condition in conditions]
        elif image is not None or video is not None:
            if not isinstance(image, list):
                image = [image]
                num_conditions = 1
            elif isinstance(image, list):
                num_conditions = len(image)
            if not isinstance(video, list):
                video = [video]
                num_conditions = 1
            elif isinstance(video, list):
                num_conditions = len(video)

            if not isinstance(frame_index, list):
                frame_index = [frame_index] * num_conditions
            if not isinstance(strength, list):
                strength = [strength] * num_conditions

        vae_dtype = self.vae.dtype

        # 3. Prepare text embeddings & conditioning image/video
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = mint.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = mint.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        conditioning_tensors = []
        is_conditioning_image_or_video = image is not None or video is not None
        if is_conditioning_image_or_video:
            for condition_image, condition_video, condition_frame_index, condition_strength in zip(
                image, video, frame_index, strength
            ):
                if condition_image is not None:
                    condition_tensor = (
                        self.video_processor.preprocess(condition_image, height, width).unsqueeze(2).to(dtype=vae_dtype)
                    )
                elif condition_video is not None:
                    condition_tensor = self.video_processor.preprocess_video(condition_video, height, width)
                    num_frames_input = condition_tensor.shape[2]
                    num_frames_output = self.trim_conditioning_sequence(
                        condition_frame_index, num_frames_input, num_frames
                    )
                    condition_tensor = condition_tensor[:, :, :num_frames_output]
                    condition_tensor = condition_tensor.to(dtype=vae_dtype)
                else:
                    raise ValueError("Either `image` or `video` must be provided for conditioning.")

                if condition_tensor.shape[2] % self.vae_temporal_compression_ratio != 1:
                    raise ValueError(
                        f"Number of frames in the video must be of the form (k * {self.vae_temporal_compression_ratio} + 1) "
                        f"but got {condition_tensor.shape[2]} frames."
                    )
                conditioning_tensors.append(condition_tensor)

        # 4. Prepare timesteps
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        if timesteps is None:
            sigmas = linear_quadratic_schedule(num_inference_steps)
            timesteps = sigmas * 1000
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps)
        sigmas = self.scheduler.sigmas
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        latent_sigma = None
        if denoise_strength < 1:
            sigmas, timesteps, num_inference_steps = self.get_timesteps(
                sigmas, timesteps, num_inference_steps, denoise_strength
            )
            latent_sigma = sigmas[:1].tile((batch_size * num_videos_per_prompt,))

        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents, conditioning_mask, video_coords, extra_conditioning_num_latents = self.prepare_latents(
            conditioning_tensors,
            strength,
            frame_index,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            sigma=latent_sigma,
            latents=latents,
            generator=generator,
            dtype=ms.float32,
        )

        video_coords = video_coords.float()
        video_coords[:, 0] = video_coords[:, 0] * (1.0 / frame_rate)

        init_latents = latents.clone() if is_conditioning_image_or_video else None

        if self.do_classifier_free_guidance:
            video_coords = mint.cat([video_coords, video_coords], dim=0)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if image_cond_noise_scale > 0 and init_latents is not None:
                    # Add timestep-dependent noise to the hard-conditioning latents
                    # This helps with motion continuity, especially when conditioned on a single frame
                    latents = self.add_noise_to_image_conditioning_latents(
                        t / 1000.0,
                        init_latents,
                        latents,
                        image_cond_noise_scale,
                        conditioning_mask,
                        generator,
                    )

                latent_model_input = mint.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                if is_conditioning_image_or_video:
                    conditioning_mask_model_input = (
                        mint.cat([conditioning_mask, conditioning_mask])
                        if self.do_classifier_free_guidance
                        else conditioning_mask
                    )
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand((latent_model_input.shape[0],)).unsqueeze(-1).float()
                if is_conditioning_image_or_video:
                    timestep = mint.min(timestep, (1 - conditioning_mask_model_input) * 1000.0)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    video_coords=video_coords,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    timestep, _ = timestep.chunk(2)

                    if self.guidance_rescale > 0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale
                        )

                denoised_latents = self.scheduler.step(
                    -noise_pred, t, latents, per_token_timesteps=timestep, return_dict=False
                )[0]
                if is_conditioning_image_or_video:
                    tokens_to_denoise_mask = (t / 1000 - 1e-6 < (1.0 - conditioning_mask)).unsqueeze(-1)
                    latents = mint.where(tokens_to_denoise_mask, denoised_latents, latents)
                else:
                    latents = denoised_latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if is_conditioning_image_or_video:
            latents = latents[:, extra_conditioning_num_latents:]

        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

        if output_type == "latent":
            video = latents
        else:
            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            latents = latents.to(prompt_embeds.dtype)

            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = randn_tensor(latents.shape, generator=generator, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = ms.tensor(decode_timestep, dtype=latents.dtype)
                decode_noise_scale = ms.tensor(decode_noise_scale, dtype=latents.dtype)[:, None, None, None, None]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            video = self.vae.decode(latents, timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        if not return_dict:
            return (video,)

        return LTXPipelineOutput(frames=video)
