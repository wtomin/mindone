# Copyright 2025 The EasyAnimate team and The HuggingFace Team.
# All rights reserved.
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
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from transformers import BertTokenizer, Qwen2Tokenizer

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import mint

from mindone.transformers import BertModel, Qwen2VLForConditionalGeneration

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLMagvit, EasyAnimateTransformer3DModel
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.mindspore_utils import pynative_context, randn_tensor
from ...video_processor import VideoProcessor
from .pipeline_output import EasyAnimatePipelineOutput

XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import mindspore as ms
        >>> from mindone.diffusers import EasyAnimateControlPipeline
        >>> from mindone.diffusers.pipelines.easyanimate.pipeline_easyanimate_control import get_video_to_video_latent
        >>> from mindone.diffusers.utils import export_to_video, load_video

        >>> pipe = EasyAnimateControlPipeline.from_pretrained(
        ...     "alibaba-pai/EasyAnimateV5.1-12b-zh-Control-diffusers", mindspore_dtype=ms.bfloat16
        ... )

        >>> control_video = load_video(
        ...     "https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-Control/blob/main/asset/pose.mp4"
        ... )
        >>> prompt = (
        ...     "In this sunlit outdoor garden, a beautiful woman is dressed in a knee-length, sleeveless white dress. "
        ...     "The hem of her dress gently sways with her graceful dance, much like a butterfly fluttering in the breeze. "
        ...     "Sunlight filters through the leaves, casting dappled shadows that highlight her soft features and clear eyes, "
        ...     "making her appear exceptionally elegant. It seems as if every movement she makes speaks of youth and vitality. "
        ...     "As she twirls on the grass, her dress flutters, as if the entire garden is rejoicing in her dance. "
        ...     "The colorful flowers around her sway in the gentle breeze, with roses, chrysanthemums, and lilies each "
        ...     "releasing their fragrances, creating a relaxed and joyful atmosphere."
        ... )
        >>> sample_size = (672, 384)
        >>> num_frames = 49

        >>> input_video, _, _ = get_video_to_video_latent(control_video, num_frames, sample_size)
        >>> video = pipe(
        ...     prompt,
        ...     num_frames=num_frames,
        ...     negative_prompt="Twisted body, limb deformities, text subtitles, comics, stillness, ugliness, errors, garbled text.",
        ...     height=sample_size[0],
        ...     width=sample_size[1],
        ...     control_video=input_video,
        ... )[0][0]
        >>> export_to_video(video, "output.mp4", fps=8)
        ```
"""


def preprocess_image(image, sample_size):
    """
    Preprocess a single image (PIL.Image, numpy.ndarray, or ms.Tensor) to a resized tensor.
    """
    if isinstance(image, ms.Tensor):
        # If input is a tensor, assume it's in CHW format and resize using interpolation
        image = mint.nn.functional.interpolate(
            image.unsqueeze(0), size=sample_size, mode="bilinear", align_corners=False
        ).squeeze(0)
    elif isinstance(image, Image.Image):
        # If input is a PIL image, resize and convert to numpy array
        image = image.resize((sample_size[1], sample_size[0]))
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        # If input is a numpy array, resize using PIL
        image = Image.fromarray(image).resize((sample_size[1], sample_size[0]))
        image = np.array(image)
    else:
        raise ValueError("Unsupported input type. Expected PIL.Image, numpy.ndarray, or ms.Tensor.")

    # Convert to tensor if not already
    if not isinstance(image, ms.Tensor):
        image = ms.tensor(image).permute(2, 0, 1).float() / 255.0  # HWC -> CHW, normalize to [0, 1]

    return image


def get_video_to_video_latent(input_video, num_frames, sample_size, validation_video_mask=None, ref_image=None):
    if input_video is not None:
        # Convert each frame in the list to tensor
        input_video = [preprocess_image(frame, sample_size=sample_size) for frame in input_video]

        # Stack all frames into a single tensor (F, C, H, W)
        input_video = mint.stack(input_video)[:num_frames]

        # Add batch dimension (B, F, C, H, W)
        input_video = input_video.permute(1, 0, 2, 3).unsqueeze(0)

        if validation_video_mask is not None:
            # Handle mask input
            validation_video_mask = preprocess_image(validation_video_mask, size=sample_size)
            input_video_mask = mint.where(validation_video_mask < 240 / 255.0, 0.0, 255)

            # Adjust mask dimensions to match video
            input_video_mask = input_video_mask.unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
            input_video_mask = mint.tile(input_video_mask, [1, 1, input_video.shape[2], 1, 1])
            input_video_mask = input_video_mask.to(input_video.dtype)
        else:
            input_video_mask = mint.zeros_like(input_video[:, :1])
            input_video_mask[:, :, :] = 255
    else:
        input_video, input_video_mask = None, None

    if ref_image is not None:
        # Convert reference image to tensor
        ref_image = preprocess_image(ref_image, size=sample_size)
        ref_image = ref_image.permute(1, 0, 2, 3).unsqueeze(0)  # Add batch dimension (B, C, H, W)
    else:
        ref_image = None

    return input_video, input_video_mask, ref_image


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


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


# Resize mask information in magvit
def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :], size=target_size, mode="trilinear", align_corners=False
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :], size=target_size, mode="trilinear", align_corners=False
            )
            resized_mask = mint.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(mask, size=target_size, mode="trilinear", align_corners=False)
    return resized_mask


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


class EasyAnimateControlPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using EasyAnimate.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    EasyAnimate uses one text encoder [qwen2 vl](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) in V5.1.

    Args:
        vae ([`AutoencoderKLMagvit`]):
            Variational Auto-Encoder (VAE) Model to encode and decode video to and from latent representations.
        text_encoder (Optional[`~transformers.Qwen2VLForConditionalGeneration`, `~transformers.BertModel`]):
            EasyAnimate uses [qwen2 vl](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) in V5.1.
        tokenizer (Optional[`~transformers.Qwen2Tokenizer`, `~transformers.BertTokenizer`]):
            A `Qwen2Tokenizer` or `BertTokenizer` to tokenize text.
        transformer ([`EasyAnimateTransformer3DModel`]):
            The EasyAnimate model designed by EasyAnimate Team.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with EasyAnimate to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKLMagvit,
        text_encoder: Union[Qwen2VLForConditionalGeneration, BertModel],
        tokenizer: Union[Qwen2Tokenizer, BertTokenizer],
        transformer: EasyAnimateTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.enable_text_attention_mask = (
            self.transformer.config.enable_text_attention_mask
            if getattr(self, "transformer", None) is not None
            else True
        )
        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 4
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_spatial_compression_ratio,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

    # Copied from diffusers.pipelines.easyanimate.pipeline_easyanimate.EasyAnimatePipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        prompt_attention_mask: Optional[ms.Tensor] = None,
        negative_prompt_attention_mask: Optional[ms.Tensor] = None,
        dtype: Optional[ms.Type] = None,
        max_sequence_length: int = 256,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            dtype (`ms.Type`):
                mindspore dtype
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            prompt_attention_mask (`ms.Tensor`, *optional*):
                Attention mask for the prompt. Required when `prompt_embeds` is passed directly.
            negative_prompt_attention_mask (`ms.Tensor`, *optional*):
                Attention mask for the negative prompt. Required when `negative_prompt_embeds` is passed directly.
            max_sequence_length (`int`, *optional*): maximum sequence length to use for the prompt.
        """
        dtype = dtype or self.text_encoder.dtype

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(prompt, str):
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": _prompt}],
                    }
                    for _prompt in prompt
                ]
            text = [
                self.tokenizer.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages
            ]

            text_inputs = self.tokenizer(
                text=text,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_attention_mask=True,
                padding_side="right",
                return_tensors="np",
            )

            text_input_ids = ms.tensor(text_inputs.input_ids)
            prompt_attention_mask = ms.tensor(text_inputs.attention_mask)
            if self.enable_text_attention_mask:
                # Inference: Generation of the output
                # text_encoder only support pynative
                with pynative_context():
                    prompt_embeds = self.text_encoder(
                        input_ids=text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
                    )[2][-2]
            else:
                raise ValueError("LLM needs attention_mask")
            prompt_attention_mask = prompt_attention_mask.tile((num_images_per_prompt, 1))

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile((1, num_images_per_prompt, 1))
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is not None and isinstance(negative_prompt, str):
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": negative_prompt}],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": _negative_prompt}],
                    }
                    for _negative_prompt in negative_prompt
                ]
            text = [
                self.tokenizer.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages
            ]

            text_inputs = self.tokenizer(
                text=text,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_attention_mask=True,
                padding_side="right",
                return_tensors="np",
            )

            text_input_ids = ms.tensor(text_inputs.input_ids)
            negative_prompt_attention_mask = ms.tensor(text_inputs.attention_mask)
            if self.enable_text_attention_mask:
                # Inference: Generation of the output
                # text_encoder only support pynative
                with pynative_context():
                    negative_prompt_embeds = self.text_encoder(
                        input_ids=text_input_ids,
                        attention_mask=negative_prompt_attention_mask,
                        output_hidden_states=True,
                    )[2][-2]
            else:
                raise ValueError("LLM needs attention_mask")
            negative_prompt_attention_mask = negative_prompt_attention_mask.tile((num_images_per_prompt, 1))

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype)

            negative_prompt_embeds = negative_prompt_embeds.tile((1, num_images_per_prompt, 1))
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

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

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, generator, latents=None
    ):
        if latents is not None:
            return latents.to(dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_temporal_compression_ratio + 1,
            height // self.vae_spatial_compression_ratio,
            width // self.vae_spatial_compression_ratio,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = (latents * self.scheduler.init_noise_sigma).to(dtype)
        return latents

    def prepare_control_latents(
        self, control, control_image, batch_size, height, width, dtype, generator, do_classifier_free_guidance
    ):
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if control is not None:
            control = control.to(dtype=dtype)
            bs = 1
            new_control = []
            for i in range(0, control.shape[0], bs):
                control_bs = control[i : i + bs]
                # vae encode only support pynative
                with pynative_context():
                    control_bs = self.vae.encode(control_bs)[0]
                control_bs = self.vae.diag_gauss_dist.mode(control_bs)
                new_control.append(control_bs)
            control = mint.cat(new_control, dim=0)
            control = control * self.vae.config.scaling_factor

        if control_image is not None:
            control_image = control_image.to(dtype=dtype)
            bs = 1
            new_control_pixel_values = []
            for i in range(0, control_image.shape[0], bs):
                control_pixel_values_bs = control_image[i : i + bs]
                # vae encode only support pynative
                with pynative_context():
                    control_pixel_values_bs = self.vae.encode(control_pixel_values_bs)[0]
                control_pixel_values_bs = self.vae.diag_gauss_dist.mode(control_pixel_values_bs)
                new_control_pixel_values.append(control_pixel_values_bs)
            control_image_latents = mint.cat(new_control_pixel_values, dim=0)
            control_image_latents = control_image_latents * self.vae.config.scaling_factor
        else:
            control_image_latents = None

        return control, control_image_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_frames: Optional[int] = 49,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        control_video: Union[ms.Tensor] = None,
        control_camera_video: Union[ms.Tensor] = None,
        ref_image: Union[ms.Tensor] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        prompt_attention_mask: Optional[ms.Tensor] = None,
        negative_prompt_attention_mask: Optional[ms.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        guidance_rescale: float = 0.0,
        timesteps: Optional[List[int]] = None,
    ):
        r"""
        Generates images or video using the EasyAnimate pipeline based on the provided prompts.

        Examples:
            prompt (`str` or `List[str]`, *optional*):
                Text prompts to guide the image or video generation. If not provided, use `prompt_embeds` instead.
            num_frames (`int`, *optional*):
                Length of the generated video (in frames).
            height (`int`, *optional*):
                Height of the generated image in pixels.
            width (`int`, *optional*):
                Width of the generated image in pixels.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of denoising steps during generation. More steps generally yield higher quality images but slow
                down inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Encourages the model to align outputs with prompts. A higher value may decrease image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                Prompts indicating what to exclude in generation. If not specified, use `negative_prompt_embeds`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images to generate for each prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Applies to DDIM scheduling. Controlled by the eta parameter from the related literature.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                A generator to ensure reproducibility in image generation.
            latents (`ms.Tensor`, *optional*):
                Predefined latent tensors to condition generation.
            prompt_embeds (`ms.Tensor`, *optional*):
                Text embeddings for the prompts. Overrides prompt string inputs for more flexibility.
            negative_prompt_embeds (`ms.Tensor`, *optional*):
                Embeddings for negative prompts. Overrides string inputs if defined.
            prompt_attention_mask (`ms.Tensor`, *optional*):
                Attention mask for the primary prompt embeddings.
            negative_prompt_attention_mask (`ms.Tensor`, *optional*):
                Attention mask for negative prompt embeddings.
            output_type (`str`, *optional*, defaults to "latent"):
                Format of the generated output, either as a PIL image or as a NumPy array.
            return_dict (`bool`, *optional*, defaults to `False`):
                If `True`, returns a structured output. Otherwise returns a simple tuple.
            callback_on_step_end (`Callable`, *optional*):
                Functions called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                Tensor names to be included in callback function calls.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Adjusts noise levels based on guidance scale.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. default height and width
        height = int((height // 16) * 16)
        width = int((width // 16) * 16)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        else:
            dtype = self.transformer.dtype

        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            dtype=dtype,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps, mu=1)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            dtype,
            generator,
            latents,
        )

        if control_camera_video is not None:
            control_video_latents = resize_mask(control_camera_video, latents, process_first_frame_only=True)
            control_video_latents = control_video_latents * 6
            control_latents = (
                mint.cat([control_video_latents] * 2) if self.do_classifier_free_guidance else control_video_latents
            ).to(dtype)
        elif control_video is not None:
            batch_size, channels, num_frames, height_video, width_video = control_video.shape
            control_video = self.image_processor.preprocess(
                control_video.permute(0, 2, 1, 3, 4).reshape(
                    batch_size * num_frames, channels, height_video, width_video
                ),
                height=height,
                width=width,
            )
            control_video = control_video.to(dtype=ms.float32)
            control_video = control_video.reshape(batch_size, num_frames, channels, height, width).permute(
                0, 2, 1, 3, 4
            )
            control_video_latents = self.prepare_control_latents(
                None,
                control_video,
                batch_size,
                height,
                width,
                dtype,
                generator,
                self.do_classifier_free_guidance,
            )[1]
            control_latents = (
                mint.cat([control_video_latents] * 2) if self.do_classifier_free_guidance else control_video_latents
            ).to(dtype)
        else:
            control_video_latents = mint.zeros_like(latents).to(dtype)
            control_latents = (
                mint.cat([control_video_latents] * 2) if self.do_classifier_free_guidance else control_video_latents
            ).to(dtype)

        if ref_image is not None:
            batch_size, channels, num_frames, height_video, width_video = ref_image.shape
            ref_image = self.image_processor.preprocess(
                ref_image.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height_video, width_video),
                height=height,
                width=width,
            )
            ref_image = ref_image.to(dtype=ms.float32)
            ref_image = ref_image.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)

            ref_image_latents = self.prepare_control_latents(
                None,
                ref_image,
                batch_size,
                height,
                width,
                prompt_embeds.dtype,
                generator,
                self.do_classifier_free_guidance,
            )[1]

            ref_image_latents_conv_in = mint.zeros_like(latents)
            if latents.shape[2] != 1:
                ref_image_latents_conv_in[:, :, :1] = ref_image_latents
            ref_image_latents_conv_in = (
                mint.cat([ref_image_latents_conv_in] * 2)
                if self.do_classifier_free_guidance
                else ref_image_latents_conv_in
            ).to(dtype)
            control_latents = mint.cat([control_latents, ref_image_latents_conv_in], dim=1)
        else:
            ref_image_latents_conv_in = mint.zeros_like(latents)
            ref_image_latents_conv_in = (
                mint.cat([ref_image_latents_conv_in] * 2)
                if self.do_classifier_free_guidance
                else ref_image_latents_conv_in
            ).to(dtype)
            control_latents = mint.cat([control_latents, ref_image_latents_conv_in], dim=1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if self.do_classifier_free_guidance:
            prompt_embeds = mint.cat([negative_prompt_embeds, prompt_embeds])
            prompt_attention_mask = mint.cat([negative_prompt_attention_mask, prompt_attention_mask])

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = mint.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
                t_expand = ms.tensor([t] * latent_model_input.shape[0]).to(dtype=latent_model_input.dtype)
                # predict the noise residual
                noise_pred = self.transformer(
                    latent_model_input,
                    t_expand,
                    encoder_hidden_states=prompt_embeds,
                    control_latents=control_latents,
                    return_dict=False,
                )[0]
                if noise_pred.shape[1] != self.vae.config.latent_channels:
                    noise_pred, _ = noise_pred.chunk(2, dim=1)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://huggingface.co/papers/2305.08891
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Convert to tensor
        if not output_type == "latent":
            # vae decode only support pynative
            with pynative_context():
                video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        if not return_dict:
            return (video,)

        return EasyAnimatePipelineOutput(frames=video)
