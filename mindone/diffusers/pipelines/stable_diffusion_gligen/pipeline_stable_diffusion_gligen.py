# Copyright 2025 The GLIGEN Authors and HuggingFace Team. All rights reserved.
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
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
from transformers import CLIPImageProcessor, CLIPTokenizer

import mindspore as ms
from mindspore import mint, ops

from ....transformers import CLIPTextModel
from ...image_processor import VaeImageProcessor
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # noqa: F401
from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.attention import GatedSelfAttentionDense
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import deprecate, logging, scale_lora_layers, unscale_lora_layers
from ...utils.mindspore_utils import randn_tensor
from ..pipeline_utils import DeprecatedPipelineMixin, DiffusionPipeline, StableDiffusionMixin
from ..stable_diffusion import StableDiffusionPipelineOutput
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker

XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import mindspore
        >>> from mindone.diffusers import StableDiffusionGLIGENPipeline
        >>> from mindone.diffusers.utils import load_image

        >>> # Insert objects described by text at the region defined by bounding boxes
        >>> pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        ...     "masterful/gligen-1-4-inpainting-text-box", variant="fp16", mindspore_dtype=mindspore.float16
        ... )

        >>> input_image = load_image(
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/livingroom_modern.png"
        ... )
        >>> prompt = "a birthday cake"
        >>> boxes = [[0.2676, 0.6088, 0.4773, 0.7183]]
        >>> phrases = ["a birthday cake"]

        >>> images = pipe(
        ...     prompt=prompt,
        ...     gligen_phrases=phrases,
        ...     gligen_inpaint_image=input_image,
        ...     gligen_boxes=boxes,
        ...     gligen_scheduled_sampling_beta=1,
        ...     output_type="pil",
        ...     num_inference_steps=50,
        ... )[0]

        >>> images[0].save("./gligen-1-4-inpainting-text-box.jpg")

        >>> # Generate an image described by the prompt and
        >>> # insert objects described by text at the region defined by bounding boxes
        >>> pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        ...     "masterful/gligen-1-4-generation-text-box", variant="fp16", mindspore_dtype=mindspore.float16
        ... )

        >>> prompt = "a waterfall and a modern high speed train running through the tunnel in a beautiful forest with fall foliage"
        >>> boxes = [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]
        >>> phrases = ["a waterfall", "a modern high speed train running through the tunnel"]

        >>> images = pipe(
        ...     prompt=prompt,
        ...     gligen_phrases=phrases,
        ...     gligen_boxes=boxes,
        ...     gligen_scheduled_sampling_beta=1,
        ...     output_type="pil",
        ...     num_inference_steps=50,
        ... )[0]

        >>> images[0].save("./gligen-1-4-generation-text-box.jpg")
        ```
"""


class StableDiffusionGLIGENPipeline(DeprecatedPipelineMixin, DiffusionPipeline, StableDiffusionMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with Grounded-Language-to-Image Generation (GLIGEN).

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) for
            more details about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    _last_supported_version = "0.33.1"

    _optional_components = ["safety_checker", "feature_extractor"]
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."  # noqa: E501
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = mint.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
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
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="np").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not np.array_equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = ms.tensor(text_inputs.attention_mask)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(ms.tensor(text_input_ids), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    ms.tensor(text_input_ids), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile((1, num_images_per_prompt, 1))
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = ms.tensor(uncond_input.attention_mask)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                ms.tensor(uncond_input.input_ids),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype)

            negative_prompt_embeds = negative_prompt_embeds.tile((1, num_images_per_prompt, 1))
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin):
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if ops.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="np")
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=ms.tensor(safety_checker_input.pixel_values).to(dtype)
            )

            # Warning for safety checker operations here as it couldn't been done in construct()
            if mint.any(has_nsfw_concept):
                logger.warning(
                    "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                    " Try again with a different prompt and/or seed."
                )
        return image, has_nsfw_concept

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
        callback_steps,
        gligen_phrases,
        gligen_boxes,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
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

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if len(gligen_phrases) != len(gligen_boxes):
            raise ValueError(
                "length of `gligen_phrases` and `gligen_boxes` has to be same, but"
                f" got: `gligen_phrases` {len(gligen_phrases)} != `gligen_boxes` {len(gligen_boxes)}"
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        else:
            latents = latents.to(dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # wtf? The above line changes the dtype of latents from fp16 to fp32, so we need a casting.
        latents = latents.to(dtype=dtype)
        return latents

    def enable_fuser(self, enabled=True):
        for sub_name, module in self.unet.cells_and_names():
            if type(module) is GatedSelfAttentionDense:
                module.enabled = enabled

    def draw_inpaint_mask_from_boxes(self, boxes, size):
        inpaint_mask = mint.ones((size[0], size[1]))
        for box in boxes:
            x0, x1 = box[0] * size[0], box[2] * size[0]
            y0, y1 = box[1] * size[1], box[3] * size[1]
            inpaint_mask[int(y0) : int(y1), int(x0) : int(x1)] = 0
        return inpaint_mask

    def crop(self, im, new_width, new_height):
        width, height = im.size
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        return im.crop((left, top, right, bottom))

    def target_size_center_crop(self, im, new_hw):
        width, height = im.size
        if width != height:
            im = self.crop(im, min(height, width), min(height, width))
        return im.resize((new_hw, new_hw), PIL.Image.LANCZOS)

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        gligen_scheduled_sampling_beta: float = 0.3,
        gligen_phrases: List[str] = None,
        gligen_boxes: List[List[float]] = None,
        gligen_inpaint_image: Optional[PIL.Image.Image] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        prompt_embeds: Optional[ms.Tensor] = None,
        negative_prompt_embeds: Optional[ms.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        callback: Optional[Callable[[int, int, ms.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            gligen_phrases (`List[str]`):
                The phrases to guide what to include in each of the regions defined by the corresponding
                `gligen_boxes`. There should only be one phrase per bounding box.
            gligen_boxes (`List[List[float]]`):
                The bounding boxes that identify rectangular regions of the image that are going to be filled with the
                content described by the corresponding `gligen_phrases`. Each rectangular box is defined as a
                `List[float]` of 4 elements `[xmin, ymin, xmax, ymax]` where each value is between [0,1].
            gligen_inpaint_image (`PIL.Image.Image`, *optional*):
                The input image, if provided, is inpainted with objects described by the `gligen_boxes` and
                `gligen_phrases`. Otherwise, it is treated as a generation task on a blank input image.
            gligen_scheduled_sampling_beta (`float`, defaults to 0.3):
                Scheduled Sampling factor from [GLIGEN: Open-Set Grounded Text-to-Image
                Generation](https://huggingface.co/papers/2301.07093). Scheduled Sampling factor is only varied for
                scheduled sampling during inference for improved quality and controllability.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only
                applies to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                A [`np.random.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`ms.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`ms.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: ms.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            gligen_phrases,
            gligen_boxes,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = mint.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        # 5.1 Prepare GLIGEN variables
        max_objs = 30
        if len(gligen_boxes) > max_objs:
            warnings.warn(
                f"More that {max_objs} objects found. Only first {max_objs} objects will be processed.",
                FutureWarning,
            )
            gligen_phrases = gligen_phrases[:max_objs]
            gligen_boxes = gligen_boxes[:max_objs]
        # prepare batched input to the GLIGENTextBoundingboxProjection (boxes, phrases, mask)
        # Get tokens for phrases from pre-trained CLIPTokenizer
        tokenizer_inputs = self.tokenizer(gligen_phrases, padding=True, return_tensors="np")
        for k, v in tokenizer_inputs.items():
            tokenizer_inputs[k] = ms.Tensor.from_numpy(v)
        # For the token, we use the same pre-trained text encoder
        # to obtain its text feature
        _text_embeddings = self.text_encoder(**tokenizer_inputs)[1]
        n_objs = len(gligen_boxes)
        # For each entity, described in phrases, is denoted with a bounding box,
        # we represent the location information as (xmin,ymin,xmax,ymax)
        boxes = mint.zeros((max_objs, 4), dtype=self.text_encoder.dtype)
        boxes[:n_objs] = ms.tensor(gligen_boxes)
        text_embeddings = mint.zeros((max_objs, self.unet.config.cross_attention_dim), dtype=self.text_encoder.dtype)
        text_embeddings[:n_objs] = _text_embeddings
        # Generate a mask for each object that is entity described by phrases
        masks = mint.zeros((max_objs,), dtype=self.text_encoder.dtype)
        masks[:n_objs] = 1

        repeat_batch = batch_size * num_images_per_prompt
        boxes = boxes.unsqueeze(0).broadcast_to((repeat_batch, -1, -1)).copy()
        text_embeddings = text_embeddings.unsqueeze(0).broadcast_to((repeat_batch, -1, -1)).copy()
        masks = masks.unsqueeze(0).broadcast_to((repeat_batch, -1)).copy()
        if do_classifier_free_guidance:
            repeat_batch = repeat_batch * 2
            boxes = mint.cat([boxes] * 2)
            text_embeddings = mint.cat([text_embeddings] * 2)
            masks = mint.cat([masks] * 2)
            masks[: repeat_batch // 2] = 0
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        cross_attention_kwargs["gligen"] = {"boxes": boxes, "positive_embeddings": text_embeddings, "masks": masks}

        # Prepare latent variables for GLIGEN inpainting
        if gligen_inpaint_image is not None:
            # if the given input image is not of the same size as expected by VAE
            # center crop and resize the input image to expected shape
            if gligen_inpaint_image.size != (self.vae.sample_size, self.vae.sample_size):
                gligen_inpaint_image = self.target_size_center_crop(gligen_inpaint_image, self.vae.sample_size)
            # Convert a single image into a batch of images with a batch size of 1
            # The resulting shape becomes (1, C, H, W), where C is the number of channels,
            # and H and W are the height and width of the image.
            # scales the pixel values to a range [-1, 1]
            gligen_inpaint_image = self.image_processor.preprocess(gligen_inpaint_image)
            gligen_inpaint_image = gligen_inpaint_image.to(dtype=self.vae.dtype)
            # Run AutoEncoder to get corresponding latents
            gligen_inpaint_latent = self.vae.encode(gligen_inpaint_image)[0]
            gligen_inpaint_latent = self.vae.diag_gauss_dist.sample(gligen_inpaint_latent)
            gligen_inpaint_latent = self.vae.config.scaling_factor * gligen_inpaint_latent
            # Generate an inpainting mask
            # pixel value = 0, where the object is present (defined by bounding boxes above)
            #               1, everywhere else
            gligen_inpaint_mask = self.draw_inpaint_mask_from_boxes(gligen_boxes, gligen_inpaint_latent.shape[2:])
            gligen_inpaint_mask = gligen_inpaint_mask.to(dtype=gligen_inpaint_latent.dtype)
            gligen_inpaint_mask = gligen_inpaint_mask[None, None]
            gligen_inpaint_mask_addition = mint.cat(
                (gligen_inpaint_latent * gligen_inpaint_mask, gligen_inpaint_mask), dim=1
            )
            # Convert a single mask into a batch of masks with a batch size of 1
            gligen_inpaint_mask_addition = gligen_inpaint_mask_addition.broadcast_to((repeat_batch, -1, -1, -1)).copy()

        num_grounding_steps = int(gligen_scheduled_sampling_beta * len(timesteps))
        self.enable_fuser(True)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the unet and will raise RuntimeError.
        lora_scale = cross_attention_kwargs.pop("scale", None) if cross_attention_kwargs is not None else None
        if lora_scale is not None:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self.unet, lora_scale)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Scheduled sampling
                if i == num_grounding_steps:
                    self.enable_fuser(False)

                if latents.shape[1] != 4:
                    latents = mint.randn_like(latents[:, :4], dtype=latents.dtype)

                if gligen_inpaint_image is not None:
                    gligen_inpaint_latent_with_noise = (
                        self.scheduler.add_noise(
                            gligen_inpaint_latent,
                            mint.randn_like(gligen_inpaint_latent, dtype=gligen_inpaint_latent.dtype),
                            t[None],
                        )
                        .broadcast_to((latents.shape[0], -1, -1, -1))
                        .copy()
                    )
                    latents = gligen_inpaint_latent_with_noise * gligen_inpaint_mask + latents * (
                        1 - gligen_inpaint_mask
                    )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = mint.cat([latents] * 2) if do_classifier_free_guidance else latents
                # TODO: method of scheduler should not change the dtype of input.
                #  Remove the casting after cuiyushi confirm that.
                tmp_dtype = latent_model_input.dtype
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = latent_model_input.to(tmp_dtype)

                if gligen_inpaint_image is not None:
                    latent_model_input = mint.cat((latent_model_input, gligen_inpaint_mask_addition), dim=1)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=ms.mutable(cross_attention_kwargs),
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # TODO: method of scheduler should not change the dtype of input.
                #  Remove the casting after cuiyushi confirm that.
                tmp_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                latents = latents.to(tmp_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if lora_scale is not None:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self.unet, lora_scale)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
