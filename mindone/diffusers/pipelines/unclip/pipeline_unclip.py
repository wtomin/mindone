# Copyright 2025 Kakao Brain and The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Tuple, Union

import numpy as np
from transformers import CLIPTokenizer

import mindspore as ms
from mindspore import mint, ops

from ....transformers.models.clip.modeling_clip import CLIPTextModelOutput, CLIPTextModelWithProjection
from ...models import PriorTransformer, UNet2DConditionModel, UNet2DModel
from ...schedulers import UnCLIPScheduler
from ...utils import logging
from ...utils.mindspore_utils import randn_tensor
from ..pipeline_utils import DeprecatedPipelineMixin, DiffusionPipeline, ImagePipelineOutput
from .text_proj import UnCLIPTextProjModel

XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class UnCLIPPipeline(DeprecatedPipelineMixin, DiffusionPipeline):
    """
    Pipeline for text-to-image generation using unCLIP.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        text_encoder ([`~transformers.CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        prior ([`PriorTransformer`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        text_proj ([`UnCLIPTextProjModel`]):
            Utility class to prepare and combine the embeddings before they are passed to the decoder.
        decoder ([`UNet2DConditionModel`]):
            The decoder to invert the image embedding into an image.
        super_res_first ([`UNet2DModel`]):
            Super resolution UNet. Used in all but the last step of the super resolution diffusion process.
        super_res_last ([`UNet2DModel`]):
            Super resolution UNet. Used in the last step of the super resolution diffusion process.
        prior_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the prior denoising process (a modified [`DDPMScheduler`]).
        decoder_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the decoder denoising process (a modified [`DDPMScheduler`]).
        super_res_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the super resolution denoising process (a modified [`DDPMScheduler`]).

    """

    _last_supported_version = "0.33.1"
    _exclude_from_cpu_offload = ["prior"]

    prior: PriorTransformer
    decoder: UNet2DConditionModel
    text_proj: UnCLIPTextProjModel
    text_encoder: CLIPTextModelWithProjection
    tokenizer: CLIPTokenizer
    super_res_first: UNet2DModel
    super_res_last: UNet2DModel

    prior_scheduler: UnCLIPScheduler
    decoder_scheduler: UnCLIPScheduler
    super_res_scheduler: UnCLIPScheduler

    model_cpu_offload_seq = "text_encoder->text_proj->decoder->super_res_first->super_res_last"

    def __init__(
        self,
        prior: PriorTransformer,
        decoder: UNet2DConditionModel,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_proj: UnCLIPTextProjModel,
        super_res_first: UNet2DModel,
        super_res_last: UNet2DModel,
        prior_scheduler: UnCLIPScheduler,
        decoder_scheduler: UnCLIPScheduler,
        super_res_scheduler: UnCLIPScheduler,
    ):
        super().__init__()

        self.register_modules(
            prior=prior,
            decoder=decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_proj=text_proj,
            super_res_first=super_res_first,
            super_res_last=super_res_last,
            prior_scheduler=prior_scheduler,
            decoder_scheduler=decoder_scheduler,
            super_res_scheduler=super_res_scheduler,
        )

    def prepare_latents(self, shape, dtype, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        latents = (latents * scheduler.init_noise_sigma).to(dtype)
        return latents

    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        text_model_output: Optional[Union[CLIPTextModelOutput, Tuple]] = None,
        text_attention_mask: Optional[ms.Tensor] = None,
    ):
        if text_model_output is None:
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            # get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = text_inputs.input_ids
            text_mask = ms.Tensor.from_numpy(text_inputs.attention_mask)  # MindSpore mask does not require bool()

            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="np").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not np.array_equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
                text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

            text_encoder_output = self.text_encoder(ms.tensor(text_input_ids))

            prompt_embeds = text_encoder_output[0]
            text_enc_hid_states = text_encoder_output[1]

        else:
            batch_size = text_model_output[0].shape[0]
            prompt_embeds, text_enc_hid_states = text_model_output[0], text_model_output[1]
            text_mask = text_attention_mask

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_enc_hid_states = text_enc_hid_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            uncond_text_mask = ms.Tensor.from_numpy(
                uncond_input.attention_mask
            )  # MindSpore mask does not require bool()
            negative_prompt_embeds_text_encoder_output = self.text_encoder(ms.Tensor.from_numpy(uncond_input.input_ids))

            negative_prompt_embeds = negative_prompt_embeds_text_encoder_output[0]
            uncond_text_enc_hid_states = negative_prompt_embeds_text_encoder_output[1]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.tile((1, num_images_per_prompt))
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_text_enc_hid_states.shape[1]
            uncond_text_enc_hid_states = uncond_text_enc_hid_states.tile((1, num_images_per_prompt, 1))
            uncond_text_enc_hid_states = uncond_text_enc_hid_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = mint.cat([negative_prompt_embeds, prompt_embeds])
            text_enc_hid_states = mint.cat([uncond_text_enc_hid_states, text_enc_hid_states])

            text_mask = mint.cat([uncond_text_mask, text_mask])

        return prompt_embeds, text_enc_hid_states, text_mask

    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        prior_num_inference_steps: int = 25,
        decoder_num_inference_steps: int = 25,
        super_res_num_inference_steps: int = 7,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        prior_latents: Optional[ms.Tensor] = None,
        decoder_latents: Optional[ms.Tensor] = None,
        super_res_latents: Optional[ms.Tensor] = None,
        text_model_output: Optional[Union[CLIPTextModelOutput, Tuple]] = None,
        text_attention_mask: Optional[ms.Tensor] = None,
        prior_guidance_scale: float = 4.0,
        decoder_guidance_scale: float = 8.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
    ):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation. This can only be left undefined if `text_model_output`
                and `text_attention_mask` is passed.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            prior_num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps for the prior. More denoising steps usually lead to a higher quality
                image at the expense of slower inference.
            decoder_num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps for the decoder. More denoising steps usually lead to a higher quality
                image at the expense of slower inference.
            super_res_num_inference_steps (`int`, *optional*, defaults to 7):
                The number of denoising steps for super resolution. More denoising steps usually lead to a higher
                quality image at the expense of slower inference.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                A [`np.random.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            prior_latents (`ms.Tensor` of shape (batch size, embeddings dimension), *optional*):
                Pre-generated noisy latents to be used as inputs for the prior.
            decoder_latents (`ms.Tensor` of shape (batch size, channels, height, width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            super_res_latents (`ms.Tensor` of shape (batch size, channels, super res height, super res width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            prior_guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            decoder_guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            text_model_output (`CLIPTextModelOutput`, *optional*):
                Pre-defined [`CLIPTextModel`] outputs that can be derived from the text encoder. Pre-defined text
                outputs can be passed for tasks like text embedding interpolations. Make sure to also pass
                `text_attention_mask` in this case. `prompt` can the be left `None`.
            text_attention_mask (`ms.Tensor`, *optional*):
                Pre-defined CLIP text attention mask that can be derived from the tokenizer. Pre-defined text attention
                masks are necessary when passing `text_model_output`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
        if prompt is not None:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        else:
            batch_size = text_model_output[0].shape[0]

        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = prior_guidance_scale > 1.0 or decoder_guidance_scale > 1.0

        prompt_embeds, text_enc_hid_states, text_mask = self._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, text_model_output, text_attention_mask
        )

        # prior

        self.prior_scheduler.set_timesteps(prior_num_inference_steps)
        prior_timesteps_tensor = self.prior_scheduler.timesteps

        embedding_dim = self.prior.config.embedding_dim

        prior_latents = self.prepare_latents(
            (batch_size, embedding_dim),
            prompt_embeds.dtype,
            generator,
            prior_latents,
            self.prior_scheduler,
        )

        for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = mint.cat([prior_latents] * 2) if do_classifier_free_guidance else prior_latents

            predicted_image_embedding = self.prior(
                latent_model_input,
                timestep=t,
                proj_embedding=prompt_embeds,
                encoder_hidden_states=text_enc_hid_states,
                attention_mask=text_mask,
            )[0]

            if do_classifier_free_guidance:
                predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                predicted_image_embedding = predicted_image_embedding_uncond + prior_guidance_scale * (
                    predicted_image_embedding_text - predicted_image_embedding_uncond
                )

            if i + 1 == prior_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = prior_timesteps_tensor[i + 1]

            prior_latents = self.prior_scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=prior_latents,
                generator=generator,
                prev_timestep=prev_timestep,
            )[0]

        prior_latents = self.prior.post_process_latents(prior_latents)

        image_embeddings = prior_latents

        # done prior

        # decoder

        text_enc_hid_states, additive_clip_time_embeddings = self.text_proj(
            image_embeddings=image_embeddings,
            prompt_embeds=prompt_embeds,
            text_encoder_hidden_states=text_enc_hid_states,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        decoder_text_mask = mint.nn.functional.pad(text_mask, (self.text_proj.clip_extra_context_tokens, 0), value=1.0)

        self.decoder_scheduler.set_timesteps(decoder_num_inference_steps)
        decoder_timesteps_tensor = self.decoder_scheduler.timesteps

        num_channels_latents = self.decoder.config.in_channels
        height = self.decoder.config.sample_size
        width = self.decoder.config.sample_size

        decoder_latents = self.prepare_latents(
            (batch_size, num_channels_latents, height, width),
            text_enc_hid_states.dtype,
            generator,
            decoder_latents,
            self.decoder_scheduler,
        )

        for i, t in enumerate(self.progress_bar(decoder_timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = mint.cat([decoder_latents] * 2) if do_classifier_free_guidance else decoder_latents

            noise_pred = self.decoder(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_enc_hid_states,
                class_labels=additive_clip_time_embeddings,
                attention_mask=decoder_text_mask,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(latent_model_input.shape[1], dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(latent_model_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + decoder_guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = mint.cat([noise_pred, predicted_variance], dim=1)

            if i + 1 == decoder_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = decoder_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            decoder_latents = self.decoder_scheduler.step(
                noise_pred, t, decoder_latents, prev_timestep=prev_timestep, generator=generator
            )[0]

        decoder_latents = decoder_latents.clamp(-1, 1)

        image_small = decoder_latents

        # done decoder

        # super res

        self.super_res_scheduler.set_timesteps(super_res_num_inference_steps)
        super_res_timesteps_tensor = self.super_res_scheduler.timesteps

        channels = self.super_res_first.config.in_channels // 2
        height = self.super_res_first.config.sample_size
        width = self.super_res_first.config.sample_size

        super_res_latents = self.prepare_latents(
            (batch_size, channels, height, width),
            image_small.dtype,
            generator,
            super_res_latents,
            self.super_res_scheduler,
        )

        interpolate_antialias = {}
        # # todo: `antialias` is not supported in `mint.nn.functional.interpolate`. Accuracy issues exist in pynative_fp16
        if "antialias" in inspect.signature(ops.interpolate).parameters:
            interpolate_antialias["antialias"] = True

        image_upscaled = ops.interpolate(
            image_small, size=[height, width], mode="bicubic", align_corners=False, **interpolate_antialias
        )

        for i, t in enumerate(self.progress_bar(super_res_timesteps_tensor)):
            # no classifier free guidance

            if i == super_res_timesteps_tensor.shape[0] - 1:
                unet = self.super_res_last
            else:
                unet = self.super_res_first

            latent_model_input = mint.cat([super_res_latents, image_upscaled], dim=1)

            noise_pred = unet(
                sample=latent_model_input,
                timestep=t,
            )[0]

            if i + 1 == super_res_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = super_res_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            super_res_latents = self.super_res_scheduler.step(
                noise_pred, t, super_res_latents, prev_timestep=prev_timestep, generator=generator
            )[0]

        image = super_res_latents
        # done super res

        # post processing
        image = image * 0.5 + 0.5
        image = image.clamp(0, 1)
        image = image.permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
