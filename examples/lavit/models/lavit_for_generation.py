import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
from utils import get_amp_model

import mindspore as ms
from mindspore import mint, nn, ops

sys.path.append(".")
mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
import PIL
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from models.modeling_decoder import build_tokenizer_decoder
from models.modeling_visual_tokenizer import VectorQuantizer, build_dynamic_tokenizer
from models.transform import LaVITImageProcessor
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor

from mindone.diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from mindone.diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from mindone.utils.amp import auto_mixed_precision


class LaVITforGeneration(nn.Cell):
    def __init__(
        self,
        model_path="",
        model_dtype="bf16",
        amp_level="O2",
        check_safety=False,
        load_tokenizer=False,
        pixel_decoding="highres",
        model_sub_dir="language_model",
        use_flash_attention=True,
        **kwargs,
    ):
        """
        model_path: The pre-trained model checkpoint path, the local path for downloaded LaVIT weight
        model_dtype: The precision of model weight during inference, should be set bf16 or fp16, default is bf16.
        load_tokenizer: Whether to load the tokenizer encoder during the image generation. For text-to-image generation,
        The visual tokenizer is not needed, so the default is False for saving the memory. When using for the
        multi-modal synthesis (the input image needs to be tokenizd to dircrete ids), the load_tokenizer must be set to True.
        check_safety: load the stable diffusion safety checker to check the safety of generated image, if not safe, output a black image
        pixel_decoding: can be set to `highres` or `lowres`, default is `highres`: using the high resolution decoding
            for generating high-quality images, if set to `lowres`, using the origin decoder to generate 512 x 512 image
        """
        super().__init__()

        # visual_vocab_size = 16384  # The visual vocab size of LaVIT is 16384
        self.model_dtype = model_dtype
        print(f"Loading LaVIT Model Weight from {model_path}, model precision: {model_dtype}")
        if model_dtype in ["fp16", "bf16"]:
            print(f"auto_mixed_precision level {amp_level}")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, model_sub_dir))
        self.llama_tokenizer.padding_side = "left"
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.llama_model, loading_info = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_path, model_sub_dir), output_loading_info=True
        )
        print(loading_info)

        for param in self.llama_model.get_parameters():
            param.requires_grad = False
        self.llama_model = get_amp_model(self.llama_model, self.dtype, amp_level)
        if load_tokenizer:
            self.visual_tokenizer = build_dynamic_tokenizer(model_path, for_understanding=False)
        else:
            # For text-to-image generation, we does not load the tokenizer for saving memory
            # For using multi-modal synthesis, please set load_tokenizer to True
            import torch

            self.visual_tokenizer = None
            self.quantize = VectorQuantizer(n_embed=16384, embedding_dim=32)
            # Load the quantize embedding weight
            weight_path = os.path.join(model_path, "visual_tokenizer", "tokenizer_encoder.bin")
            state_dict = torch.load(weight_path, map_location="cpu")
            quantize_dict = OrderedDict({"embedding.weight": state_dict["quantize.embedding.weight"]})
            self.quantize.load_state_dict(quantize_dict)

        self.tokenizer_decoder = build_tokenizer_decoder(model_path, pixel_decoding=pixel_decoding)
        img_size = 224
        self.processer = LaVITImageProcessor(image_size=img_size)
        self.check_safety = check_safety

        # The diffusion related parameters
        self.pixel_decoding = pixel_decoding

        if pixel_decoding == "lowres":
            diff_model_dir = os.path.join(model_path, "pixel_decoding")
            self.register_buffer(
                "uncond_embeddings",
                torch.load(os.path.join(diff_model_dir, "uncond_embeddings.bin"), map_location="cpu"),
            )
        else:
            diff_model_dir = os.path.join(model_path, "highres_pixel_decoding")

        self.vae = AutoencoderKL.from_pretrained(diff_model_dir, subfolder="vae")
        if self.model_dtype in ["fp16", "bf16"]:
            self.vae = auto_mixed_precision(self.vae, amp_level=amp_level, dtype=model_dtype, custom_fp32_cells=[])
        self.scheduler = DDIMScheduler.from_pretrained(diff_model_dir, subfolder="scheduler")
        self.unet = UNet2DConditionModel.from_pretrained(
            diff_model_dir,
            subfolder="unet",
            use_safetensors=False,
        )
        if self.model_dtype in ["fp16", "bf16"]:
            self.unet = auto_mixed_precision(self.unet, amp_level=amp_level, dtype=model_dtype, custom_fp32_cells=[])

        if check_safety:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                os.path.join(model_path, "pixel_decoding"),
                subfolder="feature_extractor",
            )
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                os.path.join(model_path, "pixel_decoding"),
                subfolder="safety_checker",
            )

        self.kwargs = kwargs

    @property
    def dtype(self):
        if self.model_dtype == "fp16":
            dtype = ms.float16
        elif self.model_dtype == "bf16":
            dtype = ms.bfloat16
        elif self.model_dtype == "fp32":
            # The default dtype is fp16
            dtype = ms.float32
        else:
            raise NotImplementedError
        return dtype

    def generate_image_tokenids(
        self,
        prompts,
        use_nucleus_sampling=True,
        top_p=0.9,
        top_k=50,
        num_beams=1,
        temperature=1,
        num_return_images=1,
        length_penalty=1,
        max_length=200,
        min_length=20,
        guidance_scale=3.0,
        uncond_input_ids=None,
        is_token_prompt=False,
    ):
        # Input is the multi_modal prompts, generate the image tokenids
        # If is_token_prompt, the input prompts are consiting of token ids
        self.llama_tokenizer.padding_side = "left"
        batch_size = len(prompts)

        if not is_token_prompt:
            prompt_tokens = self.llama_tokenizer(
                prompts, padding="longest", return_tensors=None, add_special_tokens=False
            )
            prompt_token_ids = ms.Tensor(prompt_tokens.input_ids)
            prompt_attn_mask = ms.Tensor(prompt_tokens.attention_mask)
        else:
            # The input prompts is already tokenized to IDs
            max_length = max([len(x) for x in prompts])
            prompt_token_ids = ops.ones((batch_size, max_length), dtype=ms.int32) * self.llama_tokenizer.pad_token_id
            prompt_attn_mask = ops.zeros((batch_size, max_length), dtype=ms.int32)
            for i in range(batch_size):
                prompt_token_ids[i, -len(prompts[i]) :] = prompts[i]
                prompt_attn_mask[i, -len(prompts[i]) :] = 1

        image_start_token = ms.Tensor([32000], dtype=ms.int32)
        image_start_token = image_start_token.broadcast_to((batch_size, -1))
        image_start_attn = ops.ones((batch_size, 1), dtype=ms.int32)  # [batch_size, 1]

        prompt_token_ids = ops.cat([prompt_token_ids, image_start_token], axis=-1)
        prompt_attn_mask = ops.cat([prompt_attn_mask, image_start_attn], axis=-1)

        prompt_embeds = self.llama_model.get_input_embeddings()(prompt_token_ids)

        # Supress the text tokens
        supress_range = range(3, 32000)
        suppress_tokens = [x for x in supress_range]

        if uncond_input_ids is None:
            outputs = self.llama_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_attn_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_new_tokens=min_length,
                suppress_tokens=suppress_tokens,
                bos_token_id=32000,
                eos_token_id=32001,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                length_penalty=length_penalty,
                num_return_sequences=num_return_images,
                guidance_scale=guidance_scale,
            )
        else:
            outputs = self.llama_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_attn_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_new_tokens=min_length,
                suppress_tokens=suppress_tokens,
                bos_token_id=32000,
                eos_token_id=32001,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                length_penalty=length_penalty,
                num_return_sequences=num_return_images,
                guidance_scale=guidance_scale,
                negative_prompt_ids=uncond_input_ids,
            )

        return outputs

    def generate_image_embeds(self, image_tokens):
        # Generate the image embeddings, that can be input to decoder to rendering pixel
        batch_size = len(image_tokens)
        tokens_prune = []
        token_nums = []

        for i_b in range(batch_size):
            image_token = image_tokens[i_b]
            image_token = image_token - 32002
            image_token = image_token[image_token >= 0]
            token_nums.append(len(image_token))
            tokens_prune.append(image_token)

        tokens_prune = ops.cat(tokens_prune, axis=0)
        token_nums = ms.Tensor(token_nums, dtype=ms.int32)

        if self.visual_tokenizer is None:
            token_quantize = self.quantize.embedding(tokens_prune)  # [np, d]
        else:
            token_quantize = self.visual_tokenizer.quantize.embedding(tokens_prune)  # [np, d]

        token_quantize = token_quantize.to(self.dtype)

        return self.tokenizer_decoder(token_quantize, token_nums)

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def run_safety_checker(self, image_array):
        safety_checker_input = self.feature_extractor(self.numpy_to_pil(image_array), return_tensors="pt")
        image_array, has_nsfw_concept = self.safety_checker(
            images=image_array, clip_input=safety_checker_input.pixel_values.to(self.dtype)
        )
        return image_array, has_nsfw_concept

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = ms.Tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        self.vae.to(ms.float32)

    def pixel_decoding_origin(
        self,
        image_tokens,
        width=512,
        height=512,
        num_inference_steps=50,
        guidance_scale_for_decoder=5.0,
    ):
        """
        For the origin LaVIT pixel decoder (based on SD v1.5), we have updated the pixel decoder. The new
        decoder supports to generate high resolution and aesthetics images. We strongly recommond you to use
        our new decoder for image synthesis.
        """

        # Take the token id as input, generate the decoded embeddings
        xrec = self.generate_image_embeds(image_tokens)
        batch_size = len(xrec)

        # To prepare the neative condition
        _, num_tokens, C = xrec.shape
        encoder_hidden_uncond = ops.zeros(batch_size, num_tokens, C, dtype=xrec.dtype)
        uncond_embeddings = self.uncond_embeddings[0].to(xrec.dtype)
        encoder_hidden_uncond[:, : len(uncond_embeddings)] = uncond_embeddings

        # To set the mask
        encoder_mask = ops.ones(batch_size, num_tokens, dtype=ms.int32)
        uncond_encoder_mask = ops.zeros(batch_size, num_tokens, dtype=ms.int32)
        uncond_encoder_mask[:, : len(uncond_embeddings)] = 1
        encoder_mask = encoder_mask.bool()
        uncond_encoder_mask = uncond_encoder_mask.bool()

        # text_embeddings, uncond_embeddings, encoder_mask, uncond_encoder_mask = self.generate_prompt_embeds(xrec)
        text_embeddings = ops.cat([encoder_hidden_uncond, xrec])
        text_embeddings_mask = ops.cat([uncond_encoder_mask, encoder_mask])

        latents = ops.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
        )
        latents = latents * self.scheduler.init_noise_sigma
        latents = latents

        self.scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = ops.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                encoder_attention_mask=text_embeddings_mask,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = mint.chunk(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale_for_decoder * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / self.vae.config.scaling_factor
        output_image = self.vae.decode(latents).sample

        output_image = output_image.float()
        output_image = (output_image / 2 + 0.5).clamp(0, 1)
        output_image = output_image.permute(0, 2, 3, 1).asnumpy()

        if self.check_safety:
            output_image, _ = self.run_safety_checker(output_image)

        output_images = self.numpy_to_pil(output_image)

        return output_images

    def generate_image(
        self,
        prompts,
        width=512,
        height=512,
        original_size=None,
        crops_coords_top_left=None,
        use_nucleus_sampling=True,
        top_p=1.0,
        top_k=50,
        num_beams=1,
        temperature=1,
        num_return_images=1,
        length_penalty=1,
        max_length=200,
        min_length=20,
        num_inference_steps=25,
        guidance_scale_for_llm=4.0,
        guidance_scale_for_decoder=7.0,
        uncond_input_ids=None,
        is_token_prompt=False,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        original_size = original_size or (height, width)
        target_size = (height, width)
        crops_coords_top_left = crops_coords_top_left or (0, 0)

        image_tokens = self.generate_image_tokenids(
            prompts,
            use_nucleus_sampling,
            top_p,
            top_k,
            num_beams,
            temperature,
            num_return_images,
            length_penalty,
            max_length,
            min_length,
            guidance_scale_for_llm,
            uncond_input_ids,
            is_token_prompt,
        )

        if self.pixel_decoding == "lowres":
            return self.pixel_decoding_origin(
                image_tokens, width, height, num_inference_steps, guidance_scale_for_decoder
            )

        # Perform pixel decoding from tokenids to RGB pixel values

        # Take the token id as input, generate the decoded embeddings
        # The negative prompt embeddings shall be forced to always be set to 0
        prompt_embeds, pooled_prompt_embeds = self.generate_image_embeds(image_tokens)
        negative_prompt_embeds = ops.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = ops.zeros_like(pooled_prompt_embeds)

        batch_size = len(prompt_embeds)

        latents = ops.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
        )
        latents = latents
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        # Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size, prompt_embeds.dtype)
        add_time_ids = add_time_ids.repeat(batch_size, axis=0)
        negative_add_time_ids = add_time_ids

        prompt_embeds = ops.cat([negative_prompt_embeds, prompt_embeds], axis=0)
        add_text_embeds = ops.cat([negative_pooled_prompt_embeds, add_text_embeds], axis=0)
        add_time_ids = ops.cat([negative_add_time_ids, add_time_ids], axis=0)

        prompt_embeds = prompt_embeds
        add_text_embeds = add_text_embeds
        add_time_ids = add_time_ids

        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = ops.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = mint.chunk(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale_for_decoder * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == ms.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()

        latents = latents.to(next(iter(self.vae.post_quant_conv.get_parameters())).dtype)
        output_image = self.vae.decode(latents / self.vae.config.scaling_factor).sample

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=ms.float16)

        output_image = output_image.float()
        output_image = (output_image / 2 + 0.5).clamp(0, 1)
        output_image = output_image.permute(0, 2, 3, 1).asnumpy()

        if self.check_safety:
            output_image, _ = self.run_safety_checker(output_image)

        output_images = self.numpy_to_pil(output_image)

        return output_images

    def multimodal_synthesis(
        self,
        prompts,
        width=512,
        height=512,
        original_size=None,
        crops_coords_top_left=None,
        use_nucleus_sampling=True,
        top_p=1.0,
        top_k=50,
        num_beams=1,
        temperature=1,
        num_return_images=1,
        length_penalty=1,
        max_length=200,
        min_length=20,
        num_inference_steps=25,
        guidance_scale_for_llm=5.0,
        guidance_scale_for_decoder=7.0,
        uncond_input_ids=None,
    ):
        # The multi-modal propmts with format:
        # Image+Text and Image+Image
        # prompts: [(img_path, 'image') or (text, 'text')]
        # Now the code only supports: batchsize=1
        assert (
            self.visual_tokenizer is not None
        ), "For multi-modal image synthesis, please set the `load_tokenizer` to True in the `build_model` method"
        input_prompts = []

        for prompt_str, prompt_type in prompts:
            assert prompt_type in ["image", "text"], "The prompt type should be image or text"
            if prompt_type == "text":
                text_tokens = self.llama_tokenizer(
                    [prompt_str], padding="longest", return_tensors=None, add_special_tokens=False
                ).input_ids[0]
                text_tokens = ms.Tensor(text_tokens)
                input_prompts.append(text_tokens)
            if prompt_type == "image":
                # image_input = Image.open(prompt_str).convert("RGB")
                image_input = cv2.cvtColor(cv2.imread(prompt_str), cv2.COLOR_BGR2RGB)
                image_tensor = self.processer(image_input).unsqueeze(0)

                image_tokens = self.visual_tokenizer.tokenize_image(image_tensor, add_special=False)[0]
                input_prompts.append(image_tokens)

        input_prompts = [ops.cat(input_prompts, axis=0)]

        output_images = self.generate_image(
            input_prompts,
            width=width,
            height=height,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            use_nucleus_sampling=use_nucleus_sampling,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            temperature=temperature,
            num_return_images=num_return_images,
            length_penalty=length_penalty,
            max_length=max_length,
            min_length=min_length,
            num_inference_steps=num_inference_steps,
            guidance_scale_for_llm=guidance_scale_for_llm,
            guidance_scale_for_decoder=guidance_scale_for_decoder,
            uncond_input_ids=uncond_input_ids,
            is_token_prompt=True,
        )

        return output_images
