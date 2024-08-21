import os

import numpy as np
import PIL
from models.modeling_decoder import build_tokenizer_decoder
from models.modeling_visual_tokenizer import build_dynamic_tokenizer
from PIL import Image
from tqdm import tqdm
from utils import load_torch_state_dict_to_ms_ckpt

import mindspore as ms
from mindspore import mint, nn, ops

from mindone.diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from mindone.utils.amp import auto_mixed_precision


class LaVITDetokenizer(nn.Cell):
    def __init__(self, model_path="", model_dtype="bf16", amp_level="O2", pixel_decoding="highres", **kwargs):
        """
        Usage:
            This aims to show the detokenize result of LaVIT (from discrete token to Original Image)
            It is used to present the reconstruction fidelity.
        params:
            model_path: The pre-trained model checkpoint path, the local path for downloaded LaVIT weight
            model_dtype: The precision of model weight during inference, should be set bf16 or fp16, default is bf16.
            pixel_decoding: can be set to `highres` or `lowres`, default is `highres`: using the high resolution decoding
                for generating high-quality images (1024 x 1024), if set to `lowres`, using the origin decoder to generate 512 x 512 image
        """
        super().__init__()

        # visual_vocab_size = 16384  # The visual vocab size of LaVIT is 16384
        self.model_dtype = model_dtype
        self.visual_tokenizer = build_dynamic_tokenizer(model_path, for_understanding=False)
        if self.model_dtype in ["fp16", "bf16"]:
            self.visual_tokenizer = auto_mixed_precision(
                self.visual_tokenizer, amp_level=amp_level, dtype=model_dtype, custom_fp32_cells=[]
            )
        for param in self.visual_tokenizer.get_parameters():
            param.requires_grad = False

        self.tokenizer_decoder = build_tokenizer_decoder(model_path, pixel_decoding=pixel_decoding)
        if self.model_dtype in ["fp16", "bf16"]:
            self.tokenizer_decoder = auto_mixed_precision(
                self.tokenizer_decoder, amp_level=amp_level, dtype=model_dtype, custom_fp32_cells=[]
            )

        for param in self.tokenizer_decoder.get_parameters():
            param.requires_grad = False

        # img_size = 224

        # The diffusion related parameters
        self.pixel_decoding = pixel_decoding

        if pixel_decoding == "lowres":
            diff_model_dir = os.path.join(model_path, "pixel_decoding")
            weight_path = os.path.join(diff_model_dir, "uncond_embeddings.bin")
            state_dict = load_torch_state_dict_to_ms_ckpt(weight_path)
            self.uncond_embeddings = ms.Parameter(state_dict, requires_grad=False)
        else:
            diff_model_dir = os.path.join(model_path, "highres_pixel_decoding")

        self.vae = AutoencoderKL.from_pretrained(
            diff_model_dir,
            subfolder="vae",
        )
        if self.model_dtype in ["fp16", "bf16"]:
            self.vae = auto_mixed_precision(self.vae, amp_level=amp_level, dtype=model_dtype, custom_fp32_cells=[])
        for param in self.vae.get_parameters():
            param.requires_grad = False

        self.scheduler = DDIMScheduler.from_pretrained(diff_model_dir, subfolder="scheduler")  # For evaluation

        self.unet = UNet2DConditionModel.from_pretrained(diff_model_dir, subfolder="unet", use_safetensors=False)
        if self.model_dtype in ["fp16", "bf16"]:
            self.unet = auto_mixed_precision(self.unet, amp_level=amp_level, dtype=model_dtype, custom_fp32_cells=[])

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

    def pre_process(self, data, process_type):
        if process_type == "vae":
            mean = ms.Tensor([0.5, 0.5, 0.5])[None, :, None, None]
            std = ms.Tensor([0.5, 0.5, 0.5])[None, :, None, None]
        elif process_type == "clip":
            data = ops.interpolate(data, size=(224, 224), mode="bicubic")
            mean = ms.Tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
            std = ms.Tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
        else:
            raise NotImplementedError

        normed_data = (data - mean) / std

        return normed_data

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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        self.vae.to(dtype=ms.float32)

    def _get_add_time_ids(self, original_size, target_size, dtype):
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = ms.Tensor([add_time_ids], dtype=dtype)
        self.add_time_ids = add_time_ids
        return add_time_ids

    def reconstruct_from_token(
        self, x, width=1024, height=1024, original_size=None, num_inference_steps=50, guidance_scale=5.0
    ):
        # Original_size is
        batch_size = len(x)

        original_size = original_size or (height, width)
        target_size = (height, width)

        x_tensor = self.pre_process(x, process_type="clip")
        quantize, token_nums = self.visual_tokenizer.tokenize_image(x_tensor, used_for_llm=False)
        prompt_embeds, pooled_prompt_embeds = self.tokenizer_decoder(quantize, token_nums)

        # The negative prompt embeddings shall be forced to always be set to 0
        negative_prompt_embeds = ops.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = ops.zeros_like(pooled_prompt_embeds)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        latents = ops.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            dtype=prompt_embeds.dtype,
        )

        latents = latents * self.scheduler.init_noise_sigma

        # Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(original_size, target_size, prompt_embeds.dtype)
        negative_add_time_ids = add_time_ids

        prompt_embeds = ops.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = ops.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = ops.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds
        add_text_embeds = add_text_embeds
        add_time_ids = add_time_ids.repeat(batch_size, axis=0)

        for t in tqdm(timesteps):
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
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == ms.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()

        latents = latents.to(ms.float32)
        output_image = self.vae.decode(latents / self.vae.config.scaling_factor).sample

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=ms.float16)

        output_image = (output_image / 2 + 0.5).clamp(0, 1)
        output_image = output_image.permute(0, 2, 3, 1).asnumpy()
        output_images = self.numpy_to_pil(output_image)

        return output_images
