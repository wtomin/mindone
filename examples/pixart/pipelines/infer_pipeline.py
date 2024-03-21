from abc import ABC

from diffusion import create_diffusion

import mindspore as ms
from mindspore import ops

__all__ = ["InferPipeline"]


class InferPipeline(ABC):
    """An Inference pipeline for diffusion model

    Args:
        model (nn.Cell): A noise prediction model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scale_factor (float): scale_factor for vae.
        guidance_rescale (float): A higher guidance scale value for noise rescale.
        num_inference_steps: (int): The number of denoising steps.
        ddim_sampling: (bool): whether to use DDIM sampling. If False, will use DDPM sampling.
    """

    def __init__(
        self,
        model,
        vae=None,
        text_encoder=None,
        scale_factor=1.0,
        guidance_rescale=1.0,
        num_inference_steps=50,
        ddim_sampling=True,
    ):
        super().__init__()
        self.model = model

        self.vae = vae
        self.scale_factor = scale_factor
        self.guidance_rescale = guidance_rescale
        if self.guidance_rescale > 1.0:
            self.use_cfg = True
        else:
            self.use_cfg = False

        self.text_encoder = text_encoder
        self.diffusion = create_diffusion(str(num_inference_steps))
        if ddim_sampling:
            self.sampling_func = self.diffusion.ddim_sample_loop
        else:
            self.sampling_func = self.diffusion.p_sample_loop

    @ms.jit
    def vae_encode(self, x):
        image_latents = self.vae.encode(x)
        image_latents = image_latents * self.scale_factor
        return image_latents.astype(ms.float16)

    @ms.jit
    def vae_decode(self, x):
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        b, c, h, w = x.shape

        y = self.vae.decode(x / self.scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        # (b 3 H W) -> (b H W 3)
        y = ops.transpose(y, (0, 2, 3, 1))

        return y

    def vae_decode_video(self, x):
        """
        Args:
            x: (b f c h w), denoised latent
        Return:
            y: (b f H W 3), batch of images, normalized to [0, 1]
        """
        b, f, c, h, w = x.shape
        x = x.reshape((b * f, c, h, w))

        y = self.vae_decode(x)
        _, h, w, c = y.shape
        y = y.reshape((b, f, h, w, c))

        return y

    def data_prepare(self, inputs):
        x = inputs["noise"]
        tokens = inputs["tokens"]
        y, attn_mask = self.get_condition_embeddings(tokens)
        if self.use_cfg:
            y_null = self.model.y_embedder.y_embedding[None].repeat(len(x), 1, 1)[:, None]
            y = ops.cat([y, y_null], axis=0)
            x_in = ops.concat([x] * 2, axis=0)
            assert y.shape[0] == x_in.shape[0], "shape mismatch!"
        else:
            x_in = x
            y = inputs["y"]
        return x_in, y, attn_mask

    def get_condition_embeddings(self, text_tokens):
        # text conditions inputs for cross-attention
        # optional: for some conditions, concat to latents, or add to time embedding
        text_emb, attn_mask = ops.stop_gradient(self.text_encoder(text_tokens))

        return text_emb, attn_mask

    def __call__(self, inputs):
        """
        args:
            inputs: dict

        return:
            images (b f H W 3)
        """
        z, y, mask = self.data_prepare(inputs)
        model_kwargs = dict(y=y, mask=mask, img_hw=inputs["img_hw"], aspect_ratio=inputs["aspect_ratio"])
        if self.use_cfg:
            model_kwargs["cfg_scale"] = inputs["scale"]
            latents = self.sampling_func(
                self.model.construct_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
            )
            latents, _ = latents.chunk(2, axis=0)
        else:
            latents = self.sampling_func(
                self.model.construct, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
            )
        if latents.dim() == 4:
            # latents: (b c h w)
            images = self.vae_decode(latents)
        else:
            # latents: (b f c h w)
            images = self.vae_decode_video(latents)
            # output (b, f, h, w, 3)
        return images
