from abc import ABC

from ad.modules.diffusionmodules.unet3d import rearrange_in
from tqdm import tqdm

import mindspore as ms
from mindspore import mint, ops


class AnimateDiffText2Video(ABC):
    """
    Text2Video inference pipeline

    Args:
        text_encoder (nn.Cell): Frozen text-encoder.
        unet (nn.Cell): A `UNet2DConditionModel` to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scheduler (nn.Cell): A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor (float): scale_factor for vae.
        guidance_rescale (float): A higher guidance scale value for noise rescale.
        num_inference_steps: (int): The number of denoising steps.
    """

    def __init__(
        self,
        text_encoder,
        unet,
        vae,
        scheduler,
        scale_factor=1.0,
        guidance_rescale=0.0,
        num_inference_steps=50,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.scale_factor = scale_factor
        self.guidance_rescale = guidance_rescale

        scheduler.set_timesteps(num_inference_steps)

        self.num_inference_steps = ms.Tensor(num_inference_steps, ms.int32)
        self.alphas_cumprod = scheduler.alphas_cumprod

    @ms.jit
    def vae_encode(self, x):
        image_latents = self.vae.encode(x)
        image_latents = image_latents * self.scale_factor
        return image_latents.astype(ms.float16)

    @ms.jit
    def vae_decode(self, x):
        """
        Args:
            x: (b c f h w), denoised latent
        Return:
            y: (b f H W 3), batch of video frames, normalized to [0, 1]
        """
        b, c, f, h, w = x.shape
        # (b c f h w) -> (b*f c h w)
        x = rearrange_in(x)

        # (b*f 4 64 64) -> (b*f 3 512 512)
        y = self.vae.decode(x / self.scale_factor)
        y = mint.clamp((y + 1.0) / 2.0, min=0.0, max=1.0)

        # (b*f 3 H W) -> (b*f H W 3) -> (b f H W 3)
        y = mint.permute(y, (0, 2, 3, 1))
        y = mint.reshape(y, (y.shape[0] // f, f, y.shape[1], y.shape[2], y.shape[3]))

        return y

    @ms.jit
    def prompt_embed(self, prompt_data, negative_prompt_data):
        pos_prompt_embeds = self.text_encoder(prompt_data)
        negative_prompt_embeds = self.text_encoder(negative_prompt_data)
        prompt_embeds = mint.concat([negative_prompt_embeds, pos_prompt_embeds], dim=0)
        return prompt_embeds

    @ms.jit
    def latents_add_noise(self, image_latents, noise, ts):
        latents = self.scheduler.add_noise(image_latents, noise, ts)
        return latents

    @ms.jit
    def scale_model_input(self, latents, t):
        return self.scheduler.scale_model_input(latents, t)

    def predict_noise(
        self,
        x,
        t_continuous,
        c_crossattn,
        guidance_scale,
        c_concat=None,
        controlnet_images=None,
        controlnet_image_index=None,
    ):
        """
        The noise predicition model function that is used for DPM-Solver.
        """

        t_continuous = mint.tile(t_continuous.reshape(1), (x.shape[0],))
        x_in = mint.concat([x] * 2, dim=0)
        t_in = mint.concat([t_continuous] * 2, dim=0)
        if c_concat is not None:
            c_concat = mint.concat([c_concat] * 2, dim=0)
        if controlnet_images is not None:
            # controlnet
            assert (
                controlnet_images.dim() == 5
            ), f"Expect to receive 5 dims for controlnet_images, but got {controlnet_images.dim()}"  # (b, c, f, h, w)
            video_length = x.shape[2]
            b, c, _, h, w = controlnet_images.shape
            if controlnet_image_index is None:
                controlnet_image_index = ms.Tensor([0])
            assert video_length >= len(controlnet_image_index), (
                f"the video length must be greater than or equal to the length of controlnet_image_index, "
                f"but got {video_length} and {len(controlnet_image_index)}"
            )
            # replace negative index by positive index
            neg_mask = controlnet_image_index < 0
            controlnet_image_index[neg_mask] = controlnet_image_index[neg_mask] + video_length
            controlnet_cond = mint.zeros((b, c, video_length, h, w), dtype=controlnet_images.dtype)
            controlnet_conditioning_mask = mint.zeros((b, 1, video_length, h, w), dtype=controlnet_images.dtype)

            controlnet_cond[:, :, controlnet_image_index] = controlnet_images[:, :, : len(controlnet_image_index)]
            controlnet_conditioning_mask[:, :, controlnet_image_index] = 1

            ctrl_kwargs = {"controlnet_cond": controlnet_cond, "conditioning_mask": controlnet_conditioning_mask}
            noise_pred = self.unet(x_in, t_in, c_concat=c_concat, c_crossattn=c_crossattn, **ctrl_kwargs)
        else:
            noise_pred = self.unet(x_in, t_in, c_concat=c_concat, c_crossattn=c_crossattn)
        # print("D--: noise pred shape: ", noise_pred.shape, noise_pred.dtype)
        noise_pred_uncond, noise_pred_text = mint.split(noise_pred, split_size_or_sections=noise_pred.shape[0] // 2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        if self.guidance_rescale > 0:
            noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text)
        return noise_pred

    def rescale_noise_cfg(self, noise_pred, noise_pred_text):
        """
        Rescale `noise_pred` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = ops.std(noise_pred_text, axis=tuple(range(1, len(noise_pred_text.shape))), keepdims=True)
        std_cfg = ops.std(noise_pred, axis=tuple(range(1, len(noise_pred.shape))), keepdims=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_pred * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_pred = self.guidance_rescale * noise_pred_rescaled + (1 - self.guidance_rescale) * noise_pred
        return noise_pred

    def data_prepare(self, inputs):
        latents = inputs["noise"]
        c_crossattn = self.prompt_embed(inputs["prompt_data"], inputs["negative_prompt_data"])
        return latents, c_crossattn, None

    def tensor_to_video(self, frames):
        """
        Args:
            frames: (b*f 3 H W)
        Return:
            frames: (b f H W 3)
        """

        frames = frames.asnumpy().transpose(0, 2, 3, 1)

    def __call__(self, inputs):
        """
        args:
            inputs: dict

        return:
            frames (b f H W 3)
        """
        latents, c_crossattn, c_concat = self.data_prepare(inputs)
        timesteps = self.scheduler.timesteps
        iterator = tqdm(timesteps, desc="Sampling", total=len(timesteps))
        controlnet_images = inputs.get("controlnet_images", None)
        controlnet_image_index = inputs.get("controlnet_image_index", ms.Tensor([0]))
        for i, t in enumerate(iterator):
            ts = ms.Tensor(t, ms.int32)
            latents = self.scale_model_input(latents, ts)
            noise_pred = self.predict_noise(
                latents,
                ts,
                c_crossattn,
                inputs["scale"],
                c_concat,
                controlnet_images,
                controlnet_image_index,
            )
            latents = self.scheduler(noise_pred, ts, latents, self.num_inference_steps)

        # latents: (b c f h w)
        frames = self.vae_decode(latents)
        return frames
