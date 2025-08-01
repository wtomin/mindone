"""
Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/
pipelines/latent_diffusion/pipeline_latent_diffusion_superresolution.py.
"""

import inspect
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image

import mindspore as ms
from mindspore import mint

from ...models import UNet2DModel, VQModel
from ...schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ...utils import PIL_INTERPOLATION
from ...utils.mindspore_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

XLA_AVAILABLE = False


def preprocess(image):
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = ms.tensor(image)
    return 2.0 * image - 1.0


class LDMSuperResolutionPipeline(DiffusionPipeline):
    r"""
    A pipeline for image super-resolution using latent diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        vqvae ([`VQModel`]):
            Vector-quantized (VQ) model to encode and decode images to and from latent representations.
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`EulerDiscreteScheduler`],
            [`EulerAncestralDiscreteScheduler`], [`DPMSolverMultistepScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vqvae: VQModel,
        unet: UNet2DModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()
        self.register_modules(vqvae=vqvae, unet=unet, scheduler=scheduler)

    def __call__(
        self,
        image: Union[ms.Tensor, PIL.Image.Image] = None,
        batch_size: Optional[int] = 1,
        num_inference_steps: Optional[int] = 100,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`ms.Tensor` or `PIL.Image.Image`):
                `Image` or tensor representing an image batch to be used as the starting point for the process.
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only
                applies to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                A [`np.random.Generator`](https://numpy.org/doc/stable/reference/random/generator.html) to make
                generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> import requests
        >>> from PIL import Image
        >>> from io import BytesIO
        >>> from mindone.diffusers import LDMSuperResolutionPipeline
        >>> import mindspore as ms

        >>> # load model and scheduler
        >>> pipeline = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")

        >>> # let's download an  image
        >>> url = (
        ...     "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
        ... )
        >>> response = requests.get(url)
        >>> low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
        >>> low_res_img = low_res_img.resize((128, 128))

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1)[0][0]
        >>> # save image
        >>> upscaled_image.save("ldm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, ms.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError(f"`image` has to be of type `PIL.Image.Image` or `ms.Tensor` but is {type(image)}")

        if isinstance(image, PIL.Image.Image):
            image = preprocess(image)

        height, width = image.shape[-2:]

        # in_channels should be 6: 3 for latents, 3 for low resolution image
        latents_shape = (batch_size, self.unet.config.in_channels // 2, height, width)
        latents_dtype = next(self.unet.get_parameters()).dtype

        latents = randn_tensor(latents_shape, generator=generator, dtype=latents_dtype)

        image = image.to(dtype=latents_dtype)

        # set timesteps and move to the correct device
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature.
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(timesteps_tensor):
            # concat latents and low resolution image in the channel dimension.
            latents_input = mint.cat([latents, image], dim=1)
            latents_input = self.scheduler.scale_model_input(latents_input, t)
            # predict the noise residual
            noise_pred = self.unet(latents_input, t)[0]
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_kwargs)[0]

        # decode the image latents with the VQVAE
        image = self.vqvae.decode(latents)[0]
        image = mint.clamp(image, -1.0, 1.0)
        image = image / 2 + 0.5
        image = image.permute(0, 2, 3, 1).asnumpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
