<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DeepFloyd IF

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

## Overview

DeepFloyd IF is a novel state-of-the-art open-source text-to-image model with a high degree of photorealism and language understanding.
The model is a modular composed of a frozen text encoder and three cascaded pixel diffusion modules:

- Stage 1: a base model that generates 64x64 px image based on text prompt,
- Stage 2: a 64x64 px => 256x256 px super-resolution model, and
- Stage 3: a 256x256 px => 1024x1024 px super-resolution model

Stage 1 and Stage 2 utilize a frozen text encoder based on the T5 transformer to extract text embeddings, which are then fed into a UNet architecture enhanced with cross-attention and attention pooling.
Stage 3 is [Stability AI's x4 Upscaling model](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler).
The result is a highly efficient model that outperforms current state-of-the-art models, achieving a zero-shot FID score of 6.66 on the COCO dataset.
Our work underscores the potential of larger UNet architectures in the first stage of cascaded diffusion models and depicts a promising future for text-to-image synthesis.

## Usage

Before you can use IF, you need to accept its usage conditions. To do so:
1. Make sure to have a [Hugging Face account](https://huggingface.co/join) and be logged in.
2. Accept the license on the model card of [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0). Accepting the license on the stage I model card will auto accept for the other IF models.
3. Make sure to login locally. Install `huggingface_hub`:
```sh
pip install huggingface_hub --upgrade
```

run the login function in a Python shell:

```py
from huggingface_hub import login

login()
```

and enter your [Hugging Face Hub access token](https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens).

The following sections give more in-detail examples of how to use IF. Specifically:

- [DeepFloyd IF](#deepfloyd-if)
  - [Overview](#overview)
  - [Usage](#usage)
    - [Text-to-Image Generation](#text-to-image-generation)
    - [Text Guided Image-to-Image Generation](#text-guided-image-to-image-generation)
    - [Text Guided Inpainting Generation](#text-guided-inpainting-generation)
    - [Converting between different pipelines](#converting-between-different-pipelines)
    - [Optimizing for speed](#optimizing-for-speed)
  - [Available Pipelines:](#available-pipelines)

**Available checkpoints**
- *Stage-1*
  - [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)
  - [DeepFloyd/IF-I-L-v1.0](https://huggingface.co/DeepFloyd/IF-I-L-v1.0)
  - [DeepFloyd/IF-I-M-v1.0](https://huggingface.co/DeepFloyd/IF-I-M-v1.0)

- *Stage-2*
  - [DeepFloyd/IF-II-L-v1.0](https://huggingface.co/DeepFloyd/IF-II-L-v1.0)
  - [DeepFloyd/IF-II-M-v1.0](https://huggingface.co/DeepFloyd/IF-II-M-v1.0)

- *Stage-3*
  - [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)

### Text-to-Image Generation

```python
from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import pt_to_pil, make_image_grid
import mindspore as ms
import numpy as np

# stage 1
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", mindspore_dtype=ms.float16)

# stage 2
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", mindspore_dtype=ms.float16
)

# stage 3
safety_modules = {
    "feature_extractor": stage_1.feature_extractor,
    "safety_checker": stage_1.safety_checker,
    "watermarker": stage_1.watermarker,
}
stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, mindspore_dtype=ms.float16
)

prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
generator = np.random.Generator(np.random.PCG64(1))

# text embeds
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

# stage 1
stage_1_output = stage_1(
    prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="ms"
)[0]
#pt_to_pil(stage_1_output)[0].save("./if_stage_I.png")

# stage 2
stage_2_output = stage_2(
    image=stage_1_output,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="ms",
)[0]
#pt_to_pil(stage_2_output)[0].save("./if_stage_II.png")

# stage 3
stage_3_output = stage_3(prompt=prompt, image=stage_2_output, noise_level=100, generator=generator)[0]
#stage_3_output[0].save("./if_stage_III.png")
make_image_grid([pt_to_pil(stage_1_output)[0], pt_to_pil(stage_2_output)[0], stage_3_output[0]], cols=1, rows=3)
```

### Text Guided Image-to-Image Generation

The same IF model weights can be used for text-guided image-to-image translation or image variation.
In this case just make sure to load the weights using the [`IFImg2ImgPipeline`](deepfloyd_if.md#mindone.diffusers.IFImg2ImgPipeline) and [`IFImg2ImgSuperResolutionPipeline`](deepfloyd_if.md#mindone.diffusers.IFImg2ImgSuperResolutionPipeline) pipelines.

**Note**: You can also directly move the weights of the text-to-image pipelines to the image-to-image pipelines
without loading them twice by making use of the [`diffusionpipeline.components`](overview.md#mindone.diffusers.DiffusionPipeline) argument as explained [here](#converting-between-different-pipelines).

```python
from mindone.diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline, DiffusionPipeline
from mindone.diffusers.utils import pt_to_pil, load_image, make_image_grid
import mindspore as ms
import numpy as np

# download image
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
original_image = load_image(url)
original_image = original_image.resize((768, 512))

# stage 1
stage_1 = IFImg2ImgPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", mindspore_dtype=ms.float16)

# stage 2
stage_2 = IFImg2ImgSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", mindspore_dtype=ms.float16
)

# stage 3
safety_modules = {
    "feature_extractor": stage_1.feature_extractor,
    "safety_checker": stage_1.safety_checker,
    "watermarker": stage_1.watermarker,
}
stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, mindspore_dtype=ms.float16
)

prompt = "A fantasy landscape in style minecraft"
generator = np.random.Generator(np.random.PCG64(1))

# text embeds
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

# stage 1
stage_1_output = stage_1(
    image=original_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="ms",
)[0]
#pt_to_pil(stage_1_output)[0].save("./if_stage_I.png")

# stage 2
stage_2_output = stage_2(
    image=stage_1_output,
    original_image=original_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="ms",
)[0]
#pt_to_pil(stage_2_output)[0].save("./if_stage_II.png")

# stage 3
stage_3_output = stage_3(prompt=prompt, image=stage_2_output, generator=generator, noise_level=100)[0]
#stage_3_output[0].save("./if_stage_III.png")
make_image_grid([original_image, pt_to_pil(stage_1_output)[0], pt_to_pil(stage_2_output)[0], stage_3_output[0]], cols=1, rows=4)
```

### Text Guided Inpainting Generation

The same IF model weights can be used for text-guided image-to-image translation or image variation.
In this case just make sure to load the weights using the [`IFInpaintingPipeline`](deepfloyd_if.md#mindone.diffusers.IFInpaintingPipeline) and [`IFInpaintingSuperResolutionPipeline`](deepfloyd_if.md#mindone.diffusers.IFInpaintingSuperResolutionPipeline) pipelines.

**Note**: You can also directly move the weights of the text-to-image pipelines to the image-to-image pipelines
without loading them twice by making use of the [`DiffusionPipeline.components()`](overview.md#mindone.diffusers.DiffusionPipeline) function as explained [here](#converting-between-different-pipelines).

```python
from mindone.diffusers import IFInpaintingPipeline, IFInpaintingSuperResolutionPipeline, DiffusionPipeline
from mindone.diffusers.utils import pt_to_pil, load_image, make_image_grid
import mindspore as ms
import numpy as np

# download image
url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/person.png"
original_image = load_image(url)

# download mask
url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/glasses_mask.png"
mask_image = load_image(url)

# stage 1
stage_1 = IFInpaintingPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", mindspore_dtype=ms.float16)

# stage 2
stage_2 = IFInpaintingSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", mindspore_dtype=ms.float16
)

# stage 3
safety_modules = {
    "feature_extractor": stage_1.feature_extractor,
    "safety_checker": stage_1.safety_checker,
    "watermarker": stage_1.watermarker,
}
stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, mindspore_dtype=ms.float16
)

prompt = "blue sunglasses"
generator = np.random.Generator(np.random.PCG64(1))

# text embeds
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

# stage 1
stage_1_output = stage_1(
    image=original_image,
    mask_image=mask_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="ms",
)[0]
#pt_to_pil(stage_1_output)[0].save("./if_stage_I.png")

# stage 2
stage_2_output = stage_2(
    image=stage_1_output,
    original_image=original_image,
    mask_image=mask_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="ms",
)[0]
#pt_to_pil(stage_1_output)[0].save("./if_stage_II.png")

# stage 3
stage_3_output = stage_3(prompt=prompt, image=stage_2_output, generator=generator, noise_level=100)[0]
#stage_3_output[0].save("./if_stage_III.png")
make_image_grid([original_image, mask_image, pt_to_pil(stage_1_output)[0], pt_to_pil(stage_2_output)[0], stage_3_output[0]], cols=1, rows=5)
```

### Converting between different pipelines

In addition to being loaded with `from_pretrained`, Pipelines can also be loaded directly from each other.

```python
from mindone.diffusers import IFPipeline, IFSuperResolutionPipeline

pipe_1 = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0")
pipe_2 = IFSuperResolutionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0")


from mindone.diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline

pipe_1 = IFImg2ImgPipeline(**pipe_1.components)
pipe_2 = IFImg2ImgSuperResolutionPipeline(**pipe_2.components)


from mindone.diffusers import IFInpaintingPipeline, IFInpaintingSuperResolutionPipeline

pipe_1 = IFInpaintingPipeline(**pipe_1.components)
pipe_2 = IFInpaintingSuperResolutionPipeline(**pipe_2.components)
```

### Optimizing for speed

The simplest optimization to run IF faster is to move all model components to the GPU.

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", mindspore_dtype=ms.float16)
```

You can also run the diffusion process for a shorter number of timesteps.

This can either be done with the `num_inference_steps` argument:

```py
pipe("<prompt>", num_inference_steps=30)
```

Or with the `timesteps` argument:

```py
from mindone.diffusers.pipelines.deepfloyd_if import fast27_timesteps

pipe("<prompt>", timesteps=fast27_timesteps)
```

When doing image variation or inpainting, you can also decrease the number of timesteps
with the strength argument. The strength argument is the amount of noise to add to the input image which also determines how many steps to run in the denoising process.
A smaller number will vary the image less but run faster.

```py
pipe = IFImg2ImgPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", mindspore_dtype=ms.float16)

image = pipe(image=image, prompt="<prompt>", strength=0.3).images
```

## Available Pipelines:

| Pipeline | Tasks |
|---|---|
| [pipeline_if.py](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/pipelines/deepfloyd_if/pipeline_if.py) | *Text-to-Image Generation* | - |
| [pipeline_if_superresolution.py](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/pipelines/deepfloyd_if/pipeline_if_superresolution.py) | *Text-to-Image Generation* |
| [pipeline_if_img2img.py](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img.py) | *Image-to-Image Generation* |
| [pipeline_if_img2img_superresolution.py](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img_superresolution.py) | *Image-to-Image Generation* |
| [pipeline_if_inpainting.py](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting.py) | *Image-to-Image Generation* |
| [pipeline_if_inpainting_superresolution.py](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting_superresolution.py) | *Image-to-Image Generation* |

::: mindone.diffusers.IFPipeline

::: mindone.diffusers.IFSuperResolutionPipeline

::: mindone.diffusers.IFImg2ImgPipeline

::: mindone.diffusers.IFImg2ImgSuperResolutionPipeline

::: mindone.diffusers.IFInpaintingPipeline

::: mindone.diffusers.IFInpaintingSuperResolutionPipeline
