# Qwen-Image
## 🌌 Introduction
This is a MindSpore implementation of [Qwen-Image](https://arxiv.org/abs/2508.02324). Qwen-Image is an advanced image generation foundation model that is part of the Qwen series. It focuses on enhancing complex text rendering and precise image editing capabilities. 



## 📦 Requirements


<div align="center">

| MindSpore | Ascend Driver |  Firmware   | CANN toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
|   2.6.0   |  24.1.RC3     | 7.6.0.1.220 |  8.0.RC3.beta1     |

</div>

1. Install
   [CANN 8.0.RC3.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1)
   and MindSpore according to the [official instructions](https://www.mindspore.cn/install).
2. Install requirements
    ```shell
    pip install -r requirements.txt
    ```
3. Install mindone
    ```
    cd mindone
    pip install -e .
    ```
    Try `python -c "import mindone"`. If no error occurs, the installation is successful.

4. Make sure your transformers>=4.51.3 (Supporting Qwen2.5-VL)

## 🚀 Quick Start

```python
from mindone.diffusers import DiffusionPipeline
import mindspore as ms

model_name = "Qwen/Qwen-Image"

# Load the pipeline
pipe = DiffusionPipeline.from_pretrained(model_name, mindspore_dtype=ms.bfloat16)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''

negative_prompt = " " # Recommended if you don't use a negative prompt.


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example.png")
```
## 🤝 Acknowledgments

We would like to thank the contributors to the [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [transformers](https://github.com/huggingface/transformers), and [diffusers](https://github.com/huggingface/diffusers)repositories, for their open research and exploration.
