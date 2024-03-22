# PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis

## Introduction of PixArt-α


## Get Started
In this tutorial, we will introduce how to run inference and finetuning experiments using MindONE.

### Environment Setup

```
pip install -r requirements.txt
```

### Pretrained Checkpoints

We refer to the [official repository of Pixart-α](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main) for pretrained checkpoints downloading.

Specifically, please from the VAE checkpoint from this [url](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema), and convert this VAE checkpoint by running:
```bash
python tools/vae_converter.py --source path/to/vae/ckpt --target models/sd-vae-ft-ema.ckpt
```

Next, please download t5 model from this [url](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl), and place the `t5-v1_1-xxl` folder under `models/`.


## Sampling

## Training


# References
