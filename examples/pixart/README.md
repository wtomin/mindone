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

Specifically, please download the VAE checkpoint from this [url](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema), and convert this VAE checkpoint by running:
```bash
python tools/vae_converter.py --source path/to/vae/ckpt --target models/sd-vae-ft-ema.ckpt
```

Pixart-α uses `t5-v1_1-xxl` model for encoding text embeddings. You can download this folder from [URL](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl), and place it under `models`.

After that, please run the checkpoint conversion with:

```bash
python tools/t5_converter.py --source models/t5-v1_1-xxl/pytorch_model-00001-of-00002.bin models/t5-v1_1-xxl/pytorch_model-00002-of-00002.bin --target models/t5-v1_1-xxl/model.ckpt
```
This will convert torch weights saved in `.bin` files to mindspore weights saved in `.ckpt` file.

So far, you have prepared all the checkpoints and files needed to run the sampling:
```bash
models/
├── sd-vae-ft-ema.ckpt
└── t5-v1_1-xxl
    ├── config.json
    ├── model.ckpt
    ├── pytorch_model.bin.index.json
    ├── special_tokens_map.json
    ├── spiece.model
    └── tokenizer_config.json
```

## Sampling

You can run text-to-image sampling using `sample.py`. Given a txt file `asset/samples.txt`, which contains lines of captions, you can run sampling with:
```bash
python sample.py --txt_file asset/samples.txt
```

It will save the generated images under `samples/{time-stamps}/`.
## Training


# References
