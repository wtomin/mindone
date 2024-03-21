# PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis

## Introduction of PixArt-α


## Get Started
In this tutorial, we will introduce how to run inference and finetuning experiments using MindONE.

### Environment Setup

```
pip install -r requirements.txt
```

`decord` is required for video generation. In case `decord` package is not available in your environment, try `pip install eva-decord`.
Instruction on ffmpeg and decord install on EulerOS:
```
1. install ffmpeg 4, referring to https://ffmpeg.org/releases
    wget wget https://ffmpeg.org/releases/ffmpeg-4.0.1.tar.bz2 --no-check-certificate
    tar -xvf ffmpeg-4.0.1.tar.bz2
    mv ffmpeg-4.0.1 ffmpeg
    cd ffmpeg
    ./configure --enable-shared         # --enable-shared is needed for sharing libavcodec with decord
    make -j 64
    make install
2. install decord, referring to https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source
    git clone --recursive https://github.com/dmlc/decord
    cd decord
    rm build && mkdir build && cd build
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
    make -j 64
    make install
    cd ../python
    python3 setup.py install --user
```

### Pretrained Checkpoints

We refer to the [official repository of Pixart-α](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main) for pretrained checkpoints downloading.

Specifically, please from the VAE checkpoint from this [url](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema), and convert this VAE checkpoint by running:
```bash
python tools/vae_converter.py --source path/to/vae/ckpt --target models/sd-vae-ft-ema.ckpt
```


## Sampling

## Training


# References
