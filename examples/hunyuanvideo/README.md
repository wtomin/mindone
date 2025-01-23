# HunyuanVideo: A Systematic Framework For Large Video Generation Model

Here we provide an efficient MindSpore implementation of [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), an open-source project that aims to foster large video generation model.

This repository is built on the models and code released by Tencent HunyuanVideo. We are grateful for their exceptional work and generous contribution to open source.


## 🎥 Demo

The following videos are generated based on MindSpore and Ascend 910*.



## 📑 Plan

- HunyuanVideo (Text-to-Video Model)
  - [x] Inference
  - [x] Training (SFT)
  - [ ] LoRA fine-tune
  - [ ] Web Demo (Gradio)
  - [ ] Multi-NPU parallel inference
- HunyuanVideo (Image-to-Video Model)
  - [ ] Training support
  - [ ] Inference


## 📜 Requirements

| mindspore | ascend driver | firmware | cann tookit/kernel |
| :---:     |   :---:       | :---:    | :---:              |
| 2.4.1     |  24.1.0     |7.35.23    |   8.0.RC3   |

```
pip install -r requirements.txt 
```

## 🧱 Prepare Pretrained Models

The details of download pretrained models are shown [here](ckpts/README.md).

Please download all checkpoints and convert them into MindSpore checkpoints following this [instruction](./ckpts/README.md).

## 📀 Inference

Currently, we support text-to-video generation with text embeddingp pre-computing. Please refer [Text embedding cache](#text-embedding-cache) to prepare the embedding before running the t2v generation.

``` bash
python sample_video.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --flow-reverse \
    --seed-type 'fixed' \
    --seed 1 \
    --save-path ./results \
    --dit-weight "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" \
    --text-embed-path /path/to/text_embeddings.npz \

```

Please run `python sample_video.py --help` to see more arguments.


## 🔑 Training 



## 3D VAE

### Reconstruction

To run a video reconstruction using the CausalVAE, please use the following command:
```bash
python hyvideo/rec_video.py \
  --video_path input_video.mp4 \
  --rec_path rec.mp4 \
  --height 360 \
  --width 640 \
  --num_frames 33 \
```
The reconstructed video is saved under `./samples/`.


### Evaluation

To evaluate VAE's PNSR, please download MCL_JCV dataset from this [URL](https://mcl.usc.edu/mcl-jcv-dataset/), and place the videos under `datasets/MCL_JCV`.

Now, to run video reconstruction on a video folder, please run:

```bash
python hyvideo/rec_video_folder.py \
  --real_video_dir datasets/MCL_JCV \
  --generated_video_dir datasets/MCL_JCV_generated \
  --height 360 \
  --width 640 \
  --num_frames 33 \
```

Afterwards, you can evaluate the PSNR via:
```bash
bash hyvideo/eval/scripts/cal_psnr.sh
```
## Embedding Cache

### Text embedding cache

```bash
cd hyvideo
python run_text_encoder.py
```

### Video embedding cache


## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://arxiv.org/abs/2412.03603), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.

