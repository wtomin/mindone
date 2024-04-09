# Latte: Latent Diffusion Transformer for Video Generation

## 1. Introduction of Latte

Latte [<a href="#references">1</a>] is a novel Latent Diffusion Transformer designed for video generation. It is built based on DiT (a diffusion transformer model for image generation). For introduction of DiT [<a href="#references">2</a>], please refer to [README of DiT](../dit/README.md).

Latte first uses a VAE (Variational AutoEncoder) to compress the video data into a latent space, and then extracts spatial-temporal tokens given the latent codes. Similar to DiT, it stacks multiple transformer blocks to model the video diffusion in the latent space. How to design the spatial and temporal blocks becomes a major question.

Through experiments and analysis, they found the best practice is structure (a) in the image below. It stacks spatial blocks and temporal blocks alternately to model spatial attentions and temporal attentions in turns.


<p align="center">
  <img src="https://raw.githubusercontent.com/Vchitect/Latte/9ededbe590a5439b6e7013d00fbe30e6c9b674b8/visuals/architecture.svg" width=550 />
</p>
<p align="center">
  <em> Figure 1. The Structure of Latte and Latte transformer blocks. [<a href="#references">1</a>] </em>
</p>

Similar to DiT, Latte supports un-conditional video generation and class-labels-conditioned video generation. In addition, it supports to generate videos given text captions.


## 2. Get Started
In this tutorial, we will introduce how to run inference and training experiments using MindONE.

This tutorial includes:
- [x] Pretrained checkpoints conversion;
- [x] Un-conditional video sampling with pretrained Latte checkpoints;
- [x] Training un-conditional Latte on Sky TimeLapse dataset: support training (1) with videos ; and (2) with embedding cache;
- [x] Mixed Precision: support (1) Float16; (2) BFloat16 (set patch_embedder to "linear");
- [x] Standalone training and distributed training.
- [ ] Text-to-Video Latte inference and training.

### 2.1 Environment Setup

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

### 2.2 Pretrained Checkpoints

We refer to the [official repository of Latte](https://github.com/Vchitect/Latte/tree/main) for pretrained checkpoints downloading. The pretrained checkpoint files trained on FaceForensics, SkyTimelapse, Taichi-HD and UCF101 (256x256) can be downloaded from [huggingface](https://huggingface.co/maxin-cn/Latte/tree/main).

After downloading the `{}.pt` file, please place it under the `models/` folder, and then run `tools/latte_converter.py`. For example, to convert `models/skytimelapse.pt`, you can run:
```bash
python tools/latte_converter.py --source models/skytimelapse.pt --target models/skytimelapse.ckpt
```

Please also download the VAE checkpoint from [huggingface/stabilityai.co](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main), and convert this VAE checkpoint by running:
```bash
python tools/vae_converter.py --source path/to/vae/ckpt --target models/sd-vae-ft-mse.ckpt
```

## 3. Sampling

For example, to run inference of `skytimelapse.ckpt` model with the `256x256` image size on Ascend devices, you can use:
```bash
python sample.py -c configs/inference/sky.yaml
```

Some of the generated results are shown here:
<table class="center">
    <tr style="line-height: 0">
    <td width=33% style="border: none; text-align: center">Example 1</td>
    <td width=33% style="border: none; text-align: center">Example 2</td>
    <td width=33% style="border: none; text-align: center">Example 3</td>
    </tr>
    <tr>
    <td width=33% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/latte/sky/generated-0.gif" style="width:100%"></td>
    <td width=33% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/latte/sky/generated-1.gif" style="width:100%"></td>
    <td width=33% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/latte/sky/generated-2.gif" style="width:100%"></td>
    </tr>
</table>
<p align="center">
  <em> Figure 2. The generated videos of the pretrained model converted from the torch checkpoint. </em>
</p>

## 4. Training (Unconditional Video Generation)

### 4.1 Training With Videos

Now, we support training Latte model on the Sky Timelapse dataset, a video dataset which can be downloaded from https://github.com/weixiong-ur/mdgan.

After uncompressing the downloaded file, you will get a folder named `sky_train/` which contains all training video frames. The folder structure is similar to:
```
sky_train/
├── video_name_0/
|   ├── frame_id_0.jpg
|   ├── frame_id_0.jpg
|   └── ...
├── video_name_1/
└── ...
```

First, edit the configuration file `configs/training/datasets/sky_video_uncond.yaml`. Change the `data_folder` from `""` to the absolute path to `sky_train/`.

Then, you can start standalone training on Ascend devices using:
```bash
python train.py -c configs/training/sky_video_uncond.yaml
```
To start training on GPU devices, simply append `--device_target GPU` to the command above.

The default training configuration is to train Latte model from scratch. The batch size is $5$, and the number of epochs is $3000$, which corresponds to around 900k steps. The learning rate is a constant value $1e^{-4}$. The model is trained under mixed precision mode. The default AMP level is `O2`. See more details in `configs/training/sky_video_uncond.yaml`.

To accelerate the training speed, we use `dataset_sink_mode: True` in the configuration file by default. You can also set `enable_flash_attention: True` to further accelerate the training speed.

After training, the checkpoints are saved under `output_dir/ckpt/`. To run inference with the checkpoint, please change `checkpoint` in `configs/inference/sky_uncond.yaml` to the path of the checkpoint, and then run `python sample.py -c configs/inference/sky_uncond.yaml`.

The number of epochs is set to a large number to ensure convergence. You can terminate training whenever it is ready. For example, we took the checkpoint which was trained for $1700$ epochs (about $500k$ steps) and ran inference with it. Here are some examples generated:
<table class="center">
    <tr style="line-height: 0">
    <td width=33% style="border: none; text-align: center">Example 1</td>
    <td width=33% style="border: none; text-align: center">Example 2</td>
    <td width=33% style="border: none; text-align: center">Example 3</td>
    </tr>
    <tr>
    <td width=33% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/latte/sky/epochs-1700-generated-0.gif" style="width:100%"></td>
    <td width=33% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/latte/sky/epochs-1700-generated-1.gif" style="width:100%"></td>
    <td width=33% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/latte/sky/epochs-1700-generated-2.gif" style="width:100%"></td>
    </tr>
</table>
<p align="center">
  <em> Figure 3. The generated videos of the Latte model trained for 1700 epochs (about 500k steps). </em>
</p>

### 4.2 Training With Embedding Cache

We can accelerate the training speed by caching the embeddings of the dataset before running the training script. This takes three steps:

- **Step 1**: Cache the embedding into a cache folder. See the following example about how to cache the embeddings. This step can take a bit long time.

<details onclose>

To cache embeddings for Sky Timelapse dataset, first, please make sure the `data_path` in `configs/training/sky_video_uncond.yaml` is set correctly to the folder named `sky_train/`.

Then you can start saving the embeddings using:
```bash
python tools/embedding_cache.py --config configs/training/sky_video_uncond.yaml --cache_folder path/to/cache/folder --cache_file_type numpy
```
You can also change `cache_file_type` to `mindrecord` to save embeddings in `.mindrecord` files.

In general, we recommend to use `mindrecord` file type because it is supported by `MindDataset` which can better accelerates data loading. However, Sky Timelapse dataset has extra long videos. Using `mindrecord` file to cache embedding increases the risk of exceeding the maximum page size of the MindRecord writer. Therefore, we recommend to use `numpy` file.

The embedding caching process can take a while depending on the size of the video dataset. Some exceptions maybe thrown during the process. If unexpected exceptions are thrown, the program will be stoped and the embedding caching writer's status will be printed on the screen:
```bash
Start Video Index: 0. # the start of video index to be processed
Saving Attempts: 0: save 120 videos, failed 0 videos. # the number of saved video files
```
In this case, you can resume the embedding cache from the video indexed at $120$ (index starts from 0). Simply append `--resume_cache_index 120`, and run `python tools/embedding_cache.py`. It will start caching the embedding from the $120^{th}$ video and save the embeddings without overwriting the existing files.

To check more usages, please use `python tools/embedding_cache.py -h`.

</details>

- **Step 2**: Change the dataset configuration file's `data_folder` to the current cache folder path.

After the embeddings have been cached, edit `configs/training/datasets/sky_numpy_uncond.yaml`, and change the `data_folder` to the folder where the cached embeddings are stored in.

- **Step 3**: Run the training script.

You can start training on the cached embedding dataset of Sky TimeLapse using:
```bash
python train.py -c configs/training/sky_numpy_uncond.yaml
```

Note that in `sky_numpy_uncond.yaml`, we use a large number of frames $128$ and a smaller sample stride $1$, which are different from the settings in `sky_video_uncond.yaml` (num_frames=16 and stride=3)· Embedding caching allows us to train Latte to generate more frames with a larger frame rate.

Due to the memory limit, we set the local batch size to $1$ and use a gradient accumulation steps $4$. The number of epochs is $3000$, which corresponds to around 800k steps. You can terminate the training when it's ready.

In case of OOM, please set `enable_flash_attention: True` in the `configs/training/sky_numpy_uncond.yaml`. It can reduce the memory cost and also accelerate the training speed.

### 4.3 Distributed Training

Taking the 4-card distributed training as an example, you can start the distributed training using:
```bash
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
mpirun -n 4 python train.py \
    -c path/to/configuration/file \
    --gradient_accumulation_steps 1 \
    --use_parallel True
```
where the configuration file can be selected from the `.yaml` files in `configs/training/` folder. By setting `gradient_accumulation_steps` to 1, and loading data into 4 cards in parallel, we have a global batch size which equals to `4 x local_batch_size`.

If you have the rank table of Ascend devices, you can take `scripts/run_distributed_sky_numpy_uncond.sh` as a reference, and start the 4-card distributed training using:
```bash
bash scripts/run_distributed_sky_numpy_uncond.sh path/to/rank/table 0 4
```

The first number `0` indicates the start index of the training devices, and the second number `4` indicates the total number of distributed processes you want to launch.


### 4.4 Performance

The training speed of the unconditional video experiments with `256x256` image size is summarized in the following table:

| Cards | Recompute | Dataset Sink mode | Embedding Cache|Train. imgs/s |
| ---   | ---       | ---               | ---          |   ---          |
| 1     | OFF       | ON                | OFF          | 62.3           |
| 1     | ON        | ON                | ON           | 93.6           |
| 4     | ON        | ON                | ON           | 368.3          |

## 5. Training (Text-To-Video Generation, Experimental)

Here we provide an experimental feature of training a text-to-video latte model using T5 model as the text encoder.

### 5.1 Text Encoder

Please download the cache folder of the `t5-v1_1-xxl` model from HuggingFace [URL](https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main), and place it under `models/`. The t5 cache folder looks like:

```bash
models/t5-v1_1-xxl/
├── config.json
├── pytorch_model-00001-of-00002.bin
├── pytorch_model-00002-of-00002.bin
├── pytorch_model.bin.index.json
├── special_tokens_map.json
├── spiece.model
└── tokenizer_config.json
```

Then, you can convert the T5 Torch checkpoint to a MindSpore checkpoint using:

```bash
python tools/t5_converter.py --source models/t5-v1_1-xxl/pytorch_model-00001-of-00002.bin  models/t5-v1_1-xxl/pytorch_model-00002-of-00002.bin --target models/t5-v1_1-xxl/model.ckpt
```

### 5.2 Dataset Preparation

In general, we need to prepare video-caption pairs to train a text-to-video latte model. As for the dataset format, we provide a toy CSV dataset which can be downloaded by:
```bash
bash scripts/download_toy_csvdataset.sh
```
Afterwards, the dataset is downloaded into `imagenet_samples/videos/`, like this:
```bash
imagenet_samples/videos/
├── n01644373_tree_frog_31.mp4
├── n02085936_Maltese_dog_153.mp4
├── n04146614_school_bus_779.mp4
└── video_caption.csv
```

Checking the CSV file, you will find:

| video | caption | class|
| ---   | ---     | ---  |
| n01644373_tree_frog_31.mp4 | "a tree frog perches on a branch in the tropical rainforest" | 3|
| n02085936_Maltese_dog_153.mp4 | "a Maltese dog squats beside the bed and looks into the camera" | 153|
...

The two columns `video` and `caption` are essential for text-to-video model training, while the third column `class` is optional. `class` is only required for class-conditioned model training.

### 5.3 Training with Videos

You can start standalone training on Ascend devices using:
```bash
python train_t2v_exp.py -c configs/training/csv_video_text.yaml
```
It is highly risky of OOM to train with raw videos and captions. Therefore, we recommend to extract embeddings before running training experiments.


### 5.4 Training with Embedding Cache

Similar to Sec [4.2](#42-training-with-embedding-cache), please extract the visual latent embeddings and the text embeddings as well as the token masks beforehand.

You can start saving the embeddings using:
```bash
python tools/embedding_cache.py --config configs/training/csv_video_text.yaml --cache_folder path/to/cache/folder --cache_file_type numpy
```

Please take the example in [4.2](#42-training-with-embedding-cache) as a reference to use `tools/embedding_cache.py`.

After the embeddings are saved in the cache folder, you can set the `data_folder` to the cache folder in `configs/training/datasets/csv_numpy_text.yaml`. Then, you can start the training using:
```bash
python train.py -c configs/training/csv_numpy_text.yaml
```

The same applies to `csv_pkl_text.yaml` if you want to save the embeddings in `.pkl` files.

Please take [4.2](#42-training-with-embedding-cache) as a reference for distributed training. After the checkpoints being saved, please edit the `checkpoint` in the file `configs/inference/csv_text.yaml`, and then run inference using:
```bash
python sample_t2v_exp.py --config configs/inference/csv_text.yaml
```

### 5.5 Performance


The training speed of the text-conditioned video experiments with `512x512` frame size is to be released soon.


# References

[1] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, Yu Qiao: Latte: Latent Diffusion Transformer for Video Generation. CoRR abs/2401.03048 (2024)

[2] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023
