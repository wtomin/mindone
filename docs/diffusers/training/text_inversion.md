 <!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Textual Inversion

[Textual Inversion](https://hf.co/papers/2208.01618) is a training technique for personalizing image generation models with just a few example images of what you want it to learn. This technique works by learning and updating the text embeddings (the new embeddings are tied to a special word you must use in the prompt) to match the example images you provide.

If you want to reduce memory footprint, you should try enabling the `gradient_checkpointing` and `mixed_precision` parameters in the training command. You can also reduce your memory footprint by using memory-efficient attention with [xFormers](../optimization/xformers.md).

This guide will explore the [textual_inversion.py](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/textual_inversion/textual_inversion.py) script to help you become more familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install .
```

Lastly, if you want to train a model on your own dataset, take a look at the [Create a dataset for training](create_dataset.md) guide to learn how to create a dataset that works with the training script.

!!! tip

    The following sections highlight parts of the training script that are important for understanding how to modify it, but it doesn't cover every aspect of the script in detail. If you're interested in learning more, feel free to read through the [script](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/textual_inversion/textual_inversion.py) and let us know if you have any questions or concerns.

## Script parameters

The training script has many parameters to help you tailor the training run to your needs. All of the parameters and their descriptions are listed in the [`parse_args()`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/textual_inversion/textual_inversion.py#L100) function. Where applicable, Diffusers provides default values for each parameter such as the training batch size and learning rate, but feel free to change these values in the training command if you'd like.

For example, to increase the number of gradient accumulation steps above the default value of 1:

```bash
python textual_inversion.py \
  --gradient_accumulation_steps=4
```

Some other basic and important parameters to specify include:

- `--pretrained_model_name_or_path`: the name of the model on the Hub or a local path to the pretrained model
- `--train_data_dir`: path to a folder containing the training dataset (example images)
- `--output_dir`: where to save the trained model
- `--push_to_hub`: whether to push the trained model to the Hub
- `--checkpointing_steps`: frequency of saving a checkpoint as the model trains; this is useful if for some reason training is interrupted, you can continue training from that checkpoint by adding `--resume_from_checkpoint` to your training command
- `--num_vectors`: the number of vectors to learn the embeddings with; increasing this parameter helps the model learn better but it comes with increased training costs
- `--placeholder_token`: the special word to tie the learned embeddings to (you must use the word in your prompt for inference)
- `--initializer_token`: a single-word that roughly describes the object or style you're trying to train on
- `--learnable_property`: whether you're training the model to learn a new "style" (for example, Van Gogh's painting style) or "object" (for example, your dog)

## Training script

Unlike some of the other training scripts, textual_inversion.py has a custom dataset class, [`TextualInversionDataset`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/textual_inversion/textual_inversion.py#L429) for creating a dataset. You can customize the image size, placeholder token, interpolation method, whether to crop the image, and more. If you need to change how the dataset is created, you can modify `TextualInversionDataset`.

Next, you'll find the dataset preprocessing code and training loop in the [`main()`](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/textual_inversion/textual_inversion.py#L514) function.

The script starts by loading the [tokenizer](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/textual_inversion/textual_inversion.py#L539), [scheduler and model](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/textual_inversion/textual_inversion.py#L545):

```py
# Load tokenizer
if args.tokenizer_name:
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
elif args.pretrained_model_name_or_path:
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)
vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
)
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
)
```

The special [placeholder token](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/textual_inversion/textual_inversion.py#L559) is added next to the tokenizer, and the embedding is readjusted to account for the new token.

Then, the script [creates a dataset](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/textual_inversion/textual_inversion.py#L635) from the `TextualInversionDataset`:

```py
train_dataset = TextualInversionDataset(
    data_root=args.train_data_dir,
    tokenizer=tokenizer,
    size=args.resolution,
    placeholder_token=" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids)),
    repeats=args.repeats,
    learnable_property=args.learnable_property,
    center_crop=args.center_crop,
    set="train",
)
train_dataloader = GeneratorDataset(
    train_dataset,
    column_names=["pixel_values", "input_ids"],
    shuffle=True,
    shard_id=args.rank,
    num_shards=args.world_size,
    num_parallel_workers=args.dataloader_num_workers,
).batch(
    batch_size=args.train_batch_size,
    num_parallel_workers=args.dataloader_num_workers,
)
```

Finally, the [training loop](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/textual_inversion/textual_inversion.py#L758) handles everything else from predicting the noisy residual to updating the embedding weights of the special placeholder token.

If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline.md) tutorial which breaks down the basic pattern of the denoising process.

## Launch the script

Once you've made all your changes or you're okay with the default configuration, you're ready to launch the training script! 🚀

For this guide, you'll download some images of a [cat toy](https://huggingface.co/datasets/diffusers/cat_toy_example) and store them in a directory. But remember, you can create and use your own dataset if you want (see the [Create a dataset for training](create_dataset.md) guide).

```py
from huggingface_hub import snapshot_download

local_dir = "./cat"
snapshot_download(
    "diffusers/cat_toy_example", local_dir=local_dir, repo_type="dataset", ignore_patterns=".gitattributes"
)
```

Set the environment variable `MODEL_NAME` to a model id on the Hub or a path to a local model, and `DATA_DIR`  to the path where you just downloaded the cat images to. The script creates and saves the following files to your repository:

- `learned_embeds.bin`: the learned embedding vectors corresponding to your example images
- `token_identifier.txt`: the special placeholder token
- `type_of_concept.txt`: the type of concept you're training on (either "object" or "style")

One more thing before you launch the script. If you're interested in following along with the training process, you can periodically save generated images as training progresses. Add the following parameters to the training command:

```bash
--validation_prompt="A <cat-toy> train"
--num_validation_images=4
--validation_steps=100
```

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATA_DIR="./cat"

python textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" \
  --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_cat" \
  --push_to_hub
```

After training is complete, you can use your newly trained model for inference like:

```py
from mindone.diffusers import StableDiffusionPipeline
import mindspore as ms

pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16)
pipeline.load_textual_inversion("sd-concepts-library/cat-toy")
image = pipeline("A <cat-toy> train", num_inference_steps=50)[0][0]
image.save("cat-train.png")
```

## Next steps

Congratulations on training your own Textual Inversion model! 🎉 To learn more about how to use your new model, the following guides may be helpful:

- Learn how to [load Textual Inversion embeddings](../using-diffusers/loading_adapters.md) and also use them as negative embeddings.
- Learn how to use [Textual Inversion](../using-diffusers/textual_inversion_inference.md) for inference with Stable Diffusion 1/2 and Stable Diffusion XL.
