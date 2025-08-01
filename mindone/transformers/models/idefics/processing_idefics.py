# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for IDEFICS.
"""

from typing import Callable, Optional, Union
from urllib.parse import urlparse

from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils.deprecation import deprecate_kwarg

import mindspore as ms
from mindspore import mint

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
    _validate_images_text_input_order,
)

IMAGE_TOKEN = "<image>"


class IdeficsImagesKwargs(ImagesKwargs, total=False):
    transform: Optional[Callable]
    image_size: Optional[dict[str, int]]
    image_mean: Optional[Union[float, list[float]]]
    image_std: Optional[Union[float, list[float]]]


class IdeficsTextKwargs(TextKwargs, total=False):
    add_eos_token: Optional[bool]
    add_end_of_utterance_token: Optional[bool]


class IdeficsProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: IdeficsTextKwargs
    images_kwargs: IdeficsImagesKwargs
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": False,
            "padding": "longest",
            "add_eos_token": False,
        },
        "images_kwargs": {},
        "common_kwargs": {"return_tensors": "ms"},
    }


# copied from m4.training.packing
def incremental_to_binary_attention_mask(incremental_mask, return_tensors, num_classes=-1):
    # Set elements >= num_classes to -1
    if num_classes != -1:
        if return_tensors == "ms":
            incremental_mask[incremental_mask >= num_classes] = -1

    # Create mask for negative values
    if return_tensors == "ms":
        negatives = incremental_mask == -1
        incremental_mask[negatives] = 0
        attn_mask = mint.nn.functional.one_hot(incremental_mask, num_classes=num_classes)
        attn_mask[negatives, :] = 0

    return attn_mask


# copied from m4.training.packing
def image_attention_mask_for_packed_input_ids(input_ids, tokenizer, return_tensors):
    if return_tensors == "ms":
        return image_attention_mask_for_packed_input_ids_ms(input_ids, tokenizer)


def image_attention_mask_for_packed_input_ids_ms(input_ids, tokenizer):
    image_attention_mask = mint.full_like(input_ids, fill_value=-1)
    next_image_attention_mask = mint.full_like(input_ids, fill_value=-1)
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    eod_token_id = tokenizer.eos_token_id
    for batch_idx in range(input_ids.shape[0]):
        count = -1
        seen_eod = False
        for idx, token_id in enumerate(input_ids[batch_idx]):
            if token_id == image_token_id:
                count += 1
                image_attention_mask[batch_idx][idx] = count
                seen_eod = False
            else:
                image_attention_mask[batch_idx][idx] = count

            if seen_eod:
                image_attention_mask[batch_idx][idx] = -1

            if token_id == eod_token_id:
                seen_eod = True

    for batch_idx in range(input_ids.shape[0]):
        count = -1
        seen_eod = False
        for idx in range(input_ids[batch_idx].shape[0] - 1, -1, -1):
            token_id = input_ids[batch_idx][idx]
            if token_id == image_token_id:
                count += 1
                next_image_attention_mask[batch_idx][idx] = count
                seen_eod = False
            else:
                next_image_attention_mask[batch_idx][idx] = count

            if token_id == eod_token_id:
                seen_eod = True

            if seen_eod:
                next_image_attention_mask[batch_idx][idx] = -1

        non_negative_indices = next_image_attention_mask[batch_idx] != -1
        next_image_attention_mask[batch_idx][non_negative_indices] -= count
        next_image_attention_mask[batch_idx][non_negative_indices] *= -1

    return image_attention_mask, next_image_attention_mask


def is_url(string):
    """Checks if the passed string contains a valid url and nothing else. e.g. if space is included it's immediately
    invalidated the url"""
    if " " in string:
        return False
    result = urlparse(string)
    return all([result.scheme, result.netloc])


class IdeficsProcessor(ProcessorMixin):
    r"""
    Constructs a IDEFICS processor which wraps a LLama tokenizer and IDEFICS image processor into a single processor.

    [`IdeficsProcessor`] offers all the functionalities of [`IdeficsImageProcessor`] and [`LlamaTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`IdeficsImageProcessor`):
            An instance of [`IdeficsImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizerFast`):
            An instance of [`LlamaTokenizerFast`]. The tokenizer is a required input.
        image_size (`int`, *optional*, defaults to 224):
            Image size (assuming a square image)
        add_end_of_utterance_token (`str`, *optional*):
            The string representation of token representing end of utterance
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "IdeficsImageProcessor"
    tokenizer_class = "LlamaTokenizerFast"

    def __init__(self, image_processor, tokenizer=None, image_size=224, add_end_of_utterance_token=None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self.image_token_id = (
            tokenizer.image_token_id
            if hasattr(tokenizer, "image_token")
            else tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        )

        self.default_image_dims = (
            self.image_processor.image_num_channels,
            self.image_processor.image_size,
            self.image_processor.image_size,
        )

        self.tokenizer_was_trained_with_end_of_utterance_token = (
            True
            if "<end_of_utterance>" in self.tokenizer.special_tokens_map.get("additional_special_tokens", [])
            else False
        )

    @deprecate_kwarg(old_name="prompts", version="5.0.0", new_name="text", raise_if_both_names=True)
    def __call__(
        self,
        images: Union[ImageInput, list[ImageInput], str, list[str], list[list[str]]] = None,
        text: Union[
            TextInput,
            PreTokenizedInput,
            list[TextInput],
            list[PreTokenizedInput],
            list[list[TextInput]],
            list[list[PreTokenizedInput]],
        ] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[IdeficsProcessorKwargs],
    ) -> BatchFeature:
        """This method takes batched or non-batched prompts made of text and images and converts them into prompts that
        the model was trained on and prepares the image pixel values for the model to process.

        Args:
            images (`Union[ImageInput, list[ImageInput], str, list[str], list[list[str]]]`):
                either a single image or a batched list of images - can be passed in when text contains only text prompts,
                in order to use the image-text-to-text behavior.
            text (`Union[list[TextInput], [list[list[TextInput]]]]`):
                either a single prompt or a batched list of prompts - see the detailed description immediately after
                the end of the arguments doc section.
            return_tensors (`str` or `TensorType`, *optional*, defaults to `TensorType.MINDSPIRE`):
                The type of tensors to return. Can be one of:
                    - `TensorType.MINDSPIRE` or `'ms'`: Return a batch of type `ms.Tensor`.

        Returns:
            a dict with entries: `input_ids`, `attention_mask`, `pixel_values`, `image_attention_mask` which can be
            directly passed to `model.generate`

        Detailed explanation:

        Each entry in `text` is either a text to be passed as is or an image that will be processed.

        An image can be either an image object (`PIL.Image`) or a url from which the image can be retrieved.

        When the processor encounters an image it'll inject `<fake_token_around_image><image><fake_token_around_image>`
        entry into the prompt.

        Example:

        ```python
        checkpoint = "HuggingFaceM4/idefics-9b"
        processor = AutoProcessor.from_pretrained(checkpoint)
        url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
        img = processor.image_processor.fetch_images([url])[0]

        prompts = [
            "User:",
            img,
            "Describe this image.\nAssistant: An image of two kittens in grass.\n",
            "User:",
            "https://hips.hearstapps.com/hmg-prod/images/dog-puns-1581708208.jpg",
            "Describe this image.\nAssistant:",
        ]

        inputs = processor(text=prompts, return_tensors="ms")
        generated_ids = model.generate(**inputs, max_length=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```

        In this example the `prompts` will be converted into:

        ```
        <s>User:<fake_token_around_image><image><fake_token_around_image>Describe this image.
        Assistant: An image of two kittens in grass.
        User:<fake_token_around_image><image><fake_token_around_image>Describe this image.
        Assistant:'
        ```

        and the two images will be massaged using [`IdeficsImageProcessor.__call__`] method and placed inside the
        `pixel_values` dict entry of the return value.

        This example also exemplifies that images can be passed as objects or as text urls. It can be seen that the
        first image is passed as object and the second one as a url.

        To do training do:

        ```python
        import mindspore.dataset.transforms as transforms
        import mindspore.dataset.vision as vision
        image_transform = transforms.Compose(
            [
                vision.RandomResizedCrop(
                    (w, h), scale=(0.9, 1.0), interpolation=vision.Inter.BICUBIC
                ),
                vision.ToTensor(),
                vision.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )
        inputs = processor(text=prompts, transform=image_transform, return_tensors="ms")
        ```

        In order to help debug prompt generation enable `debug=True` which will show you what's happening.

        """
        if images is None and text is None:
            raise ValueError("You need to specify either `text` or `images` and `text`.")
        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        if images is None:
            # assuming the user wants to use the old behavior with prompts as the only argument
            prompts = text
        elif text is not None:
            # Assuming image-text-to-text behavior:
            # Check if batched images are provided
            if not isinstance(images, (list, tuple)):
                images = [images]
            if isinstance(text, str):
                text = [text]
            # Check if batched images and text are in the correct format
            if isinstance(text, (list, tuple)) and len(text) != len(images):
                raise ValueError(
                    "When providing both images and text arguments, the number of text prompts should be the same as the number of images."
                    "If you want to have several images per prompt, images should be nested as such: "
                    "images=[[img1, img2], [img3, img4], ...] for text=[prompt1, prompt2, ...]."
                )
            # Check that only text is present in the prompts
            if not all(isinstance(i, str) for i in text):
                raise ValueError("When using the image-text-to-text behavior, the prompts should only contain text.")
            if isinstance(images[0], (list, tuple)):
                # if nested images, nest text as well
                text = [[i] for i in text]
            prompts = list(zip(images, text))

        output_kwargs = self._merge_kwargs(
            IdeficsProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        add_eos_token = output_kwargs["text_kwargs"].pop("add_eos_token", False)
        add_end_of_utterance_token = output_kwargs["text_kwargs"].pop("add_end_of_utterance_token", None)

        # if the value isn't overridden by the user, check if the tokenizer was trained with this token and then use it
        if add_end_of_utterance_token is None:
            add_end_of_utterance_token = self.tokenizer_was_trained_with_end_of_utterance_token
        # turn non-batched prompts into batched
        if not any(isinstance(i, (list, tuple)) for i in prompts):
            prompts = [prompts]

        fake_token = "<fake_token_around_image>"
        image_token = "<image>"
        end_of_utterance_token = "<end_of_utterance>"

        def image_tokens(last_was_image):
            if last_was_image:
                return image_token + fake_token
            else:
                return fake_token + image_token + fake_token

        all_prompts = []
        all_images = []
        for sample in prompts:
            # the model was trained on samples starting with <s>
            full_text = f"{self.tokenizer.bos_token}"

            # an image can either be an image object in the item or the url, everything else is a verbatim prompt text
            image_objects = []
            last_was_image = False
            last_was_text = False
            for i, item in enumerate(sample):
                if i > 0:
                    last_was_text = True if not last_was_image else False

                if isinstance(item, str):
                    item = item.strip(" ")
                    if is_url(item):
                        image = self.image_processor.fetch_images(item)
                        full_text += image_tokens(last_was_image)
                        image_objects.append(image)
                        last_was_image = True
                    else:
                        # we add end_of_utterance_token between each subsequent text prompts (but not at the last one!)
                        if add_end_of_utterance_token and last_was_text:
                            full_text += end_of_utterance_token
                        full_text += item
                        last_was_image = False
                else:
                    # must be an image obj
                    full_text += image_tokens(last_was_image)
                    image_objects.append(item)
                    last_was_image = True

            if add_eos_token:
                full_text += self.tokenizer.eos_token

            image_objects = self.image_processor(image_objects, **output_kwargs["images_kwargs"])

            all_prompts.append(full_text)
            all_images.append(image_objects)

        # For BC
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", "ms")
        assert return_tensors == "ms" or return_tensors == "np", "Only support return_tensors 'ms' or 'np' "
        text_encoding = self.tokenizer(all_prompts, **output_kwargs["text_kwargs"])
        all_texts = text_encoding["input_ids"]
        all_attention_masks = text_encoding["attention_mask"]

        # max_num_images has to be at least 1 even when there are no images
        max_num_images = max(len(x) for x in all_images)
        max_num_images = max(1, max_num_images)

        at_least_one_image = sum(len(x) for x in all_images) > 0
        output_input_ids = []
        output_images = []
        output_attention_masks = []

        for text_single, attention_mask, extracted_images in zip(all_texts, all_attention_masks, all_images):
            padded_input_ids = text_single
            image_count = padded_input_ids.count(self.image_token_id)
            local_max_num_images = min(image_count, max_num_images)

            current_images = extracted_images[:local_max_num_images]

            if len(current_images) > 0:
                if return_tensors == "ms":
                    padded_image_tensor = mint.zeros((max_num_images, *current_images.shape[1:]))
                    padded_image_tensor[: current_images.shape[0]] = current_images
            else:
                if return_tensors == "ms":
                    padded_image_tensor = mint.zeros((max_num_images, *self.default_image_dims))

            output_images.append(padded_image_tensor)
            if return_tensors == "ms":
                output_input_ids.append(ms.tensor(padded_input_ids))
                output_attention_masks.append(ms.tensor(attention_mask))

        if return_tensors == "ms":
            output_input_ids = mint.stack(output_input_ids)
            output_images = mint.stack(output_images)
            output_attention_masks = mint.stack(output_attention_masks)

        if at_least_one_image:
            image_attention_mask, _ = image_attention_mask_for_packed_input_ids(
                output_input_ids, self.tokenizer, return_tensors
            )
            image_attention_mask = incremental_to_binary_attention_mask(
                image_attention_mask, return_tensors, num_classes=max_num_images
            )
        else:
            # in full language mode we set the image mask to all-0s
            image_attention_mask = mint.zeros((output_input_ids.shape[0], output_input_ids.shape[1], 1), dtype=ms.bool_)

        if return_tensors == "np":
            output_input_ids = output_input_ids.numpy()
            output_attention_masks = output_attention_masks.numpy()
            output_images = output_images.numpy()
            image_attention_mask = image_attention_mask.numpy()

        return BatchFeature(
            data={
                "input_ids": output_input_ids,
                "attention_mask": output_attention_masks,
                "pixel_values": output_images,
                "image_attention_mask": image_attention_mask,
            }
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method constructs all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method constructs all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["IdeficsProcessor"]
