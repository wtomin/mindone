import csv
import html
import logging
import os
import random
import re
import urllib.parse as ul

import ftfy
import numpy as np
from bs4 import BeautifulSoup
from decord import VideoReader
from PIL import Image, ImageSequence

import mindspore as ms

from .dataset import create_video_transforms

logger = logging.getLogger()


def read_gif(gif_path, mode="RGB"):
    with Image.open(gif_path) as fp:
        frames = np.array([np.array(frame.convert(mode)) for frame in ImageSequence.Iterator(fp)])
    return frames


class CSVDataset:
    """A dataset using a csv file
    csv path: (str) a path to a csv file consisting of columns including video_column, class_column (optional),  caption_column(optional)
    video_folder: (str) a folder path where the videos are all stored.
    sample_size: (int, default=256) image size
    sample_stride: (int, default=4) sample stride, should be positive
    sample_n_frames: (int, default=16) the number of sampled frames, only applies when `frame_index_sampler` is None.
    transform_backend: (str, default="al") one of transformation backends in [ms, pt, al]. "al" is recommended.
    tokenizer: (object, default=None) a text tokenizer which is callable.
    video_column: (str, default=None), the column name of videos.
    caption_column: (str, default=None), the column name of text. If not provided, the returned caption/text token will be a dummy placeholder.
    class_column: (str, default=None), the column name of class labels. If not provided, the returned class label will be a dummy placeholder.
    use_safer_augment: (bool, default=False), whether to use safe augmentation. If True, it will disable random horizontal flip.
    image_video_joint: (bool, default=False), whether to use image-video-joint training. If True, the dataset will return the concatenation of `video_frames`
        and randomly-sampled `images` as the pixel values (concatenated at the frame axis). Not supported for CSVDataset.
    use_image_num: (int, default=None), the number of randomly-sampled images in image-video-joint training.
    return_token_mask: (bool, default=False), whether to return the token ids with mask in __getitem__
    """

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + r"\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    def __init__(
        self,
        csv_path,
        video_folder,
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
        transform_backend="al",  # ms, pt, al
        tokenizer=None,
        video_column=None,
        caption_column=None,
        class_column=None,
        use_safer_augment=False,
        image_video_joint=False,
        use_image_num=None,
        condition=None,
        return_token_mask=False,
    ):
        logger.info(f"loading annotations from {csv_path} ...")
        with open(csv_path, "r") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        logger.info(f"Num data samples: {self.length}")

        self.video_folder = video_folder
        self.sample_stride = sample_stride
        assert (
            isinstance(self.sample_stride, int) and self.sample_stride > 0
        ), "The sample stride should be a positive integer"
        self.sample_n_frames = sample_n_frames
        assert (
            isinstance(self.sample_n_frames, int) and self.sample_n_frames > 0
        ), "The number of sampled frames should be a positive integer"
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)

        # it should match the transformation used in SD/VAE pretraining, especially for normalization
        self.pixel_transforms = create_video_transforms(
            sample_size[0],
            sample_size[1],
            sample_n_frames,
            interpolation="bicubic",
            backend=transform_backend,
            use_safer_augment=use_safer_augment,
        )
        self.transform_backend = transform_backend
        self.tokenizer = tokenizer
        self.video_column = video_column
        assert self.video_column is not None, "The input csv file must specifiy the video column"
        # conditions: None, text, or class
        self.condition = condition
        self.caption_column = caption_column
        self.class_column = class_column
        if self.caption_column is not None and self.tokenizer is None:
            logger.warning(
                f"The caption column is provided as {self.caption_column}, but tokenizer is None",
                "The text tokens will be dummy placeholders!",
            )

        # whether to use image and video joint training
        self.image_video_joint = image_video_joint
        self.use_image_num = use_image_num
        if image_video_joint:
            # image video joint training not supported here because the total number of frames is unknown
            raise NotImplementedError
        self.image_transforms = None
        self.return_token_mask = return_token_mask
        column_names = ["video"]
        if self.condition == "text":
            column_names += ["caption"]
            if self.return_token_mask:
                column_names += ["mask"]
        elif self.condition == "class":
            column_names += ["label"]
        self.column_names = column_names  # names are mapped with the values returned by __getitem__

    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_fn = video_dict[self.video_column]
        # load caption if needed, otherwise replace it with a dummy value
        if self.caption_column is not None and self.condition == "text":
            caption = video_dict[self.caption_column]
        else:
            caption = ""
        # load class labels if needed, otherwise replace it with a dummy value
        if self.class_column is not None and self.condition == "class":
            class_label = int(video_dict[self.class_column])
        else:
            class_label = 0  # a dummy class label as a placeholder

        # pixel transformation
        video_path = os.path.join(self.video_folder, video_fn)
        if video_path.endswith(".gif"):
            video_reader = read_gif(video_path, mode="RGB")
        else:
            video_reader = VideoReader(video_path)
        # randomly sample video frames
        video_length = len(video_reader)
        clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)

        if video_path.endswith(".gif"):
            pixel_values = video_reader[batch_index]  # shape: (f, h, w, c)
        else:
            pixel_values = video_reader.get_batch(batch_index).asnumpy()  # shape: (f, h, w, c)
        del video_reader

        return pixel_values, class_label, caption

    def __len__(self):
        return self.length

    def apply_transform(self, pixel_values, video_transform=True):
        if video_transform:
            transform_func = self.pixel_transforms
        else:
            assert self.image_video_joint, "image transform requires to set image-video-joint training on"
            transform_func = self.image_transforms
        if self.transform_backend == "pt":
            import torch

            pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
            pixel_values = transform_func(pixel_values)
            pixel_values = pixel_values.numpy()
        elif self.transform_backend == "al":
            # NOTE:it's to ensure augment all frames in a video in the same way.
            # ref: https://albumentations.ai/docs/examples/example_multi_target/

            inputs = {"image": pixel_values[0]}
            num_frames = len(pixel_values)
            for i in range(num_frames - 1):
                inputs[f"image{i}"] = pixel_values[i + 1]

            output = transform_func(**inputs)

            pixel_values = np.stack(list(output.values()), axis=0)
            # (f h w c) -> (f c h w)
            pixel_values = np.transpose(pixel_values, (0, 3, 1, 2))
        else:
            raise NotImplementedError
        return pixel_values

    def __getitem__(self, idx):
        """
        Returns:
            tuple (video, text_data)
                - video: preprocessed video frames in shape (f, c, h, w)
                - text_data: if tokenizer provided, tokens shape (context_max_len,), otherwise text string
        """
        pixel_values, class_label, caption = self.get_batch(idx)
        pixel_values = self.apply_transform(pixel_values, video_transform=True)
        pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)

        if self.tokenizer is not None and self.condition == "text":
            tokens, mask = self.tokenize(caption)
            # print("D--: ", type(text_data))
            if isinstance(tokens, list):
                tokens = np.array(tokens, dtype=np.int64)
            if len(tokens.shape) == 2:  # in case, the tokenizer output [1, 77]
                tokens = tokens[0]
            if isinstance(mask, list):
                mask = np.array(mask, dtype=np.float32)
            if len(mask.shape) == 2:  # in case, the tokenizer output [1, 77]
                mask = mask[0]
            text_data = tokens

        return_values = pixel_values

        if self.condition == "text":
            return_values += text_data
            if self.return_token_mask:
                return_values += mask
        elif self.condition == "class":
            return_values += class_label
        return return_values

    def tokenize(self, text):
        # a hack to determine if use transformers.CLIPTokenizer
        # should handle it better
        if type(self.tokenizer).__name__ == "CLIPTokenizer":
            return self._clip_tokenize(text)
        elif type(self.tokenizer).__name__ == "T5TokenizerFast":
            return self._t5_tokenize(text)
        else:
            return self._mindone_tokenize(text)

    def _mindone_tokenize(self, text):
        SOT_TEXT = self.tokenizer.sot_text  # "[CLS]"
        EOT_TEXT = self.tokenizer.eot_text  # "[SEP]"
        CONTEXT_LEN = self.tokenizer.context_length

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        result = np.zeros([CONTEXT_LEN]) + eot_token
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]
        result[: len(tokens)] = tokens
        mask = np.zeros([CONTEXT_LEN])
        mask[: len(tokens)] = 1

        return result.astype(np.int64), mask.astype(np.float32)

    def _clip_tokenize(self, texts):
        batch_encoding = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.tokenizer.context_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_attention_mask=True,
            add_special_tokens=True,
        )
        tokens = np.array(batch_encoding["input_ids"], dtype=np.int32)
        mask = np.array(batch_encoding["attention_mask"], dtype=np.float32)
        return tokens, mask

    def _t5_tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = [self.text_preprocessing(text) for text in texts]
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.tokenizer.context_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
        )
        text_tokens = text_tokens_and_mask["input_ids"]
        mask = text_tokens_and_mask["attention_mask"]
        return text_tokens, mask

    def traverse_single_video_frames(self, video_index):
        video_dict = self.dataset[video_index]
        video_fn = video_dict[self.video_column]
        # load caption if needed, otherwise replace it with a dummy value
        if self.caption_column is not None and self.condition == "text":
            caption = video_dict[self.caption_column]
        else:
            caption = ""
        # load class labels if needed, otherwise replace it with a dummy value
        if self.class_column is not None and self.condition == "class":
            class_label = int(video_dict[self.class_column])
        else:
            class_label = 0  # a dummy class label as a placeholder

        if self.tokenizer is not None and self.condition == "text":
            tokens, mask = self.tokenize(caption)
            # print("D--: ", type(text_data))
            if isinstance(tokens, list):
                tokens = np.array(tokens, dtype=np.int64)
            if len(tokens.shape) == 2:  # in case, the tokenizer output [1, 77]
                tokens = tokens[0]
            if isinstance(mask, list):
                mask = np.array(mask, dtype=np.float32)
            if len(mask.shape) == 2:  # in case, the tokenizer output [1, 77]
                mask = mask[0]
            text_data = tokens

        video_name = os.path.basename(video_fn).split(".")[0]
        # read video
        video_path = os.path.join(self.video_folder, video_fn)
        if video_path.endswith(".gif"):
            video_reader = read_gif(video_path, mode="RGB")
        else:
            video_reader = VideoReader(video_path)

        video_length = len(video_reader)

        # Sampling video frames
        clips_indices = []
        start_idx = 0
        while start_idx + self.sample_n_frames < video_length:
            clips_indices.append([start_idx, start_idx + self.sample_n_frames])
            start_idx += self.sample_n_frames
        if start_idx < video_length:
            clips_indices.append([start_idx, video_length])
        assert len(clips_indices) > 0 and clips_indices[-1][-1] == video_length, "incorrect sampled clips!"

        for clip_indices in clips_indices:
            i, j = clip_indices
            frame_indice = list(range(i, j, 1))
            select_video_frames = [
                f"{index}" for index in frame_indice
            ]  # return indexes as strings, for the purpose of saving frame-wise embedding cache
            if video_path.endswith(".gif"):
                pixel_values = video_reader[frame_indice]  # shape: (f, h, w, c)
            else:
                pixel_values = video_reader.get_batch(frame_indice).asnumpy()  # shape: (f, h, w, c)
            pixel_values = self.apply_transform(pixel_values)
            pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)
            return_dict = {"video": pixel_values}
            if self.condition == "class":
                return_dict["class"] = class_label
            elif self.condition == "text":
                return_dict["caption"] = caption
                return_dict["text"] = text_data
                return_dict["mask"] = mask
            yield video_name, select_video_frames, return_dict

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    def text_preprocessing(self, text):
        # The exact text cleaning as was in the training stage:
        text = self.clean_caption(text)
        text = self.clean_caption(text)
        return text


def create_dataloader(
    config,
    tokenizer=None,
    device_num=1,
    rank_id=0,
    return_dataset=False,
):
    dataset = CSVDataset(
        config["csv_path"],
        config["data_folder"],
        sample_size=config.get("sample_size", 256),
        sample_stride=config.get("sample_stride", 4),
        sample_n_frames=config.get("sample_n_frames", 16),
        video_column=config["video_column"],
        caption_column=config["caption_column"],
        class_column=config["class_column"],
        condition=config["condition"],
        return_token_mask=config.get("return_token_mask", False),
        tokenizer=tokenizer,
    )

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=dataset.column_names,
        num_shards=device_num,
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=config["shuffle"],
        num_parallel_workers=config["num_parallel_workers"],
        max_rowsize=config["max_rowsize"],  # video data require larger rowsize
    )

    dl = dataloader.batch(
        config["batch_size"],
        drop_remainder=True,
    )
    if return_dataset:
        return dataset, dl
    else:
        return dl
