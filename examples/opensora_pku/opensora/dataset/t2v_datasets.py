import json
import logging
import os
import random
from os.path import join as opj

import numpy as np
from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing
from PIL import Image
from tqdm import tqdm

import mindspore as ms

logger = logging.getLogger()


class T2V_dataset(object):
    def __init__(
        self,
        image_data,
        video_data,
        temporal_sample,
        tokenizer,
        num_frames: int = 16,
        use_image_num: int = 4,
        use_img_from_vid: bool = False,
        model_max_length: int = 300,
        transform=None,
        filter_nonexistent: bool = True,
    ):
        self.image_data = image_data
        self.video_data = video_data
        self.num_frames = num_frames
        self.use_image_num = use_image_num
        self.use_img_from_vid = use_img_from_vid
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.filter_nonexistent = filter_nonexistent

        self.v_decoder = DecordInit()

        self.transform = transform
        if self.num_frames != 1:
            self.vid_cap_list = self.get_vid_cap_list()
            if self.use_image_num != 0 and not self.use_img_from_vid:
                self.img_cap_list = self.get_img_cap_list()
        else:
            self.img_cap_list = self.get_img_cap_list()

        logger.info(f"Number of samples: {len(self)}")

    def __len__(self):
        if self.num_frames != 1:
            return len(self.vid_cap_list)
        else:
            return len(self.img_cap_list)

    def __getitem__(self, idx):
        try:
            video_data, image_data = {}, {}
            if self.num_frames != 1:
                video_data = self.get_video(idx)
                if self.use_image_num != 0:
                    if self.use_img_from_vid:
                        image_data = self.get_image_from_video(video_data)
                    else:
                        image_data = self.get_image(idx)
            else:
                image_data = self.get_image(idx)  # 1 frame video as image
            return dict(video_data=video_data, image_data=image_data)
        except Exception as e:
            logger.info(f"Error with {e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def get_video(self, idx):
        video_path = self.vid_cap_list[idx]["path"]
        frame_idx = self.vid_cap_list[idx]["frame_idx"]
        video = self.decord_read(video_path, frame_idx)  # T H W C
        video = self.transform(video)  # T H W C

        video = video.transpose(3, 0, 1, 2)  # T H W C -> C T H W
        text = self.vid_cap_list[idx]["cap"]

        text = text_preprocessing(text)
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors=None,
        )
        input_ids = np.array(text_tokens_and_mask["input_ids"])
        cond_mask = np.array(text_tokens_and_mask["attention_mask"])
        if len(input_ids.shape) == 1:
            input_ids = input_ids[None, :]
        if len(cond_mask.shape) == 1:
            cond_mask = cond_mask[None, :]
        return dict(video=video, input_ids=input_ids, cond_mask=cond_mask)

    def get_image_from_video(self, video_data):
        select_image_idx = np.linspace(0, self.num_frames - 1, self.use_image_num, dtype=int)
        assert self.num_frames >= self.use_image_num
        image = [video_data["video"][:, i : i + 1] for i in select_image_idx]  # num_img [c, 1, h, w]
        input_ids = video_data["input_ids"].repeat(self.use_image_num, 1)  # self.use_image_num, l
        cond_mask = video_data["cond_mask"].repeat(self.use_image_num, 1)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask)

    def get_image(self, idx):
        idx = idx % len(self.img_cap_list)  # out of range
        image_data = self.img_cap_list[idx]  # [{'path': path, 'cap': cap}, ...]

        image = [Image.open(i["path"]).convert("RGB") for i in image_data]  # num_img [h, w, c]
        image = [np.array(i)[None, ...] for i in image]  # num_img [1, h, w, c]
        image = [self.transform(i) for i in image]  # num_img [1 H W C] -> num_img [1 H W C]
        image = [i.transpose(3, 0, 1, 2) for i in image]  # num_img [1 H W C] -> num_img [C 1 H W]

        caps = [i["cap"] for i in image_data]
        text = [text_preprocessing(cap) for cap in caps]
        input_ids, cond_mask = [], []
        for t in text:
            text_tokens_and_mask = self.tokenizer(
                t,
                max_length=self.model_max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors=None,
            )
            ids, mask = np.array(text_tokens_and_mask["input_ids"]), np.array(text_tokens_and_mask["attention_mask"])
            if len(ids.shape) == 1:
                ids = ids[None, :]
            if len(mask.shape) == 1:
                mask = mask[None, :]
            input_ids.append(ids)
            cond_mask.append(mask)

        input_ids = np.concatenate(input_ids)  # self.use_image_num, l
        cond_mask = np.concatenate(cond_mask)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask)

    def tv_read(self, path, frame_idx=None):
        raise NotImplementedError

    def decord_read(self, path, frame_idx=None):
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        # Sampling video frames
        if frame_idx is None:
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        else:
            start_frame_ind, end_frame_ind = frame_idx.split(":")
            # start_frame_ind, end_frame_ind = int(start_frame_ind), int(end_frame_ind)
            start_frame_ind, end_frame_ind = int(start_frame_ind), int(start_frame_ind) + self.num_frames
        # assert end_frame_ind - start_frame_ind >= self.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
        # frame_indice = np.linspace(0, 63, self.num_frames, dtype=int)

        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        return video_data  # (T H W C)

    def get_vid_cap_list(self):
        vid_cap_lists = []
        with open(self.video_data, "r") as f:
            folder_anno = [i.strip().split(",") for i in f.readlines() if len(i.strip()) > 0]
            # logger.info(folder_anno)
        for folder, anno in folder_anno:
            with open(anno, "r") as f:
                vid_cap_list = json.load(f)
            logger.info(f"Building {anno}...")

            new_vid_cap_list = []
            filtered_samples = 0
            for i in tqdm(range(len(vid_cap_list))):
                path = opj(folder, vid_cap_list[i]["path"])
                if os.path.exists(path.replace(".mp4", "_resize_1080p.mp4")):
                    path = path.replace(".mp4", "_resize_1080p.mp4")
                vid_cap_list[i]["path"] = path
                if self.filter_nonexistent:
                    if os.path.exists(path):
                        new_vid_cap_list.append(vid_cap_list[i])
                    else:
                        filtered_samples += 1
                else:
                    new_vid_cap_list.append(vid_cap_list[i])
            vid_cap_lists += new_vid_cap_list
        if self.filter_nonexistent:
            logger.info(f"Number of filtered video samples :{filtered_samples}")
        return vid_cap_lists

    def get_img_cap_list(self):
        use_image_num = self.use_image_num if self.use_image_num != 0 else 1
        img_cap_lists = []
        with open(self.image_data, "r") as f:
            folder_anno = [i.strip().split(",") for i in f.readlines() if len(i.strip()) > 0]
        filtered_samples = 0
        for folder, anno in folder_anno:
            with open(anno, "r") as f:
                img_cap_list = json.load(f)
            logger.info(f"Building {anno}...")
            new_img_cap_list = []
            for i in tqdm(range(len(img_cap_list))):
                img_cap_list[i]["path"] = opj(folder, img_cap_list[i]["path"])
                if self.filter_nonexistent:
                    if os.path.exists(img_cap_list[i]["path"]):
                        new_img_cap_list.append(img_cap_list[i])
                    else:
                        filtered_samples += 1
                else:
                    new_img_cap_list.append(img_cap_list[i])
            img_cap_lists += new_img_cap_list
        if self.filter_nonexistent:
            logger.info(f"Number of filtered image samples :{filtered_samples}")
        img_cap_lists = [img_cap_lists[i : i + use_image_num] for i in range(0, len(img_cap_lists), use_image_num)]
        return img_cap_lists[:-1]  # drop last to avoid error length


def create_dataloader(
    dataset,
    batch_size,
    num_parallel_workers=12,
    max_rowsize=64,
    shuffle=True,
    device_num=1,
    rank_id=0,
    collate_fn=None,
    output_columns=None,
    drop_remainder=True,
    return_dataset=False,
):
    column_names = ["batch"]
    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=column_names,
        num_shards=device_num,
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=max_rowsize,
    )
    if collate_fn is not None:
        assert (
            output_columns is not None
        ), "When using collate_fn, output_columns is required if the number of output lists is different from input."

    dl = dataloader.batch(
        batch_size, drop_remainder=drop_remainder, per_batch_map=collate_fn, output_columns=output_columns
    )
    if return_dataset:
        return dl, dataset
    return dl
