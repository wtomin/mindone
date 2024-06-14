from mindspore.dataset import transforms

from .t2v_datasets import T2V_dataset
from .transform import CenterCropResizeVideo, TemporalRandomCrop, ToTensorVideo

ae_norm = {
    "CausalVAEModel_4x8x8": lambda x: 2.0 * x - 1.0,
}
ae_denorm = {
    "CausalVAEModel_4x8x8": lambda x: (x + 1.0) / 2.0,
}


def getdataset(args, tokenizer):
    temporal_sample = TemporalRandomCrop(args.num_frames * args.sample_rate)  # 16 x
    norm_fun = ae_norm[args.ae]
    if args.dataset == "t2v":
        if args.multi_scale:
            raise NotImplementedError
        else:
            resize = [
                CenterCropResizeVideo(args.max_image_size),
            ]
        transform = transforms.Compose([*resize, ToTensorVideo(), norm_fun])
        return T2V_dataset(
            image_data=args.image_data,
            video_data=args.video_data,
            temporal_sample=temporal_sample,
            tokenizer=tokenizer,
            num_frames=args.num_frames,
            use_image_num=args.use_image_num,
            use_img_from_vid=args.use_img_from_vid,
            model_max_length=args.model_max_length,
            transform=transform,
            filter_nonexistent=getattr(args, "filter_nonexistent", True),
            return_text_emb=args.text_embed_cache,
        )

    raise NotImplementedError(args.dataset)
