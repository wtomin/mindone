# Adapted from https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/train/train_t2v_diffusers.py

import logging
import math
import os
import sys
import time
from copy import deepcopy

import yaml

import mindspore as ms
from mindspore import nn
from mindspore.communication.management import GlobalComm

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append("./")
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
from opensora.dataset import getdataset
from opensora.dataset.loader import create_dataloader
from opensora.models.causalvideovae import ae_channel_config, ae_stride_config, ae_wrapper
from opensora.models.diffusion import Diffusion_models
from opensora.models.diffusion.common import PatchEmbed2D
from opensora.models.diffusion.opensora.modules import Attention, LayerNorm
from opensora.models.diffusion.opensora.net_with_loss import DiffusionWithLoss
from opensora.npu_config import npu_config
from opensora.train.commons import create_loss_scaler, parse_args
from opensora.utils.dataset_utils import Collate, LengthGroupedSampler
from opensora.utils.ema import EMA
from opensora.utils.message_utils import print_banner
from opensora.utils.train_step import TrainStepOpenSoraPlan, prepare_train_network
from opensora.utils.utils import AverageMeter, get_precision, save_diffusers_json

from mindone.diffusers.models.activations import SiLU
from mindone.diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # CogVideoXDDIMScheduler,
from mindone.diffusers.schedulers import DDPMScheduler
from mindone.diffusers.training_utils import pynative_no_grad
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.transformers import CLIPTextModelWithProjection, MT5EncoderModel, T5EncoderModel
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params

logger = logging.getLogger(__name__)


def set_all_reduce_fusion(
    params,
    split_num: int = 7,
    distributed: bool = False,
    parallel_mode: str = "data",
) -> None:
    """Set allreduce fusion strategy by split_num."""

    if distributed and parallel_mode == "data":
        all_params_num = len(params)
        step = all_params_num // split_num
        split_list = [i * step for i in range(1, split_num)]
        split_list.append(all_params_num - 1)
        logger.info(f"Distribute config set: dall_params_num: {all_params_num}, set all_reduce_fusion: {split_list}")
        ms.set_auto_parallel_context(all_reduce_fusion_config=split_list)


#################################################################################
#                                  Training Loop                                #
#################################################################################


def validate_model(model, val_dataloader, rank_id, device_num):
    """Run validation on the model"""
    model.set_train(False)
    total_val_loss = 0.0
    num_val_batches = 0

    with pynative_no_grad():
        val_iter = val_dataloader.create_dict_iterator(num_epochs=1, output_numpy=True)
        for batch in val_iter:
            pixel_values = ms.Tensor(batch["pixel_values"])
            attention_mask = ms.Tensor(batch["attention_mask"])
            text_embed = ms.Tensor(batch["text_embed"])
            encoder_attention_mask = ms.Tensor(batch["encoder_attention_mask"])

            # Forward pass for validation (no gradients)
            outputs = model(
                pixel_values,
                attention_mask,
                text_embed,
                encoder_attention_mask,
            )
            val_loss = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            total_val_loss += float(val_loss.asnumpy())
            num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    model.set_train(True)
    return avg_val_loss


def main(args):
    # 1. init
    save_src_strategy = args.use_parallel and args.parallel_mode != "data"
    if args.num_frames == 1 or args.use_image_num != 0:
        args.sp_size = 1
    rank_id, device_num = npu_config.set_npu_env(args, strategy_ckpt_save_file=save_src_strategy)
    if args.profile_memory:
        if args.mode == 1:
            # maybe slow
            ms.context.set_context(pynative_synchronize=True)
        profiler = ms.Profiler(output_path="./mem_info", profile_memory=True)
        # ms.context.set_context(memory_optimize_level="O0")  # enabling it may consume more memory
        logger.info(f"Memory profiling: {profiler}")
    npu_config.print_ops_dtype_info()
    set_logger(name="", output_dir=args.output_dir, rank=rank_id, log_level=eval(args.log_level))
    LOG_FILE = os.path.join(args.output_dir, "loss.log")
    # 2. Init and load models
    # Load VAE
    train_with_vae_latent = args.vae_latent_folder is not None and len(args.vae_latent_folder) > 0
    if train_with_vae_latent:
        assert os.path.exists(
            args.vae_latent_folder
        ), f"The provided vae latent folder {args.vae_latent_folder} is not existent!"
        logger.info("Train with vae latent cache.")
        vae = None
    else:
        print_banner("vae init")
        if args.vae_fp32:
            logger.info("Force VAE running in FP32")
            args.vae_precision = "fp32"
        vae_dtype = get_precision(args.vae_precision)
        kwarg = {
            "use_safetensors": True,
            "dtype": vae_dtype,
        }
        vae = ae_wrapper[args.ae](args.ae_path, **kwarg)

        vae.set_train(False)
        for param in vae.get_parameters():  # freeze vae
            param.requires_grad = False

        if args.enable_tiling:
            vae.vae.enable_tiling()
            vae.vae.tile_overlap_factor = args.tile_overlap_factor

    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    if vae is not None:
        vae.vae_scale_factor = (ae_stride_t, ae_stride_h, ae_stride_w)
    assert (
        ae_stride_h == ae_stride_w
    ), f"Support only ae_stride_h == ae_stride_w now, but found ae_stride_h ({ae_stride_h}), ae_stride_w ({ae_stride_w})"
    args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = ae_stride_t, ae_stride_h, ae_stride_w
    args.ae_stride = args.ae_stride_h
    patch_size = args.model[-3:]
    patch_size_t, patch_size_h, patch_size_w = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
    args.patch_size = patch_size_h
    args.patch_size_t, args.patch_size_h, args.patch_size_w = patch_size_t, patch_size_h, patch_size_w
    assert (
        patch_size_h == patch_size_w
    ), f"Support only patch_size_h == patch_size_w now, but found patch_size_h ({patch_size_h}), patch_size_w ({patch_size_w})"
    assert (
        args.max_height % ae_stride_h == 0
    ), f"Height must be divisible by ae_stride_h, but found Height ({args.max_height}), ae_stride_h ({ae_stride_h})."
    assert (
        args.num_frames - 1
    ) % ae_stride_t == 0, f"(Frames - 1) must be divisible by ae_stride_t, but found num_frames ({args.num_frames}), ae_stride_t ({ae_stride_t})."
    assert (
        args.max_width % ae_stride_h == 0
    ), f"Width size must be divisible by ae_stride_h, but found Width ({args.max_width}), ae_stride_h ({ae_stride_h})."

    args.stride_t = ae_stride_t * patch_size_t
    args.stride = ae_stride_h * patch_size_h
    vae.latent_size = latent_size = (args.max_height // ae_stride_h, args.max_width // ae_stride_w)
    args.latent_size_t = latent_size_t = (args.num_frames - 1) // ae_stride_t + 1

    # Load diffusion transformer
    print_banner("Transformer model init")
    FA_dtype = get_precision(args.precision) if get_precision(args.precision) != ms.float32 else ms.bfloat16
    model = Diffusion_models[args.model](
        in_channels=ae_channel_config[args.ae],
        out_channels=ae_channel_config[args.ae],
        sample_size_h=latent_size,
        sample_size_w=latent_size,
        sample_size_t=latent_size_t,
        interpolation_scale_h=args.interpolation_scale_h,
        interpolation_scale_w=args.interpolation_scale_w,
        interpolation_scale_t=args.interpolation_scale_t,
        sparse1d=args.sparse1d,
        sparse_n=args.sparse_n,
        skip_connection=args.skip_connection,
        use_recompute=args.gradient_checkpointing,
        num_no_recompute=args.num_no_recompute,
        FA_dtype=FA_dtype,
    )
    json_name = os.path.join(args.output_dir, "config.json")
    config = deepcopy(model.config)
    if hasattr(config, "recompute"):
        del config.recompute
    save_diffusers_json(config, json_name)
    # mixed precision
    if args.precision == "fp32":
        model_dtype = get_precision(args.precision)
    else:
        model_dtype = get_precision(args.precision)
        if model_dtype == ms.float16:
            custom_fp32_cells = [LayerNorm, Attention, PatchEmbed2D, nn.SiLU, SiLU, nn.GELU]
        else:
            custom_fp32_cells = [
                nn.MaxPool2d,
                nn.MaxPool3d,
                PatchEmbed2D,
                LayerNorm,
                nn.SiLU,
                SiLU,
                nn.GELU,
            ]
        model = auto_mixed_precision(
            model,
            amp_level=args.amp_level,
            dtype=model_dtype,
            custom_fp32_cells=custom_fp32_cells,
        )
        logger.info(
            f"Set mixed precision to {args.amp_level} with dtype={args.precision}, custom fp32_cells {custom_fp32_cells}"
        )

    # load checkpoint
    if args.pretrained is not None and len(args.pretrained) > 0:
        assert os.path.exists(args.pretrained), f"Provided checkpoint file {args.pretrained} does not exist!"
        logger.info(f"Loading ckpt {args.pretrained}...")
        model = model.load_from_checkpoint(model, args.pretrained)
    else:
        logger.info("Use random initialization for transformer")
    model.set_train(True)

    # Load text encoder
    if not args.text_embed_cache:
        print_banner("text encoder init")
        text_encoder_dtype = get_precision(args.text_encoder_precision)
        if "mt5" in args.text_encoder_name_1:
            text_encoder_1, loading_info = MT5EncoderModel.from_pretrained(
                args.text_encoder_name_1,
                cache_dir=args.cache_dir,
                output_loading_info=True,
                mindspore_dtype=text_encoder_dtype,
                use_safetensors=True,
            )
            loading_info.pop("unexpected_keys")  # decoder weights are ignored
            logger.info(f"Loaded MT5 Encoder: {loading_info}")
            text_encoder_1 = text_encoder_1.set_train(False)
        else:
            text_encoder_1 = T5EncoderModel.from_pretrained(
                args.text_encoder_name_1, cache_dir=args.cache_dir, mindspore_dtype=text_encoder_dtype
            ).set_train(False)
        text_encoder_2 = None
        if args.text_encoder_name_2 is not None:
            text_encoder_2, loading_info = CLIPTextModelWithProjection.from_pretrained(
                args.text_encoder_name_2,
                cache_dir=args.cache_dir,
                mindspore_dtype=text_encoder_dtype,
                output_loading_info=True,
                use_safetensors=True,
            )
            loading_info.pop("unexpected_keys")  # only load text model, ignore vision model
            # loading_info.pop("mising_keys") # Note: missed keys when loading open-clip models
            logger.info(f"Loaded CLIP Encoder: {loading_info}")
            text_encoder_2 = text_encoder_2.set_train(False)
    else:
        text_encoder_1 = None
        text_encoder_2 = None
        text_encoder_dtype = None

    kwargs = dict(prediction_type=args.prediction_type, rescale_betas_zero_snr=args.rescale_betas_zero_snr)
    if args.cogvideox_scheduler:
        from mindone.diffusers import CogVideoXDDIMScheduler

        noise_scheduler = CogVideoXDDIMScheduler(**kwargs)
    elif args.v1_5_scheduler:
        kwargs["beta_start"] = 0.00085
        kwargs["beta_end"] = 0.0120
        kwargs["beta_schedule"] = "scaled_linear"
        noise_scheduler = DDPMScheduler(**kwargs)
    elif args.rf_scheduler:
        noise_scheduler = FlowMatchEulerDiscreteScheduler()
    else:
        noise_scheduler = DDPMScheduler(**kwargs)

    assert args.use_image_num >= 0, f"Expect to have use_image_num>=0, but got {args.use_image_num}"
    if args.use_image_num > 0:
        logger.info("Enable video-image-joint training")
    else:
        if args.num_frames == 1:
            logger.info("Training on image datasets only.")
        else:
            logger.info("Training on video datasets only.")

    latent_diffusion_with_loss = DiffusionWithLoss(
        model,
        noise_scheduler,
        vae=vae,
        text_encoder=text_encoder_1,
        text_emb_cached=args.text_embed_cache,
        video_emb_cached=False,
        use_image_num=args.use_image_num,
        dtype=model_dtype,
        noise_offset=args.noise_offset,
        snr_gamma=args.snr_gamma,
        rf_scheduler=args.rf_scheduler,
        rank_id=rank_id,
        device_num=device_num,
    )

    # 3. create dataset
    # TODO: replace it with new dataset
    assert args.dataset == "t2v", "Support t2v dataset only."
    print_banner("Training dataset Loading...")

    # Setup data:
    # TODO: to use in v1.3
    if args.trained_data_global_step is not None:
        initial_global_step_for_sampler = args.trained_data_global_step
    else:
        initial_global_step_for_sampler = 0
    total_batch_size = args.train_batch_size * device_num * args.gradient_accumulation_steps
    total_batch_size = total_batch_size // args.sp_size * args.train_sp_batch_size
    args.total_batch_size = total_batch_size
    if args.max_hxw is not None and args.min_hxw is None:
        args.min_hxw = args.max_hxw // 4

    train_dataset = getdataset(args, dataset_file=args.data)
    sampler = LengthGroupedSampler(
        args.train_batch_size,
        world_size=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
        gradient_accumulation_size=args.gradient_accumulation_steps,
        initial_global_step=initial_global_step_for_sampler,
        lengths=train_dataset.lengths,
        group_data=args.group_data,
    )
    collate_fn = Collate(args.train_batch_size, args)
    dataloader = create_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=sampler is None,
        device_num=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
        rank_id=rank_id if not get_sequence_parallel_state() else hccl_info.group_id,
        num_parallel_workers=args.dataloader_num_workers,
        max_rowsize=args.max_rowsize,
        prefetch_size=args.dataloader_prefetch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        column_names=["pixel_values", "attention_mask", "text_embed", "encoder_attention_mask"],
        dataset_iterator_no_copy=args.dataset_iterator_no_copy,
    )
    dataloader_size = dataloader.get_dataset_size()
    assert (
        dataloader_size > 0
    ), "Incorrect training dataset size. Please check your dataset size and your global batch size"

    val_dataloader = None
    if args.validate:
        assert args.val_data is not None, f"validation dataset must be specified, but got {args.val_data}"
        assert os.path.exists(args.val_data), f"validation dataset file must exist, but got {args.val_data}"
        print_banner("Validation dataset Loading...")
        val_dataset = getdataset(args, dataset_file=args.val_data)
        sampler = LengthGroupedSampler(
            args.val_batch_size,
            world_size=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
            lengths=val_dataset.lengths,
            gradient_accumulation_size=args.gradient_accumulation_steps,
            initial_global_step=initial_global_step_for_sampler,
            group_data=args.group_data,
        )

        collate_fn = Collate(args.val_batch_size, args)
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=sampler is None,
            device_num=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
            rank_id=rank_id if not get_sequence_parallel_state() else hccl_info.group_id,
            num_parallel_workers=args.dataloader_num_workers,
            max_rowsize=args.max_rowsize,
            prefetch_size=args.dataloader_prefetch_size,
            collate_fn=collate_fn,
            sampler=sampler,
            column_names=["pixel_values", "attention_mask", "text_embed", "encoder_attention_mask"],
            dataset_iterator_no_copy=args.dataset_iterator_no_copy,
        )
        val_dataloader_size = val_dataloader.get_dataset_size()
        assert (
            val_dataloader_size > 0
        ), "Incorrect validation dataset size. Please check your dataset size and your global batch size"
    # 4. build training utils: lr, optim, callbacks, trainer
    if args.scale_lr:
        learning_rate = args.start_learning_rate * args.train_batch_size * args.gradient_accumulation_steps * device_num
        end_learning_rate = (
            args.end_learning_rate * args.train_batch_size * args.gradient_accumulation_steps * device_num
        )
    else:
        learning_rate = args.start_learning_rate
        end_learning_rate = args.end_learning_rate

    if args.max_train_steps is not None:
        assert args.max_train_steps > 0, f"max_train_steps should a positive integer, but got {args.max_train_steps}"
        total_train_steps = args.max_train_steps
        args.num_train_epochs = math.ceil(total_train_steps / dataloader_size)
    else:
        # use args.num_train_epochs
        assert (
            args.num_train_epochs is not None and args.num_train_epochs > 0
        ), f"When args.max_train_steps is not provided, args.num_train_epochs must be a positive integer! but got {args.num_train_epochs}"
        total_train_steps = args.num_train_epochs * dataloader_size

    ckpt_save_interval = args.checkpointing_steps if args.checkpointing_steps is not None else args.ckpt_save_interval
    logger.info(f"ckpt_save_interval: {ckpt_save_interval} steps")
    # build learning rate scheduler
    if not args.lr_decay_steps:
        args.lr_decay_steps = total_train_steps - args.lr_warmup_steps  # fix lr scheduling
        if args.lr_decay_steps <= 0:
            logger.warning(
                f"decay_steps is {args.lr_decay_steps}, please check epochs, dataloader_size and warmup_steps. "
                f"Will force decay_steps to be set to 1."
            )
            args.lr_decay_steps = 1
    assert (
        args.lr_warmup_steps >= 0
    ), f"Expect args.lr_warmup_steps to be no less than zero,  but got {args.lr_warmup_steps}"

    lr_scheduler = create_scheduler(
        steps_per_epoch=dataloader_size,
        name=args.lr_scheduler,
        lr=learning_rate,
        end_lr=end_learning_rate,
        warmup_steps=args.lr_warmup_steps,
        decay_steps=args.lr_decay_steps,
        total_steps=total_train_steps,
    )
    set_all_reduce_fusion(
        latent_diffusion_with_loss.trainable_params(),
        split_num=7,
        distributed=args.use_parallel,
        parallel_mode=args.parallel_mode,
    )

    # build optimizer
    assert args.optim.lower() == "adamw" or args.optim.lower() == "adamw_re", f"Not support optimizer {args.optim}!"
    optimizer = create_optimizer(
        latent_diffusion_with_loss.trainable_params(),
        name=args.optim,
        betas=args.betas,
        eps=args.optim_eps,
        group_strategy=args.group_strategy,
        weight_decay=args.weight_decay,
        lr=lr_scheduler,
    )

    loss_scaler = create_loss_scaler(args)
    # resume ckpt
    ckpt_dir = os.path.join(args.output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 0
    cur_iter = 0
    if args.resume_from_checkpoint:
        resume_ckpt = (
            os.path.join(ckpt_dir, "train_resume.ckpt")
            if isinstance(args.resume_from_checkpoint, bool)
            else args.resume_from_checkpoint
        )

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(model, optimizer, resume_ckpt)
        loss_scaler.loss_scale_value = loss_scale
        loss_scaler.cur_iter = cur_iter
        loss_scaler.last_overflow_iter = last_overflow_iter
        logger.info(f"Resume training from {resume_ckpt}")

    ema = (
        EMA(
            latent_diffusion_with_loss.network,
            ema_decay=args.ema_decay,
            offloading=args.ema_offload,
            update_after_step=args.ema_start_step,
        )
        if args.use_ema
        else None
    )
    assert (
        args.gradient_accumulation_steps > 0
    ), f"Expect gradient_accumulation_steps is a positive integer, but got {args.gradient_accumulation_steps}"
    if args.parallel_mode == "zero":
        assert args.zero_stage in [0, 1, 2, 3], f"Unsupported zero stage {args.zero_stage}"
        logger.info(f"Training with zero{args.zero_stage} parallelism")
        comm_fusion_dict = None
        if args.comm_fusion:
            comm_fusion_dict = {
                "allreduce": {"openstate": True, "bucket_size": 5e8},
                "reduce_scatter": {"openstate": True, "bucket_size": 5e8},
                "allgather": {"openstate": False, "bucket_size": 5e8},
            }
        latent_diffusion_with_loss, zero_helper = prepare_train_network(
            latent_diffusion_with_loss,
            optimizer,
            zero_stage=args.zero_stage,
            optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP,
            comm_fusion=comm_fusion_dict,
        )
    else:
        zero_helper = None

    train_step_cell = TrainStepOpenSoraPlan(
        network=latent_diffusion_with_loss,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm if args.clip_grad else None,
        ema=ema,
        zero_helper=zero_helper,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # 5. log and save config
    if rank_id == 0:
        if vae is not None:
            num_params_vae, num_params_vae_trainable = count_params(vae)
        else:
            num_params_vae, num_params_vae_trainable = 0, 0
        num_params_transformer, num_params_transformer_trainable = count_params(latent_diffusion_with_loss.network)
        num_params = num_params_vae + num_params_transformer
        num_params_trainable = num_params_vae_trainable + num_params_transformer_trainable
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}"
                + (
                    f"\nParallel mode: {args.parallel_mode}"
                    + (f"{args.zero_stage}" if args.parallel_mode == "zero" else "")
                    if args.use_parallel
                    else ""
                )
                + (f"\nsp_size: {args.sp_size}" if args.sp_size != 1 else ""),
                f"Num params: {num_params} (transformer: {num_params_transformer}, vae: {num_params_vae})",
                f"Num trainable params: {num_params_trainable}",
                f"Transformer model dtype: {model_dtype}",
                f"Transformer AMP level: {args.amp_level}",
                f"VAE dtype: {vae_dtype}"
                + (f"\nText encoder dtype: {text_encoder_dtype}" if text_encoder_dtype is not None else ""),
                f"Learning rate: {learning_rate}",
                f"Instantaneous batch size per device: {args.train_batch_size}",
                f"Total train batch size (w. parallel, distributed & accumulation): {total_batch_size}",
                f"Image height: {args.max_height}",
                f"Image width: {args.max_width}",
                f"Number of frames: {args.num_frames}",
                f"Use image num: {args.use_image_num}",
                f"Optimizer: {args.optim}",
                f"Optimizer epsilon: {args.optim_eps}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num of training steps: {total_train_steps}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"EMA decay: {args.ema_decay}",
                f"EMA cpu offload: {args.ema_offload}",
                f"FA dtype: {FA_dtype}",
                f"Use recompute(gradient checkpoint): {args.gradient_checkpointing}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")

        with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    global_step = cur_iter
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(start_epoch, args.num_train_epochs):
        latent_diffusion_with_loss.set_train(True)
        data_iter = dataloader.create_dict_iterator(num_epochs=1, output_numpy=True)
        for batch in data_iter:
            pixel_values = ms.Tensor(batch["pixel_values"])  # (b, c, f, h, w)
            attention_mask = ms.Tensor(batch["attention_mask"])  # (b, f, h, w)
            text_embed = ms.Tensor(batch["text_embed"])  # (b, model_max_length, dim) or cached shape
            encoder_attention_mask = ms.Tensor(batch["encoder_attention_mask"])  # (b, model_max_length)
            data_time_m.update(time.time() - end)
            loss = train_step_cell.train_one_step(
                pixel_values,
                attention_mask,
                text_embed,
                encoder_attention_mask,
            )

            batch_time_m.update(time.time() - end)
            end = time.time()
            # save the log file
            if rank_id == 0:
                try:
                    if not os.path.exists(LOG_FILE):
                        with open(LOG_FILE, "w", encoding="utf-8") as fp:
                            fp.write("\t".join(["step", "loss", "per step time (s)"]) + "\n")
                    with open(LOG_FILE, "a", encoding="utf-8") as fp:
                        fp.write(
                            "\t".join(
                                [
                                    f"{global_step + 1:<7}",
                                    f"{loss.asnumpy().item():<10.6f}",
                                    f"{batch_time_m.val:<13.3f}",
                                ]
                            )
                            + "\n"
                        )
                except (IOError, PermissionError) as e:
                    logger.error(f"Failed to write log: {e}")

            lr_scheduler.step()
            if (global_step + 1) % args.log_interval == 0 and (rank_id == 0):
                cur_lr = lr_scheduler.get_last_lr()[0]
                cur_lr_val = float(cur_lr.asnumpy()) if hasattr(cur_lr, "asnumpy") else float(cur_lr)
                logger.info(f"Step: {global_step + 1} Loss: {float(loss.asnumpy()):.6f} LR: {cur_lr_val:.6e}")

            # Validation at step intervals (optional)
            if (
                args.validate
                and val_dataloader is not None
                and (global_step + 1) % (args.val_interval * dataloader_size) == 0
                and rank_id == 0
            ):
                logger.info(f"Running validation at step {global_step + 1}")
                val_loss = validate_model(latent_diffusion_with_loss, val_dataloader, rank_id, device_num)
                logger.info(f"Validation loss at step {global_step + 1}: {val_loss:.6f}")

            if (global_step + 1) % ckpt_save_interval == 0 and (rank_id == 0):
                save_path = os.path.join(ckpt_dir, f"train_resume_step{global_step + 1}.ckpt")
                ms.save_checkpoint(latent_diffusion_with_loss.network, save_path)
                logger.info(f"Saved checkpoint to {save_path}")

            global_step += 1
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break

        # Validation after each epoch
        if args.validate and val_dataloader is not None and (epoch + 1) % args.val_interval == 0:
            if rank_id == 0:
                logger.info(f"Running validation at epoch {epoch + 1}")
            val_loss = validate_model(latent_diffusion_with_loss, val_dataloader, rank_id, device_num)
            if rank_id == 0:
                logger.info(f"Validation loss at epoch {epoch + 1}: {val_loss:.6f}")
                # Save validation log
                val_log_file = os.path.join(args.output_dir, "val_loss.log")
                try:
                    if not os.path.exists(val_log_file):
                        with open(val_log_file, "w", encoding="utf-8") as fp:
                            fp.write("\t".join(["epoch", "val_loss"]) + "\n")
                    with open(val_log_file, "a", encoding="utf-8") as fp:
                        fp.write(f"{epoch + 1}\t{val_loss:.6f}\n")
                except (IOError, PermissionError) as e:
                    logger.error(f"Failed to write validation log: {e}")

        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break

    if rank_id == 0:
        final_path = os.path.join(ckpt_dir, f"train_final_step{global_step}.ckpt")
        ms.save_checkpoint(latent_diffusion_with_loss.network, final_path)
        logger.info(f"Final checkpoint saved to {final_path}")


def parse_t2v_train_args(parser):
    # TODO: NEW in v1.3 , but may not use
    # dataset & dataloader
    parser.add_argument("--max_hxw", type=int, default=None)
    parser.add_argument("--min_hxw", type=int, default=None)
    parser.add_argument("--ood_img_ratio", type=float, default=0.0)
    parser.add_argument("--group_data", action="store_true")
    parser.add_argument("--hw_stride", type=int, default=32)
    parser.add_argument("--force_resolution", action="store_true")
    parser.add_argument("--trained_data_global_step", type=int, default=None)
    parser.add_argument(
        "--video_reader",
        type=str,
        default="decord",
        choices=["decord", "opencv", "pyav"],
        help="what method to use to load videos. Default is decord.",
    )

    # text encoder & vae & diffusion model
    parser.add_argument("--vae_fp32", action="store_true")
    parser.add_argument("--extra_save_mem", action="store_true")
    parser.add_argument("--text_encoder_name_1", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--text_encoder_name_2", type=str, default=None)
    parser.add_argument("--sparse1d", action="store_true")
    parser.add_argument("--sparse_n", type=int, default=2)
    parser.add_argument("--skip_connection", action="store_true")
    parser.add_argument("--cogvideox_scheduler", action="store_true")
    parser.add_argument("--v1_5_scheduler", action="store_true")
    parser.add_argument("--rf_scheduler", action="store_true")
    parser.add_argument(
        "--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"]
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )

    # diffusion setting
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument("--rescale_betas_zero_snr", action="store_true")

    # validation & logs
    parser.add_argument("--enable_profiling", action="store_true")
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5)

    parser.add_argument("--output_dir", default="outputs/", help="The directory where training results are saved.")
    parser.add_argument("--dataset", type=str, default="t2v")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="The training dataset text file specifying the path of video folder, text embedding cache folder, and the annotation json file",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="The validation dataset text file, same format as the training dataset text file.",
    )
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument(
        "--filter_nonexistent",
        type=str2bool,
        default=True,
        help="Whether to filter out non-existent samples in image datasets and video datasets." "Defaults to True.",
    )
    parser.add_argument(
        "--text_embed_cache",
        type=str2bool,
        default=True,
        help="Whether to use T5 embedding cache. Must be provided in image/video_data.",
    )
    parser.add_argument("--vae_latent_folder", default=None, type=str, help="root dir for the vae latent data")
    parser.add_argument("--model", type=str, choices=list(Diffusion_models.keys()), default="OpenSoraT2V_v1_3-2B/122")
    parser.add_argument("--interpolation_scale_h", type=float, default=1.0)
    parser.add_argument("--interpolation_scale_w", type=float, default=1.0)
    parser.add_argument("--interpolation_scale_t", type=float, default=1.0)
    parser.add_argument("--downsampler", type=str, default=None)
    parser.add_argument("--ae", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--ae_path", type=str, default="LanguageBind/Open-Sora-Plan-v1.1.0")
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--train_fps", type=int, default=24)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--max_height", type=int, default=320)
    parser.add_argument("--max_width", type=int, default=240)
    parser.add_argument("--group_frame", action="store_true")
    parser.add_argument("--group_resolution", action="store_true")
    parser.add_argument("--use_rope", action="store_true")
    parser.add_argument("--pretrained", type=str, default=None)

    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--enable_tiling", action="store_true")

    parser.add_argument("--attention_mode", type=str, choices=["xformers", "math", "flash"], default="xformers")
    # parser.add_argument("--text_encoder_name", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--multi_scale", action="store_true")

    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument("--cfg", type=float, default=0.1)
    parser.add_argument(
        "--num_no_recompute",
        type=int,
        default=0,
        help="If use_recompute is True, `num_no_recompute` blocks will be removed from the recomputation list."
        "This is a positive integer which can be tuned based on the memory usage.",
    )
    parser.add_argument("--dataloader_prefetch_size", type=int, default=None, help="minddata prefetch size setting")
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument("--train_sp_batch_size", type=int, default=1, help="Batch size for sequence parallel training")
    parser.add_argument(
        "--vae_keep_gn_fp32",
        default=False,
        type=str2bool,
        help="whether keep GroupNorm in fp32. Defaults to False in inference, better to set to True when training vae",
    )
    parser.add_argument(
        "--vae_precision",
        default="fp16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for vae. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--text_encoder_precision",
        default="bf16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for T5 text encoder. Default is `bf16`, which corresponds to ms.bfloat16",
    )
    parser.add_argument(
        "--enable_parallel_fusion", default=True, type=str2bool, help="Whether to parallel fusion for AdamW"
    )

    parser.add_argument("--noise_offset", type=float, default=0.02, help="The scale of noise offset.")
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: \
            https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. \
            If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    return parser


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args(additional_parse_args=parse_t2v_train_args)
    if args.resume_from_checkpoint == "True":
        args.resume_from_checkpoint = True
    main(args)
