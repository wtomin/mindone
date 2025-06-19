import os
import uuid

import numpy as np
from PIL import ExifTags, Image
from src.flux.modules.layers import (
    DoubleStreamBlockLoraProcessor,
    DoubleStreamBlockProcessor,
    ImageProjModel,
    IPDoubleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    SingleStreamBlockProcessor,
)
from src.flux.sampling import denoise, denoise_controlnet, get_noise, get_schedule, prepare, unpack
from src.flux.util import (
    Annotator,
    get_lora_rank,
    load_ae,
    load_checkpoint,
    load_clip,
    load_controlnet,
    load_flow_model,
    load_flow_model_quintized,
    load_t5,
)
from transformers import CLIPImageProcessor

import mindspore as ms
from mindspore import Tensor

from mindone.transformers import CLIPVisionModelWithProjection


class XFluxPipeline:
    def __init__(self, model_type, device, offload: bool = False):
        self.offload = offload
        self.model_type = model_type

        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(
            model_type,
        )
        if "fp8" in model_type:
            self.model = load_flow_model_quintized(
                model_type,
            )
        else:
            self.model = load_flow_model(
                model_type,
            )

        self.image_encoder_path = "openai/clip-vit-large-patch14"
        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False
        self.ip_loaded = False

    def set_ip(self, local_path: str = None, repo_id=None, name: str = None):
        # unpack checkpoint
        checkpoint = load_checkpoint(local_path, repo_id, name)
        prefix = "double_blocks."
        blocks = {}
        proj = {}

        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix) :].replace(".processor.", ".")] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model.") :]] = value

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(dtype=ms.float16)
        self.clip_image_processor = CLIPImageProcessor()

        # setup image embedding projection model
        self.improj = ImageProjModel(4096, 768, 4)
        self.improj.load_state_dict(proj)
        self.improj = self.improj.to(dtype=ms.bfloat16)

        ip_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            ip_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    ip_state_dict[k.replace(f"{name}.", "")] = checkpoint[k]
            if ip_state_dict:
                ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
                ip_attn_procs[name].load_state_dict(ip_state_dict)
                ip_attn_procs[name].to(dtype=ms.bfloat16)
            else:
                ip_attn_procs[name] = self.model.attn_processors[name]

        self.model.set_attn_processor(ip_attn_procs)
        self.ip_loaded = True

    def set_lora(self, local_path: str = None, repo_id: str = None, name: str = None, lora_weight: int = 0.7):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 0.7):
        checkpoint = load_checkpoint(None, self.hf_lora_collection, self.lora_types_to_names[lora_type])
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1 :]] = checkpoint[k] * lora_weight

            if len(lora_state_dict):
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=3072, rank=rank)
                else:
                    lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
                lora_attn_procs[name].load_state_dict(lora_state_dict)

            else:
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockProcessor()
                else:
                    lora_attn_procs[name] = DoubleStreamBlockProcessor()

        self.model.set_attn_processor(lora_attn_procs)

    def set_controlnet(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None):
        self.controlnet = load_controlnet(self.model_type).to(ms.bfloat16)

        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.controlnet.load_state_dict(checkpoint, strict=False)
        self.annotator = Annotator(control_type)
        self.controlnet_loaded = True
        self.control_type = control_type

    def get_image_proj(
        self,
        image_prompt: Tensor,
    ):
        # encode image-prompt embeds
        image_prompt = self.clip_image_processor(images=image_prompt, return_tensors="pt").pixel_values

        image_prompt_embeds = self.image_encoder(image_prompt).image_embeds.to(
            dtype=ms.bfloat16,
        )
        # encode image
        image_proj = self.improj(image_prompt_embeds)
        return image_proj

    def __call__(
        self,
        prompt: str,
        image_prompt: Image = None,
        controlnet_image: Image = None,
        width: int = 512,
        height: int = 512,
        guidance: float = 4,
        num_steps: int = 50,
        seed: int = 123456789,
        true_gs: float = 3,
        control_weight: float = 0.9,
        ip_scale: float = 1.0,
        neg_ip_scale: float = 1.0,
        neg_prompt: str = "",
        neg_image_prompt: Image = None,
        timestep_to_start_cfg: int = 0,
    ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)
        image_proj = None
        neg_image_proj = None
        if not (image_prompt is None and neg_image_prompt is None):
            assert self.ip_loaded, "You must setup IP-Adapter to add image prompt as input"

            if image_prompt is None:
                image_prompt = np.zeros((width, height, 3), dtype=np.uint8)
            if neg_image_prompt is None:
                neg_image_prompt = np.zeros((width, height, 3), dtype=np.uint8)

            image_proj = self.get_image_proj(image_prompt)
            neg_image_proj = self.get_image_proj(neg_image_prompt)

        if self.controlnet_loaded:
            controlnet_image = self.annotator(controlnet_image, width, height)
            controlnet_image = ms.Tensor((np.array(controlnet_image) / 127.5) - 1)
            controlnet_image = controlnet_image.permute(2, 0, 1).unsqueeze(0).to(ms.bfloat16)

        return self.construct(
            prompt,
            width,
            height,
            guidance,
            num_steps,
            seed,
            controlnet_image,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            control_weight=control_weight,
            neg_prompt=neg_prompt,
            image_proj=image_proj,
            neg_image_proj=neg_image_proj,
            ip_scale=ip_scale,
            neg_ip_scale=neg_ip_scale,
        )

    def gradio_generate(
        self,
        prompt,
        image_prompt,
        controlnet_image,
        width,
        height,
        guidance,
        num_steps,
        seed,
        true_gs,
        ip_scale,
        neg_ip_scale,
        neg_prompt,
        neg_image_prompt,
        timestep_to_start_cfg,
        control_type,
        control_weight,
        lora_weight,
        local_path,
        lora_local_path,
        ip_local_path,
    ):
        if controlnet_image is not None:
            controlnet_image = Image.fromarray(controlnet_image)
            if (self.controlnet_loaded and control_type != self.control_type) or not self.controlnet_loaded:
                if local_path is not None:
                    self.set_controlnet(control_type, local_path=local_path)
                else:
                    self.set_controlnet(
                        control_type,
                        local_path=None,
                        repo_id=f"xlabs-ai/flux-controlnet-{control_type}-v3",
                        name=f"flux-{control_type}-controlnet-v3.safetensors",
                    )
        if lora_local_path is not None:
            self.set_lora(local_path=lora_local_path, lora_weight=lora_weight)
        if image_prompt is not None:
            image_prompt = Image.fromarray(image_prompt)
            if neg_image_prompt is not None:
                neg_image_prompt = Image.fromarray(neg_image_prompt)
            if not self.ip_loaded:
                if ip_local_path is not None:
                    self.set_ip(local_path=ip_local_path)
                else:
                    self.set_ip(repo_id="xlabs-ai/flux-ip-adapter", name="flux-ip-adapter.safetensors")
        seed = int(seed)
        if seed == -1:
            seed = np.random.Generator().seed()

        img = self(
            prompt,
            image_prompt,
            controlnet_image,
            width,
            height,
            guidance,
            num_steps,
            seed,
            true_gs,
            control_weight,
            ip_scale,
            neg_ip_scale,
            neg_prompt,
            neg_image_prompt,
            timestep_to_start_cfg,
        )

        filename = f"output/gradio/{uuid.uuid4()}.jpg"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Make] = "XLabs AI"
        exif_data[ExifTags.Base.Model] = self.model_type
        img.save(filename, format="jpeg", exif=exif_data, quality=95, subsampling=0)
        return img, filename

    def construct(
        self,
        prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        controlnet_image=None,
        timestep_to_start_cfg=0,
        true_gs=3.5,
        control_weight=0.9,
        neg_prompt="",
        image_proj=None,
        neg_image_proj=None,
        ip_scale=1.0,
        neg_ip_scale=1.0,
    ):
        x = get_noise(1, height, width, dtype=ms.bfloat16, seed=seed)
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )

        inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
        neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)

        if self.controlnet_loaded:
            x = denoise_controlnet(
                self.model,
                **inp_cond,
                controlnet=self.controlnet,
                timesteps=timesteps,
                guidance=guidance,
                controlnet_cond=controlnet_image,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond["txt"],
                neg_txt_ids=neg_inp_cond["txt_ids"],
                neg_vec=neg_inp_cond["vec"],
                true_gs=true_gs,
                controlnet_gs=control_weight,
                image_proj=image_proj,
                neg_image_proj=neg_image_proj,
                ip_scale=ip_scale,
                neg_ip_scale=neg_ip_scale,
            )
        else:
            x = denoise(
                self.model,
                **inp_cond,
                timesteps=timesteps,
                guidance=guidance,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond["txt"],
                neg_txt_ids=neg_inp_cond["txt_ids"],
                neg_vec=neg_inp_cond["vec"],
                true_gs=true_gs,
                image_proj=image_proj,
                neg_image_proj=neg_image_proj,
                ip_scale=ip_scale,
                neg_ip_scale=neg_ip_scale,
            )

        x = unpack(x.float(), height, width)
        x = self.ae.decode(x)

        x1 = x.clamp(-1, 1)
        # x1 = rearrange(x1[-1], "c h w -> h w c")
        x1 = x1[-1].permute(1, 2, 0)
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).asnumpy().to(np.uint8))
        return output_img


class XFluxSampler(XFluxPipeline):
    def __init__(self, clip, t5, ae, model, device):
        self.clip = clip
        self.t5 = t5
        self.ae = ae
        self.model = model
        self.model.set_train(False)
        self.controlnet_loaded = False
        self.ip_loaded = False
        self.offload = False
