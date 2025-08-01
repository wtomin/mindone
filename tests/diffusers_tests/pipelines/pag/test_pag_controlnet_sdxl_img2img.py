# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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

import random
import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from transformers import CLIPTextConfig

import mindspore as ms

from mindone.diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPAGImg2ImgPipeline
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    floats_tensor,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@ddt
class StableDiffusionXLControlNetPAGImg2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    skip_first_text_encoder = False
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=(32, 64),
                layers_per_block=2,
                sample_size=32,
                in_channels=4,
                out_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                # SD2-specific config below
                attention_head_dim=(2, 4),
                use_linear_projection=True,
                addition_embed_type="text_time",
                addition_time_embed_dim=8,
                transformer_layers_per_block=(1, 2),
                projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
                cross_attention_dim=64 if not skip_first_text_encoder else 32,
            ),
        ],
        [
            "controlnet",
            "diffusers.models.controlnets.controlnet.ControlNetModel",
            "mindone.diffusers.models.controlnets.controlnet.ControlNetModel",
            dict(
                block_out_channels=(32, 64),
                layers_per_block=2,
                in_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                conditioning_embedding_out_channels=(16, 32),
                # SD2-specific config below
                attention_head_dim=(2, 4),
                use_linear_projection=True,
                addition_embed_type="text_time",
                addition_time_embed_dim=8,
                transformer_layers_per_block=(1, 2),
                projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
                cross_attention_dim=64,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                steps_offset=1,
                beta_schedule="scaled_linear",
                timestep_spacing="leading",
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                block_out_channels=[32, 64],
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                latent_channels=4,
            ),
        ],
        [
            "text_encoder",
            "transformers.models.clip.modeling_clip.CLIPTextModel",
            "mindone.transformers.models.clip.modeling_clip.CLIPTextModel",
            dict(
                config=CLIPTextConfig(
                    bos_token_id=0,
                    eos_token_id=2,
                    hidden_size=32,
                    intermediate_size=37,
                    layer_norm_eps=1e-05,
                    num_attention_heads=4,
                    num_hidden_layers=5,
                    pad_token_id=1,
                    vocab_size=1000,
                    # SD2-specific config below
                    hidden_act="gelu",
                    projection_dim=32,
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip",
            ),
        ],
        [
            "text_encoder_2",
            "transformers.models.clip.modeling_clip.CLIPTextModelWithProjection",
            "mindone.transformers.models.clip.modeling_clip.CLIPTextModelWithProjection",
            dict(
                config=CLIPTextConfig(
                    bos_token_id=0,
                    eos_token_id=2,
                    hidden_size=32,
                    intermediate_size=37,
                    layer_norm_eps=1e-05,
                    num_attention_heads=4,
                    num_hidden_layers=5,
                    pad_token_id=1,
                    vocab_size=1000,
                    # SD2-specific config below
                    hidden_act="gelu",
                    projection_dim=32,
                ),
            ),
        ],
        [
            "tokenizer_2",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip",
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "unet",
                "controlnet",
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
                "text_encoder_2",
                "tokenizer_2",
                "image_encoder",
                "feature_extractor",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        controlnet_embedder_scale_factor = 2
        pt_image = floats_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            rng=random.Random(seed),
        )
        ms_image = ms.tensor(pt_image.numpy())

        pt_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "pag_scale": 3.0,
            "output_type": "np",
            "image": pt_image,
            "control_image": pt_image,
        }

        ms_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "pag_scale": 3.0,
            "output_type": "np",
            "image": ms_image,
            "control_image": ms_image,
        }
        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_pag_inference(self, mode, dtype):
        ms.set_context(mode=mode, jit_syntax_level=ms.STRICT, pynative_synchronize=True)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module(
            "diffusers.pipelines.pag.pipeline_pag_controlnet_sd_xl_img2img.StableDiffusionXLControlNetPAGImg2ImgPipeline"
        )
        ms_pipe_cls = get_module(
            "mindone.diffusers.pipelines.pag.pipeline_pag_controlnet_sd_xl_img2img.StableDiffusionXLControlNetPAGImg2ImgPipeline"
        )

        pt_pipe_pag = pt_pipe_cls(**pt_components, pag_applied_layers=["mid", "up", "down"])
        ms_pipe_pag = ms_pipe_cls(**ms_components, pag_applied_layers=["mid", "up", "down"])

        pt_pipe_pag.set_progress_bar_config(disable=None)
        ms_pipe_pag.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe_pag = pt_pipe_pag.to(pt_dtype)
        ms_pipe_pag = ms_pipe_pag.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe_pag(**pt_inputs).images
        torch.manual_seed(0)
        ms_image = ms_pipe_pag(**ms_inputs)[0]

        pt_image_slice = pt_image[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold


@slow
@ddt
class StableDiffusionXLControlNetPAGImg2ImgPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_pag_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small",
            variant="fp16",
            use_safetensors="True",
            mindspore_dtype=ms_dtype,
        )
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", mindspore_dtype=ms_dtype)
        pipe = StableDiffusionXLControlNetPAGImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            mindspore_dtype=ms_dtype,
            enable_pag=True,
        )

        prompt = "A robot, 4k photo"
        image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "cat.png",
            subfolder="kandinsky",
        ).resize((1024, 1024))
        controlnet_conditioning_scale = 0.5  # recommended for good generalization
        depth_image = load_downloaded_image_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            "depth_image.png",
            subfolder="pag",
        )

        torch.manual_seed(0)
        image = pipe(
            prompt,
            image=image,
            control_image=depth_image,
            strength=0.99,
            num_inference_steps=50,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"pag_controlnet_sdxl_img2img_{dtype}.npy",
            subfolder="pag",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
