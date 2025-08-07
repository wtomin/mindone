import unittest

import mindspore as ms
from mindspore import Tensor
from PIL import Image
import numpy as np

from src.flux.xflux_pipeline import XFluxPipeline
from src.flux.util import load_ae, load_clip, load_t5, load_flow_model

class TestXFluxPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_type = "flux-dev"
        cls.test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cls.test_prompt = "cyberpunk dining room, full hd, cinematic"
        
        # Create dummy tensor inputs
        cls.batch_size = 1
        cls.seq_len = 77  # Standard CLIP text sequence length
        cls.img_size = 512
        cls.latent_channels = 16
        cls.hidden_size = 768
        
        # Dummy tensors for different modules
        cls.dummy_image = Tensor(np.random.randn(cls.batch_size, 3, cls.img_size, cls.img_size), dtype=ms.float32)
        cls.dummy_latents = Tensor(np.random.randn(cls.batch_size, cls.latent_channels, cls.img_size//8, cls.img_size//8), dtype=ms.float32)
        cls.dummy_text_embeddings = Tensor(np.random.randn(cls.batch_size, cls.seq_len, cls.hidden_size), dtype=ms.float32)
        cls.dummy_timesteps = Tensor([999], dtype=ms.int64)
        
    def setUp(self):
        self.pipeline = XFluxPipeline(self.model_type)

    def test_load_ae_with_input(self):
        """Test autoencoder loading and forward pass"""
        ae = load_ae(self.model_type)
        self.assertIsNotNone(ae)
        
        # Test encode
        encoded = ae.encode(self.dummy_image)
        self.assertEqual(encoded.shape, (self.batch_size, self.latent_channels, self.img_size//8, self.img_size//8))
        
        # Test decode
        decoded = ae.decode(self.dummy_latents)
        self.assertEqual(decoded.shape, (self.batch_size, 3, self.img_size, self.img_size))

    def test_load_clip_with_input(self):
        """Test CLIP model loading and forward pass"""
        clip = load_clip()
        self.assertIsNotNone(clip)
        
        # Create dummy text input
        dummy_input_ids = Tensor(np.random.randint(0, 49408, (self.batch_size, self.seq_len)), dtype=ms.int64)
        dummy_attention_mask = Tensor(np.ones((self.batch_size, self.seq_len)), dtype=ms.int64)
        
        # Test text encoder
        text_outputs = clip(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask
        )
        
        self.assertEqual(text_outputs.last_hidden_state.shape, 
                        (self.batch_size, self.seq_len, self.hidden_size))

    def test_load_t5_with_input(self):
        """Test T5 model loading and forward pass"""
        t5 = load_t5(max_length=512)
        self.assertIsNotNone(t5)
        
        # Create dummy text input
        dummy_input_ids = Tensor(np.random.randint(0, 32128, (self.batch_size, 512)), dtype=ms.int64)
        dummy_attention_mask = Tensor(np.ones((self.batch_size, 512)), dtype=ms.int64)
        
        # Test encoder
        encoder_outputs = t5.encoder(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask
        )
        
        self.assertEqual(encoder_outputs.last_hidden_state.shape, 
                        (self.batch_size, 512, self.hidden_size))

    def test_load_flow_model_with_input(self):
        """Test flow model loading and forward pass"""
        model = load_flow_model(self.model_type)
        self.assertIsNotNone(model)
        
        # Create dummy inputs for flow model
        dummy_img = Tensor(np.random.randn(self.batch_size, self.latent_channels, 64, 64), dtype=ms.float32)
        dummy_img_ids = Tensor(np.random.randint(0, 64*64, (self.batch_size, 64*64)), dtype=ms.int64)
        dummy_txt = self.dummy_text_embeddings
        dummy_txt_ids = Tensor(np.random.randint(0, 77, (self.batch_size, 77)), dtype=ms.int64)
        dummy_timesteps = self.dummy_timesteps
        dummy_y = Tensor(np.random.randn(self.batch_size, 256), dtype=ms.float32)
        
        # Test model forward pass
        output = model(
            img=dummy_img,
            img_ids=dummy_img_ids,
            txt=dummy_txt,
            txt_ids=dummy_txt_ids,
            timesteps=dummy_timesteps,
            y=dummy_y
        )
        
        self.assertEqual(output.shape, dummy_img.shape)

    def test_set_controlnet_with_input(self):
        """Test controlnet setup and forward pass"""
        control_type = "canny"
        repo_id = "XLabs-AI/flux-controlnet-canny-v3"
        name = "flux-canny-controlnet-v3.safetensors"
        
        # Setup controlnet
        self.pipeline.set_controlnet(
            control_type=control_type,
            repo_id=repo_id,
            name=name
        )
        
        # Create dummy controlnet condition
        dummy_condition = Tensor(np.random.randn(self.batch_size, 3, self.img_size, self.img_size), 
                               dtype=ms.float32)
        
        # Test controlnet forward pass
        down_block_res_samples, mid_block_res_sample = self.pipeline.controlnet(
            hidden_states=self.dummy_latents,
            timesteps=self.dummy_timesteps,
            encoder_hidden_states=self.dummy_text_embeddings,
            controlnet_cond=dummy_condition,
            return_dict=False
        )
        
        # Verify outputs
        self.assertEqual(len(down_block_res_samples), 12)  # Typically 12 down blocks
        self.assertIsNotNone(mid_block_res_sample)

    def test_pipeline_inference_with_controlnet(self):
        """Test full pipeline inference with controlnet"""
        test_image = Image.fromarray(self.test_image)
        
        # Setup controlnet
        self.pipeline.set_controlnet(
            control_type="canny",
            repo_id="XLabs-AI/flux-controlnet-canny-v3",
            name="flux-canny-controlnet-v3.safetensors"
        )
        
        # Run inference with specific tensor shapes
        result = self.pipeline(
            prompt=self.test_prompt,
            controlnet_image=test_image,
            width=self.img_size,
            height=self.img_size,
            guidance=4.0,
            num_steps=25,
            seed=123456789,
            true_gs=4.0,
            control_weight=0.8,
            timestep_to_start_cfg=1
        )
        
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (self.img_size, self.img_size))

    def test_tensor_device_and_dtype(self):
        """Test tensor device and dtype consistency"""
        # Test if all models are using consistent dtype
        self.assertEqual(self.dummy_latents.dtype, ms.float32)
        self.assertEqual(self.dummy_text_embeddings.dtype, ms.float32)
        
        # Test if pipeline components maintain dtype
        self.pipeline.set_controlnet(
            control_type="canny",
            repo_id="XLabs-AI/flux-controlnet-canny-v3",
            name="flux-canny-controlnet-v3.safetensors"
        )
        
        self.assertEqual(self.pipeline.controlnet.dtype, ms.float32)
        self.assertEqual(self.pipeline.model.dtype, ms.float32)

if __name__ == '__main__':
    unittest.main() 