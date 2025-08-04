import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from PIL import Image
import numpy as np
import mindspore as ms
from mindspore import Tensor

# Add the example's source directory to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../examples/x-flux')))

from src.flux.xflux_pipeline import XFluxPipeline
from src.flux.sampling import prepare
from src.flux.util import Annotator

class TestPipelineComponentLoading(unittest.TestCase):
    """
    Tests the first part of the pipeline: loading components like ControlNet,
    the Canny Annotator, and preparing text embeddings.
    """

    @patch('src.flux.util.load_t5')
    @patch('src.flux.util.load_clip')
    @patch('src.flux.util.load_ae')
    @patch('src.flux.util.load_flow_model')
    def setUp(self, mock_load_flow, mock_load_ae, mock_load_clip, mock_load_t5):
        """Set up a mock pipeline for testing, avoiding large model loading."""
        self.mock_flow_model = MagicMock()
        # Provide a dummy processor dict, as the pipeline iterates over this
        self.mock_flow_model.attn_processors = {"double_blocks.0.attn1": MagicMock()}
        mock_load_flow.return_value = self.mock_flow_model

        # Mock text encoders and their outputs
        mock_t5_instance = MagicMock()
        mock_t5_instance.encode = MagicMock(return_value=(
            ms.ops.ones((1, 128, 4096), ms.bfloat16),  # txt
            ms.ops.ones((1, 128), ms.int32)           # txt_ids
        ))
        mock_clip_instance = MagicMock()
        mock_clip_instance.encode = MagicMock(return_value=ms.ops.ones((1, 1, 1, 768), ms.bfloat16)) # vec
        
        mock_load_t5.return_value = mock_t5_instance
        mock_load_clip.return_value = mock_clip_instance
        mock_load_ae.return_value = MagicMock()

        self.pipeline = XFluxPipeline(model_type="flux-dev")

    @patch('src.flux.xflux_pipeline.load_checkpoint', return_value={})
    def test_load_controlnet_and_annotator(self, mock_load_checkpoint):
        """Tests the loading of ControlNet and the Canny annotator."""
        self.assertFalse(self.pipeline.controlnet_loaded)
        
        self.pipeline.set_controlnet(control_type='canny')
        
        self.assertTrue(self.pipeline.controlnet_loaded)
        self.assertIsNotNone(self.pipeline.controlnet)
        self.assertIsNotNone(self.pipeline.annotator)
        self.assertIsInstance(self.pipeline.annotator, Annotator)
        self.assertEqual(self.pipeline.control_type, 'canny')
        
        mock_load_checkpoint.assert_called_once()

    def test_prepare_t5_and_clip(self):
        """Tests the prepare function with mock T5 and CLIP models."""
        dummy_latents = ms.ops.zeros((1, 16, 32, 32), ms.bfloat16)
        prompt = "a cyberpunk dining room"

        inp_cond = prepare(t5=self.pipeline.t5, clip=self.pipeline.clip, img=dummy_latents, prompt=prompt)

        self.pipeline.t5.encode.assert_called_once_with(prompt)
        self.pipeline.clip.encode.assert_called_once_with(dummy_latents)

        self.assertIn('txt', inp_cond)
        self.assertIn('txt_ids', inp_cond)
        self.assertIn('vec', inp_cond)
        self.assertIsInstance(inp_cond['txt'], Tensor)


class TestPipelineConstruct(unittest.TestCase):
    """
    Tests the second part of the pipeline: the end-to-end `construct` method
    to ensure it produces a valid image output.
    """
    @patch('src.flux.util.load_t5')
    @patch('src.flux.util.load_clip')
    @patch('src.flux.util.load_ae')
    @patch('src.flux.util.load_flow_model')
    def setUp(self, mock_load_flow, mock_load_ae, mock_load_clip, mock_load_t5):
        """Set up a mock pipeline for testing construct."""
        self.mock_flow_model = MagicMock()
        self.mock_flow_model.attn_processors = {"double_blocks.0.attn1": MagicMock()}
        mock_load_flow.return_value = self.mock_flow_model
        
        self.mock_ae = MagicMock()
        mock_load_ae.return_value = self.mock_ae

        self.pipeline = XFluxPipeline(model_type="flux-dev")
        
    @patch('src.flux.xflux_pipeline.load_checkpoint', return_value={})
    @patch('src.flux.xflux_pipeline.denoise_controlnet')
    def test_construct_with_canny_controlnet(self, mock_denoise, mock_load_checkpoint):
        """Tests that construct runs with controlnet and produces an image."""
        width, height = 256, 256
        latent_channels = 16
        latent_h, latent_w = height // 8, width // 8
        
        # 1. Mock the output of the denoising loop
        mock_denoise.return_value = ms.ops.zeros((1, latent_channels, latent_h, latent_w), ms.float32)

        # 2. Mock the VAE decoder output
        self.pipeline.ae.decode.return_value = ms.ops.zeros((1, 3, height, width), ms.float32)

        # 3. Setup the pipeline with a canny controlnet
        self.pipeline.set_controlnet(control_type='canny')

        # 4. Create a dummy control image and preprocess it as in the __call__ method
        dummy_control_image = Image.new('RGB', (width, height), color='red')
        processed_control_image = self.pipeline.annotator(dummy_control_image, width, height)
        control_tensor = Tensor.from_numpy((np.array(processed_control_image) / 127.5) - 1.0)
        control_tensor = control_tensor.permute(2, 0, 1).unsqueeze(0).to(ms.bfloat16)

        # 5. Execute the construct method
        result_image = self.pipeline.construct(
            prompt="a test prompt",
            width=width,
            height=height,
            guidance=4.0,
            num_steps=2, # Use few steps for testing
            seed=123,
            controlnet_image=control_tensor
        )

        # 6. Assert the results
        self.assertIsNotNone(result_image)
        self.assertIsInstance(result_image, Image.Image)
        self.assertEqual(result_image.size, (width, height))
        
        # Verify that denoise_controlnet and the VAE decoder were called
        mock_denoise.assert_called_once()
        self.pipeline.ae.decode.assert_called_once()

if __name__ == '__main__':
    unittest.main() 