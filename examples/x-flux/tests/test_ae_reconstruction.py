"""
Before running this script, please download the images via:
mkdir tests/assets
wget -P tests/assets https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg

Then run `python -m unittest tests.test_ae_reconstruction`

"""
import unittest
import os
from PIL import Image
import numpy as np
import mindspore as ms
from mindspore import Tensor

from src.flux.util import load_ae

class TestAEReconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Change this to your test image path
        cls.image_path = "tests/assets/truck.jpg"
        cls.output_path = "tests/outputs/recon.png"
        os.makedirs(os.path.dirname(cls.output_path), exist_ok=True)

        # If the test image doesn't exist, create a random one for testing
        if not os.path.exists(cls.image_path):
            raise FileNotFoundError(f"Image file not found at {cls.image_path}. Please download the image from https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg")

        cls.ae = load_ae("flux-dev")

    def load_image_tensor(self, path, size=512):
        img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
        arr = np.array(img).astype(np.float32) / 255.0  # [0,1]
        arr = arr.transpose(2, 0, 1)  # CHW
        arr = np.expand_dims(arr, 0)  # NCHW
        return Tensor(arr, dtype=ms.float32)

    def to_image(self, tensor):
        arr = tensor.asnumpy()
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
        arr = arr[0].transpose(1, 2, 0)  # HWC
        return Image.fromarray(arr)

    def test_reconstruct_image(self):
        x = self.load_image_tensor(self.image_path, size=512)

        # Encode -> Decode
        z = self.ae.encode(x)
        self.assertIsNotNone(z)
        self.assertEqual(z.ndim, 4)  # NCHW latents

        x_rec = self.ae.decode(z)
        self.assertIsNotNone(x_rec)
        self.assertEqual(x_rec.shape, x.shape)

        # Save reconstruction for inspection
        img_rec = self.to_image(x_rec)
        img_rec.save(self.output_path)

        # Basic sanity check: reconstructed image dimensions
        self.assertEqual(img_rec.size, (512, 512))

if __name__ == "__main__":
    unittest.main()
