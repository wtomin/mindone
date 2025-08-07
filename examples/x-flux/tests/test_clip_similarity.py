"""
Before running this script, please download the image via:
mkdir -p tests/assets
wget -P tests/assets https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg

Then run `pytest tests/test_clip_similarity.py`
"""
import unittest
import os
from PIL import Image
import numpy as np
import mindspore as ms
from mindspore import Tensor

from src.flux.util import load_clip

def normalize(t: Tensor, eps: float = 1e-8) -> Tensor:
    # L2 normalize along last dimension
    norm = ms.ops.sqrt(ms.ops.clip_by_value(ms.ops.reduce_sum(t * t, -1, keep_dims=True), eps, 1e9))
    return t / norm

class TestCLIPSimilarity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_path = "tests/assets/truck.jpg"
        cls.output_path = "tests/outputs/clip_truck_similarity.txt"
        os.makedirs(os.path.dirname(cls.output_path), exist_ok=True)

        if not os.path.exists(cls.image_path):
            raise FileNotFoundError(
                f"Image file not found at {cls.image_path}. Please download: "
                "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg"
            )

        # Load CLIP encoder(s). The util returns a callable that can encode image or text.
        cls.clip = load_clip()

        # Prompts
        cls.positive_text = ["a photo of a truck"]
        cls.negative_text = ["a photo of a cat"]

    def load_image_tensor(self, path, size=224):
        # Typical CLIP resolution is 224; adjust if your model expects different.
        img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
        arr = np.array(img).astype(np.float32) / 255.0  # [0,1]
        # CLIP normalization (ViT-B/32 style): mean/std per channel
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)  # CHW
        arr = np.expand_dims(arr, 0)  # NCHW
        return Tensor(arr, dtype=ms.float32)

    def test_clip_image_text_similarity(self):
        # Image feature
        img_tensor = self.load_image_tensor(self.image_path, size=224)
        # Assuming load_clip provides a callable that supports image=... or text=[...]
        img_feat = self.clip(image=img_tensor)
        self.assertIsNotNone(img_feat)
        # Shape: (batch, hidden_size)
        self.assertEqual(len(img_feat.shape), 2)

        # Text features
        txt_pos = self.clip(text=self.positive_text)
        txt_neg = self.clip(text=self.negative_text)
        self.assertIsNotNone(txt_pos)
        self.assertIsNotNone(txt_neg)
        self.assertEqual(len(txt_pos.shape), 2)
        self.assertEqual(len(txt_neg.shape), 2)

        # Normalize to unit vectors
        img_feat_n = normalize(img_feat)
        txt_pos_n = normalize(txt_pos)
        txt_neg_n = normalize(txt_neg)

        # Cosine similarity: dot product of normalized vectors
        # Shapes: (1, D) @ (D, 1) -> (1, 1)
        sim_pos = ms.ops.reduce_sum(img_feat_n * txt_pos_n, -1)  # (1,)
        sim_neg = ms.ops.reduce_sum(img_feat_n * txt_neg_n, -1)  # (1,)

        # Convert to python floats
        sim_pos_val = float(sim_pos.asnumpy()[0])
        sim_neg_val = float(sim_neg.asnumpy()[0])

        # Save results for inspection
        with open(self.output_path, "w") as f:
            f.write(f"cosine_similarity(truck_image, 'a photo of a truck') = {sim_pos_val:.4f}\n")
            f.write(f"cosine_similarity(truck_image, 'a photo of a cat')   = {sim_neg_val:.4f}\n")

        # Basic expectation: positive similarity should be higher than negative
        self.assertGreater(sim_pos_val, sim_neg_val)

if __name__ == "__main__":
    unittest.main()
