"""
This test uses only the CLIP text encoder to compare text embeddings.

Run with:
pytest tests/test_clip_text_similarity.py
"""
import unittest
import os
import mindspore as ms
from mindspore import Tensor, mint

from src.flux.util import load_clip

def normalize(t: Tensor, eps: float = 1e-8) -> Tensor:
    # L2 normalize along last dimension
    norm = mint.sqrt(
        mint.clamp(mint.sum(t * t, -1, keepdim=True), eps, 1e9)
    )
    return t / norm

class TestCLIPTextSimilarity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_path = "tests/outputs/clip_text_similarity.txt"
        os.makedirs(os.path.dirname(cls.output_path), exist_ok=True)

        # Load CLIP (text encoder only is used)
        cls.clip = load_clip()

        # Prompts
        cls.text_truck = ["a photo of a truck"]
        cls.text_car = ["a photo of a car"]
        cls.text_cat = ["a photo of a cat"]

    def test_clip_text_text_similarity(self):
        # Text features
        txt_truck = self.clip(text=self.text_truck)
        txt_car = self.clip(text=self.text_car)
        txt_cat = self.clip(text=self.text_cat)

        # Sanity checks
        for t in (txt_truck, txt_car, txt_cat):
            self.assertIsNotNone(t)
            self.assertEqual(len(t.shape), 2)  # (batch, hidden_size)

        # Normalize to unit vectors
        truck_n = normalize(txt_truck)
        car_n = normalize(txt_car)
        cat_n = normalize(txt_cat)

        # Cosine similarity = dot product of normalized vectors
        sim_truck_car = mint.sum(truck_n * car_n, -1)  # (1,)
        sim_truck_cat = mint.sum(truck_n * cat_n, -1)  # (1,)

        # Convert to floats
        sim_truck_car_val = float(sim_truck_car.asnumpy()[0])
        sim_truck_cat_val = float(sim_truck_cat.asnumpy()[0])

        # Save results
        with open(self.output_path, "w") as f:
            f.write(f"cosine_similarity('truck','car') = {sim_truck_car_val:.4f}\n")
            f.write(f"cosine_similarity('truck','cat') = {sim_truck_cat_val:.4f}\n")

        # Expect truck closer to car than to cat
        self.assertGreater(sim_truck_car_val, sim_truck_cat_val)

if __name__ == "__main__":
    unittest.main()
