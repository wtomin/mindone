"""
This test uses only the T5 text encoder to compare text embeddings.

Run with:
pytest tests/test_t5_text_similarity.py
"""
import unittest
import os
import mindspore as ms
from mindspore import Tensor, mint

from src.flux.util import load_t5

def normalize(t: Tensor, eps: float = 1e-8) -> Tensor:
    # L2 normalize along last dimension
    norm = mint.sqrt(
        mint.clamp(mint.sum(t * t, -1, keepdim=True), eps, 1e9)
    )
    return t / norm

class TestT5TextSimilarity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_path = "tests/outputs/t5_text_similarity.txt"
        os.makedirs(os.path.dirname(cls.output_path), exist_ok=True)

        # Load T5. We only use the encoder for text embeddings.
        # Adjust max_length if your tokenizer/impl expects a specific value.
        cls.t5 = load_t5(max_length=512)

        # Prompts
        cls.text_truck = ["a photo of a truck"]
        cls.text_car = ["a photo of a car"]
        cls.text_cat = ["a photo of a cat"]

    def test_t5_text_text_similarity(self):
        # Get encoder outputs; expect shape (batch, seq_len, hidden_size)
        enc_truck = self.t5(text=self.text_truck)
        enc_car = self.t5(text=self.text_car)
        enc_cat = self.t5(text=self.text_cat)

        # Sanity checks
        for e in (enc_truck, enc_car, enc_cat):
            self.assertIsNotNone(e)
            self.assertEqual(len(e.shape), 3)  # (batch, seq_len, hidden)

        # Pool to get a single embedding per prompt.
        # Use mean pooling over sequence length.
        def mean_pool(x: Tensor) -> Tensor:
            return mint.mean(x, dim=1, keepdim=False)  # (batch, hidden)

        truck_emb = mean_pool(enc_truck)
        car_emb = mean_pool(enc_car)
        cat_emb = mean_pool(enc_cat)

        # Normalize to unit vectors
        truck_n = normalize(truck_emb)
        car_n = normalize(car_emb)
        cat_n = normalize(cat_emb)

        # Cosine similarity = dot product of normalized vectors
        sim_truck_car = mint.sum(truck_n * car_n, -1)  # (1,)
        sim_truck_cat = mint.sum(truck_n * cat_n, -1)  # (1,)

        # Convert to floats
        sim_truck_car_val = float(sim_truck_car.asnumpy()[0])
        sim_truck_cat_val = float(sim_truck_cat.asnumpy()[0])

        # Save results
        with open(self.output_path, "w") as f:
            f.write(f"T5 cosine_similarity('truck','car') = {sim_truck_car_val:.4f}\n")
            f.write(f"T5 cosine_similarity('truck','cat') = {sim_truck_cat_val:.4f}\n")

        # Expect truck closer to car than to cat
        self.assertGreater(sim_truck_car_val, sim_truck_cat_val)

if __name__ == "__main__":
    unittest.main()
