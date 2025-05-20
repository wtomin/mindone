from time import time

from transformers import AutoTokenizer

import mindspore as ms
from mindspore import Tensor

from mindone.transformers import CohereForCausalLM

ms.set_context(mode=ms.PYNATIVE_MODE)


def main():
    model_id = "CohereLabs/c4ai-command-r-v01"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = CohereForCausalLM.from_pretrained(model_id, mindspore_dtype=ms.float16)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    message = [{"role": "user", "content": "How do plants make energy?"}]
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
    input_ids = (
        Tensor(input_ids) if (len(input_ids.shape) == 2 and input_ids.shape[0] == 1) else Tensor(input_ids).unsqueeze(0)
    )  # (1, L)
    infer_start = time()
    output = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.3,
        cache_implementation="static",
    )
    print(f"Inference time: {time() - infer_start:.3f}s")
    print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
