"""
usage:
export ASCEND_RT_VISIBLE_DEVICES=0,1
msrun --worker_num=2 --local_worker_num=2 --master_port=9000 --log_dir=msrun_log --join=True --cluster_time_out=300 generate.py
"""
from functools import partial
from time import time

from transformers import AutoTokenizer

import mindspore
import mindspore as ms
import mindspore.mint.distributed as dist
from mindspore import Tensor
from mindspore.communication import GlobalComm

from mindone.trainers.zero import prepare_network
from mindone.transformers import CohereForCausalLM

ms.set_context(mode=ms.PYNATIVE_MODE)
dist.init_process_group(backend="hccl")
mindspore.set_auto_parallel_context(parallel_mode="data_parallel")


def main():
    model_id = "CohereLabs/c4ai-command-r-v01"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = CohereForCausalLM.from_pretrained(model_id, mindspore_dtype=ms.float16)
    shard_fn = partial(prepare_network, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
    model = shard_fn(model)

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
    print(tokenizer.decode(output[0], skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
