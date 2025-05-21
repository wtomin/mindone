# Cohere

## Overview

Cohere Command-R is a 35B parameter multilingual large language model designed for long context tasks like retrieval-augmented generation (RAG) and calling external APIs and tools.

See the original Command-R checkpoints from [HuggingFace](https://huggingface.co/collections/CohereLabs/command-models-67652b401665205e17b192ad).

## Requirements:

|mindspore |	ascend driver | firmware | cann tookit/kernel|
|--- | --- | --- | --- |
|2.5.0 | 24.1RC3 | 7.3.0.1.231 | 8.0.RC3.beta1|

### Installation
```
# install mindone
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install .
```

### Model Checkpoint

We recommend to download the model checkpoint from [HuggingFace](https://huggingface.co/CohereLabs/c4ai-command-r-v01) using command line:

```bash
huggingface-cli download --resume-download CohereLabs/c4ai-command-r-v01
```

`CohereLabs/c4ai-command-r-v01` is a 32B model, which takes about 70GB storage room.


## Usage

The example below shows how to use the Cohere Command-R model:
```python
from time import time
from functools import partial

from transformers import AutoTokenizer

import mindspore as ms
from mindspore import Tensor

from mindone.transformers import CohereForCausalLM
import mindspore
import mindspore.mint.distributed as dist
from mindspore.communication import GlobalComm

from mindone.trainers.zero import prepare_network

ms.set_context(mode=ms.PYNATIVE_MODE)
dist.init_process_group(backend="hccl")
mindspore.set_auto_parallel_context(parallel_mode="data_parallel")

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
```

To run the script, you can use the following command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
msrun --worker_num=2 --local_worker_num=2 --master_port=9000 --log_dir=msrun_log --join=True --cluster_time_out=300 generate.py
```
