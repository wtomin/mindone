<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# MochiTransformer3DModel

A Diffusion Transformer model for 3D video-like data was introduced in [Mochi-1 Preview](https://huggingface.co/genmo/mochi-1-preview) by Genmo.

The model can be loaded with the following code snippet.

```python
from mindone.diffusers import MochiTransformer3DModel

transformer = MochiTransformer3DModel.from_pretrained("genmo/mochi-1-preview", subfolder="transformer", mindspore_dtype=ms.float16)
```

::: mindone.diffusers.MochiTransformer3DModel

::: mindone.diffusers.models.modeling_outputs.Transformer2DModelOutput
