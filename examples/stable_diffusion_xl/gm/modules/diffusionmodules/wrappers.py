# This code is adapted from https://github.com/Stability-AI/generative-models
# with modifications to run on MindSpore.

import mindspore as ms
from mindspore import Tensor, nn, ops


class IdentityWrapper(nn.Cell):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model

    @ms.jit
    def construct(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    @ms.jit
    def construct(
        self, x: Tensor, t: Tensor, concat: Tensor = None, context: Tensor = None, y: Tensor = None, **kwargs
    ) -> Tensor:
        if concat is not None:
            x = ops.concat((x, concat), axis=1)
        return self.diffusion_model(x, timesteps=t, context=context, y=y, **kwargs)
