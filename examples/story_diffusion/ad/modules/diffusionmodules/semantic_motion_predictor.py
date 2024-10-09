from ad.modules.encoders._common import LayerNorm
from ad.modules.encoders.image_encoder import Transformer

import mindspore as ms
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops


class SemanticMotionPredictor(nn.Cell):
    """
    a transformer model to first interpolate the image embedding to the target len and then project it to the output dim via transformer layers
    """

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        epsilon: float = 1e-5,
        use_quick_gelu: bool = False,
        dtype: mstype = ms.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.ln_pre = LayerNorm((width,), epsilon=epsilon)

        self.transformer = Transformer(
            width, layers, heads, epsilon=epsilon, use_quick_gelu=use_quick_gelu, dtype=self.dtype
        )
        scale = width**-0.5
        self.ln_post = LayerNorm((width,), epsilon=epsilon)
        self.proj = Parameter(scale * ms.numpy.randn(width, output_dim, dtype=self.dtype))

    def construct(self, x: Tensor, target_len: int):
        # x input shape (Bs, 2, hidden_size)
        # interpolate the image embedding to the target length
        Bs, F, D = x.shape
        x = ops.interpolate(x, size=(target_len, D), mode="linear")

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj
        x = x.reshape(Bs, F, target_len, D)

        return x
