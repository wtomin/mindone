from ad.modules.encoders._common import LayerNorm
from ad.modules.encoders.image_encoder import Transformer

import mindspore as ms
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.common.initializer import TruncatedNormal, initializer


class SemanticMotionPredictor(nn.Cell):
    """
    a transformer model to first interpolate the image embedding to the target len and then project it to the output dim via transformer layers
    """

    def __init__(
        self,
        embed_dim: int,
        width: int,
        layers: int,
        heads: int,
        num_frames: int,
        output_dim: int = None,
        epsilon: float = 1e-5,
        use_quick_gelu: bool = False,
        dtype: mstype = ms.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.ln_pre = LayerNorm((width,), epsilon=epsilon)
        self.num_frames = num_frames
        self.positional_embedding = Parameter(initializer(TruncatedNormal(0.01), [num_frames, width], dtype=self.dtype))

        self.transformer = Transformer(
            width, layers, heads, epsilon=epsilon, use_quick_gelu=use_quick_gelu, dtype=self.dtype
        )
        scale = width**-0.5
        self.ln_post = LayerNorm((width,), epsilon=epsilon)
        output_dim = output_dim if output_dim is not None else embed_dim
        self.proj_out = Parameter(scale * ms.numpy.randn(width, output_dim, dtype=self.dtype))
        self.proj_in = Parameter(scale * ms.numpy.randn(embed_dim, width, dtype=self.dtype))

    def construct(self, x: Tensor, target_len: int):
        # x input shape (Bs, 2, hidden_size)
        # interpolate the image embedding to the target length
        # Bs, F, D = x.shape
        x = x @ self.proj_in
        x = ops.interpolate(x.transpose(0, 2, 1), size=target_len, mode="linear").transpose(0, 2, 1)
        x = x + self.positional_embedding

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        if self.proj_out is not None:
            x = x @ self.proj_out
        return x
