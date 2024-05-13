import mindspore as ms
from mindspore import nn, ops


class GELU(nn.Cell):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out, has_bias=bias)
        self.approximate = approximate
        assert self.approximate != "none", "Expect approximate is not none"
        self.gelu = ops.GeLU()

    def construct(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class ApproximateGELU(nn.Cell):
    r"""
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://arxiv.org/abs/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out, has_bias=bias)
        self.sigmoid = ops.Sigmoid()

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.proj(x)
        return x * self.sigmoid(1.702 * x)
