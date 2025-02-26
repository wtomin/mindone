from dataclasses import dataclass

from diffusers.utils import BaseOutput

from mindspore import Tensor


@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    r"""
    Output class for HunyuanVideo pipelines.

    Args:
        frames (`Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: Tensor
