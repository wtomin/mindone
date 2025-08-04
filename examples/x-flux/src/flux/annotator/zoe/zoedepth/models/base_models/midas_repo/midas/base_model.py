import mindspore as ms
from mindspore import mint, nn
import mindspore


class BaseModel(mindspore.nn.Cell):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        try:
            import torch
        except ImportError:
            raise ImportError("Please install torch to load model checkpoints.")


        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
