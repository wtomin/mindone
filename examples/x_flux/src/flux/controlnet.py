from dataclasses import dataclass

import mindspore.mint as mint
from mindspore import Tensor, nn

from mindone.models.utils import zeros_

from .modules.layers import DoubleStreamBlock, EmbedND, MLPEmbedder, timestep_embedding


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


def zero_module(module):
    for p in module.get_parameters():
        zeros_(p)
    return module


class ControlNetFlux(nn.Cell):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams, controlnet_depth=2):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = mint.nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else mint.nn.Identity()
        )
        self.txt_in = mint.nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.CellList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(controlnet_depth)
            ]
        )

        # add ControlNet blocks
        self.controlnet_blocks = nn.CellList([])
        for _ in range(controlnet_depth):
            controlnet_block = mint.nn.Linear(self.hidden_size, self.hidden_size)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)
        self.pos_embed_input = mint.nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.gradient_checkpointing = False
        self.input_hint_block = nn.SequentialCell(
            mint.nn.Conv2d(3, 16, 3, padding=1),
            mint.nn.SiLU(),
            mint.nn.Conv2d(16, 16, 3, padding=1),
            mint.nn.SiLU(),
            mint.nn.Conv2d(16, 16, 3, padding=1, stride=2),
            mint.nn.SiLU(),
            mint.nn.Conv2d(16, 16, 3, padding=1),
            mint.nn.SiLU(),
            mint.nn.Conv2d(16, 16, 3, padding=1, stride=2),
            mint.nn.SiLU(),
            mint.nn.Conv2d(16, 16, 3, padding=1),
            mint.nn.SiLU(),
            mint.nn.Conv2d(16, 16, 3, padding=1, stride=2),
            mint.nn.SiLU(),
            zero_module(mint.nn.Conv2d(16, 16, 3, padding=1)),
        )

    @property
    def attn_processors(self):
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: nn.Cell, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.name_cells().items():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.name_cells().items():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: nn.Cell, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.name_cells().items():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.name_cells().items():
            fn_recursive_attn_processor(name, module, processor)

    def construct(
        self,
        img: Tensor,
        img_ids: Tensor,
        controlnet_cond: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        controlnet_cond = self.input_hint_block(controlnet_cond)
        b, c, h_p, w_p = controlnet_cond.shape
        h = h_p // 2
        w = w_p // 2
        controlnet_cond = controlnet_cond.reshape(b, c, h, 2, w, 2)
        controlnet_cond = controlnet_cond.permute(0, 2, 4, 1, 3, 5)
        controlnet_cond = controlnet_cond.reshape(b, h * w, c * 4)
        controlnet_cond = self.pos_embed_input(controlnet_cond)
        img = img + controlnet_cond
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = mint.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        block_res_samples = ()

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

            block_res_samples = block_res_samples + (img,)

        controlnet_block_res_samples = ()
        for block_res_sample, controlnet_block in zip(block_res_samples, self.controlnet_blocks):
            block_res_sample = controlnet_block(block_res_sample)
            controlnet_block_res_samples = controlnet_block_res_samples + (block_res_sample,)

        return controlnet_block_res_samples
