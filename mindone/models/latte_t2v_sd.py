#################################################################################
#                         Video DiT (Experimental)                              #
#################################################################################
import logging
import os

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import XavierUniform, initializer

from mindone.models.modules.flash_attention import FLASH_IS_AVAILABLE, MSFlashAttention

from .dit import (
    GELU,
    Attention,
    FinalLayer,
    LayerNorm,
    LinearPatchEmbed,
    Mlp,
    PatchEmbed,
    SelfAttention,
    TimestepEmbedder,
)
from .modules import get_1d_sincos_temp_embed, get_2d_sincos_pos_embed
from .utils import constant_, default, modulate, normal_, xavier_uniform_

logger = logging.getLogger(__name__)


class MultiHeadCrossAttention(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads=8,
        context_dim=None,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        dtype=ms.float32,
        enable_flash_attention=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dtype = dtype
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        context_dim = default(context_dim, dim)

        self.to_q = nn.Dense(dim, dim, has_bias=qkv_bias).to_float(self.dtype)
        self.to_kv = nn.Dense(context_dim, dim * 2, has_bias=qkv_bias).to_float(self.dtype)
        self.proj = nn.Dense(dim, dim).to_float(self.dtype)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

        self.attention = Attention(head_dim, attn_drop=attn_drop)

        self.enable_flash_attention = (
            enable_flash_attention and FLASH_IS_AVAILABLE and (ms.context.get_context("device_target") == "Ascend")
        )

        if self.enable_flash_attention:
            self.flash_attention = MSFlashAttention(
                head_dim=head_dim, head_num=num_heads, fix_head_dims=[72], attention_dropout=attn_drop
            )
        else:
            self.flash_attention = None

    @staticmethod
    def _rearange_in(x, h):
        # (b, n, h*d) -> (b*h, n, d)
        b, n, d = x.shape
        d = d // h

        x = ops.reshape(x, (b, n, h, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b * h, n, d))
        return x

    @staticmethod
    def _rearange_out(x, h):
        # (b*h, n, d) -> (b, n, h*d)
        b, n, d = x.shape
        b = b // h

        x = ops.reshape(x, (b, h, n, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, n, h * d))
        return x

    def construct(self, x, context=None, mask=None):
        x_dtype = x.dtype
        h = self.num_heads

        q = self.to_q(x)
        context = default(context, x)
        # (b, n, 2*h*d) -> (b, n, 2, h*d)  -> (b, n, h*d), (b, n, h*d)
        k, v = self.to_kv(context).reshape(context.shape[0], context.shape[1], 2, -1).unbind(2)
        q_b, q_n, _ = q.shape  # (b n h*d)
        k_b, k_n, _ = k.shape
        v_b, v_n, _ = v.shape

        head_dim = q.shape[-1] // h
        # convert sequence mask to attention mask: (b, q_n) to (b, q_n, k_n)
        if mask is not None:
            mask = self.reshape(mask, (mask.shape[0], -1))
            attn_mask = ops.zeros((q_b, q_n, k_n), self.dtype)
            mask = ops.expand_dims(mask, axis=1)  # (q_b, q_n, 1)
            attn_mask = attn_mask.masked_fill(~mask, -ms.numpy.inf)
            mask = attn_mask

        if (
            self.enable_flash_attention and q_n % 16 == 0 and k_n % 16 == 0 and head_dim <= 256
        ):  # TODO: why restrict head_dim?
            # reshape qkv shape ((b n h*d) -> (b h n d))and mask dtype for FA input format
            q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
            k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
            v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)
            if mask is not None and mask.dim() != 4:
                mask = ops.expand_dims(mask, axis=1)  # (q_b, 1, q_n, k_n)
            out = self.flash_attention(q, k, v, mask)
            b, h, n, d = out.shape
            # reshape FA output to original attn input format, (b h n d) -> (b n h*d)
            out = out.transpose(0, 2, 1, 3).view(b, n, -1)
        else:
            # (b, n, h*d) -> (b*h, n, d)
            q = self._rearange_in(q, h)
            k = self._rearange_in(k, h)
            v = self._rearange_in(v, h)
            if mask is not None and mask.shape[0] != q.shape[0]:
                mask = mask.repeat(h, axis=0)

            out = self.attention(q, k, v, mask)
            # (b*h, n, d) -> (b, n, h*d)
            out = self._rearange_out(out, h)

        return self.proj_drop(self.proj(out)).to(x_dtype)


class BasicTransformerBlock(nn.Cell):
    """The basic transformer block in stable diffusion, revised to accept Latte arguments.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, context_dim=None, **block_kwargs):
        super().__init__()

        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 6 * hidden_size, has_bias=True))
        # additional cross-attn and norm layers to handle context input
        self.attn2 = MultiHeadCrossAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=True, context_dim=context_dim, **block_kwargs
        )  # is self-attn if context is none
        self.norm_crossattn = LayerNorm(hidden_size, eps=1e-06)

    def construct(self, x, c, context=None, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, axis=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # attn 2 start
        x = x + self.attn2(self.norm_crossattn(x), context=context, mask=mask)
        # attn 2 end
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Latte_T2V_SD(nn.Cell):
    """A Latte_T2V_SD model which consists of [spatial (or temporal) attn -> cross-attention -> mlp] x (depth).
    It combines DiT Block (adaLN-zero) and Cross-Attention, which resembles PixArt-alpha, but differs in adaLN settings.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        num_classes=1000,
        learn_sigma=True,
        block_kwargs={},
        condition="text",
        context_dim=768,
        patch_embedder="conv",
        use_recompute=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes

        if condition is not None:
            assert isinstance(condition, str), f"Expect that the condition type is a string, but got {type(condition)}"
            self.condition = condition.lower()
        else:
            self.condition = condition
        assert self.condition == "text", "Latte_T2V_SD only support text condition!"
        self.patch_embedder = patch_embedder
        self.use_recompute = use_recompute
        if patch_embedder == "conv":
            PatchEmbedder = PatchEmbed
        else:
            PatchEmbedder = LinearPatchEmbed
        self.x_embedder = PatchEmbedder(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        assert self.condition in [None, "text", "class"], f"Unsupported condition type! {self.condition}"

        num_patches = self.x_embedder.num_patches
        # TODO: Text Embedding Projection
        # Will use fixed sin-cos embedding:
        self.pos_embed = ms.Parameter(ops.zeros((1, num_patches, hidden_size), dtype=ms.float32), requires_grad=False)
        self.temp_embed = ms.Parameter(ops.zeros((1, num_frames, hidden_size), dtype=ms.float32), requires_grad=False)

        self.blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio, context_dim=context_dim, **block_kwargs
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        if self.use_recompute:
            for block in self.blocks:
                self.recompute(block)

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute()
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        else:
            b.add_flags(output_no_recompute=True)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Dense):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.set_data(Tensor(pos_embed).float().unsqueeze(0))
        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.set_data(Tensor(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight
        # xavier_uniform_(w.view(w.shape[0], -1))
        w_flatted = w.view(w.shape[0], -1)
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).reshape(w.shape))
        constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        normal_(self.t_embedder.mlp[0].weight, std=0.02)
        normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            constant_(block.adaLN_modulation[-1].weight, 0)
            constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        constant_(self.final_layer.linear.weight, 0)
        constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = x.permute((0, 5, 1, 3, 2, 4))
        imgs = x.reshape((x.shape[0], c, h * p, h * p))
        return imgs

    def construct(self, x, t, text_embed=None, mask=None, **kwargs):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        text_embed: (N, L, D), tensor of text embedding. L is the number of tokens, for CLIP it should be 77
        """
        bs, num_frames, channels, height, width = x.shape
        # (b c f h w) -> (b*f c h w)
        x = x.reshape((bs * num_frames, channels, height, width))
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # time embeddings
        t = self.t_embedder(t)  # (N, D)
        # (N, D) -> (N*num_frames, D)
        t_spatial = t.repeat_interleave(repeats=self.temp_embed.shape[1], dim=0)
        # (N, D) -> (N*T, D)
        t_temp = t.repeat_interleave(repeats=self.pos_embed.shape[1], dim=0)

        # text embedding
        if text_embed is not None:
            # N, L, D = text_embed.shape
            # (N, L, D) -> (N*num_frames, L, D)
            text_embed_spatial = text_embed.repeat_interleave(repeats=self.temp_embed.shape[1], dim=0)
            if mask is not None:
                mask_spatial = mask.repeat_interleave(repeats=self.temp_embed.shape[1], dim=0)
            else:
                mask_spatial = None
            # (N, L, D) -> (N*T, L, D)
            text_embed_temp = text_embed.repeat_interleave(repeats=self.pos_embed.shape[1], dim=0)
            if mask is not None:
                mask_temp = mask.repeat_interleave(repeats=self.pos_embed.shape[1], dim=0)
            else:
                mask_temp = None
        else:
            text_embed_spatial, text_embed_temp = None, None
            mask_spatial, mask_temp = None, None

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i : i + 2]
            x = spatial_block(x, t_spatial, text_embed_spatial, mask_spatial)
            _, num_patches, channels = x.shape
            x = (
                x.reshape((bs, num_frames, num_patches, channels))
                .permute((0, 2, 1, 3))
                .reshape((bs * num_patches, num_frames, channels))
            )
            # add time embed
            if i == 0:
                x = x + self.temp_embed

            x = temp_block(x, t_temp, text_embed_temp, mask_temp)

            _, num_frames, channels = x.shape
            x = (
                x.reshape((bs, num_patches, num_frames, channels))
                .permute((0, 2, 1, 3))
                .reshape((bs * num_frames, num_patches, channels))
            )

        x = self.final_layer(x, t_spatial)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        _, channels, h, w = x.shape
        x = x.reshape((bs, num_frames, channels, h, w))
        return x

    def construct_with_cfg(self, x, t, text_embed=None, mask=None, cfg_scale=4.0, **kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        model_out = self.construct(combined, t, text_embed=text_embed, mask=mask, **kwargs)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :, : self.in_channels], model_out[:, :, self.in_channels :]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=2)

    def load_params_from_ckpt(self, ckpt):
        # load param from a ckpt file path or a parameter dictionary
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f"{ckpt} does not exist!"
            logger.info(f"Loading {ckpt} params into Latte_T2V_SD model...")
            param_dict = ms.load_checkpoint(ckpt)
        elif isinstance(ckpt, dict):
            param_dict = ckpt
        else:
            raise ValueError("Expect to receive a ckpt path or parameter dictionary as input!")
        _, ckpt_not_load = ms.load_param_into_net(
            self,
            param_dict,
        )
        if len(ckpt_not_load):
            print(f"{ckpt_not_load} not load")


def Latte_T2V_XL_2(**kwargs):
    return Latte_T2V_SD(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


Latte_T2V_models = {
    "Latte-t2v-XL/2": Latte_T2V_XL_2,
}
