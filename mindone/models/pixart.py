import logging

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import TruncatedNormal, XavierUniform, initializer

from mindone.models.modules.flash_attention import FLASH_IS_AVAILABLE, MSFlashAttention
from mindone.utils.version_control import is_old_ms_version

from .dit import GELU, Attention, LayerNorm, Mlp, PatchEmbed, SelfAttention, TimestepEmbedder
from .modules import _get_2d_sincos_pos_embed_from_grid as get_2d_sincos_pos_embed_from_grid
from .utils import constant_, modulate, normal_, xavier_uniform_

logger = logging.getLogger(__name__)


def trunc_normal_(tensor: Tensor, sigma: float = 0.01):
    tensor.set_data(initializer(TruncatedNormal(sigma), tensor.shape, tensor.dtype))


#################################################################################
#                             Modules for PixArt model                          #
#################################################################################
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = [grid_size, grid_size]
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class T2IFinalLayer(nn.Cell):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Dense(hidden_size, patch_size * patch_size * out_channels, has_bias=True)

        self.scale_shift_table = Parameter(ops.randn((2, hidden_size)) / hidden_size**0.5)
        self.out_channels = out_channels

    def construct(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, axis=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MaskFinalLayer(nn.Cell):
    """
    The final layer of PixArt.
    """

    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Dense(final_hidden_size, patch_size * patch_size * out_channels, has_bias=True)
        self.adaLN_modulation = nn.SequentialCell(nn.SiLU(), nn.Dense(c_emb_size, 2 * final_hidden_size, has_bias=True))

    def construct(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DecoderLayer(nn.Cell):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, decoder_hidden_size):
        super().__init__()
        self.norm_decoder = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Dense(hidden_size, decoder_hidden_size, has_bias=True)
        self.adaLN_modulation = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 2 * hidden_size, has_bias=True))

    def construct(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, axis=1)
        x = modulate(self.norm_decoder(x), shift, scale)
        x = self.linear(x)
        return x


class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=ms.float32):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.SequentialCell(
            nn.Dense(frequency_embedding_size, hidden_size, has_bias=True),
            nn.SiLU(),
            nn.Dense(hidden_size, hidden_size, has_bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size
        self.dtype = dtype

    def construct(self, s, bs):
        if s.dim() == 1:
            s = s[:, None]
        assert s.dim() == 2
        if s.shape[0] != bs:
            s = s.repeat(bs // s.shape[0], axis=0)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        # s = rearrange(s, "b d -> (b d)")
        s = s.reshape(
            -1,
        )
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        # s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        s_emb = s_emb.reshape((b, dims, self.outdim)).reshape((b, dims * self.outdim))
        return s_emb


class CaptionEmbedder(nn.Cell):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=GELU(approximate="tanh"), token_num=120):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0
        )
        self.y_embedding = Parameter(ops.randn(token_num, in_channels) / in_channels**0.5)
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = ops.rand(caption.shape[0]) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = ops.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def construct(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class CaptionEmbedderDoubleBr(nn.Cell):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=GELU(approximate="tanh"), token_num=120):
        super().__init__()
        self.proj = Mlp(
            in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0
        )
        self.embedding = Parameter(ops.randn(1, in_channels) / 10**0.5)
        self.y_embedding = Parameter(ops.randn(token_num, in_channels) / 10**0.5)
        self.uncond_prob = uncond_prob

    def token_drop(self, global_caption, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = ops.rand(global_caption.shape[0]) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        global_caption = ops.where(drop_ids[:, None], self.embedding, global_caption)
        caption = ops.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return global_caption, caption

    def construct(self, caption, train, force_drop_ids=None):
        assert caption.shape[2:] == self.y_embedding.shape
        global_caption = caption.mean(axis=2).squeeze()
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            global_caption, caption = self.token_drop(global_caption, caption, force_drop_ids)
        y_embed = self.proj(global_caption)
        return y_embed, caption


class WindowAttention(SelfAttention):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
        dtype=ms.float32,
        enable_flash_attention=False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, enable_flash_attention=enable_flash_attention
        )

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = Parameter(ops.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = Parameter(ops.zeros(2 * input_size[1] - 1, self.head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, sigma=0.02)
                trunc_normal_(self.rel_pos_w, sigma=0.02)


class MultiHeadCrossAttention(nn.Cell):
    """
    Flash attention doesnot work well (leading to noisy images) for SD1.5-based models on 910B up to MS2.2.1-20231122 version,
    due to the attention head dimension is 40, num heads=5. Require test on future versions
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        context_dim=None,
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
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        context_dim = dim if context_dim is None else context_dim

        self.q_linear = nn.Dense(dim, dim).to_float(dtype)
        self.kv_linear = nn.Dense(context_dim, dim * 2).to_float(dtype)

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

    def construct(self, x, context, mask=None):
        x_dtype = x.dtype
        h = self.num_heads
        B, N, C = x.shape

        q = self.q_linear(x)
        out = self.kv_linear(context)
        # (b, n, 2*h*d) -> (b, n, 2, h*d)  -> (2, b, n, h*d)
        k, v = out.reshape(out.shape[0], out.shape[1], 2, -1).unbind(2)

        q_b, q_n, _ = q.shape  # (b n h*d)
        k_b, k_n, _ = k.shape
        v_b, v_n, _ = v.shape

        head_dim = q.shape[-1] // h

        if (
            self.enable_flash_attention and q_n % 16 == 0 and k_n % 16 == 0 and head_dim <= 256
        ):  # TODO: why restrict head_dim?
            # reshape qkv shape ((b n h*d) -> (b h n d))and mask dtype for FA input format
            q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
            k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
            v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)
            out = self.flash_attention(q, k, v, mask)
            b, h, n, d = out.shape
            # reshape FA output to original attn input format, (b h n d) -> (b n h*d)
            out = out.transpose(0, 2, 1, 3).view(b, n, -1)
        else:
            # make seq_mask to attention_mask
            # TODO: xformers.ops.fmha.BlockDiagonalMask.from_seqlens
            if mask is not None:
                mask = self.reshape(mask, (mask.shape[0], -1))
                if q.dtype == ms.float16:
                    finfo_type = np.float16
                else:
                    finfo_type = np.float32
                max_neg_value = -np.finfo(finfo_type).max
                mask = mask.repeat(self.num_heads, axis=0)
                mask = ops.expand_dims(mask, axis=1)
                attn_mask = ops.zeros_like(mask, dtype=q.dtype)
                attn_mask = ops.masked_fill(attn_mask, mask == 0, Tensor(max_neg_value, dtype=q.dtype))
                mask = attn_mask
            # (b, n, h*d) -> (b*h, n, d)
            q = self._rearange_in(q, h)
            k = self._rearange_in(k, h)
            v = self._rearange_in(v, h)

            out = self.attention(q, k, v, mask)
            # (b*h, n, d) -> (b, n, h*d)
            out = self._rearange_out(out, h)

        return self.proj_drop(self.proj(out)).to(x_dtype)


class DropPath(nn.Cell):
    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
    ) -> None:
        super().__init__()
        self.keep_prob = 1.0 - drop_prob
        self.scale_by_keep = scale_by_keep
        self.dropout = nn.Dropout(1 - drop_prob) if is_old_ms_version() else nn.Dropout(p=drop_prob)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        if self.keep_prob == 1.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ops.ones(shape))
        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        return x * random_tensor


class PixArtBlock(nn.Cell):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        window_size=0,
        input_size=None,
        use_rel_pos=False,
        **block_kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rel_pos=use_rel_pos,
            **block_kwargs,
        )
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = Parameter(ops.randn(6, hidden_size) / hidden_size**0.5)

    def construct(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, axis=1)
        # assert not ops.isinf(x.min()), "input has inf!"
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        # if ops.isinf(res_x.min()):
        #     breakpoint()
        #     attn_input = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        #     attn_output = self.attn(attn_input)
        x = x + self.cross_attn(x, y, mask)
        # if ops.isinf(x.min()):
        #     breakpoint()
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        # if ops.isinf(x.min()):
        #     breakpoint()

        return x


class PixArt(nn.Cell):
    """
    Diffusion model with a Transformer backbone.
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
        class_dropout_prob=0.1,
        learn_sigma=True,
        pred_sigma=True,
        drop_path: float = 0.0,
        window_size=0,
        window_block_indexes=None,
        use_rel_pos=False,
        caption_channels=4096,
        lewei_scale=1.0,
        config=None,
        model_max_length=120,
        block_kwargs={},
    ):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.pred_sigma = pred_sigma
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = (lewei_scale,)

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        # Will use fixed sin-cos embedding:
        self.pos_embed = Parameter(ops.zeros((1, num_patches, hidden_size)), requires_grad=False)

        approx_gelu = lambda: GELU(approximate="tanh")
        self.t_block = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 6 * hidden_size, has_bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        drop_path = [x for x in np.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.CellList(
            [
                PixArtBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    input_size=(input_size // patch_size, input_size // patch_size),
                    window_size=window_size if i in window_block_indexes else 0,
                    use_rel_pos=use_rel_pos if i in window_block_indexes else False,
                    **block_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def construct(self, x, timestep, y, mask=None, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, L, C) tensor of text embeddings. L is the token length, which is 120 for T5 text encoder. C is the embedding dim.
        mask: (N, seq_q, seq_k) tensor of cross-attention mask.
        """
        pos_embed = self.pos_embed
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, 1, L, D)
        # if mask is not None:
        #     if mask.shape[0] != y.shape[0]:
        #         mask = mask.repeat(y.shape[0] // mask.shape[0], axis=0)
        #     mask = mask.squeeze(1).squeeze(1)
        #     y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])  #FIXME: dynamic shape problem!
        #     y_lens = mask.sum(axis=1).tolist()
        # else:

        # y_lens = [y.shape[2]] * y.shape[0]
        # y = y.squeeze(1).view(1, -1, x.shape[-1])
        for i, block in enumerate(self.blocks):
            # if ops.isinf(x.min()):
            #     print(f"{i} block")
            x = block(x, y, t0, mask)  # (N, T, D) #support grad checkpoint
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    # @ms.jit
    def construct_with_cfg(self, x: Tensor, timestep: Tensor, y: Tensor, cfg_scale: float, mask=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        model_out = self.construct(combined, timestep, y, mask, **kwargs)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=1)

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

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Dense):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches**0.5),
            lewei_scale=self.lewei_scale,
            base_size=self.base_size,
        )
        self.pos_embed.set_data(Tensor(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight
        # xavier_uniform_(w.view(w.shape[0], -1))
        w_flatted = w.view(w.shape[0], -1)
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).reshape(w.shape))
        constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        normal_(self.t_embedder.mlp[0].weight, std=0.02)
        normal_(self.t_embedder.mlp[2].weight, std=0.02)
        normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            constant_(block.cross_attn.proj.weight, 0)
            constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        constant_(self.final_layer.linear.weight, 0)
        constant_(self.final_layer.linear.bias, 0)


################################################################################
#                                 PixArt Configs                               #
################################################################################
def PixArt_XL_2(**kwargs):
    return PixArt(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


class PatchEmbedMS(nn.Cell):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, pad_mode="pad", has_bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def construct(self, x):
        b = x.shape[0]
        x = self.proj(x)
        if self.flatten:
            x = ops.reshape(x, (b, self.embed_dim, -1))
            x = ops.transpose(x, (0, 2, 1))  # B Ph*Pw C
        x = self.norm(x)
        return x


class PixArtMSBlock(nn.Cell):
    """
    A PixArt block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        window_size=0,
        input_size=None,
        use_rel_pos=False,
        **block_kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rel_pos=use_rel_pos,
            **block_kwargs,
        )
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = Parameter(ops.randn(6, hidden_size) / hidden_size**0.5)

    def construct(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, axis=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


class PixArtMS(PixArt):
    """
    Diffusion model with a Transformer backbone.
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
        class_dropout_prob=0.1,
        learn_sigma=True,
        pred_sigma=True,
        drop_path: float = 0.0,
        window_size=0,
        window_block_indexes=None,
        use_rel_pos=False,
        caption_channels=4096,
        lewei_scale=1.0,
        config=None,
        model_max_length=120,
        block_kwargs={},
    ):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            learn_sigma=learn_sigma,
            pred_sigma=pred_sigma,
            drop_path=drop_path,
            window_size=window_size,
            window_block_indexes=window_block_indexes,
            use_rel_pos=use_rel_pos,
            lewei_scale=lewei_scale,
            config=config,
            model_max_length=model_max_length,
            block_kwargs=block_kwargs,
        )
        self.h = self.w = 0
        approx_gelu = lambda: GELU(approximate="tanh")
        self.t_block = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 6 * hidden_size, has_bias=True))
        self.x_embedder = PatchEmbedMS(patch_size, in_channels, hidden_size, bias=True)
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        self.csize_embedder = SizeEmbedder(hidden_size // 3)  # c_size embed
        self.ar_embedder = SizeEmbedder(hidden_size // 3)  # aspect ratio embed
        drop_path = [x for x in np.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.CellList(
            [
                PixArtMSBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    input_size=(input_size // patch_size, input_size // patch_size),
                    window_size=window_size if i in window_block_indexes else 0,
                    use_rel_pos=use_rel_pos if i in window_block_indexes else False,
                    **block_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize()

    def construct(self, x, timestep, y, mask=None, img_hw=None, aspect_ratio=None, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        bs = x.shape[0]
        c_size, ar = img_hw, aspect_ratio
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        pos_embed = Tensor(
            get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], (self.h, self.w), lewei_scale=self.lewei_scale, base_size=self.base_size
            )
        ).unsqueeze(0)

        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep)  # (N, D)
        csize = self.csize_embedder(c_size, bs)  # (N, D)
        ar = self.ar_embedder(ar, bs)  # (N, D)
        t = t + ops.cat([csize, ar], axis=1)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, D)
        # if mask is not None:
        #     if mask.shape[0] != y.shape[0]:
        #         mask = mask.repeat(y.shape[0] // mask.shape[0], axis=0)
        #     mask = mask.squeeze(1).squeeze(1)
        #     y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
        #     y_lens = mask.sum(axis=1).tolist()
        # else:
        #     y_lens = [y.shape[2]] * y.shape[0]
        #     y = y.squeeze(1).view(1, -1, x.shape[-1])
        for block in self.blocks:
            x = block(x, y, t0, mask, **kwargs)  # (N, T, D) #support grad checkpoint
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def construct_with_cfg(self, x, timestep, y, cfg_scale, img_hw=None, aspect_ratio=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        model_out = self.construct(
            combined,
            timestep,
            y,
            img_hw=img_hw,
            aspect_ratio=aspect_ratio,
        )
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=1)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape((x.shape[0], self.h, self.w, p, p, c))
        x = x.permute((0, 5, 1, 3, 2, 4))
        imgs = x.reshape((x.shape[0], c, self.h * p, self.w * p))
        return imgs

    def initialize(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Dense):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight
        # xavier_uniform_(w.view(w.shape[0], -1))
        w_flatted = w.view(w.shape[0], -1)
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).reshape(w.shape))
        constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        normal_(self.t_embedder.mlp[0].weight, std=0.02)
        normal_(self.t_embedder.mlp[2].weight, std=0.02)
        normal_(self.t_block[1].weight, std=0.02)
        normal_(self.csize_embedder.mlp[0].weight, std=0.02)
        normal_(self.csize_embedder.mlp[2].weight, std=0.02)
        normal_(self.ar_embedder.mlp[0].weight, std=0.02)
        normal_(self.ar_embedder.mlp[2].weight, std=0.02)

        # Initialize caption embedding MLP:
        normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            constant_(block.cross_attn.proj.weight, 0)
            constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        constant_(self.final_layer.linear.weight, 0)
        constant_(self.final_layer.linear.bias, 0)


#################################################################################
#                                   PixArt Configs                                  #
#################################################################################


def PixArtMS_XL_2(**kwargs):
    return PixArtMS(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
