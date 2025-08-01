"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/models/autoencoders/__init__.py."""

from .autoencoder_asym_kl import AsymmetricAutoencoderKL
from .autoencoder_dc import AutoencoderDC
from .autoencoder_kl import AutoencoderKL
from .autoencoder_kl_allegro import AutoencoderKLAllegro
from .autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
from .autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
from .autoencoder_kl_ltx import AutoencoderKLLTXVideo
from .autoencoder_kl_magvit import AutoencoderKLMagvit
from .autoencoder_kl_mochi import AutoencoderKLMochi
from .autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from .autoencoder_kl_wan import AutoencoderKLWan
from .autoencoder_oobleck import AutoencoderOobleck
from .autoencoder_tiny import AutoencoderTiny
from .consistency_decoder_vae import ConsistencyDecoderVAE
from .vq_model import VQModel
