import math

from omegaconf import OmegaConf
from packaging import version

import mindspore as ms
from mindspore import ops

if version.parse(ms.__version__) <= version.parse("2.2"):
    raise ValueError("To run FFT, please use MindSpore version > 2.2, e.g., 2.3.")

gaussian_filter_param = OmegaConf.create({"method": "gaussian", "d_s": 0.25, "d_t": 0.25})
butterworth_filter_param = OmegaConf.create({"method": "butterworth", "d_s": 0.25, "d_t": 0.25, "n": 4})


def freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    if x.dtype != ms.complex64:
        x = x.to(ms.complex64)
    if noise != ms.complex64:
        noise = noise.to(ms.complex64)
    # FFT
    fftn = ops.FFTWithSize(signal_ndim=3, inverse=False, real=False)
    x_freq = fftn(x)
    x_freq = ops.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fftn(noise)
    noise_freq = ops.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high  # mix in freq domain

    # IFFT
    x_freq_mixed = ops.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    ifftn = ops.FFTWithSize(signal_ndim=3, inverse=True, real=False)
    x_mixed = ifftn(x_freq_mixed).real

    return x_mixed


def get_freq_filter(shape, filter_type, n, d_s, d_t):
    """
    Form the frequency filter for noise reinitialization.

    Args:
        shape: shape of latent (B, C, T, H, W)
        filter_type: type of the freq filter
        n: (only for butterworth) order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    if filter_type == "gaussian":
        return gaussian_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t)
    elif filter_type == "ideal":
        return ideal_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t)
    elif filter_type == "box":
        return box_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t)
    elif filter_type == "butterworth":
        return butterworth_low_pass_filter(shape=shape, n=n, d_s=d_s, d_t=d_t)
    else:
        raise NotImplementedError


def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = ops.zeros(shape)
    if d_s == 0 or d_t == 0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = ((d_s / d_t) * (2 * t / T - 1)) ** 2 + (2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2
                mask[..., t, h, w] = math.exp(-1 / (2 * d_s**2) * d_square)
    return mask


def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25):
    """
    Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = ops.zeros(shape)
    if d_s == 0 or d_t == 0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = ((d_s / d_t) * (2 * t / T - 1)) ** 2 + (2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2
                mask[..., t, h, w] = 1 / (1 + (d_square / d_s**2) ** n)
    return mask


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = ops.zeros(shape)
    if d_s == 0 or d_t == 0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = ((d_s / d_t) * (2 * t / T - 1)) ** 2 + (2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2
                mask[..., t, h, w] = 1 if d_square <= d_s * 2 else 0
    return mask


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = ops.zeros(shape)
    if d_s == 0 or d_t == 0:
        return mask

    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)

    cframe, crow, ccol = T // 2, H // 2, W // 2
    mask[
        ...,
        cframe - threshold_t : cframe + threshold_t,
        crow - threshold_s : crow + threshold_s,
        ccol - threshold_s : ccol + threshold_s,
    ] = 1.0

    return mask
