"""Remove horizontal stripes from tile planes with the SmartSPIM pipeline's filter, at any pyramid level.

The filter is aind-smartspim-destripe's ``log_space_fft_filtering``: a wavelet decomposition of log intensity whose
horizontal-detail coefficients are notch-filtered along x. Its notch width is relative to the plane's smaller
dimension, so it removes the same physical wavelengths at any level; its threshold scales with the wavelet
coefficients, which shrink by the level's downsampling factor, so it is divided by that factor.

The pipeline protects coefficients above an Otsu threshold as foreground, which also protects strong stripes.
``unprotect_finest`` drops that protection on the finest wavelet levels, where stripes dominate; coarser levels keep
it, since unprotecting them removes broad real structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pywt
from scipy import fftpack
from skimage.filters import threshold_otsu

# full-resolution settings of the pipeline, for planes with and without bright cells
_CELLS = {"sigma": 64.0, "max_threshold": 3.0}
_NO_CELLS = {"sigma": 128.0, "max_threshold": 12.0}


def log_space_fft_filtering(
    plane: np.ndarray, sigma: float, max_threshold: float, *, unprotect_finest: int = 0, wavelet: str = "db3"
) -> np.ndarray:
    """Destripe one plane.

    Parameters
    ----------
    plane : numpy.ndarray
        2D (y, x) plane, non-negative.
    sigma : float
        Notch width in voxels of a plane whose smaller side is the plane's.
    max_threshold : float
        Upper bound on the foreground threshold for wavelet coefficients.
    unprotect_finest : int
        Number of finest wavelet levels filtered without foreground protection.
    wavelet : str
        PyWavelets wavelet name.

    Returns
    -------
    numpy.ndarray
        Float64 plane of the input's shape.
    """
    coeffs = pywt.wavedec2(np.log1p(plane), wavelet=wavelet, level=None)
    width_fraction = sigma / min(plane.shape)
    out: list[Any] = [coeffs[0]]
    n_levels = len(coeffs) - 1
    for i, (ch, cv, cd) in enumerate(coeffs[1:]):  # coarsest first
        if i < n_levels - unprotect_finest:
            threshold = min(max_threshold, np.sqrt(threshold_otsu(ch**2)))
            mask = np.abs(ch) > threshold
        else:
            mask = np.zeros(ch.shape, bool)
        foreground, background = ch * mask, ch * (1 - mask)
        background_means = np.broadcast_to(np.median(background, axis=-1)[..., None], ch.shape)
        inpainted = background + background_means * mask
        fft = fftpack.rfft(inpainted, axis=-1)
        s = fft.shape[0] * width_fraction
        notch = np.broadcast_to(1 - np.exp(-(np.arange(fft.shape[-1]) ** 2) / (2 * s**2)), fft.shape)
        out.append((foreground + fftpack.irfft(fft * notch) * (1 - mask), cv, cd))
    rec = pywt.waverec2(out, wavelet)[: plane.shape[0], : plane.shape[1]]
    return np.asarray(np.exp(np.clip(rec, None, 88.0)) - 1.0)


def destripe_plane(plane: np.ndarray, level: int, *, unprotect_finest: int = 0) -> np.ndarray:
    """Destripe one plane of a tile read at ``level``, choosing the pipeline's cells or no-cells settings.

    Parameters
    ----------
    plane : numpy.ndarray
        2D (y, x) raw plane, darkfield included, as the pipeline sees it.
    level : int
        Pyramid level the plane comes from.
    unprotect_finest : int
        Number of finest wavelet levels filtered without foreground protection.

    Returns
    -------
    numpy.ndarray
        Float64 plane.
    """
    weight = 1 / (1 + np.exp(-(plane - 400) / 20))
    fg, bg = plane[weight > 0.3], plane[weight <= 0.3]
    fore, back = (fg.mean() if fg.size else 0.0), (bg.mean() if bg.size else 0.0)
    settings = _CELLS if fore > back and fore > 2500 else _NO_CELLS
    factor = 2**level
    return log_space_fft_filtering(
        plane,
        sigma=settings["sigma"] / factor,
        max_threshold=settings["max_threshold"] / factor,
        unprotect_finest=unprotect_finest,
    )


def destripe_stack(stack: np.ndarray, level: int, *, unprotect_finest: int = 2) -> np.ndarray:
    """Destripe every z plane of a raw tile.

    Parameters
    ----------
    stack : numpy.ndarray
        (z, y, x) raw tile, darkfield included.
    level : int
        Pyramid level the tile comes from.
    unprotect_finest : int
        Number of finest wavelet levels filtered without foreground protection; 0 is the pipeline's filter.

    Returns
    -------
    numpy.ndarray
        Float32 stack.
    """
    planes = stack.astype(np.float64)
    return np.stack([destripe_plane(p, level, unprotect_finest=unprotect_finest) for p in planes]).astype(np.float32)
