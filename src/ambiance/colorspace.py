"""sRGB <-> linear-light <-> OKLab / OKLCH conversions (vectorised with numpy).

OKLab (Björn Ottosson, 2020) is perceptually uniform: Euclidean distance ~ visible
difference, and straight lines give even-looking gradients. All functions accept arrays
of shape (..., 3) with sRGB / linear values in 0..1.
"""

from __future__ import annotations

import numpy as np

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M2_INV = np.array([[1.0, 0.3963377774, 0.2158037573],
                    [1.0, -0.1055613458, -0.0638541728],
                    [1.0, -0.0894841775, -1.2914855480]])
_M1_INV = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]])


def srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def linear_to_oklab(rgb):
    lms = np.asarray(rgb, dtype=float) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_linear(lab):
    lms = (np.asarray(lab, dtype=float) @ _M2_INV.T) ** 3
    return lms @ _M1_INV.T


def srgb_to_oklab(srgb):
    return linear_to_oklab(srgb_to_linear(srgb))


def oklab_to_srgb(lab):
    return linear_to_srgb(oklab_to_linear(lab))


def oklab_to_oklch(lab):
    lab = np.asarray(lab, dtype=float)
    c = np.hypot(lab[..., 1], lab[..., 2])
    h = np.arctan2(lab[..., 2], lab[..., 1])
    return np.stack([lab[..., 0], c, h], axis=-1)


def oklch_to_oklab(lch):
    lch = np.asarray(lch, dtype=float)
    return np.stack([lch[..., 0], lch[..., 1] * np.cos(lch[..., 2]),
                     lch[..., 1] * np.sin(lch[..., 2])], axis=-1)


def in_gamut(linear, eps: float = 1e-6):
    linear = np.asarray(linear)
    return np.all((linear >= -eps) & (linear <= 1 + eps), axis=-1)


def gamut_clip(lch, iterations: int = 20):
    """Reduce chroma (keeping lightness and hue) until each colour fits in sRGB."""
    lch = np.array(lch, dtype=float, copy=True)
    flat = lch.reshape(-1, 3)
    for row in flat:
        if in_gamut(oklab_to_linear(oklch_to_oklab(row))):
            continue
        lo, hi = 0.0, row[1]
        for _ in range(iterations):
            mid = (lo + hi) / 2
            if in_gamut(oklab_to_linear(oklch_to_oklab([row[0], mid, row[2]]))):
                lo = mid
            else:
                hi = mid
        row[1] = lo
    return flat.reshape(lch.shape)


def delta_e(lab1, lab2):
    """Perceptual difference (OKLab Euclidean distance; ~0.02 is just noticeable)."""
    return np.linalg.norm(np.asarray(lab1, dtype=float) - np.asarray(lab2, dtype=float), axis=-1)
