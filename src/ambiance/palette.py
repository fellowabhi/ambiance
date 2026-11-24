"""Extract an ambient-lighting palette from an image (album art, screenshots, ...).

k-means in OKLab on a thumbnail, then pick a small set of colours that are both
representative (large share of the image) and suitable for light (chromatic; huge
black/grey backgrounds don't make good LED colours), while staying visually distinct.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from ambiance.color import RGB
from ambiance.colorspace import delta_e, oklab_to_srgb, srgb_to_oklab


@dataclass(frozen=True)
class Swatch:
    rgb: RGB
    oklab: tuple[float, float, float]
    weight: float  # share of image pixels, 0..1
    score: float   # how suitable/important for lighting (relative)

    @property
    def lightness(self) -> float:
        return self.oklab[0]

    @property
    def chroma(self) -> float:
        return float(np.hypot(self.oklab[1], self.oklab[2]))


@dataclass(frozen=True)
class Palette:
    swatches: tuple[Swatch, ...]  # most important first

    @property
    def colors(self) -> list[RGB]:
        return [s.rgb for s in self.swatches]

    @property
    def is_grayscale(self) -> bool:
        return all(s.chroma < 0.03 for s in self.swatches)

    def __len__(self) -> int:
        return len(self.swatches)


def load_pixels(data: bytes, size: int = 64) -> np.ndarray:
    """Decode image bytes -> (N, 3) sRGB floats in 0..1, downsampled to ~size x size."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        img.thumbnail((size, size), Image.Resampling.BOX)
        return np.asarray(img, dtype=float).reshape(-1, 3) / 255.0


def kmeans(points: np.ndarray, k: int, iterations: int = 25, seed: int = 0):
    """Deterministic k-means with k-means++ initialisation. Returns (centres, labels)."""
    rng = np.random.default_rng(seed)
    n = len(points)
    k = min(k, n)
    centres = [points[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((points[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        if total <= 0:
            break
        centres.append(points[rng.choice(n, p=d2 / total)])
    centres = np.array(centres)
    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        d = ((points[:, None, :] - centres[None]) ** 2).sum(-1)
        new_labels = d.argmin(1)
        if _ > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(len(centres)):
            members = points[labels == j]
            if len(members):
                centres[j] = members.mean(0)
    return centres, labels


def _suitability(lab: np.ndarray, weight: np.ndarray) -> np.ndarray:
    L = lab[:, 0]
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    colourfulness = 0.12 + np.minimum(chroma, 0.25) / 0.25          # 0.12 .. 1.12
    darkness_penalty = np.clip((L - 0.08) / 0.22, 0.05, 1.0)          # near-black -> off LEDs
    pale_penalty = np.where((L > 0.9) & (chroma < 0.04), 0.5, 1.0)    # paper white
    return np.sqrt(weight) * colourfulness * darkness_penalty * pale_penalty


def extract_palette(pixels: np.ndarray, max_colors: int = 5, k: int = 10,
                    min_distance: float = 0.09) -> Palette:
    """pixels: (N, 3) sRGB 0..1. Returns up to max_colors distinct swatches."""
    lab = srgb_to_oklab(pixels)
    centres, labels = kmeans(lab, k)
    counts = np.bincount(labels, minlength=len(centres)).astype(float)
    keep = counts > 0
    centres, weights = centres[keep], counts[keep] / counts.sum()
    scores = _suitability(centres, weights)
    order = np.argsort(-scores)
    chosen: list[int] = []
    for i in order:
        if all(delta_e(centres[i], centres[j]) >= min_distance for j in chosen):
            chosen.append(int(i))
        if len(chosen) == max_colors:
            break
    swatches = []
    for i in chosen:
        srgb = oklab_to_srgb(centres[i])
        swatches.append(Swatch(RGB(*(int(round(c * 255)) for c in srgb)),
                               tuple(float(v) for v in centres[i]), float(weights[i]),
                               float(scores[i])))
    return Palette(tuple(swatches))


def palette_from_image_bytes(data: bytes, max_colors: int = 5) -> Palette:
    return extract_palette(load_pixels(data), max_colors=max_colors)
