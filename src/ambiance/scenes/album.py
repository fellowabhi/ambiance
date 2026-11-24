"""The current track's cover art as keyboard lighting.

Two looks:

* ``gradient`` (default): the extracted palette laid out as a smooth spatial gradient
  (:func:`ambiance.scenes.gradient.gradient_frame`), optionally drifting slowly.
* ``mosaic``: the cover itself projected onto the keys. Each key shows the (slightly
  blurred) average of the art underneath it, then the whole frame is quantised to at most
  ``steps`` colours with k-means in OKLab.

Both looks cap the number of distinct colours on purpose: the keyboard is painted with
HID++ writes that cost ~47 ms each over Bluetooth, and writes are grouped by colour, so
fewer distinct colours means a faster, less "tearing" frame.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from ambiance.color import RGB
from ambiance.colorspace import linear_to_oklab, oklab_to_srgb, srgb_to_linear, srgb_to_oklab
from ambiance.core import Frame, MediaState, RenderContext, Scene
from ambiance.layout import Layout
from ambiance.palette import Swatch, kmeans
from ambiance.scenes.gradient import gradient_frame

log = logging.getLogger(__name__)

STYLES = ("gradient", "mosaic")


def _swatch(rgb: RGB, weight: float) -> Swatch:
    lab = srgb_to_oklab(np.array(rgb, dtype=float) / 255.0)
    return Swatch(rgb, (float(lab[0]), float(lab[1]), float(lab[2])), weight, 1.0)


# Shown while a track has no analysed art (still loading, fetch failed, no cover at all):
# a calm candle-to-warm-white gradient instead of nothing.
WARM_FALLBACK: tuple[Swatch, ...] = (
    _swatch(RGB(255, 150, 70), 0.5),
    _swatch(RGB(255, 205, 150), 0.5),
)


class AlbumScene(Scene):
    """Lighting derived from ``media.palette`` / ``media.thumbnail``.

    ``key()`` contains the art identity (``art_url``), the style parameters and what the
    picture is currently built from (palette colours, thumbnail, or the fallback). A new
    track, or its art finishing analysis, therefore gives a new key and the engine
    crossfades; a status change alone (Playing -> Paused) keeps the key and only the
    decision's level changes.
    """

    name = "album"

    def __init__(self, media: MediaState, *, style: str = "gradient", steps: int = 10,
                 angle_deg: float = 15.0, drift_period_s: float = 0.0, band: float = 0.5,
                 blur: float = 0.35, zoom: float = 0.8, strip_cycle_s: float = 0.0) -> None:
        """
        style: "gradient" or "mosaic".
        steps: maximum number of distinct colours (gradient steps / mosaic k-means k).
        angle_deg: gradient axis; 0 = left to right.
        drift_period_s: > 0 makes the keyboard move once per that many seconds: the gradient
            drifts (ping-pong, no seam) or, for mosaic, the view pans across the cover.
        band: mosaic only. Height of the centred horizontal band of the (square) cover that
            is mapped onto the (≈2.9:1) keyboard: ~0.34 is an undistorted crop, 1.0 squashes
            the whole cover.
        blur: mosaic only. Key units added around every key when averaging the art under
            it, so neighbouring keys blend a little instead of aliasing on fine detail.
        zoom: mosaic + movement only. Share of the cover's width in view, leaving room to pan.
        strip_cycle_s: > 0 makes one-zone lights (a strip) cycle through the palette's
            colours, holding each and crossfading to the next, this many seconds per colour.
        """
        if style not in STYLES:
            raise ValueError(f"album style must be one of {', '.join(STYLES)}; got {style!r}")
        if steps < 1:
            raise ValueError(f"album steps must be >= 1; got {steps}")
        if drift_period_s < 0:
            raise ValueError(f"drift_period_s must be >= 0; got {drift_period_s}")
        if not 0.0 < band <= 1.0:
            raise ValueError(f"mosaic band must be within (0, 1]; got {band}")
        if blur < 0:
            raise ValueError(f"mosaic blur must be >= 0; got {blur}")
        if not 0.0 < zoom <= 1.0:
            raise ValueError(f"mosaic zoom must be within (0, 1]; got {zoom}")
        if strip_cycle_s < 0:
            raise ValueError(f"strip_cycle_s must be >= 0; got {strip_cycle_s}")
        self.media = media
        self.style = style
        self.steps = int(steps)
        self.angle_deg = float(angle_deg)
        self.drift_period_s = float(drift_period_s)
        self.band = float(band)
        self.blur = float(blur)
        self.zoom = float(zoom)
        self.strip_cycle_s = float(strip_cycle_s)

        swatches = tuple(getattr(media.palette, "swatches", None) or ())
        self._palette_swatches = swatches
        self._swatches = swatches or WARM_FALLBACK
        self._image = _as_rgb_image(media.thumbnail) if style == "mosaic" else None
        # Mosaic needs the thumbnail; until it exists the palette gradient stands in.
        self.mode = "mosaic" if self._image is not None else "gradient"
        self._strip_colors = _strip_colors(self._swatches)
        self.animated = self.drift_period_s > 0 or (
            self.strip_cycle_s > 0 and len(self._strip_colors) > 1)
        self._cache: dict[str, Frame] = {}
        self._mosaic_palette: tuple[np.ndarray, list[RGB]] | None = None

    # -- identity ----------------------------------------------------------------------------
    def key(self) -> Any:
        if self.mode == "mosaic":
            assert self._image is not None
            params: tuple[Any, ...] = ("mosaic", self.steps, self.band, self.blur, self.zoom,
                                       self.drift_period_s, self.strip_cycle_s)
            content: tuple[Any, ...] = ("thumbnail", self._image.shape)
        else:
            params = ("gradient", self.steps, self.angle_deg, self.drift_period_s,
                      self.strip_cycle_s)
            content = (("palette", tuple((s.rgb, round(s.weight, 4))
                                         for s in self._palette_swatches))
                       if self._palette_swatches else ("fallback",))
        return ("AlbumScene", self.media.art_url, params, content)

    @property
    def uses_fallback(self) -> bool:
        """True when neither a palette nor (for mosaic) a thumbnail is available."""
        return self.mode == "gradient" and not self._palette_swatches

    # -- rendering ---------------------------------------------------------------------------
    def render(self, ctx: RenderContext) -> Frame:
        animate = bool(ctx.options.get("animate", True))
        if len(ctx.layout.keys) <= SMALL_LAYOUT_KEYS:
            # Lights with only a few zones (a room strip, a mouse logo) can't show a gradient:
            # zone i gets the palette's i-th colour, cycling through them if animated.
            return {key.zone: self._strip_color(ctx.now if animate else None, offset=i)
                    for i, key in enumerate(ctx.layout.keys)}
        if self.drift_period_s > 0 and animate:
            phase = (ctx.now / self.drift_period_s) % 1.0
            if self.mode == "mosaic":
                return self._moving_mosaic(ctx.layout, phase)
            return gradient_frame(ctx.layout, self._swatches, self.steps, self.angle_deg, phase,
                                  cyclic=True)
        # Static picture: same key => same frame, so compute it once per layout.
        frame = self._cache.get(ctx.layout.name)
        if frame is None:
            if self.mode == "mosaic":
                assert self._image is not None
                frame = mosaic_frame(ctx.layout, self._image, self.steps, self.band, self.blur)
            else:
                frame = gradient_frame(ctx.layout, self._swatches, self.steps, self.angle_deg)
            self._cache[ctx.layout.name] = frame
        return dict(frame)

    def _strip_color(self, now: float | None, offset: int = 0) -> RGB:
        colors = self._strip_colors
        if now is None or self.strip_cycle_s <= 0 or len(colors) < 2:
            return colors[offset % len(colors)][0]
        pos = now / self.strip_cycle_s + offset
        i = int(math.floor(pos)) % len(colors)
        frac = pos - math.floor(pos)
        if frac <= _STRIP_HOLD:
            return colors[i][0]
        e = (frac - _STRIP_HOLD) / (1.0 - _STRIP_HOLD)
        e = e * e * (3.0 - 2.0 * e)  # smoothstep: no sudden starts or stops
        lab = colors[i][1] + (colors[(i + 1) % len(colors)][1] - colors[i][1]) * e
        return RGB(*(int(round(v * 255)) for v in np.clip(oklab_to_srgb(lab), 0.0, 1.0)))

    def _moving_mosaic(self, layout: Layout, phase: float) -> Frame:
        """Pan a window across the cover along a slow figure-eight (Lissajous 1:2) path.

        Keys snap to a palette computed once over the whole cover, so colours stay stable
        while the view moves (per-frame k-means would flicker) and only keys whose nearest
        palette colour changes are repainted.
        """
        assert self._image is not None
        if self._mosaic_palette is None:
            pixels = srgb_to_oklab(self._image.reshape(-1, 3))
            centres, _ = kmeans(pixels, max(1, self.steps))
            rgb = np.rint(np.clip(oklab_to_srgb(centres), 0.0, 1.0) * 255).astype(int)
            self._mosaic_palette = (centres, [RGB(*map(int, row)) for row in rgb])
        centres, colors = self._mosaic_palette
        angle = 2.0 * math.pi * phase
        cx = 0.5 + (1.0 - self.zoom) / 2.0 * math.sin(angle)
        cy = 0.5 + (1.0 - self.band) / 2.0 * math.sin(2.0 * angle + math.pi / 2.0)
        lab = linear_to_oklab(sample_keys(layout, self._image, self.band, self.blur,
                                          center=(cx, cy), zoom=self.zoom))
        nearest = ((lab[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
        return {key.zone: colors[i] for key, i in zip(layout.keys, nearest)}

    def __repr__(self) -> str:
        what = self.media.title or self.media.art_url or "?"
        return f"AlbumScene({what!r}, {self.mode}{', fallback' if self.uses_fallback else ''})"


_STRIP_HOLD = 0.55  # share of each strip cycle step spent holding the colour (rest: crossfade)
SMALL_LAYOUT_KEYS = 3  # layouts this small get palette colours per zone instead of a picture


def _strip_colors(swatches) -> list[tuple[RGB, np.ndarray]]:
    """Palette colours a room light should cycle through: up to four, skipping near-blacks
    (they read as "off" on a strip) unless nothing else exists."""
    usable = [s for s in swatches if s.lightness >= 0.2] or list(swatches[:1])
    return [(s.rgb, np.array(s.oklab, dtype=float)) for s in usable[:4]]


# ---- mosaic ------------------------------------------------------------------------------------

def _as_rgb_image(thumbnail: Any) -> np.ndarray | None:
    """Normalise a thumbnail to a float (H, W, 3) sRGB array in 0..1, or None if unusable."""
    if thumbnail is None:
        return None
    img = np.asarray(thumbnail)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] < 3 or img.shape[0] < 1 or img.shape[1] < 1:
        log.debug("ignoring thumbnail with unexpected shape %s", img.shape)
        return None
    img = img[..., :3]
    if img.dtype == np.uint8:
        img = img / 255.0
    return np.clip(img.astype(float), 0.0, 1.0)


def sample_keys(layout: Layout, image: np.ndarray, band: float = 0.5,
                blur: float = 0.35, *, center: tuple[float, float] = (0.5, 0.5),
                zoom: float = 1.0) -> np.ndarray:
    """Average the art under every key -> (len(layout.keys), 3) linear-light colours.

    The keyboard's bounding box is mapped onto a window of the image ``zoom`` x its width
    and ``band`` x its height, centred on ``center`` (fractions, clamped inside the image). Each key averages the pixels under its rectangle grown by
    ``blur`` key units on every side. Averaging happens in linear light, which is what
    optically blurring the cover would do (sRGB averaging darkens edges between colours).
    A summed-area table makes every key's box mean O(1), whatever its size (spacebar).
    """
    h, w = image.shape[:2]
    linear = srgb_to_linear(image)
    sat = np.zeros((h + 1, w + 1, 3))
    sat[1:, 1:] = linear.cumsum(axis=0).cumsum(axis=1)
    win_w, win_h = zoom * w, band * h
    win_x0 = min(max(center[0] * w - win_w / 2.0, 0.0), w - win_w)
    band_y0 = min(max(center[1] * h - win_h / 2.0, 0.0), h - win_h)
    sx, sy = win_w / layout.width, win_h / layout.height
    out = np.empty((len(layout.keys), 3))
    for i, key in enumerate(layout.keys):
        x0 = int(math.floor(win_x0 + (key.x - blur) * sx))
        x1 = int(math.ceil(win_x0 + (key.x + key.w + blur) * sx))
        y0 = int(math.floor(band_y0 + (key.y - blur) * sy))
        y1 = int(math.ceil(band_y0 + (key.y + key.h + blur) * sy))
        x0 = min(max(x0, 0), w - 1)
        y0 = min(max(y0, 0), h - 1)
        x1 = min(max(x1, x0 + 1), w)
        y1 = min(max(y1, y0 + 1), h)
        total = sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]
        out[i] = total / ((x1 - x0) * (y1 - y0))
    return out


def mosaic_frame(layout: Layout, image: np.ndarray, steps: int = 10, band: float = 0.5,
                 blur: float = 0.35) -> Frame:
    """Project ``image`` (H, W, 3 sRGB 0..1) onto the layout with at most ``steps`` colours."""
    lab = linear_to_oklab(sample_keys(layout, image, band, blur))
    centres, labels = kmeans(lab, max(1, steps))  # deterministic (fixed seed)
    srgb = np.rint(np.clip(oklab_to_srgb(centres), 0.0, 1.0) * 255).astype(int)
    colors = [RGB(*map(int, row)) for row in srgb]
    return {key.zone: colors[label] for key, label in zip(layout.keys, labels)}
