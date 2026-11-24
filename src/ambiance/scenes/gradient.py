"""Palette -> smooth spatial gradient across the keyboard."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np

from ambiance.color import RGB
from ambiance.colorspace import delta_e, oklab_to_srgb
from ambiance.layout import Layout
from ambiance.palette import Swatch


def order_stops(swatches: Sequence[Swatch]) -> list[Swatch]:
    """Order colours so neighbours are perceptually close (shortest open path)."""
    if len(swatches) <= 2:
        return sorted(swatches, key=lambda s: s.lightness)
    if len(swatches) > 7:
        return sorted(swatches, key=lambda s: s.lightness)
    best, best_cost = None, math.inf
    for perm in itertools.permutations(swatches):
        if perm[0].lightness > perm[-1].lightness:
            continue  # each path appears twice; keep the dark->light direction
        cost = sum(float(delta_e(a.oklab, b.oklab)) for a, b in zip(perm, perm[1:]))
        if cost < best_cost:
            best, best_cost = list(perm), cost
    return best or list(swatches)


def gradient_frame(layout: Layout, swatches: Sequence[Swatch], steps: int = 10,
                   angle_deg: float = 15.0, phase: float = 0.0,
                   cyclic: bool = False) -> dict[int, RGB]:
    """Colour every key by its position along an axis through the ordered palette.

    Stops get room proportional to sqrt(weight), so dominant colours cover more keys.
    `steps` quantises the gradient (fewer distinct colours = fewer HID++ writes).
    `phase` (0..1) shifts the gradient, for slow drifting animations.
    """
    stops = order_stops(swatches)
    if not stops:
        return {}
    if len(stops) == 1:
        c = stops[0].rgb
        return {k.zone: c for k in layout.keys}
    widths = np.sqrt(np.array([max(s.weight, 1e-3) for s in stops]))
    centres = (np.cumsum(widths) - widths / 2) / widths.sum()
    labs = np.array([s.oklab for s in stops])

    a = math.radians(angle_deg)
    ax, ay = math.cos(a), math.sin(a)
    proj = {k: layout.norm(k)[0] * ax * layout.width / layout.height * 0.35
            + layout.norm(k)[1] * ay for k in layout.keys}
    lo, hi = min(proj.values()), max(proj.values())
    cache: dict[float, RGB] = {}
    frame = {}
    for key, p in proj.items():
        t = (p - lo) / (hi - lo) if hi > lo else 0.5
        if cyclic:  # ping-pong through the stops, so a drifting phase never shows a seam
            t = 1.0 - abs(2.0 * ((t / 2.0 + phase) % 1.0) - 1.0)
        elif phase:
            t = (t + phase) % 1.0
        t = round(t * (steps - 1)) / (steps - 1) if steps > 1 else 0.5
        if t not in cache:
            lab = np.array([np.interp(t, centres, labs[:, i]) for i in range(3)])
            cache[t] = RGB(*(int(round(v * 255)) for v in oklab_to_srgb(lab)))
        frame[key.zone] = cache[t]
    return frame
