"""Map screen colours (sRGB) to LED drive values that *look* like the screen colour.

RGB LEDs are driven by PWM, so a value is proportional to emitted light, whereas sRGB
values are gamma-encoded. Sending sRGB straight to the LEDs makes mid-tones far too
bright and shifts hues (orange turns yellow). We therefore:

  1. optionally boost chroma in OKLCH (album colours are often muted; LEDs look
     washed out when desaturated),
  2. convert to linear light and normalise to the brightest version of that hue,
  3. apply per-channel white balance (the blue die dominates "white" on most LEDs),
  4. scale by a level derived from the colour's perceptual lightness, never below
     `min_level` so dark colours still read as colour rather than off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ambiance.color import RGB
from ambiance.colorspace import (gamut_clip, oklab_to_linear, oklab_to_oklch, oklch_to_oklab,
                                 srgb_to_oklab)


@dataclass(frozen=True)
class LedProfile:
    white_balance: tuple[float, float, float] = (1.0, 0.82, 0.62)
    vibrance: float = 1.3
    min_level: float = 0.3
    max_level: float = 1.0


DEFAULT_PROFILE = LedProfile()


def to_led(color: RGB, profile: LedProfile = DEFAULT_PROFILE) -> RGB:
    if color == (0, 0, 0):
        return RGB(0, 0, 0)
    lch = oklab_to_oklch(srgb_to_oklab(np.array(color, dtype=float) / 255.0))
    lch[1] *= profile.vibrance
    lch = gamut_clip(lch)
    linear = np.clip(oklab_to_linear(oklch_to_oklab(lch)), 0.0, 1.0)
    peak = linear.max()
    if peak <= 1e-6:
        return RGB(0, 0, 0)
    drive = (linear / peak) * np.array(profile.white_balance)
    drive /= drive.max()
    lightness = float(lch[0])
    level = profile.min_level + (profile.max_level - profile.min_level) * min(1.0, lightness / 0.8)
    return RGB(*(int(round(v)) for v in drive * level * 255))
