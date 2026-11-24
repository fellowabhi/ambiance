"""Tests for to_led: screen colour (sRGB) -> LED drive value (linear PWM).

Most properties are checked against relations, not against hard-coded outputs, so the
profile constants (white balance, vibrance, level curve) can still be tuned on the
hardware without rewriting these tests.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from ambiance.color import BLACK, RGB, WHITE
from ambiance.colorspace import linear_to_oklab, oklab_to_oklch, srgb_to_oklab
from ambiance.led import DEFAULT_PROFILE, LedProfile, to_led

NEUTRAL = LedProfile(white_balance=(1.0, 1.0, 1.0))
SATURATED = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 90, 0), (160, 0, 255), (0, 160, 140),
             (255, 60, 140), (200, 30, 40), (30, 60, 200), (255, 200, 0)]


def random_colours(n: int, seed: int) -> list[RGB]:
    rng = random.Random(seed)
    return [RGB(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(n)]


def hue_deg_of_srgb(rgb) -> float:
    return math.degrees(oklab_to_oklch(srgb_to_oklab(np.array(rgb, dtype=float) / 255.0))[2])


def hue_deg_of_drive(led: RGB, profile: LedProfile = DEFAULT_PROFILE) -> float:
    """Hue of the light an LED emits: drive values are linear, minus the white balance."""
    linear = np.array(led, dtype=float) / np.array(profile.white_balance)
    return math.degrees(oklab_to_oklch(linear_to_oklab(linear / linear.max()))[2])


def hue_gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_black_stays_off():
    assert to_led(BLACK) == BLACK
    assert to_led(RGB(0, 0, 0), NEUTRAL) == BLACK


def test_white_balance_reduces_blue_for_white():
    led = to_led(WHITE)
    assert led.r == 255
    assert led.b < led.g < led.r
    wb_r, wb_g, wb_b = DEFAULT_PROFILE.white_balance
    assert led.g / led.r == pytest.approx(wb_g / wb_r, abs=0.01)
    assert led.b / led.r == pytest.approx(wb_b / wb_r, abs=0.01)


def test_neutral_white_balance_keeps_greys_neutral():
    assert to_led(WHITE, NEUTRAL) == WHITE
    for v in (8, 60, 128, 200):
        r, g, b = to_led(RGB(v, v, v), NEUTRAL)
        assert r == g == b


def test_greys_get_the_same_white_balance_as_white():
    for v in (20, 90, 160):
        led = to_led(RGB(v, v, v))
        assert led.g / led.r == pytest.approx(0.82, abs=0.02)
        assert led.b / led.r == pytest.approx(0.62, abs=0.02)


@pytest.mark.parametrize(("rgb", "channel"), [((255, 0, 0), 0), ((0, 255, 0), 1), ((0, 0, 255), 2)])
def test_saturated_primaries_keep_their_dominant_channel(rgb, channel):
    led = to_led(RGB(*rgb))
    others = [v for i, v in enumerate(led) if i != channel]
    assert max(others) <= 0.06 * led[channel]


@pytest.mark.parametrize("rgb", SATURATED)
def test_hue_is_preserved_once_white_balance_is_undone(rgb):
    led = to_led(RGB(*rgb))
    assert hue_gap(hue_deg_of_drive(led), hue_deg_of_srgb(rgb)) < 3.0


def test_orange_does_not_turn_yellow():
    """Sending sRGB straight to PWM makes orange look yellow; drive values must be linear."""
    led = to_led(RGB(255, 90, 0))
    assert led.g / led.r < 0.15  # vs 90/255 = 0.35 in sRGB
    assert led.b == 0


def test_dark_colours_never_fall_below_min_level():
    floor = math.floor(255 * DEFAULT_PROFILE.min_level)
    for rgb in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (3, 3, 3), (10, 0, 20), (40, 0, 0)]:
        assert max(to_led(RGB(*rgb))) >= floor, rgb


def test_min_level_is_configurable():
    bright_floor = LedProfile(min_level=0.5)
    assert max(to_led(RGB(10, 0, 0), bright_floor)) >= 127
    assert max(to_led(RGB(10, 0, 0), bright_floor)) > max(to_led(RGB(10, 0, 0)))


def test_max_level_caps_the_output():
    dim = LedProfile(max_level=0.5)
    assert max(to_led(WHITE, dim)) == round(255 * 0.5)
    assert all(max(to_led(c, dim)) <= 128 for c in random_colours(100, seed=1))


def test_darker_input_is_never_brighter():
    for c in random_colours(150, seed=2):
        if c == BLACK:
            continue
        top = max(to_led(c))
        for k in (0.75, 0.5, 0.25):
            darker = c.scaled(k)
            if darker != BLACK:
                assert max(to_led(darker)) <= top, (c, darker)


def test_vibrance_boosts_muted_colours():
    muted = RGB(180, 120, 110)

    def saturation(led: RGB) -> float:
        return (max(led) - min(led)) / max(led)

    flat = saturation(to_led(muted, LedProfile(vibrance=1.0, white_balance=(1, 1, 1))))
    boosted = saturation(to_led(muted, LedProfile(vibrance=1.6, white_balance=(1, 1, 1))))
    assert boosted > flat + 0.05


def test_output_is_valid_8_bit_and_never_off_for_a_real_colour():
    for c in random_colours(300, seed=3):
        led = to_led(c)
        assert isinstance(led, RGB)
        assert all(type(v) is int and 0 <= v <= 255 for v in led)
        if c != BLACK:
            assert max(led) > 0
