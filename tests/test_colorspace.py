"""Tests for the sRGB <-> linear <-> OKLab / OKLCH conversions and gamut clipping.

Reference values come from outside this module: Ottosson's published OKLab test table
(XYZ inputs), and the OKLab coordinates of the sRGB primaries that CSS Color 4
implementations agree on. Matching them checks that the conversion matrices are
transcribed correctly. A self-consistent round trip alone would not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ambiance import colorspace as cs

# Ottosson (2020), "A perceptual color space for image processing", test table.
OTTOSSON_XYZ_TABLE = [
    ((0.950, 1.000, 1.089), (1.000, 0.000, 0.000)),
    ((1.000, 0.000, 0.000), (0.450, 1.236, -0.019)),
    ((0.000, 1.000, 0.000), (0.922, -0.671, 0.263)),
    ((0.000, 0.000, 1.000), (0.153, -1.415, -0.449)),
]
# IEC 61966-2-1 XYZ (D65) -> linear sRGB, to feed the table into linear_to_oklab.
XYZ_TO_LINEAR_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                               [-0.9692660, 1.8760108, 0.0415560],
                               [0.0556434, -0.2040259, 1.0572252]])

SRGB_PRIMARIES = [
    ((1.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.627955, 0.224863, 0.125846)),
    ((0.0, 1.0, 0.0), (0.866440, -0.233888, 0.179498)),
    ((0.0, 0.0, 1.0), (0.452014, -0.032457, -0.311528)),
]


@pytest.mark.parametrize(("xyz", "lab"), OTTOSSON_XYZ_TABLE)
def test_matches_ottosson_reference_table(xyz, lab):
    got = cs.linear_to_oklab(XYZ_TO_LINEAR_SRGB @ np.array(xyz))
    np.testing.assert_allclose(got, lab, atol=1e-3)


@pytest.mark.parametrize(("srgb", "lab"), SRGB_PRIMARIES)
def test_known_oklab_values_of_srgb_primaries(srgb, lab):
    np.testing.assert_allclose(cs.srgb_to_oklab(srgb), lab, atol=1e-5)


def test_white_is_l1_and_black_is_l0_with_no_chroma():
    white, black = cs.srgb_to_oklab([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(white, [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(black, [0.0, 0.0, 0.0], atol=1e-12)


def test_srgb_transfer_function_known_points():
    np.testing.assert_allclose(cs.srgb_to_linear([0.0, 0.5, 1.0]), [0.0, 0.214041, 1.0],
                               atol=1e-6)
    # The two pieces of the curve meet at 0.04045 <-> 0.0031308.
    assert cs.srgb_to_linear(0.04045) == pytest.approx(0.0031308, abs=1e-6)
    assert cs.linear_to_srgb(0.0031308) == pytest.approx(0.04045, abs=1e-6)


def test_linear_to_srgb_clips_out_of_range_light():
    np.testing.assert_allclose(cs.linear_to_srgb([1.5, -0.2]), [1.0, 0.0], atol=1e-12)


def test_srgb_oklab_round_trip():
    rgb = np.random.default_rng(1).random((5000, 3))
    np.testing.assert_allclose(cs.oklab_to_srgb(cs.srgb_to_oklab(rgb)), rgb, atol=1e-5)


def test_linear_oklab_round_trip():
    lin = np.random.default_rng(2).random((5000, 3))
    np.testing.assert_allclose(cs.oklab_to_linear(cs.linear_to_oklab(lin)), lin, atol=1e-6)


def test_oklch_round_trip_and_red_hue():
    lab = cs.srgb_to_oklab(np.random.default_rng(3).random((1000, 3)))
    np.testing.assert_allclose(cs.oklch_to_oklab(cs.oklab_to_oklch(lab)), lab, atol=1e-12)
    L, c, h = cs.oklab_to_oklch(cs.srgb_to_oklab([1.0, 0.0, 0.0]))
    assert (L, c, math.degrees(h)) == pytest.approx((0.62796, 0.25768, 29.2339), abs=1e-3)


def test_grey_ramp_lightness_strictly_increases_and_stays_neutral():
    v = np.linspace(0.0, 1.0, 64)
    lab = cs.srgb_to_oklab(np.stack([v, v, v], axis=-1))
    assert np.all(np.diff(lab[:, 0]) > 0)
    assert np.abs(lab[:, 1:]).max() < 1e-6


def test_delta_e():
    assert cs.delta_e([0.5, 0.1, -0.1], [0.5, 0.1, -0.1]) == 0.0
    assert cs.delta_e(cs.srgb_to_oklab([1, 1, 1]), cs.srgb_to_oklab([0, 0, 0])) == pytest.approx(
        1.0, abs=1e-6)
    np.testing.assert_allclose(cs.delta_e([[0, 0, 0], [1, 0, 0]], [0.0, 0.0, 0.0]), [0.0, 1.0])


def test_in_gamut():
    assert cs.in_gamut([0.5, 0.5, 0.5])
    assert cs.in_gamut([1.0, 0.0, 1.0])
    assert not cs.in_gamut([1.1, 0.0, 0.0])
    assert not cs.in_gamut([-0.1, 0.5, 0.5])
    np.testing.assert_array_equal(cs.in_gamut([[0.2, 0.2, 0.2], [2, 0, 0]]), [True, False])


# -- gamut_clip ----------------------------------------------------------------------------------

HUES = np.linspace(-math.pi, math.pi, 25)


@pytest.mark.parametrize("L", [0.2, 0.45, 0.7, 0.9])
def test_gamut_clip_keeps_lightness_and_hue_and_lands_in_gamut(L):
    lch = np.stack([np.full_like(HUES, L), np.full_like(HUES, 0.5), HUES], axis=-1)
    out = cs.gamut_clip(lch)
    np.testing.assert_array_equal(out[:, 0], lch[:, 0])
    np.testing.assert_array_equal(out[:, 2], lch[:, 2])
    assert np.all(out[:, 1] < lch[:, 1])
    assert np.all(cs.in_gamut(cs.oklab_to_linear(cs.oklch_to_oklab(out))))


@pytest.mark.parametrize("L", [0.2, 0.45, 0.7, 0.9])
def test_gamut_clip_keeps_as_much_chroma_as_fits(L):
    """Clipping should stop at the gamut boundary, not merely somewhere inside it."""
    lch = np.stack([np.full_like(HUES, L), np.full_like(HUES, 0.5), HUES], axis=-1)
    out = cs.gamut_clip(lch)
    more = out + np.array([0.0, 0.002, 0.0])
    assert not np.any(cs.in_gamut(cs.oklab_to_linear(cs.oklch_to_oklab(more))))


def test_gamut_clip_leaves_in_gamut_colours_untouched():
    lch = cs.oklab_to_oklch(cs.srgb_to_oklab(np.random.default_rng(4).random((200, 3)) * 0.98
                                             + 0.01))
    np.testing.assert_array_equal(cs.gamut_clip(lch), lch)


def test_gamut_clip_preserves_shape_and_does_not_mutate_input():
    lch = np.array([[[0.7, 0.4, 1.0], [0.5, 0.01, 0.2]]])
    before = lch.copy()
    out = cs.gamut_clip(lch)
    assert out.shape == lch.shape
    np.testing.assert_array_equal(lch, before)
    assert cs.gamut_clip([0.7, 0.4, 1.0]).shape == (3,)
