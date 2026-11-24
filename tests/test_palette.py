"""Tests for palette extraction from images (album art).

The images are synthetic, so the right answer is known exactly. Colour stripes are
aligned to the thumbnail's box filter, so downsampling never invents blended colours.
"""

from __future__ import annotations

import io
import itertools

import numpy as np
import pytest
from PIL import Image

from ambiance.color import RGB
from ambiance.colorspace import delta_e
from ambiance.palette import (Palette, Swatch, extract_palette, kmeans, load_pixels,
                              palette_from_image_bytes)

RED, BLUE = (220, 30, 40), (30, 60, 200)
EIGHT = [(230, 20, 20), (20, 200, 40), (30, 60, 220), (240, 220, 30), (200, 30, 200),
         (30, 200, 210), (250, 140, 20), (255, 255, 255)]


def encode(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def split_image(size: int, left: tuple[int, int, int], right: tuple[int, int, int],
                left_share: float) -> Image.Image:
    img = Image.new("RGB", (size, size), left)
    img.paste(right, (int(size * left_share), 0, size, size))
    return img


def stripes(colors, width: int = 8, height: int = 64) -> Image.Image:
    img = Image.new("RGB", (width * len(colors), height))
    for i, c in enumerate(colors):
        img.paste(c, (i * width, 0, (i + 1) * width, height))
    return img


def close(a: RGB, b: tuple[int, int, int], tol: int = 2) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


@pytest.mark.parametrize("size", [64, 640])  # 640: downsampled 10x by the box filter
def test_two_colour_image_finds_both_with_weights_matching_shares(size):
    palette = palette_from_image_bytes(encode(split_image(size, RED, BLUE, 0.75)))
    assert len(palette) == 2
    first, second = palette.swatches
    assert close(first.rgb, RED) and close(second.rgb, BLUE)
    assert first.weight == pytest.approx(0.75, abs=0.01)
    assert second.weight == pytest.approx(0.25, abs=0.01)


def test_mostly_black_image_ranks_the_small_red_patch_first():
    """Big dark backgrounds make poor LED colours: the chromatic accent must lead."""
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    img.paste((230, 20, 20), (10, 10, 26, 26))  # 16x16 = 6.25 % of the pixels
    palette = palette_from_image_bytes(encode(img))
    assert close(palette.swatches[0].rgb, (230, 20, 20))
    assert palette.swatches[0].weight == pytest.approx(0.0625, abs=0.005)
    black = [s for s in palette.swatches if s.lightness < 0.05]
    assert black and black[0].weight > 0.9  # still reported, just ranked lower


def test_dark_tinted_background_loses_to_a_bright_accent():
    """Near-black colours light LEDs poorly even when slightly chromatic (navy, maroon)."""
    img = Image.new("RGB", (64, 64), (12, 14, 48))     # navy, L ~ 0.19
    img.paste((230, 20, 20), (0, 0, 16, 16))
    palette = palette_from_image_bytes(encode(img))
    assert close(palette.swatches[0].rgb, (230, 20, 20))


def test_near_duplicate_colours_collapse_into_one_swatch():
    """Two reds 0.04 apart in OKLab look the same on LEDs: keep one, make room for blue."""
    img = Image.new("RGB", (64, 64), (220, 30, 40))
    img.paste((200, 40, 50), (24, 0, 48, 64))
    img.paste(BLUE, (48, 0, 64, 64))
    pixels = load_pixels(encode(img))
    palette = extract_palette(pixels)
    assert len(palette) == 2
    assert close(palette.swatches[1].rgb, BLUE)
    assert len(extract_palette(pixels, min_distance=0.0)) == 3


def test_same_input_gives_the_same_palette():
    rng = np.random.default_rng(7)
    data = encode(Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)))
    assert palette_from_image_bytes(data) == palette_from_image_bytes(data)
    pixels = load_pixels(data)
    assert extract_palette(pixels.copy()) == extract_palette(pixels.copy())


@pytest.mark.parametrize("max_colors", [1, 2, 3, 5, 8])
def test_max_colors_is_respected(max_colors):
    pixels = load_pixels(encode(stripes(EIGHT)))
    palette = extract_palette(pixels, max_colors=max_colors)
    assert len(palette) == max_colors  # eight well-separated colours: never fewer
    assert len(palette_from_image_bytes(encode(stripes(EIGHT)), max_colors=max_colors)) == \
        max_colors


def test_all_eight_distinct_colours_are_recovered():
    palette = extract_palette(load_pixels(encode(stripes(EIGHT))), max_colors=8)
    for color in EIGHT:
        assert any(close(s.rgb, color) for s in palette.swatches), color
    assert sum(s.weight for s in palette.swatches) == pytest.approx(1.0)


def test_swatches_are_distinct_and_sorted_by_score():
    rng = np.random.default_rng(11)
    pixels = rng.random((4096, 3))
    palette = extract_palette(pixels, max_colors=5, min_distance=0.09)
    assert 1 <= len(palette) <= 5
    for a, b in itertools.combinations(palette.swatches, 2):
        assert delta_e(a.oklab, b.oklab) >= 0.09
    scores = [s.score for s in palette.swatches]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s.weight <= 1.0 for s in palette.swatches)
    assert all(isinstance(s, Swatch) and all(0 <= c <= 255 for c in s.rgb)
               for s in palette.swatches)


def test_palette_from_png_bytes():
    palette = palette_from_image_bytes(encode(split_image(64, BLUE, RED, 0.5)), max_colors=3)
    assert isinstance(palette, Palette)
    assert {tuple(c) for c in palette.colors} == {RED, BLUE}
    assert not palette.is_grayscale


def test_palette_from_jpeg_bytes_is_close():
    palette = palette_from_image_bytes(encode(Image.new("RGB", (64, 64), RED), "JPEG"))
    assert len(palette) == 1
    assert close(palette.swatches[0].rgb, RED, tol=4)


def test_grey_image_gives_a_greyscale_palette():
    palette = palette_from_image_bytes(encode(Image.new("RGB", (32, 32), (128, 128, 128))))
    assert palette.is_grayscale
    assert palette.colors == [RGB(128, 128, 128)]
    assert palette.swatches[0].weight == 1.0


def test_load_pixels_downsamples_and_normalises():
    img = Image.new("RGBA", (640, 320), (255, 128, 0, 128))  # alpha is dropped
    pixels = load_pixels(encode(img), size=64)
    assert pixels.shape == (64 * 32, 3)
    np.testing.assert_allclose(pixels[0], [1.0, 128 / 255, 0.0])
    assert pixels.min() >= 0.0 and pixels.max() <= 1.0


def test_load_pixels_accepts_palette_mode_images():
    img = stripes(EIGHT).convert("P")
    assert load_pixels(encode(img)).shape == (64 * 64, 3)


def test_kmeans_caps_k_at_the_number_of_points_and_is_deterministic():
    points = np.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [0.5, 0.2, 0.1]])
    centres, labels = kmeans(points, k=10)
    assert len(centres) == 3
    assert sorted(labels.tolist()) == [0, 1, 2]
    again, _ = kmeans(points, k=10)
    np.testing.assert_array_equal(centres, again)
