"""Strip colour cycling, keyboard movement (gradient drift, mosaic pan), strip brightness."""

from __future__ import annotations

import numpy as np

from ambiance.color import RGB
from ambiance.core import MediaState, RenderContext
from ambiance.devices.elk_strip import STRIP_LAYOUT, brightness_commands
from ambiance.layout import TKL_ANSI
from ambiance.palette import Palette, Swatch
from ambiance.scenes.album import AlbumScene
from ambiance.colorspace import srgb_to_oklab


def swatch(rgb, weight=0.3):
    lab = srgb_to_oklab(np.array(rgb, dtype=float) / 255.0)
    return Swatch(RGB(*rgb), tuple(float(v) for v in lab), weight, 1.0)


RED, BLUE, YELLOW, NEAR_BLACK = (220, 30, 40), (30, 60, 220), (240, 210, 40), (12, 10, 14)


def media(swatches, thumbnail=None, url="https://art/1"):
    return MediaState("spotify", "Playing", "t1", "Song", ("Artist",), "Album", url,
                      Palette(tuple(swatches)), thumbnail)


def strip_color(scene, now):
    return scene.render(RenderContext(STRIP_LAYOUT, {}, now))[0]


def test_strip_static_top_colour_by_default():
    scene = AlbumScene(media([swatch(RED), swatch(BLUE)]))
    assert not scene.animated
    assert strip_color(scene, 0.0) == strip_color(scene, 123.0) == RGB(*RED)


def test_strip_cycles_hold_then_crossfade():
    scene = AlbumScene(media([swatch(RED), swatch(BLUE), swatch(YELLOW)]), strip_cycle_s=10.0)
    assert scene.animated
    assert strip_color(scene, 0.0) == RGB(*RED)
    assert strip_color(scene, 5.0) == RGB(*RED)         # still holding (hold = 55%)
    assert strip_color(scene, 10.0) == RGB(*BLUE)
    assert strip_color(scene, 20.0) == RGB(*YELLOW)
    assert strip_color(scene, 30.0) == RGB(*RED)        # wraps around
    mid = strip_color(scene, 8.0)                        # crossfading red -> blue
    assert mid not in (RGB(*RED), RGB(*BLUE))


def test_strip_skips_near_black_and_single_colour_does_not_animate():
    scene = AlbumScene(media([swatch(NEAR_BLACK), swatch(RED)]), strip_cycle_s=10.0)
    assert not scene.animated                            # only one usable colour left
    assert strip_color(scene, 0.0) == strip_color(scene, 10.0) == RGB(*RED)


def test_gradient_drift_is_seamless_and_moves():
    scene = AlbumScene(media([swatch(RED), swatch(BLUE), swatch(YELLOW)]), drift_period_s=20.0)
    assert scene.animated
    frames = [scene.render(RenderContext(TKL_ANSI, {}, t)) for t in np.arange(0, 20, 0.5)]
    assert all(set(f) == set(TKL_ANSI.zones) for f in frames)
    changes = [sum(a[z] != b[z] for z in a) for a, b in zip(frames, frames[1:])]
    assert max(changes) > 0                              # it moves
    # ping-pong: one period later the picture is identical (no seam / jump at the wrap)
    assert scene.render(RenderContext(TKL_ANSI, {}, 0.0)) == \
        scene.render(RenderContext(TKL_ANSI, {}, 20.0))


def checker_art(size=96):
    img = np.zeros((size, size, 3))
    img[: size // 2, : size // 2] = (0.9, 0.1, 0.1)
    img[: size // 2, size // 2:] = (0.1, 0.2, 0.9)
    img[size // 2:, : size // 2] = (0.95, 0.85, 0.1)
    img[size // 2:, size // 2:] = (0.1, 0.8, 0.3)
    return img


def test_mosaic_pan_uses_stable_palette_and_moves_gradually():
    scene = AlbumScene(media([swatch(RED)], checker_art()), style="mosaic", steps=6,
                       drift_period_s=40.0, zoom=0.7, band=0.5)
    assert scene.animated and scene.mode == "mosaic"
    frames = [scene.render(RenderContext(TKL_ANSI, {}, t)) for t in np.arange(0, 40, 1.0)]
    used = set().union(*(set(f.values()) for f in frames))
    assert len(used) <= 6                                # one fixed palette for the whole pan
    changes = [sum(a[z] != b[z] for z in a) for a, b in zip(frames, frames[1:])]
    assert max(changes) > 0                              # the view really pans
    assert max(changes) < len(TKL_ANSI.keys) // 2        # ...gradually (cheap diff paints)


def test_static_mosaic_unchanged_behaviour():
    scene = AlbumScene(media([swatch(RED)], checker_art()), style="mosaic", steps=4)
    assert not scene.animated
    a = scene.render(RenderContext(TKL_ANSI, {}, 0.0))
    assert a == scene.render(RenderContext(TKL_ANSI, {}, 99.0))
    assert len(set(a.values())) <= 4


def test_key_changes_with_motion_options():
    m = media([swatch(RED), swatch(BLUE)])
    assert AlbumScene(m).key() != AlbumScene(m, drift_period_s=30.0).key()
    assert AlbumScene(m).key() != AlbumScene(m, strip_cycle_s=12.0).key()


def test_brightness_commands():
    a, b = brightness_commands(100)
    assert a == bytes.fromhex("7e040164ffffff00ef") and b == bytes.fromhex("7e04016401ffff00ef")
    assert brightness_commands(250)[0][3] == 100 and brightness_commands(-5)[0][3] == 0
