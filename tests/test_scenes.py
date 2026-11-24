"""Scenes: solid/off, album gradient (with fallback and drift) and album mosaic."""

from __future__ import annotations

import numpy as np
import pytest

from ambiance.color import BLACK, RGB
from ambiance.core import MediaState, RenderContext
from ambiance.layout import TKL_ANSI, TKL_ISO, Layout
from ambiance.palette import Palette, extract_palette
from ambiance.scenes.album import AlbumScene, mosaic_frame, sample_keys
from ambiance.scenes.basic import OffScene, SolidScene

LAYOUTS = [TKL_ANSI, TKL_ISO]


def make_palette() -> Palette:
    """Three clearly different colours, like a simple album cover."""
    blocks = [(0.85, 0.10, 0.10), (0.10, 0.25, 0.85), (0.95, 0.80, 0.10)]
    pixels = np.array([c for c in blocks for _ in range(300)], dtype=float)
    return extract_palette(pixels, max_colors=3)


PALETTE = make_palette()


def make_media(art_url: str = "https://i.scdn.co/image/aaa", *, palette: object = PALETTE,
               thumbnail: object = None, status: str = "Playing") -> MediaState:
    return MediaState(player="spotify", status=status, track_id="t1", title="Song",
                      artists=("Artist",), album="Album", art_url=art_url, palette=palette,
                      thumbnail=thumbnail)


def ctx(layout: Layout = TKL_ANSI, now: float = 0.0) -> RenderContext:
    return RenderContext(layout, {}, now)


def noise_image(seed: int = 7, size: int = 48) -> np.ndarray:
    return np.random.default_rng(seed).random((size, size, 3))


def split_image(left: tuple[float, float, float], right: tuple[float, float, float],
                size: int = 40) -> np.ndarray:
    img = np.empty((size, size, 3))
    img[:, : size // 2] = left
    img[:, size // 2:] = right
    return img


# ---- basic scenes ------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda lay: lay.name)
def test_solid_scene_paints_every_key(layout: Layout) -> None:
    frame = SolidScene(RGB(10, 20, 30)).render(ctx(layout))
    assert set(frame) == layout.zones
    assert set(frame.values()) == {RGB(10, 20, 30)}


def test_solid_scene_key_is_stable_and_colour_specific() -> None:
    assert SolidScene(RGB(1, 2, 3)).key() == SolidScene(RGB(1, 2, 3)).key()
    assert SolidScene(RGB(1, 2, 3)).key() != SolidScene(RGB(1, 2, 4)).key()
    assert SolidScene((1, 2, 3)).color == RGB(1, 2, 3)  # type: ignore[arg-type]


def test_off_scene_is_black_everywhere_with_stable_key() -> None:
    frame = OffScene().render(ctx())
    assert set(frame) == TKL_ANSI.zones
    assert set(frame.values()) == {BLACK}
    assert OffScene().key() == OffScene().key() != SolidScene(BLACK).key()


# ---- album: gradient ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda lay: lay.name)
@pytest.mark.parametrize("steps", [1, 4, 10])
def test_gradient_covers_every_key_with_at_most_steps_colours(layout: Layout, steps: int) -> None:
    frame = AlbumScene(make_media(), steps=steps).render(ctx(layout))
    assert set(frame) == layout.zones
    assert 1 <= len(set(frame.values())) <= steps


def test_gradient_draws_from_the_palette() -> None:
    frame = AlbumScene(make_media(), steps=10).render(ctx())
    fallback = AlbumScene(make_media(palette=None), steps=10).render(ctx())
    assert frame != fallback
    # the gradient passes through (close to) every palette colour
    for swatch in PALETTE.swatches:
        best = min(sum(abs(a - b) for a, b in zip(c, swatch.rgb)) for c in frame.values())
        assert best < 60, (swatch.rgb, best)


def test_missing_palette_falls_back_to_a_warm_white_gradient() -> None:
    scene = AlbumScene(make_media(palette=None))
    frame = scene.render(ctx())
    assert scene.uses_fallback
    assert set(frame) == TKL_ANSI.zones
    assert len(set(frame.values())) > 1  # a gradient, not a flat fill
    for c in frame.values():
        assert c.r >= c.g >= c.b and c.r > 200, c  # warm


def test_empty_palette_counts_as_missing() -> None:
    assert AlbumScene(make_media(palette=Palette(()))).uses_fallback


def test_static_gradient_is_not_animated_and_drifting_one_is() -> None:
    static = AlbumScene(make_media())
    assert not static.animated
    assert static.render(ctx(now=0.0)) == static.render(ctx(now=123.0))
    drifting = AlbumScene(make_media(), drift_period_s=10.0)
    assert drifting.animated
    assert drifting.render(ctx(now=0.0)) != drifting.render(ctx(now=5.0))
    assert drifting.render(ctx(now=1.0)) == drifting.render(ctx(now=11.0))  # periodic


def test_render_returns_a_copy_of_the_cached_frame() -> None:
    scene = AlbumScene(make_media())
    first = scene.render(ctx())
    first.clear()
    assert set(scene.render(ctx())) == TKL_ANSI.zones


# ---- album: identity ---------------------------------------------------------------------------

def test_key_is_equal_for_identical_scenes_and_changes_with_art_url() -> None:
    a = AlbumScene(make_media("https://art/1"))
    assert a.key() == AlbumScene(make_media("https://art/1")).key()
    assert a.key() != AlbumScene(make_media("https://art/2")).key()


def test_key_ignores_playback_status_so_pause_only_changes_level() -> None:
    playing = AlbumScene(make_media(status="Playing"))
    paused = AlbumScene(make_media(status="Paused"))
    assert playing.key() == paused.key()


@pytest.mark.parametrize("kwargs", [
    {"steps": 6}, {"angle_deg": 45.0}, {"drift_period_s": 30.0}, {"style": "mosaic"},
])
def test_key_changes_with_style_parameters(kwargs: dict) -> None:
    media = make_media(thumbnail=noise_image())
    assert AlbumScene(media).key() != AlbumScene(media, **kwargs).key()


def test_key_changes_when_the_palette_or_thumbnail_arrives() -> None:
    pending = AlbumScene(make_media(palette=None))
    ready = AlbumScene(make_media())
    assert pending.key() != ready.key()
    no_thumb = AlbumScene(make_media(), style="mosaic")
    with_thumb = AlbumScene(make_media(thumbnail=noise_image()), style="mosaic")
    assert no_thumb.key() != with_thumb.key()


def test_key_is_hashable() -> None:
    hash(AlbumScene(make_media()).key())
    hash(AlbumScene(make_media(thumbnail=noise_image()), style="mosaic").key())


# ---- album: mosaic -----------------------------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda lay: lay.name)
@pytest.mark.parametrize("steps", [1, 3, 6, 10])
def test_mosaic_covers_every_key_and_caps_distinct_colours(layout: Layout, steps: int) -> None:
    scene = AlbumScene(make_media(thumbnail=noise_image()), style="mosaic", steps=steps)
    assert scene.mode == "mosaic" and not scene.animated
    frame = scene.render(ctx(layout))
    assert set(frame) == layout.zones
    assert 1 <= len(set(frame.values())) <= steps


def test_mosaic_projects_the_image_left_to_right() -> None:
    img = split_image((0.9, 0.05, 0.05), (0.05, 0.05, 0.9))
    frame = AlbumScene(make_media(thumbnail=img), style="mosaic", steps=4).render(ctx())
    for key in TKL_ANSI.keys:
        x, _ = TKL_ANSI.norm(key)
        c = frame[key.zone]
        if x < 0.4:
            assert c.r > c.b, (key.name, c)
        elif x > 0.6:
            assert c.b > c.r, (key.name, c)


def test_mosaic_band_maps_a_centred_horizontal_strip() -> None:
    img = np.empty((60, 60, 3))
    img[:] = (0.1, 0.8, 0.1)       # green top and bottom quarters...
    img[15:45] = (0.85, 0.1, 0.1)  # ...red middle half
    narrow = mosaic_frame(TKL_ANSI, img, steps=4, band=0.34, blur=0.35)
    assert all(c.r > c.g for c in narrow.values())  # only the red band is used
    whole = mosaic_frame(TKL_ANSI, img, steps=4, band=1.0, blur=0.0)
    top_row = [k for k in TKL_ANSI.keys if k.y == 0.0]
    assert all(whole[k.zone].g > whole[k.zone].r for k in top_row)


def test_mosaic_averages_in_linear_light() -> None:
    img = np.zeros((256, 256, 3))
    img[:, ::2] = 1.0  # fine black/white stripes: optically a mid grey
    lin = sample_keys(TKL_ANSI, img, band=1.0, blur=0.35)
    assert np.allclose(lin, 0.5, atol=0.05)  # sRGB averaging would give ~0.21


def test_mosaic_accepts_uint8_and_grayscale_thumbnails() -> None:
    rgb8 = (noise_image() * 255).astype(np.uint8)
    frame = AlbumScene(make_media(thumbnail=rgb8), style="mosaic").render(ctx())
    assert set(frame) == TKL_ANSI.zones
    gray = np.linspace(0, 1, 32 * 32).reshape(32, 32)
    frame = AlbumScene(make_media(thumbnail=gray), style="mosaic").render(ctx())
    assert all(c.r == c.g == c.b for c in frame.values())


def test_mosaic_without_thumbnail_falls_back_to_the_palette_gradient() -> None:
    scene = AlbumScene(make_media(thumbnail=None), style="mosaic", steps=5)
    assert scene.mode == "gradient"
    assert scene.render(ctx()) == AlbumScene(make_media(), steps=5).render(ctx())


def test_mosaic_ignores_unusable_thumbnails() -> None:
    scene = AlbumScene(make_media(thumbnail=np.zeros((4,))), style="mosaic")
    assert scene.mode == "gradient"


@pytest.mark.parametrize("kwargs", [
    {"style": "spiral"}, {"steps": 0}, {"drift_period_s": -1.0}, {"band": 0.0}, {"band": 1.5},
    {"blur": -0.1},
])
def test_invalid_parameters_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AlbumScene(make_media(), **kwargs)
