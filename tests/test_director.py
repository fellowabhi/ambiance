"""DefaultDirector: rule priorities, levels, transitions and extension."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from ambiance.color import RGB
from ambiance.core import MEDIA, PRESENCE, Decision, MediaState, Presence
from ambiance.director import (DEFAULT_RULES, AlbumStyle, DefaultDirector, DirectorConfig,
                               rule_idle, rule_media)
from ambiance.palette import extract_palette
from ambiance.scenes.album import AlbumScene
from ambiance.scenes.basic import OffScene, SolidScene

PALETTE = extract_palette(np.array([(0.9, 0.2, 0.1)] * 50 + [(0.1, 0.3, 0.9)] * 50), max_colors=2)


def media(status: str = "Playing", *, art_url: str = "https://art/1", palette: object = PALETTE,
          thumbnail: object = None) -> MediaState:
    return MediaState(player="spotify", status=status, track_id="t", title="Song",
                      artists=("A",), album="B", art_url=art_url, palette=palette,
                      thumbnail=thumbnail)


def decide(state: Mapping[str, Any], **cfg: Any) -> Decision:
    return DefaultDirector(DirectorConfig(**cfg)).decide(state)


def is_release(d: Decision) -> bool:
    return d.release and isinstance(d.scene, OffScene) and d.level == 0.0


# ---- no media ----------------------------------------------------------------------------------

def test_empty_state_releases_by_default() -> None:
    d = decide({})
    assert is_release(d)
    assert d.transition_s == DirectorConfig().no_media_transition_s


def test_no_media_solid_shows_the_fallback_colour() -> None:
    d = decide({}, when_no_media="solid", fallback_color=RGB(1, 2, 3), playing_level=0.8)
    assert isinstance(d.scene, SolidScene) and d.scene.color == RGB(1, 2, 3)
    assert d.level == 0.8 and not d.release


def test_no_media_off_paints_black_but_keeps_control() -> None:
    d = decide({}, when_no_media="off")
    assert isinstance(d.scene, OffScene) and d.level == 0.0 and not d.release


@pytest.mark.parametrize("m", [
    media("Stopped"),
    media(art_url="", palette=None),  # a track without any cover art
    None,
], ids=["stopped", "no-art", "none"])
def test_media_without_anything_to_show_falls_through(m: MediaState | None) -> None:
    assert is_release(decide({MEDIA: m}))


# ---- media -------------------------------------------------------------------------------------

def test_playing_with_palette_shows_album_at_playing_level() -> None:
    cfg = DirectorConfig()
    d = decide({MEDIA: media("Playing")})
    assert isinstance(d.scene, AlbumScene) and not d.release
    assert d.level == cfg.playing_level and d.transition_s == cfg.track_transition_s


def test_paused_keeps_the_scene_key_and_dims() -> None:
    cfg = DirectorConfig(paused_level=0.2, pause_transition_s=0.7)
    playing = DefaultDirector(cfg).decide({MEDIA: media("Playing")})
    paused = DefaultDirector(cfg).decide({MEDIA: media("Paused")})
    assert paused.scene.key() == playing.scene.key()
    assert paused.level == 0.2 and paused.transition_s == 0.7


def test_thumbnail_alone_is_enough_for_album() -> None:
    d = decide({MEDIA: media(palette=None, thumbnail=np.zeros((8, 8, 3)))},
               album=AlbumStyle(style="mosaic"))
    assert isinstance(d.scene, AlbumScene) and d.scene.mode == "mosaic"


def test_art_still_loading_shows_album_fallback_instead_of_releasing() -> None:
    d = decide({MEDIA: media(palette=None)})
    assert isinstance(d.scene, AlbumScene) and d.scene.uses_fallback and not d.release
    ready = decide({MEDIA: media()})
    assert ready.scene.key() != d.scene.key()  # palette arrival => crossfade


def test_album_style_is_passed_to_the_scene() -> None:
    style = AlbumStyle(style="gradient", steps=4, angle_deg=30.0, drift_period_s=12.0)
    scene = decide({MEDIA: media()}, album=style).scene
    assert isinstance(scene, AlbumScene)
    assert (scene.steps, scene.angle_deg, scene.drift_period_s) == (4, 30.0, 12.0)
    assert scene.animated


# ---- idle --------------------------------------------------------------------------------------

def test_locked_screen_turns_lights_off_with_a_slow_fade() -> None:
    cfg = DirectorConfig()
    d = decide({PRESENCE: Presence(locked=True), MEDIA: media()})
    assert isinstance(d.scene, OffScene) and d.level == 0.0 and not d.release
    assert d.transition_s == cfg.idle_transition_s


def test_idle_can_release_instead() -> None:
    assert is_release(decide({PRESENCE: Presence(locked=True)}, when_idle="release"))


@pytest.mark.parametrize("idle_seconds, expect_idle", [(0.0, False), (299.9, False),
                                                       (300.0, True), (5000.0, True)])
def test_idle_threshold(idle_seconds: float, expect_idle: bool) -> None:
    d = decide({PRESENCE: Presence(idle_seconds=idle_seconds), MEDIA: media()}, idle_off_s=300)
    assert isinstance(d.scene, OffScene if expect_idle else AlbumScene)


def test_idle_off_s_zero_disables_the_timer_but_not_the_lock() -> None:
    state: dict[str, Any] = {PRESENCE: Presence(idle_seconds=1e6), MEDIA: media()}
    assert isinstance(decide(state, idle_off_s=0).scene, AlbumScene)
    state[PRESENCE] = Presence(idle_seconds=0.0, locked=True)
    assert isinstance(decide(state, idle_off_s=0).scene, OffScene)


def test_idle_beats_playing_media_and_media_beats_no_media() -> None:
    playing = {MEDIA: media("Playing")}
    assert isinstance(decide(playing).scene, AlbumScene)
    away = {**playing, PRESENCE: Presence(idle_seconds=10_000)}
    assert isinstance(decide(away).scene, OffScene)


def test_presence_is_duck_typed() -> None:
    class OtherPresence:
        idle_seconds = 0.0
        locked = True

    assert isinstance(decide({PRESENCE: OtherPresence()}).scene, OffScene)


# ---- purity & extension ------------------------------------------------------------------------

def test_decide_is_a_pure_function_of_the_snapshot() -> None:
    director = DefaultDirector()
    state = {MEDIA: media(), PRESENCE: Presence(idle_seconds=3.0)}
    before = copy.copy(state)
    a, b = director.decide(state), director.decide(state)
    assert a.scene.key() == b.scene.key() and a.level == b.level
    assert state == before


def test_custom_rule_slots_into_the_priority_order() -> None:
    def rule_party(state: Mapping[str, Any], cfg: DirectorConfig) -> Decision | None:
        if not state.get("party"):
            return None
        return Decision(SolidScene(RGB(255, 0, 255)), level=1.0, transition_s=0.3)

    director = DefaultDirector(rules=(rule_idle, rule_party, rule_media))
    party = {"party": True, MEDIA: media()}
    assert director.decide(party).scene.key() == SolidScene(RGB(255, 0, 255)).key()
    assert isinstance(director.decide({**party, PRESENCE: Presence(locked=True)}).scene, OffScene)
    assert isinstance(director.decide({MEDIA: media()}).scene, AlbumScene)
    assert DEFAULT_RULES == (rule_idle, rule_media)


def test_custom_fallback() -> None:
    director = DefaultDirector(rules=(), fallback=lambda s, c: Decision(SolidScene(RGB(9, 9, 9))))
    assert director.decide({MEDIA: media()}).scene.key() == SolidScene(RGB(9, 9, 9)).key()


# ---- config validation -------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, field", [
    ({"paused_level": 1.5}, "paused_level"),
    ({"playing_level": -0.1}, "playing_level"),
    ({"when_idle": "sleep"}, "when_idle"),
    ({"when_no_media": "rainbow"}, "when_no_media"),
    ({"track_transition_s": -1.0}, "track_transition_s"),
])
def test_director_config_rejects_bad_values(kwargs: dict, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        DirectorConfig(**kwargs)


@pytest.mark.parametrize("kwargs, field", [
    ({"style": "spiral"}, "style"), ({"steps": 0}, "steps"), ({"band": 0.0}, "band"),
    ({"drift_period_s": -5.0}, "drift_period_s"),
])
def test_album_style_rejects_bad_values(kwargs: dict, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        AlbumStyle(**kwargs)


def test_fallback_colour_is_normalised_to_rgb() -> None:
    assert DirectorConfig(fallback_color=(300, 2, 3)).fallback_color == RGB(255, 2, 3)  # type: ignore[arg-type]
