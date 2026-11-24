"""The director: what should the lighting show, given a State snapshot?

``DefaultDirector.decide(snapshot)`` walks a tuple of *rules* in priority order and
returns the first Decision a rule produces; the fallback rule always decides.

1. ``rule_idle``     screen locked, or nobody touched the computer for ``idle_off_s``
                     -> OffScene at level 0 with a slow fade (``when_idle = "off"``), or hand
                     the keyboard back to its firmware (``when_idle = "release"``).
                     Off is *painted* black: while the host holds the keyboard its firmware
                     never dims the LEDs on its own, so black is what saves the battery.
2. ``rule_media``    a player is Playing or Paused and its track has cover art
                     -> AlbumScene at ``playing_level`` / ``paused_level``. Pausing keeps the
                     scene key, so the engine only fades the level.
3. ``rule_no_media`` (fallback) -> release (default), a solid fallback colour, or off.

Adding a rule
-------------
A rule is a pure function ``(snapshot, cfg) -> Decision | None``: return None to let
lower-priority rules decide. Insert it where it belongs in the priority order::

    def rule_meeting(state: Mapping[str, Any], cfg: DirectorConfig) -> Decision | None:
        if not state.get("in_meeting"):        # published by a (hypothetical) CalendarSource
            return None
        return Decision(SolidScene(RGB(255, 0, 0)), level=0.6, transition_s=0.5)

    director = DefaultDirector(cfg, rules=(rule_idle, rule_meeting, rule_media))

Keep rules pure (no I/O, no clock, no memory of previous calls). Anything a rule needs to
know about the world belongs in a Source that publishes it into the State; the engine
re-evaluates the director whenever the State changes. That keeps every decision
reproducible from a snapshot, which is what makes the director trivial to test.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ambiance.color import RGB
from ambiance.core import MEDIA, PRESENCE, Decision, Director
from ambiance.scenes.album import STYLES, AlbumScene
from ambiance.scenes.basic import OffScene, SolidScene

WHEN_IDLE = ("off", "release")
WHEN_NO_MEDIA = ("release", "solid", "off")


def _check_range(name: str, value: float, lo: float, hi: float | None = None) -> None:
    if hi is None:
        if not value >= lo:
            raise ValueError(f"{name} must be >= {lo:g}, got {value!r}")
    elif not lo <= value <= hi:
        raise ValueError(f"{name} must be within {lo:g}..{hi:g}, got {value!r}")


def _check_choice(name: str, value: str, choices: Sequence[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(map(repr, choices))}, got {value!r}")


@dataclass(frozen=True)
class AlbumStyle:
    """How AlbumScene looks (see ambiance.scenes.album)."""

    style: str = "gradient"       # "gradient" | "mosaic"
    steps: int = 10               # max distinct colours (fewer = faster HID++ frames)
    angle_deg: float = 15.0       # gradient axis, 0 = left to right
    drift_period_s: float = 0.0   # > 0: keyboard moves (gradient drifts / art pans) per period
    band: float = 0.5             # mosaic: share of the cover's height mapped onto the keys
    zoom: float = 0.8             # mosaic: share of the cover's width shown (room to pan)
    strip_cycle_s: float = 0.0    # > 0: one-colour lights cycle the palette, s per colour

    def __post_init__(self) -> None:
        _check_choice("style", self.style, STYLES)
        _check_range("steps", self.steps, 1)
        _check_range("drift_period_s", self.drift_period_s, 0.0)
        _check_range("strip_cycle_s", self.strip_cycle_s, 0.0)
        if not 0.0 < self.band <= 1.0:
            raise ValueError(f"band must be within (0, 1], got {self.band!r}")
        if not 0.0 < self.zoom <= 1.0:
            raise ValueError(f"zoom must be within (0, 1], got {self.zoom!r}")

    def scene(self, media: Any) -> AlbumScene:
        return AlbumScene(media, style=self.style, steps=self.steps, angle_deg=self.angle_deg,
                          drift_period_s=self.drift_period_s, band=self.band, zoom=self.zoom,
                          strip_cycle_s=self.strip_cycle_s)


@dataclass(frozen=True)
class DirectorConfig:
    idle_off_s: float = 300.0          # idle this long -> idle behaviour; <= 0 disables
    paused_level: float = 0.35         # LED drive multiplier (linear light) while paused
    playing_level: float = 1.0
    when_idle: str = "off"             # "off" (paint black, keep control) | "release"
    when_no_media: str = "release"     # "release" | "solid" | "off"
    fallback_color: RGB = RGB(255, 150, 60)   # for when_no_media = "solid"
    album: AlbumStyle = field(default_factory=AlbumStyle)
    track_transition_s: float = 1.5    # new track / back to playing
    pause_transition_s: float = 1.0    # playing -> paused
    idle_transition_s: float = 2.0     # fade to off (or out before release) when idle
    no_media_transition_s: float = 2.0  # fade into the no-media behaviour

    def __post_init__(self) -> None:
        object.__setattr__(self, "fallback_color", RGB(*self.fallback_color).clamped())
        _check_range("paused_level", self.paused_level, 0.0, 1.0)
        _check_range("playing_level", self.playing_level, 0.0, 1.0)
        _check_choice("when_idle", self.when_idle, WHEN_IDLE)
        _check_choice("when_no_media", self.when_no_media, WHEN_NO_MEDIA)
        for name in ("track_transition_s", "pause_transition_s", "idle_transition_s",
                     "no_media_transition_s"):
            _check_range(name, getattr(self, name), 0.0)


Rule = Callable[[Mapping[str, Any], DirectorConfig], Decision | None]


def rule_idle(state: Mapping[str, Any], cfg: DirectorConfig) -> Decision | None:
    """Locked screen or idle user -> lights off (or released)."""
    presence = state.get(PRESENCE)
    if presence is None:
        return None
    locked = bool(getattr(presence, "locked", False))
    idle_s = float(getattr(presence, "idle_seconds", 0.0) or 0.0)
    if not (locked or (cfg.idle_off_s > 0 and idle_s >= cfg.idle_off_s)):
        return None
    return Decision(OffScene(), level=0.0, transition_s=cfg.idle_transition_s,
                    release=cfg.when_idle == "release")


def rule_media(state: Mapping[str, Any], cfg: DirectorConfig) -> Decision | None:
    """Playing / paused track with cover art -> album lighting.

    A track whose art is still being fetched/analysed (``art_url`` set, no palette or
    thumbnail yet) also gets an AlbumScene, which renders a warm fallback until the
    palette arrives (the key then changes and the engine crossfades). Treating that gap
    as "no media" would release the keyboard and re-claim it a moment later on every
    track change.
    """
    media = state.get(MEDIA)
    if media is None:
        return None
    if media.status == "Playing":
        level, transition = cfg.playing_level, cfg.track_transition_s
    elif media.status == "Paused":
        level, transition = cfg.paused_level, cfg.pause_transition_s
    else:
        return None
    if media.palette is None and media.thumbnail is None and not media.art_url:
        return None  # a track without any cover art: nothing to show
    return Decision(cfg.album.scene(media), level=level, transition_s=transition)


def rule_no_media(state: Mapping[str, Any], cfg: DirectorConfig) -> Decision:
    """Fallback: nothing worth showing."""
    if cfg.when_no_media == "solid":
        return Decision(SolidScene(cfg.fallback_color), level=cfg.playing_level,
                        transition_s=cfg.no_media_transition_s)
    return Decision(OffScene(), level=0.0, transition_s=cfg.no_media_transition_s,
                    release=cfg.when_no_media == "release")


DEFAULT_RULES: tuple[Rule, ...] = (rule_idle, rule_media)


class DefaultDirector(Director):
    """First matching rule wins; ``fallback`` decides when no rule does."""

    def __init__(self, cfg: DirectorConfig | None = None, *, rules: Sequence[Rule] | None = None,
                 fallback: Callable[[Mapping[str, Any], DirectorConfig], Decision] = rule_no_media
                 ) -> None:
        self.cfg = cfg if cfg is not None else DirectorConfig()
        self.rules: tuple[Rule, ...] = DEFAULT_RULES if rules is None else tuple(rules)
        self.fallback = fallback

    def decide(self, state: Mapping[str, Any]) -> Decision:
        for rule in self.rules:
            decision = rule(state, self.cfg)
            if decision is not None:
                return decision
        return self.fallback(state, self.cfg)
