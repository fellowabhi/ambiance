"""The engine: turns director decisions into LED frames over time.

One iteration (``Engine.step``) runs on every State change, and every tick while
something moves (a transition, an animated scene, a fade-out before a release)::

    snapshot -> director.decide -> new scene key / level / release? -> start a transition
             -> scene.render (only when needed) -> blend -> LED mapping -> device.submit

Between changes the engine sleeps in ``State.wait_changed`` so an idle daemon costs ~0 CPU.

Colour model
------------
Each composited zone is a :class:`Pixel` ``(color, gain)``:

* ``color`` is the scene's sRGB colour, *before* LED mapping. Crossfading two scenes mixes
  colours in linear light (srgb_to_linear / linear_to_srgb), like optically dissolving two
  lights; black zones have no colour, so a fade from/to black only changes the gain.
* ``gain`` is the decision's level (0 for black). It multiplies the LED drive values that
  ``led.to_led`` returns, and those are linear in emitted light (PWM duty).

Why not scale the sRGB colour by the level before ``to_led``? ``to_led`` renormalises every
colour to its hue's peak and derives brightness from OKLab lightness with a floor at
``min_level`` (so dark album colours still read as colour). Fed dimmed colours it barely
dims (level 0.35 -> 65-80 % drive) and a fade to black would stay above ~30 % drive
until the last frame, then pop to off. Gains are therefore applied in drive space, and
interpolated in cube-root (≈ perceptual lightness) space so fades look even instead of
lingering bright and collapsing at the end.

Hardware economics
------------------
A per-key frame costs ~0.4-0.8 s of HID++ writes over Bluetooth and devices group writes
by colour, so the engine never submits an identical frame twice, maps each distinct
colour once (``to_led`` costs ~0.4 ms), and blends per distinct (from, to) pair so an
intermediate frame has no more distinct colours than the transition needs. Devices keep
only the latest submitted frame, so ticking faster than they can paint is harmless.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from ambiance.color import BLACK, RGB
from ambiance.colorspace import linear_to_srgb, srgb_to_linear
from ambiance.core import (Decision, Director, Frame, LightDevice, RenderContext, Scene, State,
                           Stats)
from ambiance.layout import Layout
from ambiance.led import DEFAULT_PROFILE, LedProfile, to_led

log = logging.getLogger(__name__)


class Pixel(NamedTuple):
    """One composited zone: pre-LED sRGB colour and a brightness gain (0..1)."""

    color: RGB
    gain: float


OFF = Pixel(BLACK, 0.0)


def make_pixel(color: RGB, gain: float) -> Pixel:
    """Canonical pixel: anything black or at gain 0 is OFF (so it compares equal)."""
    if gain <= 0.0 or color == BLACK:
        return OFF
    return Pixel(color if type(color) is RGB else RGB(*color), min(1.0, float(gain)))


def ease_in_out(p: float) -> float:
    """Smoothstep: zero slope at both ends, so transitions start and land softly."""
    p = min(1.0, max(0.0, p))
    return p * p * (3.0 - 2.0 * p)


def mix_gain(a: float, b: float, e: float) -> float:
    """Interpolate a brightness gain in cube-root (≈ OKLab lightness) space."""
    if a == b or e <= 0.0:
        return a
    if e >= 1.0:
        return b
    ca, cb = math.cbrt(a), math.cbrt(b)
    return (ca + (cb - ca) * e) ** 3


_SRGB8_TO_LINEAR = srgb_to_linear(np.arange(256) / 255.0)  # LUT: 8-bit sRGB -> linear


def mix_colors(pairs: Sequence[tuple[RGB, RGB]], e: float) -> list[RGB]:
    """Mix colour pairs in linear light (vectorised); e = 0 -> first, 1 -> second."""
    if not pairs:
        return []
    a = _SRGB8_TO_LINEAR[np.array([p[0] for p in pairs], dtype=int)]
    b = _SRGB8_TO_LINEAR[np.array([p[1] for p in pairs], dtype=int)]
    mixed = np.rint(linear_to_srgb(a + (b - a) * e) * 255.0).astype(int)
    return [RGB(*row) for row in mixed.tolist()]


def blend_frames(src: Mapping[int, Pixel], dst: Mapping[int, Pixel], e: float) -> dict[int, Pixel]:
    """Per-zone blend from ``src`` to ``dst`` (zones of ``dst``; missing src zones are OFF).

    Zones sharing the same (from, to) pair blend to the same pixel, so the intermediate
    frame has at most as many distinct colours as there are distinct pairs.
    """
    groups: dict[tuple[Pixel, Pixel], list[int]] = {}
    for zone, b in dst.items():
        groups.setdefault((src.get(zone, OFF), b), []).append(zone)
    pairs = list(groups)
    colors: list[RGB | None] = []
    to_mix: list[int] = []
    for i, (a, b) in enumerate(pairs):
        if a.gain <= 0.0:
            colors.append(b.color)
        elif b.gain <= 0.0 or a.color == b.color:
            colors.append(a.color)
        else:
            colors.append(None)
            to_mix.append(i)
    for i, mixed in zip(to_mix, mix_colors([(pairs[i][0].color, pairs[i][1].color)
                                            for i in to_mix], e)):
        colors[i] = mixed
    out: dict[int, Pixel] = {}
    for (pair, zones), color in zip(groups.items(), colors):
        assert color is not None
        px = make_pixel(color, mix_gain(pair[0].gain, pair[1].gain, e))
        for zone in zones:
            out[zone] = px
    return out


class LedMapper:
    """``led.to_led`` with a bounded per-colour cache (to_led gamut-clips in Python, ~0.4 ms)."""

    def __init__(self, profile: LedProfile = DEFAULT_PROFILE, max_entries: int = 4096) -> None:
        self.profile = profile
        self._max = max_entries
        self._cache: dict[RGB, RGB] = {}

    def __call__(self, color: RGB) -> RGB:
        led = self._cache.get(color)
        if led is None:
            if len(self._cache) >= self._max:
                self._cache.clear()  # crossfades mint new colours every tick; stay bounded
            led = self._cache[color] = to_led(color, self.profile)
        return led

    def pixel(self, px: Pixel) -> RGB:
        """LED drive values for a pixel: map the colour, then scale (linear PWM) by gain."""
        if px.gain <= 0.0:
            return BLACK
        led = self(px.color)
        return led if px.gain >= 1.0 else led.scaled(px.gain)

    def frame(self, composite: Mapping[int, Pixel]) -> Frame:
        per_pixel: dict[Pixel, RGB] = {}
        out: Frame = {}
        for zone, px in composite.items():
            led = per_pixel.get(px)
            if led is None:
                led = per_pixel[px] = self.pixel(px)
            out[zone] = led
        return out


@dataclass
class EngineStats(Stats):
    """Counters for logging/diagnostics (``frames_submitted`` counts per layout group)."""

    renders: int = 0
    duplicates_skipped: int = 0
    releases: int = 0
    errors: int = 0
    scene: str = ""
    level: float = 0.0
    released: bool = False


@dataclass(eq=False)
class _Plan:
    """The decision currently being shown."""

    ident: tuple[Any, ...]   # (release?, scene key, level): a change starts a transition
    scene: Scene
    level: float
    release: bool


@dataclass(eq=False)
class _Output:
    """Devices sharing one layout, and what they currently show."""

    layout: Layout
    devices: list[LightDevice]
    off: dict[int, Pixel]                      # every zone OFF
    shown: dict[int, Pixel]                    # composite displayed now (pre-LED)
    fade_from: dict[int, Pixel] | None = None  # `shown` when the running transition began
    target: dict[int, Pixel] | None = None     # latest render of the plan, at its level
    last_led: Frame | None = None              # last submitted LED drive values
    led: LedMapper | None = None               # per-device colour mapping (None: engine's)


class _ErrorGate:
    """Log a failure once with its traceback; identical repeats go to debug (4 Hz ticks!)."""

    def __init__(self, what: str, stats: EngineStats) -> None:
        self.what = what
        self._stats = stats
        self._last: str | None = None

    def failed(self, exc: BaseException) -> None:
        self._stats.errors += 1
        signature = f"{type(exc).__name__}: {exc}"
        if signature != self._last:
            self._last = signature
            log.error("%s failed (%s); keeping the last frame", self.what, signature,
                      exc_info=exc)
        else:
            log.debug("%s failed again: %s", self.what, signature)

    def ok(self) -> None:
        if self._last is not None:
            log.info("%s recovered", self.what)
            self._last = None


def _look(out: _Output) -> Mapping[str, Any]:
    return getattr(out.devices[0], "look", None) or {}


def _brightness(out: _Output) -> float:
    try:
        return min(1.0, max(0.0, float(getattr(out.devices[0], "brightness", 1.0))))
    except (TypeError, ValueError):
        return 1.0


class Engine:
    """Drives ``devices`` from ``director`` decisions about ``state``."""

    def __init__(self, state: State, director: Director, devices: Sequence[LightDevice], *,
                 led_profile: LedProfile = DEFAULT_PROFILE, tick_hz: float = 4.0,
                 clock: Callable[[], float] = time.monotonic,
                 idle_timeout_s: float | None = 5.0) -> None:
        """
        tick_hz: frame rate while transitioning/animating. Devices paint at ~1-2 frames/s
            over Bluetooth and drop superseded frames, so a few Hz is plenty.
        idle_timeout_s: when nothing moves, re-ask the director at least this often even
            without State changes (cheap; lets custom time-based rules work). None = only
            on State changes.
        """
        if not tick_hz > 0:
            raise ValueError(f"tick_hz must be > 0, got {tick_hz!r}")
        self._state = state
        self._director = director
        self._clock = clock
        self._tick_s = 1.0 / tick_hz
        self._idle_timeout_s = idle_timeout_s
        self.led = LedMapper(led_profile)
        self._outputs = self._group(devices)
        self._plan: _Plan | None = None
        self._fade_t0 = 0.0
        self._fade_s = 0.0
        self._released = False
        self._stats = EngineStats()
        self._director_errors = _ErrorGate("director", self._stats)
        self._scene_errors = _ErrorGate("scene", self._stats)
        self._device_errors = _ErrorGate("device", self._stats)
        self._engine_errors = _ErrorGate("engine step", self._stats)

    def _group(self, devices: Sequence[LightDevice]) -> list[_Output]:
        """One output per (layout, LED profile): a device may bring its own `led_profile`
        (different LEDs need different white balance), else the engine's applies."""
        outputs: dict[tuple[int, int | None], _Output] = {}
        for dev in devices:
            profile = getattr(dev, "led_profile", None)
            key = (id(dev.layout), id(profile) if profile is not None else None)
            out = outputs.get(key)
            if out is None:
                off = {zone: OFF for zone in sorted(dev.layout.zones)}
                mapper = LedMapper(profile) if profile is not None else None
                out = outputs[key] = _Output(dev.layout, [], off, off, led=mapper)
            out.devices.append(dev)
        return list(outputs.values())

    # -- public ------------------------------------------------------------------------------
    @property
    def stats(self) -> EngineStats:
        return self._stats

    @property
    def released(self) -> bool:
        return self._released

    def shown(self, layout: Layout | None = None) -> dict[int, Pixel]:
        """The composite currently displayed (pre-LED) for a layout (default: the first)."""
        for out in self._outputs:
            if layout is None or out.layout is layout:
                return dict(out.shown)
        raise KeyError(layout.name if layout else "no outputs")

    async def run(self) -> None:
        """Run until cancelled. Never raises because of a director, scene or device bug."""
        log.info("engine running: %d device(s), tick %.1f Hz",
                 sum(len(o.devices) for o in self._outputs), 1.0 / self._tick_s)
        version = self._state.version
        timeout = self._safe_step(True)
        while True:
            new_version = await self._state.wait_changed(version, timeout)
            changed = new_version != version
            version = self._state.version
            timeout = self._safe_step(changed)

    def step(self, state_changed: bool = True) -> float | None:
        """One iteration at ``clock()``. Returns how long to wait before the next one
        (None = until the State changes). Public so tests can drive time explicitly."""
        now = self._clock()
        snapshot = self._state.snapshot()
        replanned = False
        try:
            decision = self._director.decide(snapshot)
            if not isinstance(decision, Decision):
                raise TypeError(f"decide() returned {decision!r}, not a Decision")
        except Exception as exc:  # keep showing the current plan
            self._director_errors.failed(exc)
        else:
            self._director_errors.ok()
            replanned = self._adopt(decision, now)
        plan = self._plan
        if plan is None or self._released:
            return self._idle_timeout_s
        if plan.release:
            self._fade_out_then_release(now)
        else:
            for out in self._outputs:
                rerender = (state_changed or replanned
                            or plan.scene.animated_for(_look(out)))
                self._update(out, plan, snapshot, now, rerender)
        return self._next_timeout(now)

    # -- planning ----------------------------------------------------------------------------
    def _adopt(self, decision: Decision, now: float) -> bool:
        """Make ``decision`` the plan; returns True if that starts a transition."""
        level = min(1.0, max(0.0, float(decision.level)))
        if decision.release:
            ident: tuple[Any, ...] = ("release",)
        else:
            try:
                ident = ("show", decision.scene.key(), level)
            except Exception as exc:
                self._scene_errors.failed(exc)
                return False
        old = self._plan
        if old is not None and old.ident == ident:
            return False  # same picture: keep the current scene instance and its caches
        scene = decision.scene
        if (old is not None and not decision.release and not old.release
                and old.ident[1] == ident[1]):
            scene = old.scene  # only the level changed
        self._plan = _Plan(ident, scene, level, decision.release)
        self._stats.transitions += 1
        self._stats.scene = "release" if decision.release else repr(scene)
        self._stats.level = level
        log.info("transition -> %s at level %.2f over %.1f s", self._stats.scene, level,
                 decision.transition_s)
        if not decision.release and self._released:
            # The firmware owns the LEDs; the next submit re-claims them. Fade in from black
            # and make sure that first frame is sent even if it equals the pre-release one.
            self._released = self._stats.released = False
            for out in self._outputs:
                out.shown, out.last_led = out.off, None
        self._fade_t0 = now
        self._fade_s = max(0.0, float(decision.transition_s))
        for out in self._outputs:
            out.fade_from = out.shown
            out.target = None if not decision.release else out.off
        return True

    def _progress(self, now: float) -> float:
        if self._fade_s <= 0.0:
            return 1.0
        return ease_in_out((now - self._fade_t0) / self._fade_s)

    def _fading(self, now: float) -> bool:
        return now - self._fade_t0 < self._fade_s

    def _next_timeout(self, now: float) -> float | None:
        """Tick while something moves; land the last transition frame on time."""
        plan = self._plan
        if plan is None or self._released:
            return self._idle_timeout_s
        if not plan.release and any(plan.scene.animated_for(_look(o)) for o in self._outputs):
            return self._tick_s
        if self._fading(now) and (plan.release
                                  or any(o.fade_from is not None for o in self._outputs)):
            return min(self._tick_s, self._fade_t0 + self._fade_s - now)
        return self._idle_timeout_s

    # -- rendering ---------------------------------------------------------------------------
    def _render(self, scene: Scene, out: _Output, snapshot: Mapping[str, Any],
                level: float, now: float) -> dict[int, Pixel] | None:
        try:
            frame = scene.render(RenderContext(out.layout, snapshot, now, _look(out)))
            # Zones the scene left out go black; zones the layout doesn't have are dropped.
            target = {zone: make_pixel(frame.get(zone, BLACK), level) for zone in out.off}
        except Exception as exc:
            self._scene_errors.failed(exc)
            return None
        self._scene_errors.ok()
        self._stats.renders += 1
        return target

    def _update(self, out: _Output, plan: _Plan, snapshot: Mapping[str, Any], now: float,
                rerender: bool) -> None:
        new_target = False
        if out.target is None or rerender:
            target = self._render(plan.scene, out, snapshot, plan.level, now)
            if target is not None:
                new_target = target != out.target
                out.target = target
        if out.target is None:
            return  # the scene never rendered: keep showing the last frame
        if not (new_target or out.fade_from is not None or out.last_led is None):
            return  # nothing that could change the picture happened
        self._compose(out, out.target, now)

    def _compose(self, out: _Output, target: dict[int, Pixel], now: float) -> None:
        e = self._progress(now)
        if out.fade_from is not None and e < 1.0:
            composite = blend_frames(out.fade_from, target, e)
        else:
            out.fade_from = None
            composite = target
        out.shown = composite
        led = (out.led or self.led).frame(composite)
        brightness = _brightness(out)
        if brightness < 1.0:
            led = {zone: c.scaled(brightness) for zone, c in led.items()}
        self._submit(out, led)

    def _submit(self, out: _Output, led: Frame) -> None:
        if led == out.last_led:
            self._stats.duplicates_skipped += 1
            return
        out.last_led = led
        self._stats.frames_submitted += 1
        for dev in out.devices:
            try:
                dev.submit(dict(led))  # own copy: the device thread keeps it
            except Exception as exc:
                self._device_errors.failed(exc)
                out.last_led = None  # retry at the next opportunity
            else:
                self._device_errors.ok()

    def refresh(self) -> None:
        """Re-render and re-submit everything on the next step (after a device's brightness
        or look changed at runtime). Wake the engine by changing the State afterwards."""
        for out in self._outputs:
            out.target = None
            out.last_led = None

    # -- release -----------------------------------------------------------------------------
    def _fade_out_then_release(self, now: float) -> None:
        """Fade whatever is shown to black, then hand every device back to its firmware.

        The final black frame isn't sent: release() replaces it immediately anyway, and
        every frame costs HID++ writes. With nothing visible we release straight away
        (also at startup, so a keyboard left claimed by a crashed run is handed back).
        """
        visible = any(px.gain > 0.0 for out in self._outputs for px in out.shown.values())
        if visible and self._progress(now) < 1.0:
            for out in self._outputs:
                if out.fade_from is None:
                    out.fade_from = out.shown
                self._compose(out, out.off, now)
            return
        for out in self._outputs:
            for dev in out.devices:
                try:
                    dev.release()
                except Exception as exc:
                    self._device_errors.failed(exc)
            out.shown, out.fade_from, out.target, out.last_led = out.off, None, None, None
        self._released = self._stats.released = True
        self._stats.releases += 1
        log.info("lighting released to the device firmware")

    def _safe_step(self, state_changed: bool) -> float | None:
        try:
            timeout = self.step(state_changed)
        except Exception as exc:  # a bug in the engine itself must not kill the daemon
            self._engine_errors.failed(exc)
            return self._tick_s
        self._engine_errors.ok()
        return timeout
