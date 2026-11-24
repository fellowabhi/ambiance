"""Engine: transitions, LED mapping, duplicate suppression, release and robustness.

The bottom section covers the daemon wiring around it (ambiance.app): supervision of
crashing coroutines and a full run with fake devices/sources. Nothing here touches
hardware, the network or a real D-Bus session.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import signal
import threading
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest

from ambiance import engine as engine_mod
from ambiance.app import main_run, run_daemon, supervise
from ambiance.color import BLACK, RGB
from ambiance.config import config_from_dict
from ambiance.core import (MEDIA, Decision, Director, Frame, LightDevice, MediaState,
                           RenderContext, Scene, Source, State)
from ambiance.director import DefaultDirector, DirectorConfig
from ambiance.engine import (OFF, Engine, LedMapper, Pixel, blend_frames, ease_in_out, mix_gain,
                             make_pixel)
from ambiance.layout import TKL_ANSI, TKL_ISO, Layout
from ambiance.led import DEFAULT_PROFILE, to_led
from ambiance.palette import extract_palette
from ambiance.scenes.album import AlbumScene
from ambiance.scenes.basic import OffScene, SolidScene

RED, BLUE, GREEN = RGB(220, 30, 20), RGB(20, 40, 220), RGB(30, 200, 60)
ZONES = TKL_ANSI.zones
A_ZONE = TKL_ANSI.by_name["a"].zone
PALETTE = extract_palette(np.array([(0.9, 0.2, 0.1)] * 50 + [(0.1, 0.3, 0.9)] * 50), max_colors=2)


# ---- fakes -------------------------------------------------------------------------------------

class FakeDevice(LightDevice):
    """Records everything the engine asks of it."""

    def __init__(self, layout: Layout = TKL_ANSI, name: str = "fake") -> None:
        self.layout = layout
        self.name = name
        self.frames: list[Frame] = []
        self.events: list[str] = []
        self.releases = 0
        self.started = False
        self.stopped_with: bool | None = None

    def start(self) -> None:
        self.started = True

    def submit(self, frame: Frame) -> None:
        self.frames.append(dict(frame))
        self.events.append("submit")

    def release(self) -> None:
        self.releases += 1
        self.events.append("release")

    def stop(self, release: bool = True) -> None:
        self.stopped_with = release

    @property
    def connected(self) -> bool:
        return True


class Clock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class ScriptedDirector(Director):
    """Returns ``self.decision`` (raises it if it is an exception)."""

    def __init__(self, decision: Decision | Exception | None = None) -> None:
        self.decision = decision

    def decide(self, state: Mapping[str, Any]) -> Decision:
        if isinstance(self.decision, Exception):
            raise self.decision
        assert self.decision is not None
        return self.decision


class CountingScene(Scene):
    def __init__(self, color: RGB) -> None:
        self.color = color
        self.renders = 0

    def render(self, ctx: RenderContext) -> Frame:
        self.renders += 1
        return {k.zone: self.color for k in ctx.layout.keys}

    def key(self) -> Any:
        return ("counting", self.color)


class ClockScene(Scene):
    """Animated: the colour follows the render time."""

    animated = True

    def __init__(self) -> None:
        self.renders = 0

    def render(self, ctx: RenderContext) -> Frame:
        self.renders += 1
        v = int(ctx.now * 40) % 256
        return {k.zone: RGB(v, 255 - v, 128) for k in ctx.layout.keys}

    def key(self) -> Any:
        return ("clock",)


class ExplodingScene(Scene):
    def render(self, ctx: RenderContext) -> Frame:
        raise RuntimeError("boom")

    def key(self) -> Any:
        return ("exploding",)


class BadKeyScene(Scene):
    def render(self, ctx: RenderContext) -> Frame:
        return {}

    def key(self) -> Any:
        raise RuntimeError("no key")


class PartialScene(Scene):
    def render(self, ctx: RenderContext) -> Frame:
        return {A_ZONE: RED, 250: BLUE}  # 250: not a key of the layout


def led(color: RGB, level: float = 1.0) -> RGB:
    """What the engine must send for ``color`` shown at ``level``."""
    c = to_led(color, DEFAULT_PROFILE)
    return c if level >= 1.0 else c.scaled(level)


def uniform(color: RGB, zones: frozenset[int] = ZONES) -> Frame:
    return {z: color for z in zones}


def make(decision: Decision | None = None, *, tick_hz: float = 4.0, state: State | None = None,
         director: Director | None = None, **kwargs: Any):
    clock = Clock()
    scripted = ScriptedDirector(decision)
    dev = FakeDevice()
    eng = Engine(state or State(), director or scripted, [dev], tick_hz=tick_hz, clock=clock,
                 **kwargs)
    return eng, scripted, dev, clock


def run_at(eng: Engine, clock: Clock, times: list[float], changed: bool = False) -> None:
    for t in times:
        clock.t = t
        eng.step(state_changed=changed)


def assert_no_consecutive_duplicates(frames: list[Frame]) -> None:
    for a, b in zip(frames, frames[1:]):
        assert a != b


def put(state: State, key: str, value: Any) -> None:
    """State.set needs a running loop; engine.step() itself is synchronous."""
    async def _set() -> None:
        state.set(key, value)
        await asyncio.sleep(0)
    asyncio.run(_set())


def media(status: str = "Playing", art_url: str = "https://art/1") -> MediaState:
    return MediaState(player="spotify", status=status, track_id="t", title="Song",
                      artists=("A",), album="B", art_url=art_url, palette=PALETTE)


async def eventually(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


# ---- blending primitives -----------------------------------------------------------------------

def test_ease_and_gain_curves_hit_their_endpoints() -> None:
    assert ease_in_out(0.0) == 0.0 and ease_in_out(1.0) == 1.0 and ease_in_out(0.5) == 0.5
    assert ease_in_out(-1.0) == 0.0 and ease_in_out(2.0) == 1.0
    assert mix_gain(1.0, 0.35, 0.0) == 1.0 and mix_gain(1.0, 0.35, 1.0) == pytest.approx(0.35)
    assert mix_gain(0.0, 1.0, 0.5) == pytest.approx(0.125)  # perceptual: half-way looks half


def test_make_pixel_canonicalises_black() -> None:
    assert make_pixel(BLACK, 1.0) is OFF and make_pixel(RED, 0.0) is OFF
    assert make_pixel(RED, 3.0) == Pixel(RED, 1.0)


def test_blend_frames_mixes_per_distinct_pair() -> None:
    src = {1: Pixel(RED, 1.0), 2: Pixel(RED, 1.0), 3: OFF, 4: Pixel(GREEN, 1.0)}
    dst = {1: Pixel(BLUE, 1.0), 2: Pixel(BLUE, 1.0), 3: Pixel(GREEN, 0.5), 4: OFF}
    mid = blend_frames(src, dst, 0.5)
    assert mid[1] == mid[2] and mid[1].color not in (RED, BLUE)
    assert mid[3].color == GREEN and 0 < mid[3].gain < 0.5        # fade in: colour kept
    assert mid[4].color == GREEN and 0 < mid[4].gain < 1.0        # fade out: colour kept
    assert blend_frames(src, dst, 0.0)[1] == Pixel(RED, 1.0)
    assert blend_frames(src, dst, 1.0) == {1: dst[1], 2: dst[2], 3: dst[3], 4: OFF}


def test_led_mapper_caches_and_stays_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[RGB] = []
    monkeypatch.setattr(engine_mod, "to_led", lambda c, p: calls.append(c) or to_led(c, p))
    mapper = LedMapper(max_entries=2)
    for c in (RED, RED, BLUE, RED):
        mapper(c)
    assert calls == [RED, BLUE]
    mapper(GREEN)  # full -> cleared, then cached
    mapper(RED)
    assert calls == [RED, BLUE, GREEN, RED]


# ---- transitions -------------------------------------------------------------------------------

def test_first_decision_fades_in_from_black_without_jumping_to_the_led_floor() -> None:
    eng, _, dev, clock = make(Decision(SolidScene(RED), level=1.0, transition_s=1.0))
    eng.step()
    run_at(eng, clock, [0.25, 0.5, 0.75])
    peaks = [max(f[A_ZONE]) for f in dev.frames]
    assert peaks[0] == 0 and peaks == sorted(peaks) and len(set(peaks)) == len(peaks)
    # to_led() alone would never go below min_level (0.3 x 255) for a non-black colour
    assert peaks[1] < 0.1 * 255 and peaks[2] < 0.3 * 255
    run_at(eng, clock, [1.0])
    assert dev.frames[-1] == uniform(led(RED))


def test_crossfade_has_intermediate_frames_and_ends_exactly_at_led_target_times_level() -> None:
    eng, director, dev, clock = make(Decision(SolidScene(RED), level=0.8, transition_s=0.0))
    eng.step()
    assert dev.frames == [uniform(led(RED, 0.8))]
    director.decision = Decision(SolidScene(BLUE), level=0.8, transition_s=1.0)
    clock.t = 10.0
    eng.step()  # transition starts: picture unchanged yet, nothing sent
    assert len(dev.frames) == 1
    run_at(eng, clock, [10.25, 10.5, 10.75])
    mids = dev.frames[1:]
    assert len(mids) == 3
    for f in mids:
        assert len(set(f.values())) == 1  # one (from, to) pair -> one colour, one HID++ write
        assert f[A_ZONE] not in (led(RED, 0.8), led(BLUE, 0.8))
    hue = [f[A_ZONE].b / (f[A_ZONE].r + 1) for f in mids]
    assert hue == sorted(hue)  # moving from red towards blue
    run_at(eng, clock, [11.0])
    assert dev.frames[-1] == uniform(led(BLUE, 0.8))
    assert eng.stats.transitions == 2
    n = len(dev.frames)
    run_at(eng, clock, [12.0, 13.0])
    eng.step(state_changed=True)
    assert len(dev.frames) == n
    assert_no_consecutive_duplicates(dev.frames)


def test_new_decision_mid_transition_continues_from_what_is_shown() -> None:
    eng, director, dev, clock = make(Decision(SolidScene(RED), transition_s=0.0))
    eng.step()
    director.decision = Decision(SolidScene(BLUE), transition_s=1.0)
    clock.t = 1.0
    eng.step()
    run_at(eng, clock, [1.5])
    halfway = dev.frames[-1]
    director.decision = Decision(SolidScene(GREEN), transition_s=1.0)
    clock.t = 1.6
    eng.step()  # e = 0 of the new transition: still exactly the half-way picture
    assert dev.frames[-1] == halfway
    run_at(eng, clock, [2.6])
    assert dev.frames[-1] == uniform(led(GREEN))


def test_pause_fades_the_level_and_keeps_the_colours() -> None:
    state = State()
    put(state, MEDIA, media("Playing"))
    cfg = DirectorConfig(paused_level=0.35, pause_transition_s=1.0, track_transition_s=0.0)
    eng, _, dev, clock = make(state=state, director=DefaultDirector(cfg))
    eng.step()
    target = AlbumScene(media()).render(RenderContext(TKL_ANSI, {}, 0.0))
    assert dev.frames[-1] == {z: led(c) for z, c in target.items()}
    put(state, MEDIA, dataclasses.replace(media(), status="Paused"))
    clock.t = 5.0
    eng.step()
    run_at(eng, clock, [5.5])
    mid = dev.frames[-1]
    for z, c in target.items():
        lo, hi = led(c, 0.35), led(c)
        assert all(a <= m <= b for a, m, b in zip(lo, mid[z], hi)), (z, lo, mid[z], hi)
    run_at(eng, clock, [6.0])
    assert dev.frames[-1] == {z: led(c, 0.35) for z, c in target.items()}


def test_level_fades_reuse_the_led_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[RGB] = []
    monkeypatch.setattr(engine_mod, "to_led", lambda c, p: calls.append(c) or to_led(c, p))
    eng, director, dev, clock = make(Decision(SolidScene(RED), level=1.0, transition_s=0.0))
    eng.step()
    director.decision = Decision(SolidScene(RED), level=0.2, transition_s=1.0)
    run_at(eng, clock, [0.0, 0.3, 0.6, 1.0], changed=True)
    assert calls == [RED]
    assert dev.frames[-1] == uniform(led(RED, 0.2))


def test_frames_cover_exactly_the_layout_zones() -> None:
    eng, _, dev, _ = make(Decision(PartialScene(), transition_s=0.0))
    eng.step()
    frame = dev.frames[-1]
    assert set(frame) == ZONES
    assert frame[A_ZONE] == led(RED)
    assert all(c == BLACK for z, c in frame.items() if z != A_ZONE)


def test_each_layout_gets_its_own_frame() -> None:
    ansi, iso = FakeDevice(TKL_ANSI), FakeDevice(TKL_ISO)
    eng = Engine(State(), ScriptedDirector(Decision(SolidScene(RED), transition_s=0.0)),
                 [ansi, iso], clock=Clock())
    eng.step()
    assert ansi.frames == [uniform(led(RED), TKL_ANSI.zones)]
    assert iso.frames == [uniform(led(RED), TKL_ISO.zones)]


# ---- duplicate suppression & scheduling --------------------------------------------------------

def test_identical_frames_are_never_submitted_twice() -> None:
    eng, director, dev, clock = make(Decision(SolidScene(RED), transition_s=0.0))
    run_at(eng, clock, [0.0, 1.0, 2.0, 3.0], changed=True)
    assert len(dev.frames) == 1
    director.decision = Decision(SolidScene(RED), transition_s=3.0)  # same picture
    eng.step()
    assert len(dev.frames) == 1 and eng.stats.transitions == 1


def test_slow_transitions_skip_ticks_that_round_to_the_same_frame() -> None:
    eng, director, dev, clock = make(Decision(SolidScene(RED), transition_s=0.0))
    eng.step()
    director.decision = Decision(SolidScene(RGB(222, 30, 20)), transition_s=10.0)
    run_at(eng, clock, [i * 0.25 for i in range(41)])
    assert 1 < len(dev.frames) < 41
    assert eng.stats.duplicates_skipped > 0
    assert_no_consecutive_duplicates(dev.frames)


def test_static_scene_renders_only_when_the_state_or_decision_changes() -> None:
    scene = CountingScene(GREEN)
    eng, _, dev, clock = make(Decision(scene, transition_s=1.0), idle_timeout_s=5.0)
    eng.step()
    run_at(eng, clock, [0.25, 0.5, 0.75, 1.0, 2.0])
    assert scene.renders == 1
    assert dev.frames[-1] == uniform(led(GREEN))
    n = len(dev.frames)
    eng.step(state_changed=True)
    assert scene.renders == 2 and len(dev.frames) == n


def test_animated_scene_rerenders_every_tick() -> None:
    scene = ClockScene()
    eng, _, dev, clock = make(Decision(scene, transition_s=0.0), tick_hz=4.0)
    assert eng.step() == pytest.approx(0.25)
    for i in range(1, 5):
        clock.t = i * 0.25
        assert eng.step(state_changed=False) == pytest.approx(0.25)
    assert scene.renders == 5 and len(dev.frames) == 5
    assert_no_consecutive_duplicates(dev.frames)


def test_step_asks_to_tick_only_while_something_moves() -> None:
    eng, director, _, clock = make(Decision(SolidScene(RED), transition_s=1.0), tick_hz=4.0,
                                   idle_timeout_s=None)
    assert eng.step() == pytest.approx(0.25)
    clock.t = 0.9
    assert eng.step(False) == pytest.approx(0.1)  # lands the last frame on time
    clock.t = 1.0
    assert eng.step(False) is None  # done: sleep until the State changes


# ---- release -----------------------------------------------------------------------------------

def test_release_fades_out_releases_once_and_resumes_on_a_new_decision() -> None:
    eng, director, dev, clock = make(Decision(SolidScene(RED), transition_s=0.0))
    eng.step()
    director.decision = Decision(OffScene(), level=0.0, transition_s=1.0, release=True)
    clock.t = 1.0
    eng.step()
    run_at(eng, clock, [1.25, 1.5, 1.75])
    assert dev.releases == 0
    fade = [max(f[A_ZONE]) for f in dev.frames[1:]]
    assert len(fade) == 3 and fade == sorted(fade, reverse=True) and fade[0] < max(led(RED))
    run_at(eng, clock, [2.0])
    assert dev.releases == 1 and eng.released and eng.stats.released
    assert dev.events[-1] == "release" and len(dev.frames) == 4  # no final black frame
    run_at(eng, clock, [3.0, 4.0, 5.0], changed=True)
    assert dev.releases == 1 and len(dev.frames) == 4  # once; nothing sent while released
    director.decision = Decision(SolidScene(RED), transition_s=0.0)
    clock.t = 6.0
    eng.step()
    assert dev.frames[-1] == uniform(led(RED))  # same as before the release, sent anyway
    assert dev.releases == 1 and not eng.released


def test_release_with_nothing_shown_is_immediate() -> None:
    eng, _, dev, clock = make(Decision(OffScene(), level=0.0, transition_s=2.0, release=True))
    eng.step()  # also hands back a keyboard left claimed by a crashed previous run
    assert dev.releases == 1 and dev.frames == []
    run_at(eng, clock, [1.0, 2.0, 3.0], changed=True)
    assert dev.releases == 1


def test_a_new_decision_during_the_fade_out_cancels_the_release() -> None:
    eng, director, dev, clock = make(Decision(SolidScene(RED), transition_s=0.0))
    eng.step()
    director.decision = Decision(OffScene(), level=0.0, transition_s=2.0, release=True)
    clock.t = 1.0
    eng.step()
    run_at(eng, clock, [1.5])
    director.decision = Decision(SolidScene(BLUE), transition_s=1.0)
    clock.t = 1.6
    eng.step()
    run_at(eng, clock, [2.0, 2.6, 4.0])
    assert dev.releases == 0
    assert dev.frames[-1] == uniform(led(BLUE))


def test_idle_off_paints_black_and_keeps_control() -> None:
    eng, director, dev, clock = make(Decision(SolidScene(RED), transition_s=0.0))
    eng.step()
    director.decision = Decision(OffScene(), level=0.0, transition_s=2.0)
    clock.t = 1.0
    eng.step()
    run_at(eng, clock, [2.0, 3.0])
    assert dev.frames[-1] == uniform(BLACK) and dev.releases == 0


# ---- robustness --------------------------------------------------------------------------------

def test_scene_exception_keeps_the_last_frame_and_the_engine_alive(
        caplog: pytest.LogCaptureFixture) -> None:
    eng, director, dev, clock = make(Decision(SolidScene(RED), transition_s=0.0))
    eng.step()
    director.decision = Decision(ExplodingScene(), transition_s=0.5)
    with caplog.at_level(logging.DEBUG, logger="ambiance.engine"):
        run_at(eng, clock, [1.0, 1.25, 1.5, 2.0], changed=True)
    assert dev.frames == [uniform(led(RED))]
    assert eng.stats.errors == 4
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1 and "boom" in errors[0].getMessage()  # not once per tick
    director.decision = Decision(SolidScene(BLUE), transition_s=0.0)
    clock.t = 3.0
    eng.step()
    assert dev.frames[-1] == uniform(led(BLUE))


def test_scene_key_exception_is_contained() -> None:
    eng, director, dev, _ = make(Decision(SolidScene(RED), transition_s=0.0))
    eng.step()
    director.decision = Decision(BadKeyScene())
    eng.step()
    assert dev.frames == [uniform(led(RED))] and eng.stats.errors == 1


def test_director_exception_keeps_the_current_plan_running() -> None:
    eng, director, dev, clock = make(Decision(SolidScene(RED), transition_s=1.0))
    eng.step()
    run_at(eng, clock, [0.5])
    director.decision = RuntimeError("director bug")
    run_at(eng, clock, [0.75, 1.0], changed=True)
    assert dev.frames[-1] == uniform(led(RED))  # the fade still finished
    assert eng.stats.errors == 2


def test_director_failing_from_the_start_shows_nothing() -> None:
    eng, director, dev, _ = make(RuntimeError("nope"))  # type: ignore[arg-type]
    assert eng.step() == 5.0
    assert dev.frames == [] and dev.releases == 0


def test_device_errors_do_not_kill_the_engine() -> None:
    class FlakyDevice(FakeDevice):
        def submit(self, frame: Frame) -> None:
            if not self.events:
                self.events.append("fail")
                raise OSError("device thread died")
            super().submit(frame)

    dev = FlakyDevice()
    eng = Engine(State(), ScriptedDirector(Decision(SolidScene(RED), transition_s=0.0)), [dev],
                 clock=Clock())
    eng.step()
    eng.step()  # the failed frame is retried at the next opportunity
    assert dev.frames == [uniform(led(RED))] and eng.stats.errors == 1


# ---- the asyncio loop --------------------------------------------------------------------------

def test_run_follows_state_changes_and_sleeps_when_idle() -> None:
    class ColourDirector(Director):
        def decide(self, state: Mapping[str, Any]) -> Decision:
            color = state.get("color")
            if color is None:
                return Decision(OffScene(), level=0.0, release=True)
            return Decision(SolidScene(color), transition_s=0.05)

    async def main() -> None:
        state = State()
        dev = FakeDevice()
        eng = Engine(state, ColourDirector(), [dev], tick_hz=50.0, idle_timeout_s=None)
        steps = 0
        real_step = eng.step

        def counting_step(state_changed: bool = True) -> float | None:
            nonlocal steps
            steps += 1
            return real_step(state_changed)

        eng.step = counting_step  # type: ignore[method-assign]
        task = asyncio.create_task(eng.run())
        try:
            await eventually(lambda: dev.releases == 1)
            state.set("color", RED)
            await eventually(lambda: bool(dev.frames) and dev.frames[-1] == uniform(led(RED)))
            state.set("color", BLUE)
            await eventually(lambda: dev.frames[-1] == uniform(led(BLUE)))
            settled = steps
            await asyncio.sleep(0.3)
            assert steps - settled <= 1  # idle: no ticking, ~0 CPU
            assert_no_consecutive_duplicates(dev.frames)
        finally:
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())


# ---- daemon wiring (ambiance.app) --------------------------------------------------------------

def test_supervise_restarts_a_crashing_coroutine_with_backoff() -> None:
    async def main() -> None:
        delays: list[float] = []
        runs = 0

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        async def flaky() -> None:
            nonlocal runs
            runs += 1
            if runs == 3:
                return  # returning is as unexpected as crashing
            if runs < 5:
                raise RuntimeError(f"crash {runs}")
            await asyncio.Event().wait()

        task = asyncio.create_task(supervise("flaky", flaky, min_backoff=1.0, max_backoff=3.0,
                                             sleep=fake_sleep, clock=lambda: 0.0))
        await eventually(lambda: runs == 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert delays == [1.0, 2.0, 3.0, 3.0]

    asyncio.run(main())


def test_supervise_resets_the_backoff_after_a_healthy_run() -> None:
    async def main() -> None:
        delays: list[float] = []
        times = iter([0.0, 0.0, 0.0, 0.0, 10.0, 200.0, 200.0])
        runs = 0

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        async def flaky() -> None:
            nonlocal runs
            runs += 1
            if runs <= 3:
                raise RuntimeError("crash")
            await asyncio.Event().wait()

        task = asyncio.create_task(supervise("flaky", flaky, min_backoff=1.0, healthy_after=60.0,
                                             sleep=fake_sleep, clock=lambda: next(times)))
        await eventually(lambda: runs == 4)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert delays == [1.0, 2.0, 1.0]

    asyncio.run(main())


class MediaSource(Source):
    name = "fake-media"

    async def run(self, state: State) -> None:
        state.set(MEDIA, media())
        await asyncio.Event().wait()


class CrashingSource(Source):
    name = "crashy"

    def __init__(self) -> None:
        self.runs = 0

    async def run(self, state: State) -> None:
        self.runs += 1
        raise RuntimeError("source bug")


def test_run_daemon_wires_everything_and_shuts_down_cleanly() -> None:
    async def main() -> None:
        cfg = config_from_dict({"director": {"track_transition_s": 0.0}})
        dev, crashy, stop = FakeDevice(), CrashingSource(), asyncio.Event()
        daemon = asyncio.create_task(run_daemon(cfg, devices=[dev], sources=[MediaSource(), crashy],
                                                stop=stop, handle_signals=False))
        await eventually(lambda: len(dev.frames) >= 1)
        assert dev.started and crashy.runs == 1  # crashed, logged, waiting to restart
        expected = AlbumScene(media()).render(RenderContext(TKL_ANSI, {}, 0.0))
        assert dev.frames[-1] == {z: led(c) for z, c in expected.items()}
        stop.set()
        await asyncio.wait_for(daemon, 5.0)
        assert dev.stopped_with is True  # release_on_exit default

    asyncio.run(main())


@pytest.mark.skipif(threading.current_thread() is not threading.main_thread(),
                    reason="signal handlers need the main thread")
def test_sigterm_stops_the_daemon() -> None:
    async def main() -> None:
        cfg = config_from_dict({"device": {"release_on_exit": False}})
        dev = FakeDevice()
        daemon = asyncio.create_task(run_daemon(cfg, devices=[dev], sources=[]))
        await eventually(lambda: dev.started)  # handlers are installed before start()
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(daemon, 5.0)
        assert dev.stopped_with is False

    asyncio.run(main())


def test_main_run_rejects_a_bad_config(tmp_path: Any) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[director]\nnope = 1\n")
    assert main_run(str(bad), verbose=False) == 2
