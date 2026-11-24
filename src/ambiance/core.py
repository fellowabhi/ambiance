"""Core contracts shared by every part of ambiance.

Data flow (one asyncio event loop, plus one thread per hardware device):

    Source --writes--> State --snapshot--> Director --picks--> Scene --render--> Frame
                                                                                  |
          LightDevice (own thread) <--submit-- LED mapping <-- crossfade/level <--+

* Sources observe the world (media players, desktop presence, time, ...) and publish
  immutable values into the State under well-known keys.
* The Director maps a State snapshot to a Decision: which Scene to show, at what level
  (0..1 brightness multiplier) and how long the transition into it should take.
* Scenes are pure: render(ctx) -> Frame. Frames are sRGB colours keyed by zone id.
* The engine owns timing: transitions, animation ticks, LED mapping, and hands finished
  frames (LED drive values) to devices. Devices handle reconnects themselves.

Adding a new input = write a Source and teach the Director (or a rule) about its key.
Adding a new look = write a Scene. Adding hardware = implement LightDevice.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ambiance.color import RGB
from ambiance.layout import Layout

Frame = dict[int, RGB]  # zone id -> colour

# ---- well-known state keys ----------------------------------------------------------------
MEDIA = "media"        # MediaState | None
PRESENCE = "presence"  # Presence


@dataclass(frozen=True)
class MediaState:
    """What a media player is doing. Published by MprisSource."""

    player: str            # "spotify", "chromium.instance42", ...
    status: str            # "Playing" | "Paused" | "Stopped"
    track_id: str
    title: str
    artists: tuple[str, ...]
    album: str
    art_url: str
    palette: Any = None    # ambiance.palette.Palette once the art is analysed, else None
    thumbnail: Any = None  # numpy (H, W, 3) sRGB 0..1 thumbnail of the art, or None

    @property
    def playing(self) -> bool:
        return self.status == "Playing"


@dataclass(frozen=True)
class Presence:
    """Is someone at the computer? Published by PresenceSource."""

    idle_seconds: float = 0.0
    locked: bool = False


class State:
    """Observable key -> immutable value store. Single-threaded (event-loop) use only."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._version = 0
        self._cond = asyncio.Condition()

    @property
    def version(self) -> int:
        return self._version

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._values)

    def set(self, key: str, value: Any) -> bool:
        """Store a value; wakes waiters only if it actually changed (==)."""
        if key in self._values and self._values[key] == value:
            return False
        self._values[key] = value
        self._version += 1
        asyncio.get_running_loop().create_task(self._notify())
        return True

    async def _notify(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    async def wait_changed(self, since_version: int, timeout: float | None = None) -> int:
        """Return once version > since_version (or on timeout); returns current version."""
        async with self._cond:
            try:
                await asyncio.wait_for(
                    self._cond.wait_for(lambda: self._version > since_version), timeout)
            except TimeoutError:
                pass
        return self._version


class Source(ABC):
    """Long-running producer of state. run() is cancelled on shutdown."""

    name: str = "source"

    @abstractmethod
    async def run(self, state: State) -> None: ...


@dataclass(frozen=True)
class RenderContext:
    layout: Layout
    state: Mapping[str, Any]
    now: float  # monotonic seconds
    # Per-device look options (LightDevice.look), e.g. {"animate": False} for a light that
    # should hold still while others move.
    options: Mapping[str, Any] = field(default_factory=dict)


class Scene(ABC):
    name: str = "scene"
    animated: bool = False  # True -> the engine re-renders it every tick

    @abstractmethod
    def render(self, ctx: RenderContext) -> Frame: ...

    def key(self) -> Any:
        """Identity for change detection: same key => same picture, no transition."""
        return (type(self).__name__, self.name)

    def animated_for(self, options: Mapping[str, Any]) -> bool:
        """Does this scene move on a device with these look options? (default: animated)"""
        return self.animated and options.get("animate", True)


@dataclass(frozen=True)
class Decision:
    scene: Scene
    level: float = 1.0          # brightness multiplier applied after rendering
    transition_s: float = 1.2   # crossfade duration when the scene or level changes
    release: bool = False       # True: hand the hardware back to its own firmware


class Director(ABC):
    @abstractmethod
    def decide(self, state: Mapping[str, Any]) -> Decision: ...


class LightDevice(ABC):
    """A lighting output. Implementations run their own I/O thread and reconnect.

    Optional attributes the engine reads live (so they can change at runtime):
      brightness: 0..1 multiplier on LED drive values (default 1.0)
      look:       dict handed to scenes as RenderContext.options (e.g. {"animate": False})
      led_profile: LedProfile for this device's LEDs (default: the engine's)
    """

    name: str = "device"
    layout: Layout
    brightness: float = 1.0
    look: Mapping[str, Any] = MappingProxyType({})  # read-only default; set per instance

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def submit(self, frame: Frame) -> None:
        """Queue LED drive values; latest frame wins if the device is busy."""

    @abstractmethod
    def release(self) -> None:
        """Hand the LEDs back to the device firmware (until the next submit)."""

    @abstractmethod
    def stop(self, release: bool = True) -> None: ...

    @property
    @abstractmethod
    def connected(self) -> bool: ...


@dataclass
class Stats:
    frames_submitted: int = 0
    transitions: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
