"""Logitech devices with a few whole-zone lights: COLOR_LED_EFFECTS (0x8070), e.g. G502 HERO.

Verified on a G502 SE HERO (046d:c08b, USB):

    0x8070 fn0                  -> [zone count, nv caps...]          2 zones
    0x8070 fn1 [zone]           -> [zone, location(2), effect count]  1 = primary (DPI), 2 = logo
    0x8070 fn2 [zone, effect]   -> [zone, effect, effect id(2), ...]  0 off, 1 static, 0x0a, 0x03
    0x8070 fn14 [zone]          -> [zone, effect index, 10 params]    current effect
    0x8070 fn3 [zone, effect index, 10 params, persist]               set effect
            static params: R G B 02 0 0 0 0 0 0;  persist 00 = RAM only (01 writes EEPROM)

fn3 works in onboard-profile mode: no host mode, so the mouse's DPI/buttons profile is never
touched. The G logo is the visible light; the DPI LEDs only flash in the primary colour
when DPI buttons are pressed. Loading a profile (the button behind the wheel) re-applies
the profile's LED effects without any HID++ notification, so we re-assert periodically.
On release, the effects found at connect are restored (RAM only).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

from ambiance.color import BLACK, RGB
from ambiance.core import Frame, LightDevice
from ambiance.devices.hidraw import enumerate_hidraw, release_claim, try_claim
from ambiance.layout import Key, Layout
from ambiance.led import LedProfile

from . import features as F
from .hidpp import DeviceGone, HidppDevice, HidppError, HidppTimeout

log = logging.getLogger(__name__)

LOCATION_PRIMARY, LOCATION_LOGO = 1, 2
CLAIM_OWNER = "logitech-zones"  # ambiance.devices.hidraw node ownership
EFFECT_STATIC = 0x0001
# Layout zone ids are the 0x8070 zone *locations*; the logo comes first so it gets the
# palette's top colour.
ZONES_LAYOUT = Layout("zones", (
    Key("logo", "Logo", LOCATION_LOGO, 0.0, 0.0, 1.0, 1.0),
    Key("primary", "DPI", LOCATION_PRIMARY, 0.0, 1.0, 1.0, 1.0),
))


@dataclass
class _Zone:
    index: int
    location: int
    static_index: int
    original: bytes  # [effect index, 10 params] found at connect, restored on release


class _Session:
    def __init__(self, dev: HidppDevice, idx: int, zones: list[_Zone], name: str) -> None:
        self.dev, self.idx, self.zones, self.name = dev, idx, zones, name
        self.by_location = {z.location: z for z in zones}

    def set_static(self, zone: _Zone, c: RGB) -> None:
        self.dev.request(self.idx, 3, bytes((zone.index, zone.static_index, *c, 0x02,
                                              0, 0, 0, 0, 0, 0, 0x00)))

    def restore(self) -> None:
        for z in self.zones:
            self.dev.request(self.idx, 3, bytes((z.index,)) + z.original + b"\x00")


class _NotZoneDevice(Exception):
    """The node answered, and it has no zone lighting (or is a per-key keyboard)."""


def _open_session(path: str) -> _Session | None:
    dev = HidppDevice(path)
    try:
        dev.ping()
        idx = dev.feature_index(F.COLOR_LED_EFFECTS)
        if idx is None or dev.feature_index(F.PER_KEY_LIGHTING_V2) is not None:
            dev.close()
            raise _NotZoneDevice(path)  # per-key keyboards are LogitechKeyboardDevice's
        zones = []
        for z in range(dev.request(idx, 0)[0]):
            info = dev.request(idx, 1, bytes((z,)))
            location, count = info[1] << 8 | info[2], info[3]
            static = None
            for e in range(count):
                r = dev.request(idx, 2, bytes((z, e)))
                if (r[2] << 8 | r[3]) == EFFECT_STATIC:
                    static = e
                    break
            if static is None:
                continue
            current = dev.request(idx, 0xE, bytes((z,)))
            zones.append(_Zone(z, location, static, bytes(current[1:12])))
        if not zones:
            dev.close()
            raise _NotZoneDevice(path)
        return _Session(dev, idx, zones, dev.name())
    except (OSError, HidppError, HidppTimeout):
        dev.close()
        return None


# Nodes that answered but have no zone lighting (the keyboard, receivers), while they exist.
# Without this, a missing mouse would mean HID++ probes to the BT keyboard every rescan.
_not_zone: set[tuple[str, int, str]] = set()


def _node_key(info) -> tuple[str, int, str]:
    return info.path, info.product_id, info.uniq


def find_zone_device(product_ids: Sequence[int] | None = None) -> _Session | None:
    """First Logitech node with zone lighting. Nodes another driver holds are skipped, and
    the claim is kept for the returned session (release it with close_session)."""
    candidates = [i for i in enumerate_hidraw(0x046D) if i.has_hidpp_long()
                  and not (product_ids and i.product_id not in product_ids)]
    _not_zone.intersection_update({_node_key(i) for i in candidates})  # forget vanished nodes
    for info in candidates:
        if _node_key(info) in _not_zone or not try_claim(info.path, CLAIM_OWNER):
            continue
        try:
            session = _open_session(info.path)
        except _NotZoneDevice:
            _not_zone.add(_node_key(info))  # answered "no": don't poke it every rescan
            session = None
        except OSError:
            session = None
        if session is not None:
            return session
        release_claim(info.path, CLAIM_OWNER)
    return None


def close_session(session: _Session) -> None:
    try:
        session.dev.close()
    except OSError:
        pass
    finally:
        release_claim(session.dev.path, CLAIM_OWNER)


class LogitechZoneDevice(LightDevice):
    def __init__(self, *, product_ids: Sequence[int] | None = None, rescan_interval: float = 3.0,
                 reassert_s: float = 4.0, max_rate_hz: float = 30.0, smoothing_s: float = 0.25,
                 brightness: float = 1.0, animate: bool = False, name: str = "mouse",
                 led_profile: LedProfile | None = None) -> None:
        self.name = name
        self.kind = "mouse"
        self.layout = ZONES_LAYOUT
        self.led_profile = led_profile
        self.brightness = brightness
        self.look = {"animate": animate}
        self.product_ids = tuple(product_ids or ())
        self.rescan_interval = rescan_interval
        self.reassert_s = reassert_s
        self.period = 1.0 / max(1.0, max_rate_hz)
        self.smoothing_s = max(0.0, smoothing_s)
        self._cond = threading.Condition()
        self._target: Frame | None = None
        self._released = False
        self._stopping = False
        self._release_on_stop = True
        self._thread: threading.Thread | None = None
        self._connected = False
        self.device_name = ""
        self.writes = 0
        self.last_error = ""

    # -- LightDevice ---------------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name=f"ambiance-{self.name}", daemon=True)
        self._thread.start()

    def submit(self, frame: Frame) -> None:
        with self._cond:
            self._target = {z: RGB(*c) for z, c in frame.items()}
            self._released = False
            self._cond.notify_all()

    def release(self) -> None:
        with self._cond:
            self._released = True
            self._cond.notify_all()

    def stop(self, release: bool = True) -> None:
        with self._cond:
            self._stopping, self._release_on_stop = True, release
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=5)

    def rescan(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def status(self) -> dict:
        return {"connected": self._connected, "device": self.device_name, "writes": self.writes,
                "last_error": self.last_error}

    # -- worker ---------------------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stopping:
            session = find_zone_device(self.product_ids)
            if session is None:
                with self._cond:
                    self._cond.wait(self.rescan_interval)
                continue
            self.device_name = session.name
            self._connected = True
            log.info("%s: connected to %s (%s, %d zones)", self.name, session.name,
                     session.dev.path, len(session.zones))
            try:
                self._drive(session)
            except (DeviceGone, HidppTimeout, OSError, HidppError) as e:
                self.last_error = f"{type(e).__name__}: {e}"
                log.info("%s: lost (%s); rescanning", self.name, self.last_error)
            finally:
                self._connected = False
                close_session(session)

    def _drive(self, s: _Session) -> None:
        shown: dict[int, RGB] = {}
        owns_lights = False
        last_write = 0.0
        while True:
            with self._cond:
                if self._stopping:
                    if owns_lights and self._release_on_stop:
                        s.restore()
                    return
                target, released = self._target, self._released
            if released or target is None:
                if owns_lights:
                    s.restore()
                    owns_lights, shown = False, {}
                with self._cond:
                    if self._released == released and self._target is target and not self._stopping:
                        self._cond.wait(1.0)
                continue
            wanted = {loc: target.get(loc, BLACK) for loc in s.by_location}
            if wanted == shown:
                with self._cond:
                    if self._target is target and not self._stopping and not self._released:
                        self._cond.wait(self.reassert_s)
                if time.monotonic() - last_write >= self.reassert_s:
                    shown = {}  # a profile switch may have reloaded the profile's lighting
                continue
            start = dict(shown) if shown else dict(wanted)
            steps = max(1, round(self.smoothing_s / self.period)) if shown else 1
            for i in range(1, steps + 1):
                with self._cond:
                    if self._target is not target or self._stopping or self._released:
                        break
                for loc, zone in s.by_location.items():
                    c = start[loc].lerp(wanted[loc], i / steps)
                    if shown.get(loc) != c:
                        s.set_static(zone, c)
                        self.writes += 1
                        shown[loc] = c
                owns_lights = True
                last_write = time.monotonic()
                if i < steps:
                    time.sleep(self.period)
