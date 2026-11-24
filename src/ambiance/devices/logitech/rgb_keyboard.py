"""Per-key RGB for Logitech keyboards with RGB_EFFECTS (0x8071) + PER_KEY_LIGHTING_V2 (0x8081).

Verified against the G515 LS TKL (046d:b389 over Bluetooth). The sequence matches what
Solaar, logirgb and G HUB captures do:

  claim     0x8101 fn6 [05]            profile management -> host mode (firmware timers off)
            0x8071 fn5 [01 03 flags]   SetSWControl(set, mode 3 = host owns LEDs, event flags)
  base      0x8071 fn1 [cluster, static_idx, R G B, 02, 0*6, 01]   static effect, RAM only
  paint     0x8081 fn5 [first last R G B]*3   ranges
            0x8081 fn6 [R G B z1..z13]        one colour, many zones
            0x8081 fn1 [z R G B]*4            individual zones
  commit    0x8081 fn7 [00 ...]               frameEnd, RAM only (01 would touch flash)
  release   0x8071 fn5 [01 00 00], then 0x8101 fn6 [03]

While the host holds control the firmware never dims or sleeps the LEDs by itself; with
flag 0x04 it sends RGB_EFFECTS fn1 notifications (data[0]: 0 = idle, else active) so the
host can do it. Host mode is runtime state: a power cycle / deep sleep returns the
keyboard to its onboard profile, so callers must re-claim after reconnecting.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from ambiance.color import RGB

from . import features as F
from .hidpp import HidppDevice, HidppError

log = logging.getLogger(__name__)

# SetSWControl flags (0x8071 fn5)
FLAG_EFFECT = 0x01
FLAG_ACTIVITY_EVENTS = 0x02  # firmware reports user activity (use while idle)
FLAG_IDLE_EVENTS = 0x04      # firmware reports idle-timeout expiry (use while active)

EFFECT_STATIC = 0x0001
ZONE_MIN, ZONE_MAX = 1, 254

FN_SET_INDIVIDUAL = 1
FN_SET_RANGE = 5
FN_SET_SINGLE_VALUE = 6
FN_FRAME_END = 7


class UnsupportedDevice(Exception):
    pass


@dataclass(frozen=True)
class EffectInfo:
    index: int
    effect_id: int
    default_period_ms: int


def encode_frame(colors: Mapping[int, RGB], fill: RGB | None = None, *,
                 auto_fill: bool = False) -> list[tuple[int, bytes]]:
    """Encode zone colours as a minimal list of (0x8081 function, params) writes.

    Only the zones in `colors` change, unless a fill is used: `fill` first paints EVERY
    zone 1..254 with one range write. With auto_fill=True (caller accepts that zones absent
    from `colors` get repainted) the most common colour becomes the fill when that saves
    writes. FrameEnd is NOT included.
    """
    for zone in colors:
        if not ZONE_MIN <= zone <= ZONE_MAX:
            raise ValueError(f"zone {zone} out of range")
    if fill is None and auto_fill and colors:
        common = Counter(RGB(*c) for c in colors.values()).most_common(1)[0][0]
        with_fill = _encode(colors, common)
        without = _encode(colors, None)
        return with_fill if len(with_fill) < len(without) else without
    return _encode(colors, fill)


def _encode(colors: Mapping[int, RGB], base: RGB | None) -> list[tuple[int, bytes]]:
    by_color: dict[RGB, list[int]] = defaultdict(list)
    for zone, c in colors.items():
        by_color[RGB(*c)].append(zone)
    writes: list[tuple[int, bytes]] = []
    if base is not None:
        writes.append((FN_SET_RANGE, bytes((ZONE_MIN, ZONE_MAX, *base))))
        by_color.pop(base, None)

    singles: list[tuple[int, RGB]] = []
    for color, zones in by_color.items():
        zones.sort()
        while len(zones) >= 4:
            chunk, zones = zones[:13], zones[13:]
            writes.append((FN_SET_SINGLE_VALUE, bytes((*color, *chunk))))  # 0-padded = end
        singles.extend((z, color) for z in zones)
    singles.sort()
    for i in range(0, len(singles), 4):
        params = b"".join(bytes((z, *c)) for z, c in singles[i:i + 4])
        if len(params) < 16:
            params += b"\xff"  # end-of-list marker for short individual writes
        writes.append((FN_SET_INDIVIDUAL, params))
    return writes


class LogitechRgbKeyboard:
    def __init__(self, dev: HidppDevice):
        self.dev = dev
        self.idx_rgb = dev.feature_index(F.RGB_EFFECTS)
        self.idx_perkey = dev.feature_index(F.PER_KEY_LIGHTING_V2)
        if self.idx_rgb is None or self.idx_perkey is None:
            raise UnsupportedDevice(f"{dev.path}: needs HID++ features 0x8071 and 0x8081")
        self.idx_profile = dev.feature_index(F.PROFILE_MANAGEMENT)
        self.idx_onboard = None if self.idx_profile else dev.feature_index(F.ONBOARD_PROFILES)
        self.idx_brightness = dev.feature_index(F.BRIGHTNESS_CONTROL)
        self.zones: frozenset[int] = self._read_zone_bitmap()
        self.effects: list[list[EffectInfo]] = self._read_effects()
        static = [e for e in self.effects[0] if e.effect_id == EFFECT_STATIC] if self.effects else []
        if not static:
            raise UnsupportedDevice(f"{dev.path}: no static effect in cluster 0")
        self.static_index = static[0].index
        self.window = 1  # requests in flight; raise once pipelining is verified on a device
        self._fresh_claim = False

    # -- discovery ---------------------------------------------------------------------------
    def _read_zone_bitmap(self) -> frozenset[int]:
        zones = set()
        for page in range(3):
            bitmap = self.dev.request(self.idx_perkey, 0, bytes((0x00, page)))[2:16]
            for byte_i, byte in enumerate(bitmap):
                for bit in range(8):
                    zone = page * 112 + byte_i * 8 + bit
                    if byte >> bit & 1 and ZONE_MIN <= zone <= ZONE_MAX:
                        zones.add(zone)
        return frozenset(zones)

    def _read_effects(self) -> list[list[EffectInfo]]:
        clusters = []
        count = self.dev.request(self.idx_rgb, 0, bytes((0xFF, 0xFF, 0x00)))[2]
        for c in range(count):
            n = self.dev.request(self.idx_rgb, 0, bytes((c, 0xFF, 0x00)))[4]
            effects = []
            for e in range(n):
                r = self.dev.request(self.idx_rgb, 0, bytes((c, e, 0x00)))
                effects.append(EffectInfo(e, r[2] << 8 | r[3], r[6] << 8 | r[7]))
            clusters.append(effects)
        return clusters

    # -- control -----------------------------------------------------------------------------
    def sw_control(self) -> tuple[int, int]:
        """-> (mode, flags); mode 0 = firmware/onboard, 3 = host."""
        r = self.dev.request(self.idx_rgb, 5, bytes((0x00, 0x00, 0x00)))
        return r[1], r[2]

    def claim(self, flags: int = FLAG_IDLE_EVENTS) -> None:
        if self.idx_profile is not None:
            self.dev.request(self.idx_profile, 6, b"\x05")
        elif self.idx_onboard is not None:
            self.dev.request(self.idx_onboard, 1, b"\x02")
        self.dev.request(self.idx_rgb, 5, bytes((0x01, 0x03, flags)))
        time.sleep(0.2)  # let the firmware leave onboard mode before the first frame
        self._fresh_claim = True

    def set_event_flags(self, flags: int) -> None:
        self.dev.request(self.idx_rgb, 5, bytes((0x01, 0x03, flags)))

    def release(self) -> None:
        self.dev.request(self.idx_rgb, 5, bytes((0x01, 0x00, 0x00)))
        if self.idx_profile is not None:
            self.dev.request(self.idx_profile, 6, b"\x03")
        elif self.idx_onboard is not None:
            self.dev.request(self.idx_onboard, 1, b"\x01")

    def set_static(self, color: RGB, cluster: int = 0) -> None:
        """Whole-cluster static colour (also the layer per-key paint sits on)."""
        params = bytes((cluster, self.static_index, *color, 0x02, 0, 0, 0, 0, 0, 0, 0x01))
        self.dev.request(self.idx_rgb, 1, params)

    def paint(self, colors: Mapping[int, RGB], fill: RGB | None = None) -> int:
        """Write a per-key frame and commit it. Returns the number of HID++ writes.

        Changes exactly the zones in `colors` (plus every zone if `fill` is given). A frame
        that covers all of the keyboard's zones may be range-filled automatically.
        """
        colors = {z: c for z, c in colors.items() if z in self.zones}
        writes = encode_frame(colors, fill, auto_fill=self.zones <= colors.keys())
        writes.append((FN_FRAME_END, b"\x00"))
        reqs = [(self.idx_perkey, fn, params) for fn, params in writes]
        self.dev.request_many(reqs, window=self.window)
        if self._fresh_claim:
            # the first frame after a claim can land mid-transition; replay it once
            self._fresh_claim = False
            self.dev.request_many(reqs, window=self.window)
        return len(reqs)

    # -- brightness & power timers ----------------------------------------------------------------
    def brightness(self) -> int | None:
        if self.idx_brightness is None:
            return None
        r = self.dev.request(self.idx_brightness, 1)
        return r[0] << 8 | r[1]

    def set_brightness(self, value: int) -> None:
        if self.idx_brightness is None:
            raise UnsupportedDevice("no brightness control")
        self.dev.request(self.idx_brightness, 2, bytes((value >> 8 & 0xFF, value & 0xFF)))

    def power_timers(self) -> tuple[int, int]:
        """-> (idle_seconds, sleep_seconds) the firmware uses."""
        r = self.dev.request(self.idx_rgb, 7, b"\x00")
        return r[3] << 8 | r[4], r[5] << 8 | r[6]

    def is_activity_event(self, n) -> bool | None:
        """For a Notification: True = user active, False = idle, None = unrelated."""
        if n.feature_index == self.idx_rgb and n.function == 1:
            return bool(n.data[0]) if n.data else None
        return None

    def describe(self) -> dict:
        mode, flags = self.sw_control()
        idle, sleep = self.power_timers()
        return {
            "zones": sorted(self.zones),
            "effects": [[(e.index, f"{e.effect_id:#06x}") for e in cl] for cl in self.effects],
            "static_index": self.static_index,
            "sw_control": {"mode": mode, "flags": flags},
            "brightness": self.brightness(),
            "power_timers_s": {"idle": idle, "sleep": sleep},
        }


__all__ = ["LogitechRgbKeyboard", "encode_frame", "UnsupportedDevice", "HidppError"]
