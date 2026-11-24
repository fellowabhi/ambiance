"""Per-device brightness/look in the engine, and the 0x8070 zone (mouse) device."""

from __future__ import annotations

import threading
import time

import numpy as np

from ambiance.color import RGB
from ambiance.colorspace import srgb_to_oklab
from ambiance.core import Decision, Director, LightDevice, MediaState, State
from ambiance.devices.logitech import zones as zones_mod
from ambiance.devices.logitech.zones import LOCATION_LOGO, LOCATION_PRIMARY, LogitechZoneDevice
from ambiance.engine import Engine
from ambiance.layout import Key, Layout
from ambiance.palette import Palette, Swatch
from ambiance.scenes.album import AlbumScene
from ambiance.scenes.basic import SolidScene


class FakeDevice(LightDevice):
    def __init__(self, layout, brightness=1.0, animate=True):
        self.layout, self.brightness, self.look = layout, brightness, {"animate": animate}
        self.frames = []
        self.name = layout.name

    def start(self): pass
    def submit(self, frame): self.frames.append(dict(frame))
    def release(self): pass
    def stop(self, release=True): pass

    @property
    def connected(self): return True


class Fixed(Director):
    def __init__(self, scene): self.scene = scene
    def decide(self, state): return Decision(self.scene, 1.0, transition_s=0.0)


ONE = Layout("one", (Key("only", "x", 0, 0, 0),))


def test_engine_scales_by_device_brightness_live():
    dim, full = FakeDevice(ONE, brightness=0.5), FakeDevice(Layout("one-b", ONE.keys))
    engine = Engine(State(), Fixed(SolidScene(RGB(255, 255, 255))), [dim, full], clock=lambda: 0.0)
    engine.step()
    bright = full.frames[-1][0]
    assert dim.frames[-1][0] == bright.scaled(0.5)
    dim.brightness = 0.25            # the UI changes it at runtime...
    engine.refresh()
    engine.step()                    # ...and the next step re-submits
    assert dim.frames[-1][0] == bright.scaled(0.25)


def swatch(rgb):
    lab = srgb_to_oklab(np.array(rgb, dtype=float) / 255.0)
    return Swatch(RGB(*rgb), tuple(float(v) for v in lab), 0.3, 1.0)


def test_look_animate_false_keeps_a_cycling_scene_static_on_that_device():
    media = MediaState("spotify", "Playing", "t", "S", ("A",), "Al", "u",
                       Palette((swatch((220, 30, 40)), swatch((30, 60, 220)))))
    scene = AlbumScene(media, strip_cycle_s=1.0)
    t = [0.0]
    still, moving = FakeDevice(ONE, animate=False), FakeDevice(Layout("one-c", ONE.keys))
    engine = Engine(State(), Fixed(scene), [still, moving], clock=lambda: t[0])
    for step in range(12):
        t[0] = step * 0.25
        engine.step(state_changed=False)
    assert len({f[0] for f in still.frames}) == 1          # held still
    assert len({f[0] for f in moving.frames}) > 2          # cycled


# ---- zone device ----------------------------------------------------------------------------

class FakeHidpp:
    path = "/dev/fake"

    def close(self): pass


class FakeSession:
    def __init__(self):
        self.dev = FakeHidpp()
        self.name = "G502 HERO Gaming Mouse"
        self.zones = [zones_mod._Zone(0, LOCATION_PRIMARY, 1, b"\x03" + bytes(10)),
                      zones_mod._Zone(1, LOCATION_LOGO, 1, b"\x03" + bytes(10))]
        self.by_location = {z.location: z for z in self.zones}
        self.calls = []
        self.lock = threading.Lock()

    def set_static(self, zone, c):
        with self.lock:
            self.calls.append(("static", zone.location, c))

    def restore(self):
        with self.lock:
            self.calls.append(("restore",))

    def logo(self):
        with self.lock:
            return [c for kind, *rest in self.calls if kind == "static"
                    for loc, c in [rest] if loc == LOCATION_LOGO]


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def make_device(monkeypatch, **kw):
    session = FakeSession()
    monkeypatch.setattr(zones_mod, "find_zone_device", lambda product_ids=None: session)
    dev = LogitechZoneDevice(max_rate_hz=200, smoothing_s=0.05, **kw)
    dev.start()
    assert wait_for(lambda: dev.connected)
    return dev, session


def test_zone_device_paints_fades_and_restores_on_stop(monkeypatch):
    dev, s = make_device(monkeypatch)
    dev.submit({LOCATION_LOGO: RGB(0, 0, 0), LOCATION_PRIMARY: RGB(0, 0, 0)})
    assert wait_for(lambda: s.logo()[-1:] == [RGB(0, 0, 0)])
    dev.submit({LOCATION_LOGO: RGB(200, 100, 0), LOCATION_PRIMARY: RGB(0, 0, 50)})
    assert wait_for(lambda: s.logo()[-1:] == [RGB(200, 100, 0)])
    assert len(s.logo()) > 3                                # faded in steps
    dev.stop(release=True)
    assert s.calls[-1] == ("restore",)


def test_zone_device_release_restores_then_resumes(monkeypatch):
    dev, s = make_device(monkeypatch)
    dev.submit({LOCATION_LOGO: RGB(9, 9, 9)})
    assert wait_for(lambda: s.logo()[-1:] == [RGB(9, 9, 9)])
    dev.release()
    assert wait_for(lambda: ("restore",) in s.calls)
    dev.submit({LOCATION_LOGO: RGB(1, 2, 3)})
    assert wait_for(lambda: s.logo()[-1:] == [RGB(1, 2, 3)])
    dev.stop(release=False)
    assert s.calls.count(("restore",)) == 1                  # stop(release=False) left it


def test_zone_device_reasserts_after_profile_switch(monkeypatch):
    dev, s = make_device(monkeypatch, reassert_s=0.2)
    dev.submit({LOCATION_LOGO: RGB(5, 6, 7)})
    assert wait_for(lambda: s.logo()[-1:] == [RGB(5, 6, 7)])
    n = len(s.logo())
    assert wait_for(lambda: len(s.logo()) > n, timeout=2.0)  # re-sent without a new frame
    dev.stop(release=False)


def test_hidraw_claims_are_exclusive_per_owner():
    from ambiance.devices.hidraw import release_claim, try_claim

    assert try_claim("/dev/x", "a") and try_claim("/dev/x", "a")   # re-claim is fine
    assert not try_claim("/dev/x", "b")                              # held by another driver
    release_claim("/dev/x", "b")                                     # not the owner: no-op
    assert not try_claim("/dev/x", "b")
    release_claim("/dev/x", "a")
    assert try_claim("/dev/x", "b")
    release_claim("/dev/x", "b")
