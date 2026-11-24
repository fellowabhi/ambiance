"""ElkStripDevice tests with a fake bleak client (no Bluetooth needed)."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types

import pytest

from ambiance.color import RGB
from ambiance.devices import elk_strip
from ambiance.devices.elk_strip import (MELK_HANDSHAKE, POWER_ON, STRIP_ZONE, ElkStripDevice,
                                        color_command)


class FakeChar:
    pass


class FakeServices:
    def get_characteristic(self, uuid):
        return FakeChar() if uuid == elk_strip.WRITE_UUIDS[0] else None


class FakeBle:
    """Stands in for the `bleak` module: one fake device and client."""

    def __init__(self, name="MELK-OA10   40", fail_connects=0):
        self.writes: list[bytes] = []
        self.lock = threading.Lock()
        self.name = name
        self.fail_connects = fail_connects
        self.connects = 0
        self.client = None
        fake = self

        class Dev:
            address = "AA:BB:CC:DD:EE:FF"

        Dev.name = name

        class Scanner:
            @staticmethod
            async def find_device_by_address(addr, timeout=10):
                return Dev()

            @staticmethod
            async def find_device_by_filter(fn, timeout=10):
                return Dev() if fn(Dev(), None) else None

        class Client:
            def __init__(self, dev, timeout=20.0, disconnected_callback=None):
                self.services = FakeServices()
                self._cb = disconnected_callback
                fake.client = self

            async def __aenter__(self):
                fake.connects += 1
                if fake.fail_connects > 0:
                    fake.fail_connects -= 1
                    raise OSError("simulated connect failure")
                return self

            async def __aexit__(self, *exc):
                return False

            async def write_gatt_char(self, uuid, data, response=False):
                with fake.lock:
                    fake.writes.append(bytes(data))

            def drop(self):
                self._cb(self)

        self.module = types.SimpleNamespace(BleakScanner=Scanner, BleakClient=Client)

    def colors(self) -> list[RGB]:
        with self.lock:
            return [RGB(w[4], w[5], w[6]) for w in self.writes if w[:4] == bytes((0x7E, 0, 5, 3))]


@pytest.fixture
def fake_ble(monkeypatch):
    fake = FakeBle()
    monkeypatch.setitem(sys.modules, "bleak", fake.module)
    return fake


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_handshake_power_on_then_colour(fake_ble):
    d = ElkStripDevice(address="AA:BB:CC:DD:EE:FF", max_rate_hz=50, smoothing_s=0.0)
    d.start()
    assert wait_for(lambda: d.connected)
    d.submit({STRIP_ZONE: RGB(10, 20, 30)})
    assert wait_for(lambda: fake_ble.colors() == [RGB(10, 20, 30)])
    with fake_ble.lock:
        assert fake_ble.writes[:3] == [*MELK_HANDSHAKE, POWER_ON]
    d.stop(release=False)
    assert not d._thread.is_alive()


def test_fades_between_keyframes_and_ends_on_target(fake_ble):
    d = ElkStripDevice(max_rate_hz=100, smoothing_s=0.1)  # 10 steps per fade
    d.start()
    assert wait_for(lambda: d.connected)
    d.submit({STRIP_ZONE: RGB(0, 0, 0)})
    assert wait_for(lambda: fake_ble.colors()[-1:] == [RGB(0, 0, 0)])
    d.submit({STRIP_ZONE: RGB(200, 100, 0)})
    assert wait_for(lambda: fake_ble.colors()[-1:] == [RGB(200, 100, 0)])
    fade = fake_ble.colors()[1:]
    assert len(fade) >= 5  # intermediate steps were sent
    reds = [c.r for c in fade]
    assert reds == sorted(reds)  # monotonic ramp
    d.stop(release=False)


def test_latest_wins_and_rate_limited(fake_ble):
    d = ElkStripDevice(max_rate_hz=20, smoothing_s=0.0)
    d.start()
    assert wait_for(lambda: d.connected)
    t0 = time.monotonic()
    for i in range(50):
        d.submit({STRIP_ZONE: RGB(i, 0, 0)})
        time.sleep(0.002)
    assert wait_for(lambda: fake_ble.colors()[-1:] == [RGB(49, 0, 0)])
    elapsed = time.monotonic() - t0
    n = len(fake_ble.colors())
    assert n < 50 and n <= elapsed * 20 + 2  # skipped stale frames, respected the rate
    d.stop(release=False)


def test_frame_for_other_layout_uses_average(fake_ble):
    d = ElkStripDevice(max_rate_hz=50, smoothing_s=0.0)
    d.start()
    assert wait_for(lambda: d.connected)
    d.submit({1: RGB(100, 0, 0), 2: RGB(0, 100, 0)})
    assert wait_for(lambda: fake_ble.colors()[-1:] == [RGB(50, 50, 0)])
    d.stop(release=False)


def test_reconnects_after_disconnect_and_repaints(fake_ble):
    d = ElkStripDevice(max_rate_hz=50, smoothing_s=0.0)
    d.start()
    assert wait_for(lambda: d.connected)
    d.submit({STRIP_ZONE: RGB(1, 2, 3)})
    assert wait_for(lambda: fake_ble.colors()[-1:] == [RGB(1, 2, 3)])
    fake_ble.client.drop()
    assert wait_for(lambda: fake_ble.connects >= 2, timeout=5)
    # after reconnecting the last colour is painted again
    assert wait_for(lambda: fake_ble.colors().count(RGB(1, 2, 3)) >= 2, timeout=5)
    d.stop(release=False)


def test_retries_failed_connect(monkeypatch):
    fake = FakeBle(fail_connects=1)
    monkeypatch.setitem(sys.modules, "bleak", fake.module)
    d = ElkStripDevice(max_rate_hz=50, smoothing_s=0.0)
    d.start()
    assert wait_for(lambda: d.connected, timeout=5)
    assert fake.connects == 2 and "simulated" in d.last_error
    d.stop(release=False)


def test_stop_writes_release_colour(fake_ble):
    d = ElkStripDevice(max_rate_hz=50, smoothing_s=0.0, release_color=RGB(255, 140, 60))
    d.start()
    assert wait_for(lambda: d.connected)
    d.submit({STRIP_ZONE: RGB(0, 0, 255)})
    assert wait_for(lambda: fake_ble.colors()[-1:] == [RGB(0, 0, 255)])
    d.stop(release=True)
    assert fake_ble.colors()[-1] == RGB(255, 140, 60)


def test_non_melk_skips_handshake(monkeypatch):
    fake = FakeBle(name="ELK-BLEDOM")
    monkeypatch.setitem(sys.modules, "bleak", fake.module)
    d = ElkStripDevice(max_rate_hz=50, smoothing_s=0.0)
    d.start()
    assert wait_for(lambda: d.connected)
    with fake.lock:
        assert fake.writes[0] == POWER_ON
    d.stop(release=False)


def test_color_command_bytes():
    assert color_command(RGB(1, 2, 3)) == bytes.fromhex("7e00050301020300ef")
