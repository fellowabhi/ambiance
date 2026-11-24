"""LogitechKeyboardDevice worker tests.

Everything below the device is faked; an autouse fixture fails any attempt to open a
/dev/hidraw* node, because the real keyboard is in use while these run.
"""

from __future__ import annotations

import errno
import logging
import os
import threading
import time
from collections.abc import Callable

import pytest

from ambiance.color import BLACK, RGB
from ambiance.devices.hidraw import HidrawDevice
from ambiance.devices.logitech import device as device_mod
from ambiance.devices.logitech.device import (HidrawConnector, KeyboardSession,
                                              LogitechKeyboardDevice)
from ambiance.devices.logitech.hidpp import DeviceGone, HidppError, HidppTimeout, Notification
from ambiance.devices.logitech.rgb_keyboard import (FN_FRAME_END, FN_SET_INDIVIDUAL, FN_SET_RANGE,
                                                    FN_SET_SINGLE_VALUE, LogitechRgbKeyboard,
                                                    UnsupportedDevice)
from ambiance.layout import TKL_ANSI

RED, GREEN, BLUE = RGB(255, 0, 0), RGB(0, 255, 0), RGB(0, 0, 255)
ZONES = TKL_ANSI.zones
RGB_EFFECTS_INDEX = 0x0A


@pytest.fixture(autouse=True)
def no_hidraw(monkeypatch):
    """Safety net: no test may open a real hidraw node."""
    real_open = os.open

    def guarded(path, *args, **kwargs):
        if os.fsdecode(path).startswith("/dev/hidraw"):
            raise AssertionError(f"test tried to open {path}")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded)


# ---- fakes --------------------------------------------------------------------------------------

def activity(active: bool) -> Notification:
    """RGB_EFFECTS fn1 notification as the keyboard sends it (data[0]: 0 = idle, else active)."""
    return Notification(RGB_EFFECTS_INDEX, 1, bytes((int(active),)) + bytes(15))


class FakeTransport:
    """Stands in for HidppDevice: notifications pushed by a test come out of poll()."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.closed = False
        self.gone = False
        self._pending: list[Notification] = []
        self._cond = threading.Condition()

    def push(self, *notes: Notification) -> None:
        with self._cond:
            self._pending.extend(notes)
            self._cond.notify_all()

    def vanish(self) -> None:
        """The node disappears (deep sleep, power off, out of range)."""
        with self._cond:
            self.gone = True
            self._cond.notify_all()

    def poll(self, timeout: float) -> list[Notification]:
        with self._cond:
            self._cond.wait_for(lambda: self._pending or self.gone or self.closed, timeout)
            if self.gone or self.closed:
                raise DeviceGone(errno.ENODEV, f"{self.path}: gone")
            out, self._pending = self._pending, []
            return out

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()


class FakeKeyboard:
    """Records the lighting calls the worker makes; can be told to fail or to stall."""

    def __init__(self, dev: FakeTransport, zones=ZONES) -> None:
        self.dev = dev
        self.zones = frozenset(zones)
        self.mode = 0  # sw_control mode: onboard until claimed
        self.calls: list[tuple] = []  # (name, *args); a failed call is recorded as "name!"
        self.fail: dict[str, list[BaseException]] = {}
        self.paint_gate: threading.Event | None = None
        self.paint_started = threading.Event()
        self.on_claim: Callable[[], None] | None = None
        self._lock = threading.Lock()

    def _call(self, name: str, *args) -> None:
        with self._lock:
            pending = self.fail.get(name)
            err = pending.pop(0) if pending else None
            self.calls.append((name + "!" if err else name, *args))
        if err is not None:
            raise err

    def claim(self) -> None:
        self._call("claim")
        self.mode = 3
        if self.on_claim is not None:
            self.on_claim()

    def release(self) -> None:
        self._call("release")
        self.mode = 0

    def set_static(self, color: RGB) -> None:
        self._call("set_static", color)

    def paint(self, colors, fill=None) -> int:
        self.paint_started.set()
        if self.paint_gate is not None and not self.paint_gate.wait(5):
            raise RuntimeError("paint gate never opened")
        self._call("paint", dict(colors), fill)
        return 1 + len(colors) // 4

    def sw_control(self) -> tuple[int, int]:
        self._call("sw_control")
        return self.mode, 0x04

    def is_activity_event(self, n: Notification) -> bool | None:
        if n.feature_index == RGB_EFFECTS_INDEX and n.function == 1:
            return bool(n.data[0])
        return None

    def names(self) -> list[str]:
        with self._lock:
            return [c[0] for c in self.calls]

    def paints(self) -> list[tuple[dict, RGB | None]]:
        with self._lock:
            return [(c[1], c[2]) for c in self.calls if c[0] == "paint"]


class FakeConnector:
    """Hands out the given sessions in order, then None (no keyboard around)."""

    def __init__(self, *sessions: KeyboardSession) -> None:
        self.sessions = list(sessions)

    def __call__(self) -> KeyboardSession | None:
        return self.sessions.pop(0) if self.sessions else None


def keyboard(path: str = "/dev/fake0") -> tuple[KeyboardSession, FakeKeyboard]:
    dev = FakeTransport(path)
    kb = FakeKeyboard(dev)
    return KeyboardSession(dev, kb), kb


@pytest.fixture
def make_device():
    made: list[LogitechKeyboardDevice] = []

    def factory(*sessions: KeyboardSession, **kwargs) -> LogitechKeyboardDevice:
        kwargs.setdefault("rescan_interval", 0.02)
        kwargs.setdefault("verify_interval", 60.0)
        dev = LogitechKeyboardDevice(TKL_ANSI, connector=FakeConnector(*sessions), **kwargs)
        dev.poll_interval = 0.01
        dev.claim_quiet_s = 0.05
        dev.backoff_base = 0.01
        dev.join_timeout = 2.0
        made.append(dev)
        return dev

    yield factory
    for dev in made:
        dev.stop(release=False)


def wait_until(predicate: Callable[[], object], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the worker")
        time.sleep(0.005)


def painted(dev: LogitechKeyboardDevice, n: int) -> Callable[[], bool]:
    return lambda: dev.status()["frames_painted"] >= n


def solid(color: RGB) -> dict[int, RGB]:
    return {z: color for z in ZONES}


def two_tone() -> dict[int, RGB]:
    """Letters blue, the other keys red (the most common colour)."""
    return {z: BLUE if z <= 26 else RED for z in ZONES}


# ---- worker -------------------------------------------------------------------------------------

def test_claims_on_first_frame_and_paints_it_in_full(make_device):
    session, kb = keyboard()
    dev = make_device(session)
    dev.start()
    wait_until(lambda: dev.connected)
    time.sleep(0.05)
    assert kb.names() == []  # connected, but nothing to show yet: the firmware keeps the LEDs

    frame = two_tone()
    dev.submit(frame)
    wait_until(painted(dev, 1))
    assert kb.names() == ["claim", "set_static", "paint"]
    assert kb.calls[1] == ("set_static", BLACK)
    assert kb.paints() == [(frame, RED)]  # the whole frame, its most common colour as fill
    status = dev.status()
    assert status["connected"] and status["claimed"] and status["path"] == "/dev/fake0"
    assert status["writes"] == 1 + len(frame) // 4 and status["reconnects"] == 0


def test_latest_frame_wins_while_the_keyboard_is_busy(make_device):
    session, kb = keyboard()
    kb.paint_gate = threading.Event()  # the first paint stalls like a slow BLE frame
    dev = make_device(session)
    dev.start()
    dev.submit(solid(RGB(1, 0, 0)))
    assert kb.paint_started.wait(2)
    for i in range(2, 11):
        dev.submit(solid(RGB(i, 0, 0)))
    kb.paint_gate.set()
    wait_until(painted(dev, 2))
    time.sleep(0.05)
    assert kb.paints() == [(solid(RGB(1, 0, 0)), RGB(1, 0, 0)),
                           (solid(RGB(10, 0, 0)), RGB(10, 0, 0))]
    assert dev.status()["frames_submitted"] == 10


def test_small_changes_are_painted_as_a_diff_without_fill(make_device):
    session, kb = keyboard()
    dev = make_device(session)
    dev.start()
    base = solid(RED)
    dev.submit(base)
    wait_until(painted(dev, 1))

    changed = {**base, 1: BLUE, 2: BLUE, 3: BLUE}
    dev.submit(changed)
    wait_until(painted(dev, 2))
    assert kb.paints()[1] == ({1: BLUE, 2: BLUE, 3: BLUE}, None)

    dev.submit(dict(changed))  # identical frame: costs nothing
    time.sleep(0.05)
    assert len(kb.paints()) == 2

    dev.submit(solid(GREEN))  # most keys changed: one full frame is cheaper than a diff
    wait_until(painted(dev, 3))
    assert kb.paints()[2] == (solid(GREEN), GREEN)


@pytest.mark.parametrize("error", [DeviceGone(errno.ENODEV, "/dev/fake1: gone"),
                                   HidppTimeout("no reply"),
                                   OSError(errno.EIO, "I/O error")],
                         ids=["gone", "timeout", "oserror"])
def test_reconnects_and_repaints_after_losing_the_keyboard_mid_paint(make_device, error):
    first, kb1 = keyboard("/dev/fake1")
    second, kb2 = keyboard("/dev/fake2")
    dev = make_device(first, second)
    dev.start()
    dev.submit(solid(RED))
    wait_until(painted(dev, 1))

    kb1.fail["paint"] = [error]
    frame = {**solid(RED), 1: BLUE}
    dev.submit(frame)
    wait_until(lambda: kb2.paints())
    assert kb1.dev.closed and kb1.names()[-1] == "paint!"
    assert kb2.names() == ["claim", "set_static", "paint"]
    assert kb2.paints() == [(frame, RED)]  # a full repaint of the latest frame, not the diff
    status = dev.status()
    assert status["connected"] and status["path"] == "/dev/fake2" and status["reconnects"] == 1
    assert status["last_error"].startswith(type(error).__name__)


def test_verify_reclaims_when_the_keyboard_silently_reverts(make_device):
    session, kb = keyboard()
    dev = make_device(session, verify_interval=0.05)
    dev.start()
    frame = two_tone()
    dev.submit(frame)
    wait_until(lambda: "sw_control" in kb.names())  # periodic check while holding control
    assert kb.names().count("claim") == 1

    kb.mode = 0  # e.g. deep sleep without a reconnect: onboard lighting is back
    wait_until(lambda: len(kb.paints()) == 2)
    names = kb.names()
    last_claim = len(names) - 1 - names[::-1].index("claim")
    assert names.count("claim") == 2
    assert names[last_claim - 1:] == ["sw_control", "claim", "set_static", "paint"]
    assert kb.paints()[1] == (frame, RED)


def test_spurious_activity_right_after_claim_is_ignored(make_device):
    events: list[bool] = []
    session, kb = keyboard()
    kb.on_claim = lambda: kb.dev.push(activity(False), activity(True))
    dev = make_device(session, on_activity=events.append)
    dev.claim_quiet_s = 0.3
    dev.start()
    dev.submit(solid(RED))
    wait_until(painted(dev, 1))
    time.sleep(0.1)
    assert events == []
    assert kb.names() == ["claim", "set_static", "paint"]  # and no host-mode check either

    time.sleep(0.35)  # quiet window over: a real idle event gets through
    kb.dev.push(activity(False))
    wait_until(lambda: events == [False])
    assert dev.status()["user_active"] is False


def test_active_after_idle_checks_host_mode_and_reclaims(make_device):
    events: list[bool] = []
    session, kb = keyboard()
    dev = make_device(session, on_activity=events.append)
    dev.start()
    frame = two_tone()
    dev.submit(frame)
    wait_until(painted(dev, 1))
    time.sleep(0.1)  # past the post-claim quiet window

    kb.dev.push(activity(False))
    wait_until(lambda: events == [False])
    assert "sw_control" not in kb.names()

    kb.mode = 0  # the keyboard slept and dropped host mode without disconnecting
    kb.dev.push(activity(True))
    wait_until(lambda: len(kb.paints()) == 2)
    assert events == [False, True]
    assert kb.names()[3:] == ["sw_control", "claim", "set_static", "paint"]
    assert kb.paints()[1] == (frame, RED)


def test_reconnect_after_idle_reports_the_user_back(make_device):
    events: list[bool] = []
    first, kb1 = keyboard("/dev/fake1")
    second, kb2 = keyboard("/dev/fake2")
    dev = make_device(first, second, on_activity=events.append)
    dev.start()
    dev.submit(solid(RED))
    wait_until(painted(dev, 1))
    time.sleep(0.1)
    kb1.dev.push(activity(False))
    wait_until(lambda: events == [False])

    kb1.dev.vanish()  # drops off Bluetooth; it comes back when someone presses a key
    wait_until(lambda: kb2.paints())
    assert events == [False, True]
    assert kb2.paints() == [(solid(RED), RED)]


def test_release_hands_back_until_the_next_submit(make_device):
    session, kb = keyboard()
    dev = make_device(session)
    dev.start()
    frame = two_tone()
    dev.submit(frame)
    wait_until(painted(dev, 1))

    dev.release()
    wait_until(lambda: "release" in kb.names())
    dev.release()  # already handed back: no second request
    time.sleep(0.05)
    assert kb.names() == ["claim", "set_static", "paint", "release"]
    assert dev.connected and not dev.status()["claimed"]

    dev.submit(frame)  # even the same frame means "take the LEDs back"
    wait_until(painted(dev, 2))
    assert kb.names()[4:] == ["claim", "set_static", "paint"]
    assert kb.paints()[1] == (frame, RED)


def test_stop_releases_closes_and_joins(make_device):
    session, kb = keyboard()
    dev = make_device(session)
    dev.start()
    dev.submit(solid(RED))
    wait_until(painted(dev, 1))

    dev.stop()
    assert kb.names()[-1] == "release" and kb.dev.closed
    status = dev.status()
    assert not status["running"] and not status["connected"]
    dev.stop()
    dev.submit(solid(BLUE))  # ignored once stopped
    assert kb.names().count("release") == 1 and len(kb.paints()) == 1
    with pytest.raises(RuntimeError):
        dev.start()


def test_stop_without_release_leaves_the_frame_on(make_device):
    session, kb = keyboard()
    dev = make_device(session)
    dev.start()
    dev.submit(solid(RED))
    wait_until(painted(dev, 1))
    dev.stop(release=False)
    assert "release" not in kb.names() and kb.dev.closed


def test_stop_is_safe_when_never_started_or_never_connected(make_device):
    make_device().stop()

    dev = make_device()  # no keyboard anywhere
    dev.start()
    dev.submit(solid(RED))
    dev.release()
    time.sleep(0.05)
    started = time.monotonic()
    dev.stop()
    dev.stop()
    assert time.monotonic() - started < 1.0
    assert not dev.status()["running"] and not dev.connected


def test_zones_the_keyboard_lacks_are_ignored(make_device):
    session, kb = keyboard()
    dev = make_device(session)
    dev.start()
    dev.submit({**solid(RED), 0: BLUE, 200: BLUE, 999: BLUE})
    wait_until(painted(dev, 1))
    assert kb.paints() == [(solid(RED), RED)]

    dev.submit({200: GREEN})  # nothing this keyboard can show
    time.sleep(0.05)
    dev.submit(solid(BLUE))
    wait_until(painted(dev, 2))
    assert kb.paints()[1] == (solid(BLUE), BLUE)


def test_hidpp_error_is_retried_with_a_full_repaint(make_device):
    session, kb = keyboard()
    dev = make_device(session)
    dev.start()
    dev.submit(solid(RED))
    wait_until(painted(dev, 1))

    kb.fail["paint"] = [HidppError(0x04, 0x0B, 1)]
    frame = {**solid(RED), 1: BLUE}
    dev.submit(frame)
    wait_until(painted(dev, 2))
    assert kb.names()[3:] == ["paint!", "paint"]
    assert kb.paints()[1] == (frame, RED)  # full: the failed diff may have left junk behind
    status = dev.status()
    assert status["reconnects"] == 0 and status["last_error"].startswith("HidppError")


def test_repeated_hidpp_errors_count_as_a_disconnect(make_device):
    first, kb1 = keyboard("/dev/fake1")
    second, kb2 = keyboard("/dev/fake2")
    dev = make_device(first, second)
    dev.start()
    dev.submit(solid(RED))
    wait_until(painted(dev, 1))

    kb1.fail["paint"] = [HidppError(0x04, 0x0B, 1) for _ in range(3)]
    dev.submit(solid(BLUE))
    wait_until(lambda: kb2.paints())
    assert kb1.names().count("paint!") == 3 and kb1.dev.closed
    assert kb2.names() == ["claim", "set_static", "paint"]
    assert kb2.paints() == [(solid(BLUE), BLUE)]
    assert dev.status()["reconnects"] == 1


# ---- the real LogitechRgbKeyboard.paint(): diffs must never flood other zones ----------------

class RecordingHidpp:
    """Just enough HidppDevice for LogitechRgbKeyboard.paint(): records every request."""

    path = "/dev/fake"

    def __init__(self) -> None:
        self.requests: list[tuple[int, int, bytes]] = []

    def request_many(self, requests, *, window=1, timeout=1.5, busy_retries=3):
        reqs = [(fi, fn, bytes(p)) for fi, fn, p in requests]
        self.requests.extend(reqs)
        return [bytes(16)] * len(reqs)


def real_keyboard(zones) -> LogitechRgbKeyboard:
    """A real LogitechRgbKeyboard on RecordingHidpp, skipping its discovery requests."""
    kb = LogitechRgbKeyboard.__new__(LogitechRgbKeyboard)
    kb.dev = RecordingHidpp()
    kb.idx_rgb, kb.idx_perkey = RGB_EFFECTS_INDEX, 0x0B
    kb.zones = frozenset(zones)
    kb.window = 1
    kb._fresh_claim = False
    return kb


def apply_writes(requests, shown: dict[int, RGB]) -> None:
    """Replay PER_KEY_LIGHTING_V2 writes onto a zone -> colour map like the keyboard does."""
    for _index, fn, p in requests:
        if fn == FN_SET_RANGE:
            for z in range(p[0], p[1] + 1):
                shown[z] = RGB(*p[2:5])
        elif fn == FN_SET_SINGLE_VALUE:
            for z in p[3:16]:
                if z == 0:
                    break
                shown[z] = RGB(*p[0:3])
        elif fn == FN_SET_INDIVIDUAL:
            for i in range(0, len(p) - 3, 4):
                if p[i] in (0x00, 0xFF):
                    break
                shown[p[i]] = RGB(*p[i + 1:i + 4])


def test_partial_paint_never_range_fills():
    raw = real_keyboard(range(1, 120))
    raw.paint({1: BLUE, 2: BLUE, 50: GREEN})
    assert FN_SET_RANGE not in [fn for _, fn, _ in raw.dev.requests]


def test_paint_covering_every_zone_may_auto_fill():
    raw = real_keyboard(range(1, 120))
    raw.paint({**{z: RED for z in range(1, 120)}, 7: BLUE})
    assert raw.dev.requests[0][1:] == (FN_SET_RANGE, bytes((1, 254, *RED)))


def test_paint_touches_only_the_given_zones():
    raw = real_keyboard(range(1, 120))
    diff = {1: BLUE, 2: BLUE, 3: BLUE, 4: BLUE, 5: BLUE, 9: RED, 10: RED, 50: GREEN, 200: GREEN}
    shown = {z: RGB(9, 9, 9) for z in range(1, 255)}
    expected = {**shown, **{z: c for z, c in diff.items() if z in raw.zones}}

    writes = raw.paint(diff)

    sent = raw.dev.requests
    assert writes == len(sent) and sent[-1][1:] == (FN_FRAME_END, b"\x00")
    assert FN_SET_RANGE not in [fn for _, fn, _ in sent]
    apply_writes(sent, shown)
    assert shown == expected


def test_paint_keeps_explicit_fills_and_distinct_colours_as_they_were():
    raw = real_keyboard(range(1, 120))
    frame = {**{z: RED for z in range(1, 100)}, 7: BLUE}
    raw.paint(frame, fill=RED)
    assert raw.dev.requests[0][1:] == (FN_SET_RANGE, bytes((1, 254, *RED)))
    shown: dict[int, RGB] = {}
    apply_writes(raw.dev.requests, shown)
    assert all(shown[z] == c for z, c in frame.items())

    raw.dev.requests.clear()
    raw.paint({3: BLUE, 4: GREEN})
    assert [fn for _, fn, _ in raw.dev.requests] == [FN_SET_INDIVIDUAL, FN_FRAME_END]


def test_session_wraps_a_plain_logitech_keyboard():
    raw = real_keyboard(range(1, 10))
    session = KeyboardSession(raw.dev, raw)
    assert session.kb is raw  # paint() is exact for diffs; no wrapper needed


# ---- default connector (sysfs discovery and HidppDevice faked) ----------------------------------

# Vendor page 0xFF00, report 0x11 with 19-byte input and output: a HID++ long-report node.
HIDPP_DESCRIPTOR = bytes.fromhex("0600ff0902a101851175089513150026ff00090281000902910" "0c0")


def hidraw_node(n: int, product_id: int, descriptor: bytes = HIDPP_DESCRIPTOR) -> HidrawDevice:
    return HidrawDevice(path=f"/dev/hidraw{n}", sys_path=f"/sys/class/hidraw/hidraw{n}",
                        bus=0x05, vendor_id=0x046D, product_id=product_id, name="Logitech",
                        phys="", uniq="aa:bb:cc", report_descriptor=descriptor)


def test_default_connector_finds_the_keyboard_and_stops_probing_other_devices(monkeypatch):
    nodes = [hidraw_node(3, 0xB023, descriptor=b""),  # no HID++ reports: never opened
             hidraw_node(5, 0xB023),                  # a HID++ mouse
             hidraw_node(7, 0xB389)]                  # the keyboard
    opened: list = []

    class FakeHidpp:
        def __init__(self, path, device_index):
            assert device_index == 0xFF
            self.path, self.closed = path, False
            opened.append(self)

        def ping(self):
            return 4, 2

        def close(self):
            self.closed = True

    class FakeRgb:
        def __init__(self, dev):
            if dev.path != "/dev/hidraw7":
                raise UnsupportedDevice(f"{dev.path}: needs HID++ features 0x8071 and 0x8081")
            self.dev, self.zones = dev, ZONES

    monkeypatch.setattr(device_mod, "enumerate_hidraw", lambda vendor_id=None: list(nodes))
    monkeypatch.setattr(device_mod, "HidppDevice", FakeHidpp)
    monkeypatch.setattr(device_mod, "LogitechRgbKeyboard", FakeRgb)
    connect = HidrawConnector()

    session = connect()
    assert session is not None and session.path == "/dev/hidraw7"
    assert isinstance(session.kb, FakeRgb)
    assert [d.path for d in opened] == ["/dev/hidraw5", "/dev/hidraw7"]
    assert opened[0].closed and not opened[1].closed

    session.close()
    assert connect() is not None  # reconnect: the mouse is not probed again
    assert [d.path for d in opened] == ["/dev/hidraw5", "/dev/hidraw7", "/dev/hidraw7"]
    nodes.pop()
    assert connect() is None


def test_default_connector_reports_missing_permission_once(monkeypatch, caplog):
    def denied(path, device_index):
        raise PermissionError(f"no permission for {path}; install udev/60-ambiance-logitech.rules")

    monkeypatch.setattr(device_mod, "enumerate_hidraw", lambda vendor_id=None: [hidraw_node(7, 1)])
    monkeypatch.setattr(device_mod, "HidppDevice", denied)
    connect = HidrawConnector()
    with caplog.at_level(logging.WARNING, logger=device_mod.__name__):
        assert connect() is None
        assert connect() is None
    assert ["no permission" in r.getMessage() for r in caplog.records] == [True]
