"""Tests for the HID++ 2.0 transport (HidppDevice) against a scripted fake keyboard.

The fake keyboard sits on the far end of an AF_UNIX SOCK_SEQPACKET socketpair. Like a
hidraw node, that socket keeps report boundaries, so the real select()/os.read()/
os.write() paths run unchanged. HidppDevice.__init__ itself runs too: inside hidpp,
`os` is swapped for a shim whose open() returns the socket's fd. Nothing ever opens
/dev/hidraw*.

The fake answers synchronously from inside the host's write(). By the time the host
blocks in select(), the reply is already queued, together with any noise a test injects
ahead of it. On hardware that answer would take one ~47 ms BLE round trip.
"""

from __future__ import annotations

import errno
import os
import socket
import threading
import time
from collections.abc import Callable, Iterable

import pytest

from ambiance.devices.logitech import features as F
from ambiance.devices.logitech import hidpp
from ambiance.devices.logitech.hidpp import (DeviceGone, HidppDevice, HidppError, HidppTimeout,
                                             Notification)

FAKE_PATH = "fake-hidraw-node"  # not a filesystem path; the shim refuses to open anything else
SHORT = 0.05                    # seconds; reply timeout for tests that expect silence

Responder = Callable[[bytes], Iterable[bytes]]


# -- report builders (device -> host) ---------------------------------------------------------------

def reply(req: bytes, payload: bytes = b"", *, device_index: int | None = None) -> bytes:
    """A reply echoes request bytes 1..3: device index, feature index, function<<4|sw_id."""
    index = req[1] if device_index is None else device_index
    return bytes((hidpp.REPORT_LONG, index, req[2], req[3])) + payload.ljust(16, b"\x00")


def echo(req: bytes) -> list[bytes]:
    """Default fake behaviour: reply with the request's own parameters as the payload."""
    return [reply(req, req[4:])]


def silent(_req: bytes) -> list[bytes]:
    return []


def error(req: bytes, code: int) -> bytes:
    """HID++ 2.0 error: [0x11, index, 0xFF, feature_index, function<<4|sw_id, code, ...]."""
    return bytes((hidpp.REPORT_LONG, req[1], 0xFF, req[2], req[3], code)).ljust(hidpp.LONG_LEN,
                                                                               b"\x00")


def notification(feature_index: int, function: int, data: bytes = b"",
                 device_index: int = hidpp.DEVICE_INDEX_DIRECT) -> bytes:
    """Device-initiated event: sw_id 0."""
    return bytes((hidpp.REPORT_LONG, device_index, feature_index, function << 4)) + data.ljust(
        16, b"\x00")


def keyboard_report(req: bytes) -> bytes:
    """17-byte BT keyboard input report (id 0x18) whose next bytes mimic req's header.

    The HID++ node also carries key presses. If the transport ignored the report ID,
    this report would look exactly like the reply it is waiting for.
    """
    return bytes((0x18, req[1], req[2], req[3])) + b"\xee" * 13


# -- the fake keyboard and the plumbing -------------------------------------------------------------

class FakeKeyboard:
    """Device end of the socketpair: records every request and answers via `respond`."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.received: list[bytes] = []
        self.respond: Responder = echo

    def inject(self, *reports: bytes) -> None:
        """Queue unsolicited reports for the host (notifications, key presses, ...)."""
        for report in reports:
            self.sock.send(report)

    def service(self) -> None:
        while True:
            try:
                req = self.sock.recv(64)
            except BlockingIOError:
                return
            self.received.append(req)
            for out in self.respond(req):
                self.sock.send(out)


class OsShim:
    """Stands in for `os` inside hidpp: real os, except open/read/write are instrumented."""

    def __init__(self, link: Link) -> None:
        self._link = link
        self.events: list[str] = []  # "W" per write, "R" per read: shows pipelining depth
        self.open_error: OSError | None = None
        self.read_error: OSError | None = None
        self.write_error: OSError | None = None

    def __getattr__(self, name: str):
        return getattr(os, name)

    def open(self, path: str, flags: int, mode: int = 0o777) -> int:
        if path != FAKE_PATH:
            raise AssertionError(f"test tried to open a real node: {path!r}")
        if self.open_error is not None:
            raise self.open_error
        return self._link.take_host_fd()

    def write(self, fd: int, data: bytes) -> int:
        if self.write_error is not None:
            raise self.write_error
        n = os.write(fd, data)
        self.events.append("W")
        self._link.keyboard.service()
        return n

    def read(self, fd: int, n: int) -> bytes:
        if self.read_error is not None:
            raise self.read_error
        data = os.read(fd, n)
        self.events.append("R")
        return data


class Link:
    """One socketpair + shim; hands its host fd to exactly one HidppDevice."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        host, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self._host_fd = host.detach()
        os.set_blocking(self._host_fd, False)  # HidppDevice opens hidraw with O_NONBLOCK
        peer.setblocking(False)
        self._fd_taken = False
        self.keyboard = FakeKeyboard(peer)
        self.os = OsShim(self)
        self.device: HidppDevice | None = None
        monkeypatch.setattr(hidpp, "os", self.os)

    def take_host_fd(self) -> int:
        assert not self._fd_taken, "one HidppDevice per link: two would share (and close) one fd"
        self._fd_taken = True
        return self._host_fd

    def open(self, device_index: int = hidpp.DEVICE_INDEX_DIRECT) -> HidppDevice:
        self.device = HidppDevice(FAKE_PATH, device_index)
        return self.device

    def close(self) -> None:
        if self.device is not None:
            self.device.close()           # the device owns (and closes) the host fd
        elif not self._fd_taken:
            os.close(self._host_fd)
        self.keyboard.sock.close()


@pytest.fixture
def link(monkeypatch: pytest.MonkeyPatch):
    lnk = Link(monkeypatch)
    yield lnk
    lnk.close()


@pytest.fixture
def dev(link: Link) -> HidppDevice:
    return link.open()


def max_in_flight(events: list[str]) -> int:
    """Largest number of writes not yet matched by reads (valid when every read is a reply)."""
    depth = peak = 0
    for event in events:
        depth += 1 if event == "W" else -1
        peak = max(peak, depth)
    return peak


# -- framing and basics ---------------------------------------------------------------------------

def test_request_sends_long_report_and_returns_16_byte_payload(link, dev):
    out = dev.request(0x07, 2, [1, 2, 3])
    (req,) = link.keyboard.received
    assert len(req) == hidpp.LONG_LEN
    assert req[:3] == bytes((hidpp.REPORT_LONG, hidpp.DEVICE_INDEX_DIRECT, 0x07))
    assert req[3] >> 4 == 2 and 1 <= req[3] & 0x0F <= 15
    assert req[4:] == bytes((1, 2, 3)) + bytes(13)
    assert out == bytes((1, 2, 3)) + bytes(13)


def test_sw_ids_cycle_1_to_15_and_never_use_0(link, dev):
    """sw_id 0 marks device notifications, so requests must never use it."""
    for _ in range(40):
        dev.request(0x01, 0)
    sw_ids = [req[3] & 0x0F for req in link.keyboard.received]
    assert set(sw_ids) == set(range(1, 16))
    assert all(a != b for a, b in zip(sw_ids, sw_ids[1:]))


def test_more_than_16_parameter_bytes_is_rejected_before_sending(link, dev):
    with pytest.raises(ValueError):
        dev.request(0x01, 0, bytes(17))
    assert link.keyboard.received == []


def test_send_no_reply_writes_once_and_does_not_wait(link, dev):
    link.keyboard.respond = silent
    t0 = time.monotonic()
    dev.send_no_reply(0x0C, 7, b"\x00")
    assert time.monotonic() - t0 < 0.5  # far below request()'s 1.5 s reply timeout
    (req,) = link.keyboard.received
    assert req[2] == 0x0C and req[3] >> 4 == 7
    assert link.os.events == ["W"]


# -- reply matching ---------------------------------------------------------------------------------

def test_keyboard_input_reports_are_not_mistaken_for_replies(link, dev):
    link.keyboard.respond = lambda req: [keyboard_report(req), keyboard_report(req),
                                         reply(req, b"real")]
    assert dev.request(0x05, 1).startswith(b"real")
    assert not dev.notifications


def test_stale_and_foreign_replies_are_ignored(link, dev):
    def respond(req: bytes) -> list[bytes]:
        fi, fnsw = req[2], req[3]
        other_sw = (fnsw & 0xF0) | ((fnsw & 0x0F) % 15 + 1)
        return [
            bytes((0x11, 0xFF, fi, other_sw)) + b"wrong-sw-id".ljust(16, b"\x00"),
            bytes((0x11, 0xFF, fi + 1, fnsw)) + b"wrong-feature".ljust(16, b"\x00"),
            bytes((0x11, 0xFF, fi, fnsw ^ 0x10)) + b"wrong-function".ljust(16, b"\x00"),
            error(bytes((0x11, 0xFF, fi, other_sw)), 0x02),   # error for someone else
            reply(req, b"other-device", device_index=0x01),
            b"\x11\xff",                                       # runt report
            reply(req, b"real"),
        ]

    link.keyboard.respond = respond
    assert dev.request(0x05, 1).startswith(b"real")


def test_late_reply_to_a_timed_out_request_is_dropped(link, dev):
    link.keyboard.respond = silent
    with pytest.raises(HidppTimeout):
        dev.request(0x05, 1, timeout=SHORT, timeout_retries=0)
    late = link.keyboard.received[-1]
    link.keyboard.respond = lambda req: [reply(late, b"late"), reply(req, b"fresh")]
    assert dev.request(0x05, 1).startswith(b"fresh")


def test_reply_on_device_index_0_accepted_for_direct_device(link, dev):
    """Some BT firmwares answer requests sent to index 0xFF on index 0x00."""
    link.keyboard.respond = lambda req: [reply(req, b"idx0", device_index=0x00)]
    assert dev.request(0x05, 1).startswith(b"idx0")


def test_reply_on_device_index_0_ignored_for_a_receiver_slot(link):
    dev = link.open(device_index=0x01)
    link.keyboard.respond = lambda req: [reply(req, b"idx0", device_index=0x00),
                                         reply(req, b"idx1")]
    assert dev.request(0x05, 1).startswith(b"idx1")
    assert link.keyboard.received[0][1] == 0x01


# -- errors -------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("code", "name"), [(0x02, "InvalidArgument"), (0x05, "LogitechInternal"),
                                            (0x09, "Unsupported"), (0x42, "0x42")])
def test_hidpp20_error_raises_hidpp_error_with_code(link, dev, code, name):
    link.keyboard.respond = lambda req: [error(req, code)]
    with pytest.raises(HidppError) as exc:
        dev.request(0x0B, 3, b"\x01")
    assert (exc.value.code, exc.value.feature_index, exc.value.function) == (code, 0x0B, 3)
    assert name in str(exc.value)
    assert len(link.keyboard.received) == 1  # errors other than BUSY are not retried


def test_hidpp10_style_error_also_raises(link, dev):
    link.keyboard.respond = lambda req: [bytes((hidpp.REPORT_SHORT, req[1], 0x8F, req[2], req[3],
                                                0x07, 0x00))]
    with pytest.raises(HidppError) as exc:
        dev.request(0x02, 1)
    assert exc.value.code == 0x07


def test_busy_is_retried_with_the_same_report_then_succeeds(link, dev):
    def respond(req: bytes) -> list[bytes]:
        busy = len(link.keyboard.received) <= 2
        return [error(req, F.ERR_BUSY)] if busy else [reply(req, b"ok")]

    link.keyboard.respond = respond
    assert dev.request(0x0C, 5, b"\x09").startswith(b"ok")
    first, second, third = link.keyboard.received
    assert first == second == third  # identical re-send: same sw_id, same parameters


def test_busy_gives_up_after_busy_retries(link, dev):
    link.keyboard.respond = lambda req: [error(req, F.ERR_BUSY)]
    with pytest.raises(HidppError) as exc:
        dev.request(0x0C, 5, busy_retries=1)
    assert exc.value.code == F.ERR_BUSY
    assert len(link.keyboard.received) == 2


# -- timeouts ---------------------------------------------------------------------------------------

def test_no_reply_raises_hidpp_timeout(link, dev):
    link.keyboard.respond = silent
    t0 = time.monotonic()
    with pytest.raises(HidppTimeout):
        dev.request(0x05, 1, timeout=SHORT, timeout_retries=0)
    assert SHORT * 0.8 <= time.monotonic() - t0 < 1.0
    assert len(link.keyboard.received) == 1
    assert issubclass(HidppTimeout, TimeoutError)


def test_missing_reply_is_resent_timeout_retries_times(link, dev):
    """Requests can be lost over the air; the same report is repeated before giving up."""
    link.keyboard.respond = silent
    t0 = time.monotonic()
    with pytest.raises(HidppTimeout):
        dev.request(0x05, 1, timeout=SHORT, timeout_retries=2)
    assert 3 * SHORT * 0.8 <= time.monotonic() - t0 < 2.0
    assert len(link.keyboard.received) == 3
    assert len(set(link.keyboard.received)) == 1


def test_lost_request_is_recovered_by_the_resend(link, dev):
    link.keyboard.respond = lambda req: [] if len(link.keyboard.received) == 1 else [
        reply(req, b"second try")]
    assert dev.request(0x05, 1, timeout=SHORT).startswith(b"second try")  # default: 1 retry
    assert len(link.keyboard.received) == 2


def test_noise_does_not_postpone_the_timeout(link, dev):
    link.keyboard.respond = lambda req: [keyboard_report(req), notification(0x0B, 1)]
    t0 = time.monotonic()
    with pytest.raises(HidppTimeout):
        dev.request(0x05, 1, timeout=SHORT, timeout_retries=0)
    assert time.monotonic() - t0 < 1.0


# -- notifications ----------------------------------------------------------------------------------

def test_notifications_during_a_request_are_queued_for_poll(link, dev):
    link.keyboard.respond = lambda req: [notification(0x0B, 1, b"\x00"), reply(req, b"ok"),
                                         notification(0x0B, 1, b"\x01")]
    assert dev.request(0x05, 1).startswith(b"ok")
    t0 = time.monotonic()
    assert dev.poll(1.0) == [Notification(0x0B, 1, b"\x00" + bytes(15))]  # already queued
    assert time.monotonic() - t0 < 0.5
    assert dev.poll(1.0) == [Notification(0x0B, 1, b"\x01" + bytes(15))]  # still on the wire
    assert not dev.notifications


def test_poll_returns_notifications_and_skips_other_traffic(link, dev):
    req = bytes((0x11, 0xFF, 0x05, 0x13)) + bytes(16)
    link.keyboard.inject(keyboard_report(req), reply(req, b"stray"),
                         notification(0x0B, 1, b"\x01\x02", device_index=0x02),  # other device
                         notification(0x0B, 1, b"\x01\x02"))
    t0 = time.monotonic()
    assert dev.poll(1.0) == [Notification(0x0B, 1, b"\x01\x02" + bytes(14))]
    assert time.monotonic() - t0 < 0.5


def test_poll_times_out_empty(dev):
    t0 = time.monotonic()
    assert dev.poll(SHORT) == []
    assert SHORT * 0.8 <= time.monotonic() - t0 < 1.0


# -- pipelining -------------------------------------------------------------------------------------

def reversed_batches(batch: int, total: int) -> Responder:
    """Hold requests until `batch` are outstanding (or all `total` arrived); answer newest first.

    Also asserts that every request in flight is distinguishable by
    (feature_index, function<<4|sw_id): otherwise replies could not be matched.
    """
    held: list[bytes] = []
    seen = 0

    def respond(req: bytes) -> list[bytes]:
        nonlocal seen
        seen += 1
        held.append(req)
        if len(held) < batch and seen < total:
            return []
        assert len({r[2:4] for r in held}) == len(held), "two in-flight requests share a key"
        out = [reply(r, r[4:]) for r in reversed(held)]
        held.clear()
        return out

    return respond


def test_request_many_matches_out_of_order_replies(link, dev):
    n = 10
    link.keyboard.respond = reversed_batches(4, n)
    # Same feature and function for all: only the sw_id tells the replies apart.
    out = dev.request_many([(0x0A, 1, bytes((i,))) for i in range(n)], window=4)
    assert [r[0] for r in out] == list(range(n))
    assert max_in_flight(link.os.events) == 4


def test_window_is_clamped_to_14_in_flight(link, dev):
    """15 sw ids exist, so more than 14 identical requests in flight could not be matched."""
    n = 30
    link.keyboard.respond = reversed_batches(14, n)
    out = dev.request_many([(0x0A, 1, bytes((i,))) for i in range(n)], window=100)
    assert [r[0] for r in out] == list(range(n))
    assert max_in_flight(link.os.events) == 14


def test_window_1_is_strict_request_reply(link, dev):
    out = dev.request_many([(0x0A, 1, bytes((i,))) for i in range(5)])
    assert [r[0] for r in out] == list(range(5))
    assert link.os.events == ["W", "R"] * 5


def test_request_many_empty(dev):
    assert dev.request_many([]) == []


def test_concurrent_requests_are_serialised(link, dev):
    """Two threads must never read each other's replies (which would look stale -> timeout)."""
    results: dict[int, list[bytes]] = {}

    def worker(tag: int) -> None:
        results[tag] = [dev.request(0x0A, 1, bytes((tag, i))) for i in range(25)]

    threads = [threading.Thread(target=worker, args=(t,)) for t in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    for tag in (1, 2):
        assert [r[:2] for r in results[tag]] == [bytes((tag, i)) for i in range(25)]


# -- device gone --------------------------------------------------------------------------------------

def test_enodev_on_write_raises_device_gone(link, dev):
    link.os.write_error = OSError(errno.ENODEV, "No such device")
    with pytest.raises(DeviceGone) as exc:
        dev.request(0x05, 1)
    assert isinstance(exc.value, OSError)
    assert exc.value.errno == errno.ENODEV


@pytest.mark.parametrize("err", [errno.ENODEV, errno.EIO])
def test_gone_errno_on_read_raises_device_gone(link, dev, err):
    link.os.read_error = OSError(err, os.strerror(err))
    with pytest.raises(DeviceGone):
        dev.request(0x05, 1)


def test_other_write_errors_are_not_device_gone(link, dev):
    link.os.write_error = OSError(errno.EINVAL, "Invalid argument")
    with pytest.raises(OSError) as exc:
        dev.request(0x05, 1)
    assert not isinstance(exc.value, DeviceGone)


def test_use_after_close_raises_device_gone(link, dev):
    dev.close()
    assert dev.fileno() == -1
    with pytest.raises(DeviceGone):
        dev.request(0x05, 1)
    with pytest.raises(DeviceGone):
        dev.poll(0.01)
    dev.close()  # idempotent


def test_permission_error_points_at_the_udev_rule(link):
    link.os.open_error = PermissionError(errno.EACCES, "Permission denied")
    with pytest.raises(PermissionError, match="udev"):
        link.open()


def test_missing_node_error_propagates_unchanged(link):
    link.os.open_error = FileNotFoundError(errno.ENOENT, "No such file or directory")
    with pytest.raises(FileNotFoundError):
        link.open()


# -- feature helpers built on request() -------------------------------------------------------------

def _root_get_feature(table: dict[int, int]) -> Responder:
    def respond(req: bytes) -> list[bytes]:
        assert req[2] == 0x00 and req[3] >> 4 == 0, "expected ROOT.getFeature"
        return [reply(req, bytes((table.get(req[4] << 8 | req[5], 0), 0, 0)))]
    return respond


def test_feature_index_is_looked_up_once_and_cached(link, dev):
    link.keyboard.respond = _root_get_feature({F.RGB_EFFECTS: 0x0B})
    assert dev.feature_index(F.RGB_EFFECTS) == 0x0B
    assert dev.feature_index(F.RGB_EFFECTS) == 0x0B
    assert dev.feature_index(F.ONBOARD_PROFILES) is None   # index 0 means unsupported
    assert dev.feature_index(F.ONBOARD_PROFILES) is None
    assert dev.feature_index(F.ROOT) == 0                  # always index 0, never asked
    assert [req[4:6] for req in link.keyboard.received] == [b"\x80\x71", b"\x81\x00"]


def test_ping_returns_protocol_version_and_checks_the_echo(link, dev):
    link.keyboard.respond = lambda req: [reply(req, bytes((4, 2, req[6])))]
    assert dev.ping() == (4, 2)
    link.keyboard.respond = lambda req: [reply(req, bytes((4, 2, 0x00)))]
    with pytest.raises(HidppError):
        dev.ping(0x5A)


def test_name_is_assembled_from_16_byte_chunks(link, dev):
    name = b"G515 LS TKL Bluetooth Kbd"  # 25 bytes -> two getDeviceName calls

    def respond(req: bytes) -> list[bytes]:
        if req[2] == 0x00:
            return _root_get_feature({F.DEVICE_NAME: 0x03})(req)
        if req[3] >> 4 == 0:
            return [reply(req, bytes((len(name),)))]
        return [reply(req, name[req[4]:req[4] + 16])]

    link.keyboard.respond = respond
    assert dev.name() == name.decode()


def test_name_and_battery_without_the_features(link, dev):
    link.keyboard.respond = _root_get_feature({})
    assert dev.name() == ""
    assert dev.battery() is None
