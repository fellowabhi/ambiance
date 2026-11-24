"""HID++ 2.0 transport over a Linux hidraw node (USB, LIGHTSPEED or Bluetooth).

Message (long report, the only kind Logitech BLE devices expose):

    [0x11, device_index, feature_index, function<<4 | sw_id, 16 parameter bytes]

Replies echo bytes 1..3. Errors come back as
    [0x11, device_index, 0xFF, feature_index, function<<4 | sw_id, error_code, ...]
Device-initiated notifications carry sw_id 0, so requests use sw ids 1..15.
"""

from __future__ import annotations

import collections
import errno
import logging
import os
import select
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from . import features as F

log = logging.getLogger(__name__)

REPORT_SHORT = 0x10
REPORT_LONG = 0x11
LONG_LEN = 20
DEVICE_INDEX_DIRECT = 0xFF  # directly connected (Bluetooth / USB cable)

_GONE_ERRNOS = {errno.ENODEV, errno.EIO, errno.ENXIO, errno.EPIPE, errno.ESHUTDOWN, errno.EBADF}


class HidppError(Exception):
    def __init__(self, code: int, feature_index: int, function: int):
        self.code = code
        self.feature_index = feature_index
        self.function = function
        super().__init__(f"HID++ error {F.ERRORS.get(code, hex(code))} "
                         f"(feature index {feature_index:#04x}, fn {function})")


class HidppTimeout(TimeoutError):
    pass


class DeviceGone(OSError):
    """The hidraw node went away (sleep, power-off, out of range, unplugged)."""


@dataclass(frozen=True)
class Notification:
    feature_index: int
    function: int
    data: bytes


class HidppDevice:
    """One HID++ 2.0 device on a hidraw node. Thread-safe: requests are serialised."""

    def __init__(self, path: str, device_index: int = DEVICE_INDEX_DIRECT):
        self.path = path
        self.device_index = device_index
        try:
            self._fd = os.open(path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EPERM):
                raise PermissionError(
                    f"no permission for {path}; install udev/60-ambiance-logitech.rules") from e
            raise
        self._lock = threading.RLock()
        self._sw_id = 0
        self._feature_index: dict[int, int | None] = {F.ROOT: 0}
        self.notifications: collections.deque[Notification] = collections.deque(maxlen=512)

    # -- lifecycle -------------------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1

    def __enter__(self) -> "HidppDevice":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def fileno(self) -> int:
        return self._fd

    # -- low level -------------------------------------------------------------------------
    def _next_sw_id(self) -> int:
        self._sw_id = self._sw_id % 15 + 1
        return self._sw_id

    def _write(self, msg: bytes) -> None:
        try:
            os.write(self._fd, msg)
        except OSError as e:
            if e.errno in _GONE_ERRNOS:
                raise DeviceGone(e.errno, f"{self.path}: {e.strerror}") from e
            raise

    def _read(self, timeout: float) -> bytes | None:
        try:
            ready, _, _ = select.select([self._fd], [], [], max(0.0, timeout))
            if not ready:
                return None
            return os.read(self._fd, 64)
        except BlockingIOError:
            return None
        except (OSError, ValueError) as e:
            if isinstance(e, ValueError) or e.errno in _GONE_ERRNOS:
                raise DeviceGone(getattr(e, "errno", errno.EBADF), f"{self.path}: gone") from e
            raise

    def _frame(self, feature_index: int, fnsw: int, params: bytes) -> bytes:
        if len(params) > 16:
            raise ValueError("HID++ long report carries at most 16 parameter bytes")
        return bytes((REPORT_LONG, self.device_index, feature_index, fnsw)) + params.ljust(16, b"\x00")

    def _classify(self, msg: bytes):
        """-> ('reply'|'error'|'notify', feature_index, fnsw, payload|code) or None."""
        if len(msg) < 5 or msg[0] not in (REPORT_SHORT, REPORT_LONG):
            return None  # keyboard/mouse/consumer input reports share the node
        if msg[1] != self.device_index and not (self.device_index == DEVICE_INDEX_DIRECT and msg[1] == 0):
            return None  # some BT firmwares answer requests sent to 0xFF on index 0x00
        if msg[2] in (0xFF, 0x8F):
            return ("error", msg[3], msg[4], msg[5] if len(msg) > 5 else 0)
        if msg[3] & 0x0F == 0:
            self.notifications.append(Notification(msg[2], msg[3] >> 4, bytes(msg[4:])))
            return ("notify", msg[2], msg[3], None)
        return ("reply", msg[2], msg[3], bytes(msg[4:]))

    # -- requests --------------------------------------------------------------------------
    def request(self, feature_index: int, function: int, params: bytes | Sequence[int] = b"",
                *, timeout: float = 1.5, busy_retries: int = 3, timeout_retries: int = 1) -> bytes:
        """Send one request and wait for its reply payload (16 bytes)."""
        return self.request_many([(feature_index, function, bytes(params))], window=1,
                                 timeout=timeout, busy_retries=busy_retries,
                                 timeout_retries=timeout_retries)[0]

    def request_many(self, requests: Iterable[tuple[int, int, bytes]], *, window: int = 1,
                     timeout: float = 1.5, busy_retries: int = 3,
                     timeout_retries: int = 1) -> list[bytes]:
        """Send requests keeping up to `window` in flight; returns replies in order.

        Replies are matched on (feature_index, function<<4 | sw_id), so window <= 14.
        A BUSY error re-sends that request after a short back-off. A missing reply is
        re-sent `timeout_retries` times: Logitech's x8081 guidelines note that requests
        can be lost over the air and must be repeated when the acknowledgement is missing.
        """
        reqs = [(fi, fn, bytes(p)) for fi, fn, p in requests]
        window = max(1, min(window, 14))
        results: list[bytes | None] = [None] * len(reqs)
        with self._lock:
            # key -> [index, msg, deadline, busy_count, timeout_count]
            pending: dict[tuple[int, int], list] = {}
            nxt = 0
            while nxt < len(reqs) or pending:
                while nxt < len(reqs) and len(pending) < window:
                    fi, fn, params = reqs[nxt]
                    while True:
                        fnsw = ((fn & 0x0F) << 4) | self._next_sw_id()
                        if (fi, fnsw) not in pending:
                            break
                    msg = self._frame(fi, fnsw, params)
                    self._write(msg)
                    pending[(fi, fnsw)] = [nxt, msg, time.monotonic() + timeout, 0, 0]
                    nxt += 1
                deadline = min(p[2] for p in pending.values())
                raw = self._read(deadline - time.monotonic())
                if raw is None:
                    now = time.monotonic()
                    for (fi, fnsw), p in pending.items():
                        if p[2] <= now:
                            if p[4] >= timeout_retries:
                                raise HidppTimeout(f"{self.path}: no reply to feature index "
                                                   f"{fi:#04x} fn {fnsw >> 4}")
                            p[4] += 1
                            log.debug("%s: no reply to %#04x fn %d, re-sending", self.path,
                                      fi, fnsw >> 4)
                            self._write(p[1])
                            p[2] = time.monotonic() + timeout
                    continue
                kind = self._classify(raw)
                if kind is None or kind[0] == "notify":
                    continue
                what, fi, fnsw, payload = kind
                p = pending.get((fi, fnsw))
                if p is None:
                    continue  # stale reply (e.g. from a timed-out earlier call)
                if what == "error":
                    if payload == F.ERR_BUSY and p[3] < busy_retries:
                        p[3] += 1
                        time.sleep(0.03 * p[3])
                        self._write(p[1])
                        p[2] = time.monotonic() + timeout
                        continue
                    raise HidppError(payload, fi, fnsw >> 4)
                results[p[0]] = payload
                del pending[(fi, fnsw)]
        return results  # type: ignore[return-value]

    def send_no_reply(self, feature_index: int, function: int, params: bytes | Sequence[int] = b"") -> None:
        with self._lock:
            self._write(self._frame(feature_index, ((function & 0x0F) << 4) | self._next_sw_id(),
                                    bytes(params)))

    def poll(self, timeout: float) -> list[Notification]:
        """Wait up to `timeout` for device notifications; returns those received."""
        end = time.monotonic() + timeout
        with self._lock:
            while not self.notifications:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                raw = self._read(remaining)
                if raw is not None:
                    self._classify(raw)
            out = list(self.notifications)
            self.notifications.clear()
        return out

    # -- features --------------------------------------------------------------------------
    def feature_index(self, feature_id: int) -> int | None:
        """Index of a feature (ROOT.getFeature), cached; None if unsupported."""
        if feature_id not in self._feature_index:
            reply = self.request(0x00, 0, bytes((feature_id >> 8, feature_id & 0xFF)))
            self._feature_index[feature_id] = reply[0] or None
        return self._feature_index[feature_id]

    def ping(self, data: int = 0x5A) -> tuple[int, int]:
        """ROOT.getProtocolVersion -> (major, minor)."""
        r = self.request(0x00, 1, bytes((0, 0, data)))
        if r[2] != data:
            raise HidppError(0x01, 0, 1)
        return r[0], r[1]

    def name(self) -> str:
        idx = self.feature_index(F.DEVICE_NAME)
        if idx is None:
            return ""
        length = self.request(idx, 0)[0]
        out = b""
        while len(out) < length:
            out += self.request(idx, 1, bytes((len(out),)))
        return out[:length].decode(errors="replace")

    def battery(self) -> int | None:
        idx = self.feature_index(F.UNIFIED_BATTERY)
        return None if idx is None else self.request(idx, 1)[0]
