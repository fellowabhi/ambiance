"""Threaded LightDevice for Logitech per-key RGB keyboards (verified on the G515 LS TKL).

Why a worker thread: every HID++ request is a blocking ~47 ms round trip over Bluetooth
LE and pipelining does not help (~21 writes/s in total), so one full per-key frame takes
0.4-0.8 s. The asyncio engine must never wait for that. It only drops its newest frame
into a one-slot mailbox (submit); the worker paints whatever is newest when it is free,
so a slow keyboard skips stale frames instead of falling further and further behind.

The worker also owns the keyboard's lifecycle, because the firmware forgets host mode and
the picture after deep sleep, a power cycle or a Bluetooth reconnect (which also gives it
a new /dev/hidrawN):

* discovery by vendor id and capabilities, never by path, retried every rescan_interval;
* claiming lazily, on the first frame after start() or release(), then a black base and a
  full repaint;
* diff painting against what the keyboard acknowledged, to spend as few writes as possible;
* activity notifications: while the host holds control the firmware never dims or sleeps
  the LEDs, so the host must learn when the user is away (on_activity);
* a periodic host-mode check, because waking from sleep can silently drop host mode.

Nothing here writes flash: claims and frames are RAM-only (see rgb_keyboard), and "off"
is a black frame, never a power-mode command.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ambiance.color import BLACK, RGB
from ambiance.core import Frame, LightDevice
from ambiance.devices.hidraw import HidrawDevice, enumerate_hidraw, release_claim, try_claim
from ambiance.layout import Layout

from .hidpp import DEVICE_INDEX_DIRECT, HidppDevice, HidppError, Notification
from .rgb_keyboard import LogitechRgbKeyboard, UnsupportedDevice, encode_frame

log = logging.getLogger(__name__)

LOGITECH_VENDOR_ID = 0x046D
CLAIM_OWNER = "logitech-keyboard"  # ambiance.devices.hidraw node ownership
HOST_MODE = 3  # RGB_EFFECTS GetSWControl mode while the host owns the LEDs (0 = onboard)


# ---- what the worker needs from a keyboard ----------------------------------------------------

class HidppTransport(Protocol):
    """The part of HidppDevice the worker uses directly."""

    path: str

    def poll(self, timeout: float) -> list[Notification]: ...

    def close(self) -> None: ...


class RgbKeyboard(Protocol):
    """The lighting calls the worker makes (LogitechRgbKeyboard).

    paint() must change exactly the zones in `colors`, plus every other zone when `fill`
    is given, and return the number of HID++ writes it took.
    """

    @property
    def zones(self) -> frozenset[int]: ...

    def claim(self) -> None: ...

    def release(self) -> None: ...

    def set_static(self, color: RGB) -> None: ...

    def paint(self, colors: Mapping[int, RGB], fill: RGB | None = None) -> int: ...

    def sw_control(self) -> tuple[int, int]: ...

    def is_activity_event(self, n: Notification) -> bool | None: ...


@dataclass
class KeyboardSession:
    """One opened keyboard: the HID++ transport and the lighting API on top of it."""

    dev: HidppTransport
    kb: RgbKeyboard

    @property
    def path(self) -> str:
        return str(getattr(self.dev, "path", "?"))

    def close(self) -> None:
        try:
            self.dev.close()
        except OSError as e:
            log.debug("closing %s: %s", self.path, e)
        finally:
            release_claim(self.path, CLAIM_OWNER)


class HidrawConnector:
    """Default connector: the first Logitech HID++ node with per-key RGB, or None.

    It runs again on every reconnect: Bluetooth gives the keyboard a new /dev/hidrawN each
    time, so it is found by vendor id and capabilities, never by path. Nodes that answered
    but lack per-key lighting (mice, receivers) are remembered while their node exists, so
    a missing keyboard does not mean HID++ traffic to other devices on every scan.
    """

    def __init__(self, vendor_id: int = LOGITECH_VENDOR_ID) -> None:
        self.vendor_id = vendor_id
        self._not_keyboards: set[tuple[str, int, str]] = set()
        self._denied: set[str] = set()

    def __call__(self) -> KeyboardSession | None:
        candidates = [d for d in enumerate_hidraw(self.vendor_id) if d.has_hidpp_long()]
        self._not_keyboards &= {self._key(d) for d in candidates}
        self._denied &= {d.path for d in candidates}
        for info in candidates:
            if self._key(info) in self._not_keyboards:
                continue
            session = self._open(info)
            if session is not None:
                return session
        return None

    @staticmethod
    def _key(info: HidrawDevice) -> tuple[str, int, str]:
        return info.path, info.product_id, info.uniq

    def _open(self, info: HidrawDevice) -> KeyboardSession | None:
        if not try_claim(info.path, CLAIM_OWNER):
            return None  # another driver is using this node; look again next scan
        session = self._probe(info)
        if session is None:
            release_claim(info.path, CLAIM_OWNER)
        return session

    def _probe(self, info: HidrawDevice) -> KeyboardSession | None:
        try:
            dev = HidppDevice(info.path, DEVICE_INDEX_DIRECT)
        except PermissionError as e:
            if info.path not in self._denied:  # actionable, but don't repeat it every scan
                self._denied.add(info.path)
                log.warning("%s", e)
            return None
        except OSError as e:
            log.debug("%s: cannot open: %s", info, e)
            return None
        try:
            dev.ping()
            return KeyboardSession(dev, LogitechRgbKeyboard(dev))
        except UnsupportedDevice as e:
            log.debug("%s", e)
            self._not_keyboards.add(self._key(info))
        except (HidppError, OSError) as e:
            log.debug("%s: probe failed: %s", info, e)
        dev.close()
        return None


# ---- the device -------------------------------------------------------------------------------

@dataclass
class _Link:
    """Worker-only state of one open session; a fresh one on every (re)connect."""

    session: KeyboardSession
    host_mode: bool | None = None   # None: unknown (a crashed run may have left it claimed)
    claimed: bool = False           # claim + black base done: frames can be painted
    painted: dict[int, RGB] = field(default_factory=dict)  # as acknowledged by the keyboard
    painted_gen: int | None = None  # mailbox generation `painted` is up to date with
    needs_full: bool = True         # the next paint must be a full frame
    post_claim: bool = False        # the next paint is the first one after a claim
    quiet_until: float = 0.0        # ignore activity events before this (monotonic)
    next_verify: float = math.inf   # next host-mode check (monotonic)
    failures: int = 0               # consecutive HID++ errors

    @property
    def kb(self) -> RgbKeyboard:
        return self.session.kb


@dataclass
class _Status:
    connected: bool = False
    claimed: bool = False
    path: str | None = None
    frames_painted: int = 0
    writes: int = 0                 # HID++ writes reported by paint()
    reconnects: int = 0             # sessions lost to errors
    last_error: str | None = None
    user_active: bool = True


class LogitechKeyboardDevice(LightDevice):
    """LightDevice for one Logitech per-key RGB keyboard, driven by its own worker thread.

    submit(), release(), stop(), status() and `connected` may be called from any thread
    (normally the asyncio engine) and never wait for the keyboard; only the worker thread
    talks HID++. `on_activity(active)` runs ON THE WORKER THREAD whenever the keyboard
    reports the user going idle (False) or coming back (True): keep it short and hand it
    to the event loop, e.g. with loop.call_soon_threadsafe.
    """

    # Tunables; class attributes so a caller (or a test) can adjust a single instance.
    poll_interval: float = 0.1      # s; dev.poll() can't be woken by submit(): bounds latency
    claim_quiet_s: float = 1.5      # s; a claim makes the firmware send a spurious idle/active pair
    full_repaint_ratio: float = 0.6  # repaint the whole frame when more of it changed than this
    backoff_base: float = 0.5       # s; wait after the first HID++ error, doubled per retry
    max_backoff: float = 5.0        # s
    max_hidpp_failures: int = 3     # consecutive HID++ errors that count as a lost keyboard
    join_timeout: float = 3.0       # s; how long stop() waits for the worker

    def __init__(self, layout: Layout, *,
                 connector: Callable[[], KeyboardSession | None] | None = None,
                 rescan_interval: float = 2.0, verify_interval: float = 30.0,
                 on_activity: Callable[[bool], None] | None = None,
                 name: str = "g515", brightness: float = 1.0, animate: bool = True) -> None:
        self.layout = layout
        self.name = name
        self.kind = "keyboard"
        self.brightness = brightness          # read live by the engine
        self.look = {"animate": animate}      # read live by the engine (RenderContext.options)
        self.rescan_interval = rescan_interval
        self.verify_interval = verify_interval
        self._connector = connector if connector is not None else HidrawConnector()
        self._on_activity = on_activity
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        # Guarded by _cond: the mailbox, the caller's intent and what status() reports.
        self._desired: Frame | None = None
        self._generation = 0
        self._want_control = False
        self._release_pending = False
        self._stopping = False
        self._rescan_now = False
        self._release_on_stop = True
        self._st = _Status()
        # Worker thread only.
        self._link: _Link | None = None
        self._user_active = True  # presumed: claims happen while the computer is in use
        self._waiting_logged = False

    # ---- LightDevice API (any thread) ---------------------------------------------------------
    def rescan(self) -> None:
        """Look for the keyboard now instead of at the next rescan tick."""
        with self._cond:
            self._rescan_now = True
            self._cond.notify_all()

    def start(self) -> None:
        """Start the worker. It connects right away but claims the LEDs only on submit()."""
        with self._cond:
            if self._stopping:
                raise RuntimeError(f"{self.name}: device was stopped")
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=f"ambiance-{self.name}",
                                                daemon=True)
                self._thread.start()

    def submit(self, frame: Frame) -> None:
        """Make `frame` the one to show; replaces any frame not painted yet (latest wins).

        A frame is the whole picture (LED drive values). Zones the keyboard lacks are
        ignored; keyboard zones the frame leaves out are don't-care (a full repaint gives
        them the frame's most common colour).
        """
        frame = dict(frame)
        with self._cond:
            if self._stopping or (self._want_control and frame == self._desired):
                return
            self._desired = frame
            self._generation += 1
            self._want_control = True
            self._release_pending = False  # a frame after release() means "take them back"
            self._cond.notify_all()

    def release(self) -> None:
        """Hand the LEDs back to the keyboard's own lighting until the next submit()."""
        with self._cond:
            if self._stopping:
                return
            self._want_control = False
            self._release_pending = True
            self._cond.notify_all()

    def stop(self, release: bool = True) -> None:
        """Stop the worker; with `release`, hand the LEDs back first if we hold them.

        Idempotent and safe before start() or when no keyboard was ever found. Blocks up to
        join_timeout while an in-flight HID++ exchange finishes. stop(release=False) leaves
        the last frame lit, and the firmware will not dim or sleep it (battery).
        """
        with self._cond:
            if not self._stopping:
                self._stopping = True
                self._release_on_stop = release
            self._cond.notify_all()
            thread = self._thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(self.join_timeout)
        if thread.is_alive():
            log.warning("%s: worker still busy after %.1f s; abandoning it (daemon thread)",
                        self.name, self.join_timeout)

    @property
    def connected(self) -> bool:
        with self._cond:
            return self._st.connected

    def status(self) -> dict[str, Any]:
        with self._cond:
            running = self._thread is not None and self._thread.is_alive()
            return {"name": self.name, "running": running,
                    "frames_submitted": self._generation, **asdict(self._st)}

    # ---- worker thread ------------------------------------------------------------------------
    def _run(self) -> None:
        log.debug("%s: worker started", self.name)
        try:
            while not self._should_stop():
                try:
                    if self._link is None:
                        self._discover()
                    else:
                        self._step(self._link)
                except Exception as e:  # a bug rather than the keyboard: log, don't spin, go on
                    log.exception("%s: unexpected worker error", self.name)
                    if self._link is not None:
                        self._drop(self._link, e)
                    else:
                        self._note_error(e)
                    self._sleep(self.rescan_interval)
        finally:
            self._shutdown()
            log.debug("%s: worker stopped", self.name)

    def _discover(self) -> None:
        try:
            session = self._connector()
        except Exception as e:
            log.warning("%s: keyboard discovery failed: %s", self.name, _describe(e))
            self._note_error(e)
            session = None
        if session is None:
            if not self._waiting_logged:
                log.info("%s: waiting for the keyboard (rescanning every %g s)",
                         self.name, self.rescan_interval)
                self._waiting_logged = True
            self._sleep(self.rescan_interval)
            return
        self._waiting_logged = False
        self._attach(session)

    def _attach(self, session: KeyboardSession) -> None:
        link = _Link(session)
        self._link = link
        with self._cond:
            self._st.connected, self._st.path = True, session.path
        zones = session.kb.zones
        log.info("%s: connected to %s (%d lighting zones)", self.name, session.path, len(zones))
        missing = self.layout.zones - zones
        if missing:
            log.debug("%s: layout zones the keyboard lacks (ignored): %s",
                      self.name, sorted(missing))
        if not self._user_active:
            self._set_user_active(True, link)  # a Bluetooth keyboard reconnects on a key press

    def _step(self, link: _Link) -> None:
        try:
            if self._act(link):
                link.failures = 0
                # Notifications that arrived during the exchange wait in the device's queue;
                # poll(0) hands them over without blocking.
                self._handle_notifications(link, link.session.dev.poll(0))
        except HidppError as e:
            self._on_hidpp_error(link, e)
        except OSError as e:  # DeviceGone, HidppTimeout, other I/O errors: the keyboard is gone
            self._drop(link, e)

    def _act(self, link: _Link) -> bool:
        """Do the single most urgent thing; False if there was nothing but events to wait for."""
        with self._cond:
            release, want = self._release_pending, self._want_control
            frame, gen = self._desired, self._generation
        if release:
            self._release(link)
        elif want and not link.claimed:
            self._claim(link)
        elif link.claimed and time.monotonic() >= link.next_verify:
            self._verify(link)
        elif want and frame is not None and gen != link.painted_gen:
            self._paint(link, frame, gen)
        else:
            self._idle(link)
            return False
        return True

    def _claim(self, link: _Link) -> None:
        link.host_mode = True  # from the first request on, the keyboard may be ours
        link.kb.claim()
        link.quiet_until = time.monotonic() + self.claim_quiet_s
        # Per-key paint sits on the static-effect layer: a black base hides whatever the
        # onboard effect was doing in zones the frame doesn't cover.
        link.kb.set_static(BLACK)
        link.claimed = link.post_claim = link.needs_full = True
        link.painted, link.painted_gen = {}, None
        link.next_verify = time.monotonic() + self.verify_interval
        with self._cond:
            self._st.claimed = True
        log.info("%s: took over the lighting on %s", self.name, link.session.path)

    def _paint(self, link: _Link, frame: Frame, gen: int) -> None:
        try:
            target = {z: RGB(*c).clamped() for z, c in frame.items() if z in link.kb.zones}
        except (TypeError, ValueError) as e:
            log.error("%s: dropping malformed frame: %s", self.name, e)
            link.painted_gen = gen
            return
        changed = {z: c for z, c in target.items() if link.painted.get(z) != c}
        if not target or not (changed or link.needs_full):
            link.painted_gen = gen  # nothing visible to do
            return
        full = link.needs_full or len(changed) > self.full_repaint_ratio * len(target)
        link.needs_full = True  # until acknowledged: a failed write leaves the keyboard unknown
        if full:
            writes = link.kb.paint(target, _background(target))
            link.painted = dict(target)
        else:
            writes = link.kb.paint(changed)
            link.painted.update(changed)
        link.needs_full = False
        link.painted_gen = gen
        if link.post_claim:
            # The spurious idle/active pair may have arrived while this (long) first frame
            # was written; it is only read now, so the quiet window starts over from here.
            link.post_claim = False
            link.quiet_until = time.monotonic() + self.claim_quiet_s
        with self._cond:
            self._st.frames_painted += 1
            self._st.writes += int(writes or 0)
        log.debug("%s: frame %d painted (%s, %d zones, %s writes)", self.name, gen,
                  "full" if full else "diff", len(target if full else changed), writes)

    def _verify(self, link: _Link) -> None:
        mode, _flags = link.kb.sw_control()
        link.next_verify = time.monotonic() + self.verify_interval
        if mode != HOST_MODE:
            log.info("%s: keyboard went back to its own lighting (mode %d); taking it back",
                     self.name, mode)
            self._lose_claim(link)

    def _release(self, link: _Link) -> None:
        if link.host_mode is not False:  # also undoes a claim left behind by a crashed run
            link.kb.release()
            log.info("%s: lighting handed back to the keyboard", self.name)
        link.host_mode = False
        self._lose_claim(link)
        with self._cond:
            self._release_pending = False

    def _lose_claim(self, link: _Link) -> None:
        link.claimed = False
        link.next_verify = math.inf
        with self._cond:
            self._st.claimed = False

    def _idle(self, link: _Link) -> None:
        wait = min(self.poll_interval, max(0.0, link.next_verify - time.monotonic()))
        self._handle_notifications(link, link.session.dev.poll(wait))

    def _handle_notifications(self, link: _Link, notes: list[Notification]) -> None:
        for n in notes:
            active = link.kb.is_activity_event(n)
            if active is None:
                continue
            if time.monotonic() < link.quiet_until:
                log.debug("%s: ignoring %s event right after the claim",
                          self.name, "active" if active else "idle")
                continue
            self._set_user_active(active, link)

    def _set_user_active(self, active: bool, link: _Link) -> None:
        if active == self._user_active:
            return
        self._user_active = active
        with self._cond:
            self._st.user_active = active
        log.info("%s: user %s", self.name, "is back" if active else "went idle")
        if active and link.claimed:
            link.next_verify = 0.0  # waking up can silently drop host mode: check right away
        if self._on_activity is not None:
            try:
                self._on_activity(active)
            except Exception:
                log.exception("%s: on_activity callback failed", self.name)

    def _on_hidpp_error(self, link: _Link, err: HidppError) -> None:
        link.failures += 1
        self._note_error(err)
        if link.failures >= self.max_hidpp_failures:
            log.warning("%s: %d HID++ errors in a row (%s); reconnecting",
                        self.name, link.failures, err)
            self._drop(link, err)
            self._sleep(self.rescan_interval)  # the node is still there: don't hammer it
            return
        delay = min(self.max_backoff, self.backoff_base * 2 ** (link.failures - 1))
        log.warning("%s: %s; retrying in %.2f s", self.name, err, delay)
        self._sleep(delay)

    def _drop(self, link: _Link, err: BaseException) -> None:
        log.info("%s: lost %s: %s", self.name, link.session.path, _describe(err))
        link.session.close()
        self._link = None
        with self._cond:
            self._st.connected = self._st.claimed = False
            self._st.path = None
            self._st.reconnects += 1
            self._st.last_error = _describe(err)

    def _shutdown(self) -> None:
        link, self._link = self._link, None
        if link is not None:
            with self._cond:
                release = self._release_on_stop and (
                    link.host_mode is True or (self._release_pending and link.host_mode is None))
            if release:
                try:
                    link.kb.release()
                    log.info("%s: lighting handed back to the keyboard", self.name)
                except Exception as e:  # best effort: we are going away either way
                    log.warning("%s: could not hand the lighting back: %s",
                                self.name, _describe(e))
            link.session.close()
        with self._cond:
            self._st.connected = self._st.claimed = False
            self._st.path = None

    def _should_stop(self) -> bool:
        with self._cond:
            return self._stopping

    def _sleep(self, seconds: float) -> None:
        """Wait, but wake at once for stop() or rescan()."""
        with self._cond:
            self._cond.wait_for(lambda: self._stopping or self._rescan_now, seconds)
            self._rescan_now = False

    def _note_error(self, err: BaseException) -> None:
        with self._cond:
            self._st.last_error = _describe(err)


def _background(colors: Mapping[int, RGB]) -> RGB | None:
    """Fill for a full repaint: the most common colour, if one range write saves writes.

    A fill repaints every zone (also those the frame leaves out, giving them a defined
    colour), so the worker only uses it for full frames.
    """
    if not colors:
        return None
    color = Counter(colors.values()).most_common(1)[0][0]
    return color if len(encode_frame(colors, color)) < len(encode_frame(colors)) else None


def _describe(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}"


__all__ = ["LogitechKeyboardDevice", "KeyboardSession", "HidrawConnector",
           "HidppTransport", "RgbKeyboard", "LOGITECH_VENDOR_ID"]
