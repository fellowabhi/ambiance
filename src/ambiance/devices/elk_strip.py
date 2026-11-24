"""ELK-BLEDOM-family Bluetooth LE RGB strip controllers ("MELK-*", "ELK-BLEDOM", ...).

These are the controllers the "Lotus Lamp X" / "duoCo Strip" phone apps drive. Verified on
a MELK-OA10 (GATT service FFF0, write characteristic FFF3, notify FFF4):

    7e 07 83                     MELK handshake (device notifies 12 13 14), then
    7e 04 04                     second handshake frame
    7e 00 04 f0 00 01 ff 00 ef   power on
    7e 00 04 00 00 00 ff 00 ef   power off
    7e 00 05 03 RR GG BB 00 ef   set colour (whole strip)

Over BLE the controller shows one colour on the whole strip (its per-LED animations are
firmware presets only), so the device exposes a single-zone layout. Colour writes are
write-without-response: ~30/s are accepted but arrive in visible bursts, so we pace
updates at `max_rate_hz` and fade between the engine's keyframes ourselves, in linear
light (LED drive values are proportional to emitted light).

All BLE I/O runs on a private asyncio loop in a daemon thread; the engine only calls
submit(), which stores the latest colour (latest wins).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Sequence

from ambiance.color import RGB
from ambiance.core import Frame, LightDevice
from ambiance.layout import Key, Layout
from ambiance.led import LedProfile

log = logging.getLogger(__name__)

WRITE_UUIDS = ("0000fff3-0000-1000-8000-00805f9b34fb", "0000ffe1-0000-1000-8000-00805f9b34fb")
DEFAULT_PREFIXES = ("MELK-", "ELK-", "LEDBLE")

STRIP_ZONE = 0
STRIP_LAYOUT = Layout("strip", (Key("strip", "Strip", STRIP_ZONE, 0.0, 0.0, 1.0, 1.0),))

MELK_HANDSHAKE = (bytes.fromhex("7e0783"), bytes.fromhex("7e0404"))
POWER_ON = bytes.fromhex("7e0004f00001ff00ef")
POWER_OFF = bytes.fromhex("7e0004000000ff00ef")


def color_command(c: RGB) -> bytes:
    return bytes((0x7E, 0x00, 0x05, 0x03, c.r, c.g, c.b, 0x00, 0xEF))


def brightness_commands(percent: int) -> tuple[bytes, ...]:
    """Controller (hardware) brightness, 0-100. Two framings are in the wild for this
    family; a controller ignores the one it doesn't speak, so both are sent."""
    p = max(0, min(100, int(percent)))
    return (bytes((0x7E, 0x04, 0x01, p, 0xFF, 0xFF, 0xFF, 0x00, 0xEF)),
            bytes((0x7E, 0x04, 0x01, p, 0x01, 0xFF, 0xFF, 0x00, 0xEF)))


async def _wait_any(*events: asyncio.Event, timeout: float) -> None:
    tasks = [asyncio.ensure_future(e.wait()) for e in events]
    try:
        await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()


class ElkStripDevice(LightDevice):
    def __init__(self, *, address: str | None = None,
                 name_prefixes: Sequence[str] = DEFAULT_PREFIXES, max_rate_hz: float = 12.0,
                 smoothing_s: float = 0.3, release_color: RGB | None = None,
                 power_off_on_stop: bool = False, name: str = "strip",
                 scan_timeout: float = 10.0, led_profile: LedProfile | None = None,
                 hw_brightness: int = 100, brightness: float = 1.0, animate: bool = True):
        self.name = name
        self.kind = "strip"
        self.layout = STRIP_LAYOUT
        self.led_profile = led_profile  # the engine maps colours with this (None: its default)
        self.hw_brightness = hw_brightness  # controller brightness set on connect; 0 = leave it
        self.brightness = brightness  # software level, read live by the engine
        self.look = {"animate": animate}  # read live by the engine (RenderContext.options)
        self.address = address
        self.name_prefixes = tuple(name_prefixes)
        self.max_rate_hz = max(1.0, max_rate_hz)
        self.smoothing_s = max(0.0, smoothing_s)
        self.release_color = release_color
        self.power_off_on_stop = power_off_on_stop
        self.scan_timeout = scan_timeout
        self._lock = threading.Lock()
        self._target: RGB | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._release_on_stop = True
        self._connected = False
        self._device_name = ""
        self.writes = 0
        self.reconnects = 0
        self.last_error = ""

    # -- LightDevice ---------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._thread_main, name=f"ambiance-{self.name}",
                                        daemon=True)
        self._thread.start()

    def submit(self, frame: Frame) -> None:
        color = frame.get(STRIP_ZONE)
        if color is None and frame:  # frame rendered for another layout: use its average
            vals = list(frame.values())
            color = RGB(*(sum(c[i] for c in vals) // len(vals) for i in range(3)))
        if color is None:
            return
        with self._lock:
            self._target = RGB(*color)
        self._kick()

    def release(self) -> None:
        """No firmware mode to return to: fade to `release_color` if configured, else hold."""
        if self.release_color is not None:
            with self._lock:
                self._target = self.release_color
            self._kick()

    def stop(self, release: bool = True) -> None:
        self._release_on_stop = release
        self._stopping.set()
        self._kick()
        if self._thread:
            self._thread.join(timeout=8)

    def rescan(self) -> None:
        """Retry the connection now instead of after the back-off."""
        self._kick()

    def status(self) -> dict:
        return {"connected": self._connected, "device": self._device_name, "writes": self.writes,
                "reconnects": self.reconnects, "last_error": self.last_error}

    # -- worker --------------------------------------------------------------------------
    def _kick(self) -> None:
        loop, wake = self._loop, self._wake
        if loop is not None and wake is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:  # loop closed between the check and the call
                pass

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception:  # last-resort guard: never let the thread die silently
            log.exception("%s: worker crashed", self.name)

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        never_connected = asyncio.Event()
        backoff = 1.0
        while not self._stopping.is_set():
            try:
                await self._session()
                backoff = 1.0
            except Exception as e:  # bleak raises a zoo of exception types across BlueZ versions
                self.last_error = f"{type(e).__name__}: {e}"
                log.info("%s: %s; retrying in %.0fs", self.name, self.last_error, backoff)
            if self._connected:
                self.reconnects += 1
            self._connected = False
            if self._stopping.is_set():
                break
            self._wake.clear()
            await _wait_any(self._wake, never_connected, timeout=backoff)
            backoff = min(backoff * 2, 30.0)

    async def _find(self):
        from bleak import BleakScanner

        if self.address:
            return await BleakScanner.find_device_by_address(self.address, timeout=self.scan_timeout)
        return await BleakScanner.find_device_by_filter(
            lambda d, _adv: bool(d.name) and d.name.startswith(self.name_prefixes),
            timeout=self.scan_timeout)

    async def _session(self) -> None:
        from bleak import BleakClient

        dev = await self._find()
        if dev is None:
            raise ConnectionError("strip not found (powered? phone app connected to it?)")
        loop = asyncio.get_running_loop()
        gone = asyncio.Event()

        def on_disconnect(_client) -> None:
            loop.call_soon_threadsafe(gone.set)

        async with BleakClient(dev, timeout=20.0, disconnected_callback=on_disconnect) as client:
            uuid = next((u for u in WRITE_UUIDS if client.services.get_characteristic(u)), None)
            if uuid is None:
                raise ConnectionError(f"{dev.name}: no known write characteristic")

            async def write(data: bytes) -> None:
                await client.write_gatt_char(uuid, data, response=False)
                self.writes += 1

            self._device_name = dev.name or dev.address
            if (dev.name or "").startswith("MELK"):
                for frame in MELK_HANDSHAKE:
                    await write(frame)
                    await asyncio.sleep(0.4)
            await write(POWER_ON)
            if self.hw_brightness > 0:
                for frame in brightness_commands(self.hw_brightness):
                    await write(frame)
                    await asyncio.sleep(0.05)
            self._connected = True
            log.info("%s: connected to %s (%s)", self.name, self._device_name, dev.address)
            await self._pump(write, gone)
            if self._stopping.is_set() and not gone.is_set():
                if self._release_on_stop and self.release_color is not None:
                    await write(color_command(self.release_color))
                if self.power_off_on_stop:
                    await write(POWER_OFF)

    async def _pump(self, write, gone: asyncio.Event) -> None:
        """Fade from the shown colour to the latest target at max_rate_hz until stopped."""
        loop = asyncio.get_running_loop()
        period = 1.0 / self.max_rate_hz
        shown: RGB | None = None
        last_write = -1e9
        while not self._stopping.is_set() and not gone.is_set():
            self._wake.clear()  # clear BEFORE reading the target, so no submit is missed
            with self._lock:
                target = self._target
            if target is None or target == shown:
                await _wait_any(self._wake, gone, timeout=30.0)
                continue
            start = shown if shown is not None else target
            steps = max(1, round(self.smoothing_s * self.max_rate_hz)) if shown is not None else 1
            for i in range(1, steps + 1):
                if self._stopping.is_set() or gone.is_set():
                    return
                with self._lock:
                    if self._target != target:
                        break  # newer keyframe: the next pass fades from where we are
                c = start.lerp(target, i / steps)
                delay = last_write + period - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                if c != shown:
                    await write(color_command(c))
                    last_write = loop.time()
                    shown = c
