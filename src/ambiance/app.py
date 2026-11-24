"""Daemon wiring: State + sources + director + engine + devices, with a clean shutdown.

Every long-running coroutine (each source, the engine) runs under :func:`supervise`, so a
crash is logged and the coroutine restarted with exponential backoff instead of taking
the daemon (and the keyboard's lighting) down with it. SIGINT/SIGTERM cancel everything,
then devices are stopped, by default handing the lighting back to the firmware.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections.abc import Awaitable, Callable, Sequence

from ambiance.config import Config, ConfigError, load_config
from ambiance.core import MEDIA, LightDevice, Source, State
from ambiance.director import DefaultDirector
from ambiance.engine import Engine
from ambiance.layout import LAYOUTS, Layout

log = logging.getLogger(__name__)


async def supervise(name: str, run: Callable[[], Awaitable[object]], *,
                    min_backoff: float = 1.0, max_backoff: float = 60.0,
                    healthy_after: float = 60.0,
                    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
                    clock: Callable[[], float] = time.monotonic) -> None:
    """Run ``run()`` forever: restart it after it raises or returns, until cancelled.

    The delay doubles after each quick failure (up to ``max_backoff``) so a source whose
    D-Bus service is gone doesn't spin, and resets once a run lasted ``healthy_after``.
    """
    delay = min_backoff
    while True:
        started = clock()
        error: Exception | None = None
        try:
            await run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = exc
        if clock() - started >= healthy_after:
            delay = min_backoff
        if error is not None:
            log.error("%s crashed (%s: %s); restarting in %.1f s", name, type(error).__name__,
                      error, delay, exc_info=error)
        else:
            log.warning("%s stopped unexpectedly; restarting in %.1f s", name, delay)
        await sleep(delay)
        delay = min(delay * 2.0, max_backoff)


def make_devices(cfg: Config, layout: Layout) -> list[LightDevice]:
    # Imported lazily: the device stack is optional for tests and other front-ends.
    from ambiance.devices.logitech.device import LogitechKeyboardDevice

    devices: list[LightDevice] = [
        LogitechKeyboardDevice(layout, rescan_interval=cfg.device.rescan_interval,
                               verify_interval=cfg.device.verify_interval,
                               brightness=cfg.device.brightness, animate=cfg.device.animate)]
    if cfg.strip.enabled:
        from ambiance.devices.elk_strip import ElkStripDevice

        devices.append(ElkStripDevice(
            address=cfg.strip.address or None, name_prefixes=cfg.strip.name_prefixes,
            max_rate_hz=cfg.strip.max_rate_hz, smoothing_s=cfg.strip.smoothing_s,
            release_color=cfg.strip.release_rgb(), power_off_on_stop=cfg.strip.power_off_on_exit,
            led_profile=cfg.strip.led.profile(), hw_brightness=cfg.strip.hw_brightness,
            brightness=cfg.strip.brightness, animate=cfg.strip.animate))
    if cfg.mouse.enabled:
        from ambiance.devices.logitech.zones import LogitechZoneDevice

        devices.append(LogitechZoneDevice(brightness=cfg.mouse.brightness,
                                          animate=cfg.mouse.animate))
    return devices


def make_sources(cfg: Config) -> list[Source]:
    try:
        from ambiance.sources import MprisSource, PresenceSource
    except ImportError:  # not re-exported by the package: use the modules directly
        from ambiance.sources.mpris import MprisSource  # type: ignore[attr-defined,no-redef]
        from ambiance.sources.presence import PresenceSource  # type: ignore[no-redef]
    return [MprisSource(prefer=cfg.media.prefer, ignore=cfg.media.ignore), PresenceSource()]


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> list[int]:
    installed = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop, stop, sig)
        except (NotImplementedError, RuntimeError, ValueError):
            continue  # not the main thread / unsupported platform: KeyboardInterrupt still works
        installed.append(sig)
    return installed


def _request_stop(stop: asyncio.Event, sig: int) -> None:
    if stop.is_set():
        log.info("%s received again; still shutting down", signal.Signals(sig).name)
        return
    log.info("%s received; shutting down", signal.Signals(sig).name)
    stop.set()


async def run_daemon(cfg: Config, *, devices: Sequence[LightDevice] | None = None,
                     sources: Sequence[Source] | None = None,
                     stop: asyncio.Event | None = None, handle_signals: bool = True,
                     startup_grace_s: float = 3.0, with_control: bool = False) -> None:
    """Run until SIGINT/SIGTERM (or until ``stop`` is set).

    ``devices`` / ``sources`` default to the Logitech keyboard and the MPRIS + presence
    sources; tests and other front-ends can inject their own.
    """
    loop = asyncio.get_running_loop()
    layout = LAYOUTS[cfg.device.layout]
    devices = list(devices) if devices is not None else make_devices(cfg, layout)
    sources = list(sources) if sources is not None else make_sources(cfg)
    stop = stop if stop is not None else asyncio.Event()
    state = State()
    director = DefaultDirector(cfg.director)
    engine = Engine(state, director, devices,
                    led_profile=cfg.led.profile(), tick_hz=cfg.engine.tick_hz)
    control = None
    if with_control:
        from ambiance.control import ControlService, Controller

        control = ControlService(Controller(cfg, state, engine, director, devices, sources,
                                            stop.set))
        await control.start()  # raises AlreadyRunning before we touch any device
    installed = _install_signal_handlers(loop, stop) if handle_signals else []
    log.info("ambiance starting: layout %s, %d device(s), sources: %s", layout.name,
             len(devices), ", ".join(s.name for s in sources) or "none")

    for dev in devices:
        try:
            dev.start()
        except Exception:
            log.exception("device %s failed to start", dev.name)
    tasks = [asyncio.create_task(supervise(f"source {src.name}", lambda src=src: src.run(state)),
                                 name=f"source:{src.name}") for src in sources]
    if control is not None:
        tasks.append(asyncio.create_task(supervise("control", control.publish_forever),
                                         name="control"))
    try:
        # Give sources a moment to publish their first state: an engine that decides before
        # MPRIS has reported sees "no media" and hands the keyboard back to its firmware
        # for a second, only to grab it again.
        deadline = loop.time() + startup_grace_s
        while MEDIA not in state.snapshot() and not stop.is_set():
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await state.wait_changed(state.version, timeout=remaining)
        tasks.append(asyncio.create_task(supervise("engine", engine.run), name="engine"))
        await stop.wait()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for sig in installed:
            loop.remove_signal_handler(sig)
        if control is not None:
            control.close()
        release = cfg.device.release_on_exit
        for dev in devices:
            try:
                # stop() joins the device's I/O thread (and may send HID++ release
                # requests, ~50 ms each), so keep it off the event loop.
                await asyncio.to_thread(dev.stop, release=release)
            except Exception:
                log.exception("device %s failed to stop cleanly", dev.name)
        log.info("stopped (%s); engine stats: %s",
                 "lighting released" if release else "lighting left as is", engine.stats)


def _setup_logging(verbose: bool) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    elif verbose:
        root.setLevel(logging.DEBUG)


def main_run(config_path: str | None, verbose: bool) -> int:
    """Entry point for ``ambiance run``: 0 on a clean stop, 2 on a bad config."""
    _setup_logging(verbose)
    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        log.error("%s", e)
        return 2
    from ambiance.control import AlreadyRunning

    try:
        asyncio.run(run_daemon(cfg, with_control=True))
    except KeyboardInterrupt:  # only without signal handlers (e.g. not the main thread)
        pass
    except AlreadyRunning as e:
        log.error("%s", e)
        return 3
    except ImportError as e:
        log.error("cannot start: %s", e)
        return 1
    return 0
