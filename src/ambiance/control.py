"""The running daemon's D-Bus control interface (session bus), used by the ambiance app.

    bus name  dev.ambiance.Daemon      (owning it = "the daemon is running"; one instance)
    object    /dev/ambiance/Daemon
    interface dev.ambiance.Daemon1
        GetStatus() -> s               JSON: media, devices (+ live colours), options, ...
        SetOptions(s patch) -> s       JSON patch of options; applied live and saved
        Rescan()                       look for devices right away
        Quit()                         fade out, hand lights back, exit
        signal StatusChanged(s)        same JSON as GetStatus, at most a few times a second

JSON keeps the interface trivially extensible: new options/devices don't change signatures.
Note: no `from __future__ import annotations` here, dbus-fast reads the signature strings
from the annotations at runtime.
"""

import asyncio
import dataclasses
import json
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

from dbus_fast import BusType, NameFlag, RequestNameReply
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, method, signal

from ambiance import __version__
from ambiance.colorspace import linear_to_srgb, srgb_to_linear
from ambiance.config import Config, save_config
from ambiance.core import MEDIA, PRESENCE, LightDevice, State

log = logging.getLogger(__name__)

BUS_NAME = "dev.ambiance.Daemon"
OBJECT_PATH = "/dev/ambiance/Daemon"
INTERFACE = "dev.ambiance.Daemon1"
DEVICE_SECTIONS = {"keyboard": "device", "strip": "strip", "mouse": "mouse"}  # kind -> [section]


class AlreadyRunning(RuntimeError):
    pass


def _hex(color, gain: float = 1.0) -> str:
    """Display colour: sRGB scaled by a light gain in linear space (what the eye sees)."""
    lin = srgb_to_linear([c / 255.0 for c in color]) * max(0.0, min(1.0, gain))
    r, g, b = (int(round(v * 255)) for v in linear_to_srgb(lin))
    return f"#{r:02x}{g:02x}{b:02x}"


class Controller:
    """Daemon-side logic behind the interface (plain Python, unit-testable)."""

    def __init__(self, cfg: Config, state: State, engine, director, devices: Sequence[LightDevice],
                 sources: Sequence[Any], request_stop: Callable[[], None], *,
                 config_path=None, persist: bool = True) -> None:
        self.cfg = cfg
        self.state, self.engine, self.director = state, engine, director
        self.devices, self.sources = list(devices), list(sources)
        self.request_stop = request_stop
        self.config_path = config_path
        self.persist = persist
        self._rev = 0

    # -- status ---------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        media, presence = self.state.get(MEDIA), self.state.get(PRESENCE)
        stats = self.engine.stats
        return {
            "pid": os.getpid(), "version": __version__,
            "scene": stats.scene, "level": stats.level, "released": self.engine.released,
            "media": None if media is None else {
                "player": media.player, "status": media.status, "title": media.title,
                "artists": list(media.artists), "album": media.album, "art_url": media.art_url,
                "art_path": _art_path(media.art_url),
                "palette": [s.rgb.hex for s in getattr(media.palette, "swatches", ()) or ()],
            },
            "presence": None if presence is None else {
                "idle_seconds": presence.idle_seconds, "locked": presence.locked},
            "devices": [self._device_status(d) for d in self.devices],
            "options": self.options(),
        }

    def _device_status(self, dev: LightDevice) -> dict[str, Any]:
        info = {}
        try:
            info = dev.status() if hasattr(dev, "status") else {}
        except Exception:  # status is best effort, never let it break the interface
            pass
        colors = {}
        try:
            brightness = float(getattr(dev, "brightness", 1.0))
            for zone, px in self.engine.shown(dev.layout).items():
                colors[str(zone)] = _hex(px.color, px.gain * brightness)
        except (KeyError, AttributeError):
            pass
        return {
            "name": dev.name, "kind": getattr(dev, "kind", dev.name),
            "connected": bool(dev.connected), "layout": dev.layout.name,
            "detail": info.get("device") or info.get("path") or "",
            "brightness": float(getattr(dev, "brightness", 1.0)),
            "animate": bool((getattr(dev, "look", None) or {}).get("animate", True)),
            "colors": colors,
        }

    def options(self) -> dict[str, Any]:
        album, d = self.cfg.director.album, self.cfg.director
        opts: dict[str, Any] = {
            "album": {"style": album.style, "drift_period_s": album.drift_period_s,
                      "strip_cycle_s": album.strip_cycle_s},
            "behaviour": {"paused_level": d.paused_level, "idle_off_s": d.idle_off_s,
                          "ignore": list(self.cfg.media.ignore)},
        }
        for dev in self.devices:
            kind = getattr(dev, "kind", dev.name)
            opts[kind] = {"brightness": float(getattr(dev, "brightness", 1.0)),
                          "animate": bool((getattr(dev, "look", None) or {}).get("animate", True))}
        return opts

    # -- changes --------------------------------------------------------------------------------
    def set_options(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a (partial) options patch; all-or-nothing validation, then save."""
        try:
            cfg = self._patched_config(patch)
        except (ValueError, TypeError, KeyError) as e:
            return {"ok": False, "error": str(e)}
        self.cfg = cfg
        self.director.cfg = cfg.director
        for src in self.sources:
            if hasattr(src, "ignore"):
                src.ignore = tuple(cfg.media.ignore)
        for dev in self.devices:
            kind = getattr(dev, "kind", dev.name)
            section = DEVICE_SECTIONS.get(kind)
            if section is not None:
                sec = getattr(cfg, section)
                dev.brightness = sec.brightness
                dev.look = {**(getattr(dev, "look", None) or {}), "animate": sec.animate}
        self.engine.refresh()
        self._rev += 1
        self.state.set("_options_rev", self._rev)  # wakes the engine
        if self.persist:
            try:
                save_config(cfg, self.config_path)
            except OSError as e:
                log.warning("could not save settings: %s", e)
        return {"ok": True, "options": self.options()}

    def _patched_config(self, patch: dict[str, Any]) -> Config:
        return apply_patch(self.cfg, patch)

    def rescan(self) -> None:
        for dev in self.devices:
            fn = getattr(dev, "rescan", None)
            if callable(fn):
                fn()


def apply_patch(cfg: Config, patch: dict[str, Any]) -> Config:
    """Return cfg with an options patch applied (validated by the config dataclasses)."""
    for kind, section in DEVICE_SECTIONS.items():
        if kind in patch:
            p = patch[kind]
            sec = getattr(cfg, section)
            changes = {k: (float(v) if k == "brightness" else bool(v))
                       for k, v in p.items() if k in ("brightness", "animate")}
            cfg = dataclasses.replace(cfg, **{section: dataclasses.replace(sec, **changes)})
    if "album" in patch:
        p = patch["album"]
        changes = {}
        if "style" in p:
            changes["style"] = str(p["style"])
        for k in ("drift_period_s", "strip_cycle_s"):
            if k in p:
                changes[k] = float(p[k])
        album = dataclasses.replace(cfg.director.album, **changes)
        cfg = dataclasses.replace(cfg, director=dataclasses.replace(cfg.director, album=album))
    if "behaviour" in patch:
        p = patch["behaviour"]
        dchanges = {k: float(p[k]) for k in ("paused_level", "idle_off_s") if k in p}
        cfg = dataclasses.replace(cfg, director=dataclasses.replace(cfg.director, **dchanges))
        if "ignore" in p:
            ignore = tuple(str(x) for x in p["ignore"])
            cfg = dataclasses.replace(cfg, media=dataclasses.replace(cfg.media, ignore=ignore))
    return cfg


def _art_path(url: str) -> str:
    if not url:
        return ""
    if url.startswith("file://"):
        from urllib.parse import unquote, urlparse

        return unquote(urlparse(url).path)
    import hashlib

    from ambiance.art import cache_dir

    path = cache_dir() / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".img")
    return str(path) if path.exists() else ""


class _Interface(ServiceInterface):
    def __init__(self, ctl: Controller) -> None:
        super().__init__(INTERFACE)
        self._ctl = ctl
        self._last = ""

    @method()
    def GetStatus(self) -> "s":
        return json.dumps(self._ctl.status())

    @method()
    def SetOptions(self, patch: "s") -> "s":
        try:
            data = json.loads(patch)
            if not isinstance(data, dict):
                raise ValueError("options patch must be a JSON object")
        except ValueError as e:
            return json.dumps({"ok": False, "error": str(e)})
        return json.dumps(self._ctl.set_options(data))

    @method()
    def Rescan(self):
        self._ctl.rescan()

    @method()
    def Quit(self):
        self._ctl.request_stop()

    @signal()
    def StatusChanged(self) -> "s":
        return self._last

    def publish(self, status_json: str) -> None:
        self._last = status_json
        self.StatusChanged()


class ControlService:
    """Owns the bus name and publishes status changes; start() fails if already running."""

    def __init__(self, ctl: Controller, *, bus_address: str | None = None,
                 interval_s: float = 0.4) -> None:
        self.ctl = ctl
        self.bus_address = bus_address
        self.interval_s = interval_s
        self.bus: MessageBus | None = None
        self._iface = _Interface(ctl)

    async def start(self) -> None:
        bus = (await MessageBus(bus_address=self.bus_address).connect() if self.bus_address
               else await MessageBus(bus_type=BusType.SESSION).connect())
        bus.export(OBJECT_PATH, self._iface)
        reply = await bus.request_name(BUS_NAME, NameFlag.DO_NOT_QUEUE)
        if reply not in (RequestNameReply.PRIMARY_OWNER, RequestNameReply.ALREADY_OWNER):
            bus.disconnect()
            raise AlreadyRunning("ambiance is already running (bus name taken)")
        self.bus = bus
        log.info("control interface on the session bus as %s", BUS_NAME)

    async def publish_forever(self) -> None:
        last = ""
        while True:
            try:
                status = json.dumps(self.ctl.status(), sort_keys=True)
                if status != last:
                    last = status
                    self._iface.publish(status)
            except Exception:  # keep publishing even if one snapshot fails
                log.exception("status publishing failed")
            await asyncio.sleep(self.interval_s)

    def close(self) -> None:
        if self.bus is not None:
            self.bus.disconnect()
            self.bus = None
