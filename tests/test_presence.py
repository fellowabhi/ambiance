"""PresenceSource against fake GNOME services (Mutter IdleMonitor, ScreenSaver) on a
private dbus-daemon. No real session bus is touched."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from dbus_fast import RequestNameReply
from dbus_fast.aio import MessageBus
from dbus_fast.annotations import DBusBool, DBusUInt64
from dbus_fast.service import ServiceInterface, dbus_method, dbus_signal

from ambiance.core import PRESENCE, Presence, State
from ambiance.sources.presence import IDLE_MONITOR, SCREENSAVER, PresenceSource


class FakeIdleMonitor(ServiceInterface):
    def __init__(self, idle_ms: int) -> None:
        super().__init__(IDLE_MONITOR[2])
        self.idle_ms = idle_ms

    @dbus_method()
    def GetIdletime(self) -> DBusUInt64:
        return self.idle_ms


class FakeScreenSaver(ServiceInterface):
    def __init__(self, active: bool) -> None:
        super().__init__(SCREENSAVER[2])
        self.active = active

    @dbus_method()
    def GetActive(self) -> DBusBool:
        return self.active

    @dbus_signal()
    def ActiveChanged(self, active: bool) -> DBusBool:
        return active

    def set_active(self, active: bool) -> None:
        self.active = active
        self.ActiveChanged(active)


class RecordingState(State):
    def __init__(self) -> None:
        super().__init__()
        self.published: list[Presence] = []

    def set(self, key: str, value: Any) -> bool:
        changed = super().set(key, value)
        if changed and key == PRESENCE:
            self.published.append(value)
        return changed


class Harness:
    """A PresenceSource plus fake GNOME services, each on its own connection."""

    def __init__(self, address: str, poll_interval: float = 0.05) -> None:
        self.address = address
        self.state = RecordingState()
        self.source = PresenceSource(poll_interval, bus_address=address)
        self.task: asyncio.Task[None] | None = None
        self._services: dict[str, MessageBus] = {}

    async def start_idle_monitor(self, idle_ms: int = 0) -> FakeIdleMonitor:
        return await self._serve(IDLE_MONITOR, FakeIdleMonitor(idle_ms))

    async def start_screensaver(self, active: bool = False) -> FakeScreenSaver:
        return await self._serve(SCREENSAVER, FakeScreenSaver(active))

    async def _serve(self, where: tuple[str, str, str], iface: Any) -> Any:
        name, path, _ = where
        bus = await MessageBus(bus_address=self.address).connect()
        bus.export(path, iface)
        assert await bus.request_name(name) == RequestNameReply.PRIMARY_OWNER
        self._services[name] = bus
        return iface

    async def stop_service(self, name: str) -> None:
        await _close(self._services.pop(name))

    def start_source(self) -> None:
        self.task = asyncio.create_task(self.source.run(self.state))

    async def wait(self, predicate: Callable[[Presence | None], bool],
                   timeout: float = 3.0) -> Presence:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not predicate(value := self.state.get(PRESENCE)):
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise AssertionError(f"timed out; state[PRESENCE] = {value!r}")
            await self.state.wait_changed(self.state.version, remaining)
        return value

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task  # re-raises if the source crashed
        for bus in self._services.values():
            await _close(bus)


async def _close(bus: MessageBus) -> None:
    bus.disconnect()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(bus.wait_for_disconnect(), 1.0)


def run(address: str, scenario: Callable[[Harness], Awaitable[None]], **kwargs: Any) -> None:
    async def main() -> None:
        harness = Harness(address, **kwargs)
        try:
            await scenario(harness)
        finally:
            await harness.close()
    asyncio.run(main())


def warnings_about(caplog: pytest.LogCaptureFixture, text: str) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and text in r.getMessage()]


@pytest.fixture(autouse=True)
def _no_logged_errors(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger="ambiance")
    yield
    # caplog.records would only hold the teardown phase here.
    records = caplog.get_records("setup") + caplog.get_records("call")
    errors = [r.getMessage() for r in records if r.levelno >= logging.ERROR]
    assert not errors


def test_idle_time_and_lock_are_published(private_bus):
    async def scenario(h: Harness) -> None:
        idle = await h.start_idle_monitor(12_400)
        saver = await h.start_screensaver(active=False)
        h.start_source()
        assert await h.wait(lambda p: p is not None) == Presence(12.0, False)  # whole seconds
        idle.idle_ms = 61_600
        await h.wait(lambda p: p == Presence(62.0, False))
        saver.set_active(True)
        await h.wait(lambda p: p == Presence(62.0, True))
        idle.idle_ms = 200  # someone touched the keyboard on the lock screen
        saver.set_active(False)
        await h.wait(lambda p: p == Presence(0.0, False))
    run(private_bus.address, scenario)


def test_lock_changes_do_not_wait_for_the_next_poll(private_bus):
    async def scenario(h: Harness) -> None:
        await h.start_idle_monitor(5_000)
        saver = await h.start_screensaver(active=False)
        h.start_source()
        await h.wait(lambda p: p == Presence(5.0, False))
        saver.set_active(True)
        await h.wait(lambda p: p is not None and p.locked, timeout=1.0)
    run(private_bus.address, scenario, poll_interval=60.0)


def test_initial_lock_state_is_read_at_start(private_bus):
    async def scenario(h: Harness) -> None:
        await h.start_idle_monitor(0)
        await h.start_screensaver(active=True)
        h.start_source()
        await h.wait(lambda p: p is not None)
        assert h.state.published[0] == Presence(0.0, True)
    run(private_bus.address, scenario)


def test_missing_gnome_services_fall_back_and_log_once(private_bus, caplog):
    async def scenario(h: Harness) -> None:
        h.start_source()
        assert await h.wait(lambda p: p is not None) == Presence()
        await asyncio.sleep(0.4)  # ~8 polls
        assert not h.task.done()
        assert h.state.published == [Presence()]
        assert len(warnings_about(caplog, "org.gnome.Mutter.IdleMonitor")) == 1
        assert len(warnings_about(caplog, "org.gnome.ScreenSaver")) == 1
        # GNOME shows up later (session still starting): picked up without a restart.
        await h.start_idle_monitor(3_000)
        await h.start_screensaver(active=True)
        await h.wait(lambda p: p == Presence(3.0, True))
        assert any("available again" in r.getMessage() for r in caplog.records)
    run(private_bus.address, scenario)


def test_screensaver_restart_is_followed(private_bus, caplog):
    async def scenario(h: Harness) -> None:
        await h.start_idle_monitor(0)
        await h.start_screensaver(active=True)
        h.start_source()
        await h.wait(lambda p: p == Presence(0.0, True))
        await h.stop_service(SCREENSAVER[0])  # e.g. gnome-shell restarting
        await h.wait(lambda p: p == Presence(0.0, False))
        assert len(warnings_about(caplog, "org.gnome.ScreenSaver")) == 1
        await h.start_screensaver(active=True)
        await h.wait(lambda p: p == Presence(0.0, True))
    run(private_bus.address, scenario)


def test_active_changed_from_a_non_owner_is_ignored(private_bus):
    async def scenario(h: Harness) -> None:
        await h.start_idle_monitor(0)
        await h.start_screensaver(active=False)
        h.start_source()
        await h.wait(lambda p: p == Presence(0.0, False))
        impostor_bus = await MessageBus(bus_address=h.address).connect()
        try:
            impostor = FakeScreenSaver(active=True)
            impostor_bus.export(SCREENSAVER[1], impostor)  # same path/interface, no name
            impostor.set_active(True)
            await asyncio.sleep(0.3)
            assert h.state.get(PRESENCE) == Presence(0.0, False)
        finally:
            await _close(impostor_bus)
    run(private_bus.address, scenario)


def test_unreachable_bus_publishes_defaults_and_keeps_retrying(tmp_path, caplog):
    async def scenario(h: Harness) -> None:
        h.source.reconnect_initial_s = 0.02
        h.start_source()
        assert await h.wait(lambda p: p is not None) == Presence()
        await asyncio.sleep(0.3)
        assert not h.task.done()
        assert len(warnings_about(caplog, "session bus")) == 1
    run(f"unix:path={tmp_path}/no-such-bus", scenario)
