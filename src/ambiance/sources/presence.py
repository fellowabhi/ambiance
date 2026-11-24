"""Is someone at the computer? Idle time and screen lock from GNOME Shell over D-Bus.

* Idle time: org.gnome.Mutter.IdleMonitor GetIdletime (ms since the last input), polled.
  The monitor only offers threshold watches, not a stream of values, and one tiny call
  every couple of seconds is cheaper than juggling watches for every threshold a rule
  might care about.
* Lock: org.gnome.ScreenSaver GetActive once, then the ActiveChanged signal. GNOME reports
  the shield (lock screen or idle blank) as "active", which is exactly "nobody is looking".

Other desktops provide neither; then Presence() ("someone is here, unlocked") keeps being
published, so presence-based rules degrade to "always present" instead of failing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from dbus_fast import BusType, Message, MessageFlag, MessageType
from dbus_fast.aio import MessageBus

from ambiance.core import PRESENCE, Presence, Source, State

log = logging.getLogger(__name__)

DBUS = ("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus")
IDLE_MONITOR = ("org.gnome.Mutter.IdleMonitor", "/org/gnome/Mutter/IdleMonitor/Core",
                "org.gnome.Mutter.IdleMonitor")
SCREENSAVER = ("org.gnome.ScreenSaver", "/org/gnome/ScreenSaver", "org.gnome.ScreenSaver")

# sender= with a well-known name makes the bus daemon forward only ActiveChanged signals
# sent by the connection that currently owns org.gnome.ScreenSaver (no spoofing).
ACTIVE_CHANGED_RULE = (
    f"type='signal',sender='{SCREENSAVER[0]}',interface='{SCREENSAVER[2]}',"
    f"member='ActiveChanged',path='{SCREENSAVER[1]}'")
SCREENSAVER_OWNER_RULE = (
    "type='signal',sender='org.freedesktop.DBus',interface='org.freedesktop.DBus',"
    f"member='NameOwnerChanged',path='/org/freedesktop/DBus',arg0='{SCREENSAVER[0]}'")


async def _try_call(bus: MessageBus, destination: str, path: str, interface: str, member: str,
                    signature: str = "", body: Sequence[Any] = (), *, timeout: float,
                    autostart: bool = False) -> Message | None:
    """A method call that cannot hang (dbus-fast has no call timeout of its own).

    Returns the reply, or None for an error reply / no reply in time, which for the
    optional desktop services means "not available". autostart=False sets NO_AUTO_START
    so polling never makes the bus daemon try to D-Bus-activate a missing service.
    """
    msg = Message(destination=destination, path=path, interface=interface, member=member,
                  signature=signature, body=list(body),
                  flags=MessageFlag.NONE if autostart else MessageFlag.NO_AUTOSTART)
    try:
        reply = await asyncio.wait_for(bus.call(msg), timeout)
    except TimeoutError:
        log.debug("%s.%s on %s: no reply within %.1f s", interface, member, destination, timeout)
        return None
    if reply.message_type is MessageType.ERROR:
        log.debug("%s.%s on %s: %s %s", interface, member, destination, reply.error_name,
                  reply.body)
        return None
    return reply


class _Backoff:
    """Reconnect delays: initial, 2x, 4x, ... capped at maximum."""

    def __init__(self, initial: float, maximum: float) -> None:
        self._initial = initial
        self._maximum = maximum
        self._next = initial

    def reset(self) -> None:
        self._next = self._initial

    def next(self) -> float:
        delay = self._next
        self._next = min(self._next * 2, self._maximum)
        return delay


class _Availability:
    """Log once when an optional service is missing, and once when it comes back."""

    def __init__(self) -> None:
        self._missing: set[str] = set()

    def missing(self, service: str, consequence: str) -> None:
        if service not in self._missing:
            self._missing.add(service)
            log.warning("presence: %s is not available (%s)", service, consequence)

    def present(self, service: str) -> None:
        if service in self._missing:
            self._missing.discard(service)
            log.info("presence: %s is available again", service)


class _PresenceWatch:
    """Presence over one bus connection: a poll loop plus the ActiveChanged signal."""

    def __init__(self, bus: MessageBus, source: PresenceSource, state: State) -> None:
        self._bus = bus
        self._source = source
        self._state = state
        self._locked = False
        self._lock_signals = 0           # ActiveChanged count, to spot stale GetActive replies
        self._lock_query_due = True      # ask GetActive at the next tick
        self._may_activate_saver = True  # only the very first GetActive may D-Bus-activate
        self._wake = asyncio.Event()

    async def run(self) -> None:
        """Publish presence until the connection drops (returns or raises)."""
        self._bus.add_message_handler(self._on_message)
        for rule in (ACTIVE_CHANGED_RULE, SCREENSAVER_OWNER_RULE):
            if await self._call(*DBUS, "AddMatch", "s", [rule]) is None:
                raise RuntimeError(f"AddMatch refused: {rule}")
        # The poller handles its own failures (missing services are not errors), so a
        # plain task is enough; it is cancelled when the connection goes away.
        poller = asyncio.create_task(self._poll_loop(), name="presence-poll")
        try:
            await self._bus.wait_for_disconnect()
        finally:
            poller.cancel()

    async def _call(self, *args: Any, autostart: bool = False) -> Message | None:
        return await _try_call(self._bus, *args, timeout=self._source.call_timeout_s,
                               autostart=autostart)

    def _on_message(self, msg: Message) -> None:
        """dbus-fast hook for every incoming message. Must return None: a truthy result
        marks the message handled, which would swallow the replies to our own calls."""
        if msg.message_type is not MessageType.SIGNAL:
            return None
        if (msg.member == "ActiveChanged" and msg.interface == SCREENSAVER[2]
                and msg.path == SCREENSAVER[1] and msg.body):
            self._lock_signals += 1
            self._locked = bool(msg.body[0])
            self._wake.set()  # publish a lock change now, not at the next poll
        elif (msg.member == "NameOwnerChanged" and msg.sender == DBUS[0]
              and msg.body and msg.body[0] == SCREENSAVER[0]):
            if msg.body[2]:  # (re)started, e.g. gnome-shell restart: ask it directly
                self._lock_query_due = True
            else:            # gone: nobody can tell us about the lock any more
                self._lock_query_due = False
                self._locked = False
                self._source.availability.missing("org.gnome.ScreenSaver",
                                                  "screen lock not detected")
            self._wake.set()
        return None

    async def _poll_loop(self) -> None:
        while True:
            if self._lock_query_due:
                self._lock_query_due = False
                await self._read_lock()
            idle = await self._read_idle()
            self._state.set(PRESENCE, Presence(idle_seconds=idle, locked=self._locked))
            try:
                await asyncio.wait_for(self._wake.wait(), self._source.poll_interval)
            except TimeoutError:
                pass
            self._wake.clear()

    async def _read_lock(self) -> None:
        # Only the very first query may D-Bus-activate the service (GNOME starts its
        # org.gnome.ScreenSaver helper on demand); re-activating something that just went
        # away could loop, so afterwards we wait for NameOwnerChanged instead.
        autostart, self._may_activate_saver = self._may_activate_saver, False
        signals_before = self._lock_signals
        reply = await self._call(*SCREENSAVER, "GetActive", autostart=autostart)
        if reply is None:
            self._source.availability.missing("org.gnome.ScreenSaver", "screen lock not detected")
            return
        self._source.availability.present("org.gnome.ScreenSaver")
        if self._lock_signals == signals_before:  # else a newer ActiveChanged already told us
            self._locked = bool(reply.body[0])

    async def _read_idle(self) -> float:
        reply = await self._call(*IDLE_MONITOR, "GetIdletime")
        if reply is None:
            self._source.availability.missing("org.gnome.Mutter.IdleMonitor",
                                              "idle time reported as 0")
            return 0.0
        self._source.availability.present("org.gnome.Mutter.IdleMonitor")
        # Whole seconds: State.set notifies on every change, sub-second jitter is noise.
        return float(round(int(reply.body[0]) / 1000))


async def _disconnect(bus: MessageBus, timeout: float = 1.0) -> None:
    """Close the connection and give dbus-fast a moment to finalize (closes the socket)."""
    bus.disconnect()
    try:
        await asyncio.wait_for(bus.wait_for_disconnect(), timeout)
    except Exception:  # already broken, or slow: either way nothing left for us to do
        pass


class PresenceSource(Source):
    """Publishes state[PRESENCE] = Presence(idle_seconds, locked).

    One State.set per poll tick (State only notifies on change); a lock change starts a
    tick early so it is published at once. Without GNOME, or without a session bus at all,
    it keeps publishing Presence() and retries in the background instead of failing.
    """

    name = "presence"

    # Tunables (instance-overridable, e.g. by tests).
    call_timeout_s = 2.0
    reconnect_initial_s = 0.5
    reconnect_max_s = 30.0

    def __init__(self, poll_interval: float = 2.0, *, bus_address: str | None = None) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.poll_interval = poll_interval
        self._bus_address = bus_address
        self.availability = _Availability()

    async def run(self, state: State) -> None:
        backoff = _Backoff(self.reconnect_initial_s, self.reconnect_max_s)
        while True:
            try:
                bus = await MessageBus(bus_address=self._bus_address,
                                       bus_type=BusType.SESSION).connect()
            except Exception as e:  # no session bus (headless, misconfigured address, ...)
                self.availability.missing("the session bus", f"{type(e).__name__}: {e}")
            else:
                self.availability.present("the session bus")
                backoff.reset()
                try:
                    await _PresenceWatch(bus, self, state).run()
                    log.warning("presence: session bus connection closed; reconnecting")
                except Exception as e:
                    log.warning("presence: session bus connection lost (%s: %s); reconnecting",
                                type(e).__name__, e)
                finally:
                    await _disconnect(bus)
            state.set(PRESENCE, Presence())  # unknown again: assume someone is here
            await asyncio.sleep(backoff.next())
