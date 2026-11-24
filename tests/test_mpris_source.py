"""MprisSource against fake MPRIS players on a private dbus-daemon.

No real session bus, network or hardware: each fake player is a dbus-fast ServiceInterface
on its own connection (like a separate app) and cover-art analysis is a recording stub.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
import pytest
from dbus_fast import Message, PropertyAccess, RequestNameReply, Variant
from dbus_fast.aio import MessageBus
from dbus_fast.annotations import DBusDict, DBusStr
from dbus_fast.service import ServiceInterface, dbus_property

from ambiance.color import RGB
from ambiance.core import MEDIA, MediaState, State
from ambiance.palette import Palette, Swatch
from ambiance.sources.mpris import MPRIS_PATH, MPRIS_PREFIX, PLAYER_IFACE, MprisSource


def art(n: int) -> str:
    return f"https://art.example/{n}.jpg"


def metadata(n: int, *, art_url: str | None = None, track_id: str | None = None
             ) -> dict[str, Variant]:
    """Spotify-shaped Metadata for track number n."""
    return {
        "mpris:trackid": Variant("s", track_id or f"/com/spotify/track/{n:04d}"),
        "xesam:title": Variant("s", f"Song {n}"),
        "xesam:artist": Variant("as", [f"Artist {n}"]),
        "xesam:album": Variant("s", f"Album {n}"),
        "mpris:artUrl": Variant("s", art(n) if art_url is None else art_url),
        "mpris:length": Variant("t", 180_000_000),
    }


class FakePlayer(ServiceInterface):
    """org.mpris.MediaPlayer2.Player with just the properties MprisSource reads."""

    def __init__(self, status: str, track: int) -> None:
        super().__init__(PLAYER_IFACE)
        self._status = status
        self._metadata = metadata(track)

    @dbus_property(access=PropertyAccess.READ)
    def PlaybackStatus(self) -> DBusStr:
        return self._status

    @dbus_property(access=PropertyAccess.READ)
    def Metadata(self) -> DBusDict:
        return self._metadata

    def change(self, *, status: str | None = None,
               metadata: dict[str, Variant] | None = None) -> None:
        """Update properties and emit PropertiesChanged with them, as players do."""
        changed: dict[str, Any] = {}
        if status is not None:
            self._status = status
            changed["PlaybackStatus"] = status
        if metadata is not None:
            self._metadata = metadata
            changed["Metadata"] = metadata
        self.emit_properties_changed(changed)


def palette_for(url: str) -> Palette:
    r, g, b = hashlib.sha256(url.encode()).digest()[:3]
    return Palette((Swatch(RGB(r, g, b), (0.6, 0.1, 0.1), 1.0, 1.0),))


class StubAnalyzer:
    """Stands in for fetch + palette extraction; runs in MprisSource's worker thread."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail: set[str] = set()
        self.gate = threading.Event()  # cleared -> analysis blocks until set (max 5 s)
        self.gate.set()
        self._lock = threading.Lock()

    def __call__(self, url: str) -> tuple[Palette, np.ndarray]:
        with self._lock:
            self.calls.append(url)
        self.gate.wait(timeout=5)
        if url in self.fail:
            raise OSError(f"cannot fetch {url}")
        return palette_for(url), np.full((24, 24, 3), 0.5)


class RecordingState(State):
    """State that remembers every MEDIA value that was actually published."""

    def __init__(self) -> None:
        super().__init__()
        self.published: list[MediaState | None] = []

    def set(self, key: str, value: Any) -> bool:
        changed = super().set(key, value)
        if changed and key == MEDIA:
            self.published.append(value)
        return changed


class Harness:
    """A MprisSource plus fake player apps on the private bus, torn down after the test."""

    def __init__(self, address: str, **source_kwargs: Any) -> None:
        self.address = address
        self.state = RecordingState()
        self.analyze = StubAnalyzer()
        self.source = MprisSource(analyze=self.analyze, bus_address=address, **source_kwargs)
        self.source.debounce_s = 0.05  # keep the suite fast; the burst test restores 0.15
        self.task: asyncio.Task[None] | None = None
        self._players: dict[str, MessageBus] = {}

    async def start_player(self, name: str = "spotify", *, status: str = "Playing",
                           track: int = 1) -> FakePlayer:
        bus = await MessageBus(bus_address=self.address).connect()
        player = FakePlayer(status, track)
        bus.export(MPRIS_PATH, player)  # export first: the name announces a ready player
        assert await bus.request_name(MPRIS_PREFIX + name) == RequestNameReply.PRIMARY_OWNER
        self._players[name] = bus
        return player

    async def quit_player(self, name: str) -> None:
        await _close(self._players.pop(name))

    async def forget_players(self) -> None:
        """After the daemon died: reap the dead connections (retrieves their EOFError)."""
        players, self._players = self._players, {}
        for bus in players.values():
            await _close(bus)

    def start_source(self) -> None:
        self.task = asyncio.create_task(self.source.run(self.state))

    async def stop_source(self) -> None:
        if self.task is not None:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task  # re-raises if the source crashed
            self.task = None

    async def wait(self, predicate: Callable[[Any], bool], timeout: float = 3.0) -> Any:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not predicate(media := self.state.get(MEDIA)):
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise AssertionError(f"timed out; state[MEDIA] = {media!r}")
            await self.state.wait_changed(self.state.version, remaining)
        return media

    async def close(self) -> None:
        self.analyze.gate.set()  # never leave a worker thread blocked
        await self.stop_source()
        for bus in self._players.values():
            await _close(bus)


async def _close(bus: MessageBus) -> None:
    bus.disconnect()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(bus.wait_for_disconnect(), 1.0)


def track(n: int, status: str = "Playing", player: str = "spotify") -> Callable[[Any], bool]:
    """Predicate: MEDIA shows track n with this status from this player."""
    return lambda m: (m is not None and m.title == f"Song {n}" and m.status == status
                      and m.player == player)


def run(address: str, scenario: Callable[[Harness], Awaitable[None]], **source_kwargs: Any) -> None:
    async def main() -> None:
        harness = Harness(address, **source_kwargs)
        try:
            await scenario(harness)
        finally:
            await harness.close()
    asyncio.run(main())


SETTLE_S = 0.3  # long enough for any stray (debounced) publication to show up


@pytest.fixture(autouse=True)
def _no_logged_errors(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger="ambiance")
    yield
    # caplog.records would only hold the teardown phase here.
    records = caplog.get_records("setup") + caplog.get_records("call")
    errors = [r.getMessage() for r in records if r.levelno >= logging.ERROR]
    assert not errors


def test_initial_pickup_publishes_track_with_palette_and_thumbnail(private_bus):
    async def scenario(h: Harness) -> None:
        await h.start_player("spotify", track=1)
        h.start_source()
        media = await h.wait(track(1))
        assert media.track_id == "/com/spotify/track/0001"
        assert media.artists == ("Artist 1",) and media.album == "Album 1"
        assert media.art_url == art(1)
        assert media.palette == palette_for(art(1))
        assert media.thumbnail.shape == (24, 24, 3)
        await asyncio.sleep(SETTLE_S)
        assert h.state.published == [media]  # one publication, palette included
        assert h.analyze.calls == [art(1)]
    run(private_bus.address, scenario)


def test_player_appearing_later_is_picked_up(private_bus):
    async def scenario(h: Harness) -> None:
        h.start_source()
        await h.wait(lambda m: m is None and h.state.published == [None])
        await h.start_player("spotify", track=3)
        media = await h.wait(track(3))
        assert media.palette == palette_for(art(3))
    run(private_bus.address, scenario)


def test_track_change_publishes_once_with_new_palette(private_bus):
    async def scenario(h: Harness) -> None:
        player = await h.start_player("spotify", track=1)
        h.start_source()
        await h.wait(track(1))
        player.change(metadata=metadata(2))
        media = await h.wait(track(2))
        assert media.palette == palette_for(art(2))
        await asyncio.sleep(SETTLE_S)
        assert [m.title for m in h.state.published] == ["Song 1", "Song 2"]
        assert h.analyze.calls == [art(1), art(2)]
    run(private_bus.address, scenario)


def test_burst_of_property_changes_is_coalesced(private_bus):
    async def scenario(h: Harness) -> None:
        h.source.debounce_s = 0.15  # the production value
        player = await h.start_player("spotify", track=1)
        h.start_source()
        await h.wait(track(1))
        # A skip storm: signals 20 ms apart. Without coalescing the source would read
        # (and analyse, and publish) the intermediate tracks.
        for change in (dict(metadata=metadata(2)), dict(status="Paused"),
                       dict(metadata=metadata(3)), dict(status="Playing"),
                       dict(metadata=metadata(4))):
            player.change(**change)
            await asyncio.sleep(0.02)
        await h.wait(track(4))
        await asyncio.sleep(SETTLE_S)
        assert [m.title for m in h.state.published] == ["Song 1", "Song 4"]
        assert h.analyze.calls == [art(1), art(4)]
    run(private_bus.address, scenario)


def test_pause_and_play_are_status_only_updates(private_bus):
    async def scenario(h: Harness) -> None:
        player = await h.start_player("spotify", track=1)
        h.start_source()
        playing = await h.wait(track(1))
        player.change(status="Paused")
        paused = await h.wait(track(1, "Paused"))
        assert not paused.playing
        assert paused.palette is playing.palette and paused.thumbnail is playing.thumbnail
        player.change(status="Playing")
        resumed = await h.wait(track(1))
        assert resumed.palette is playing.palette
        assert h.analyze.calls == [art(1)]  # no re-analysis
    run(private_bus.address, scenario)


def test_unchanged_properties_do_not_republish(private_bus):
    # Regression guard: re-publishing an equal MediaState whose thumbnail is an ndarray
    # would make State.set raise (MediaState.__eq__ compares the arrays).
    async def scenario(h: Harness) -> None:
        player = await h.start_player("spotify", track=1)
        h.start_source()
        await h.wait(track(1))
        player.change(status="Playing", metadata=metadata(1))
        await asyncio.sleep(SETTLE_S)
        assert len(h.state.published) == 1
    run(private_bus.address, scenario)


def test_player_exit_publishes_none(private_bus):
    async def scenario(h: Harness) -> None:
        await h.start_player("spotify", track=1)
        h.start_source()
        await h.wait(track(1))
        await h.quit_player("spotify")
        await h.wait(lambda m: m is None)
        assert h.state.published[-1] is None
    run(private_bus.address, scenario)


def test_preferred_player_wins_but_playing_beats_paused(private_bus):
    async def scenario(h: Harness) -> None:
        spotify = await h.start_player("spotify", track=1)
        await h.start_player("vlc", track=2)
        h.start_source()
        await h.wait(track(1, player="spotify"))  # both playing: preference decides
        spotify.change(status="Paused")
        await h.wait(track(2, player="vlc"))      # playing beats paused-but-preferred
        spotify.change(status="Playing")
        await h.wait(track(1, player="spotify"))
        assert h.analyze.calls == [art(1), art(2)]  # the cache served the way back
    run(private_bus.address, scenario)


def test_ignored_players_are_never_followed(private_bus):
    async def scenario(h: Harness) -> None:
        await h.start_player("vlc", track=2)
        h.start_source()
        await h.wait(lambda m: m is None and h.state.published == [None])
        await h.start_player("spotify", status="Paused", track=1)
        await h.wait(track(1, "Paused"))
        await asyncio.sleep(SETTLE_S)
        assert h.analyze.calls == [art(1)]
    run(private_bus.address, scenario, ignore=("vlc",))


def test_spotify_ads_keep_the_previous_artwork(private_bus):
    async def scenario(h: Harness) -> None:
        player = await h.start_player("spotify", track=1)
        h.start_source()
        before = await h.wait(track(1))
        player.change(metadata=metadata(9, track_id="/com/spotify/ad/42",
                                        art_url="https://ads.example/ad.jpg"))
        ad = await h.wait(track(9))
        assert ad.track_id == "/com/spotify/ad/42"
        assert ad.art_url == art(1) and ad.palette is before.palette
        player.change(metadata=metadata(2))
        after = await h.wait(track(2))
        assert after.palette == palette_for(art(2))
        assert h.analyze.calls == [art(1), art(2)]  # the ad's art was never analysed
    run(private_bus.address, scenario)


def test_new_art_is_held_back_until_analysed(private_bus):
    async def scenario(h: Harness) -> None:
        player = await h.start_player("spotify", track=1)
        h.start_source()
        await h.wait(track(1))
        h.analyze.gate.clear()  # slow network
        player.change(metadata=metadata(2))
        await asyncio.sleep(SETTLE_S)
        assert h.state.get(MEDIA).title == "Song 1"  # still the old track and palette
        h.analyze.gate.set()
        media = await h.wait(track(2))
        assert media.palette == palette_for(art(2))
        assert all(m.palette is not None for m in h.state.published)  # never flashed
    run(private_bus.address, scenario)


def test_analysis_timeout_publishes_without_palette_then_late_palette(private_bus):
    async def scenario(h: Harness) -> None:
        h.source.art_timeout_s = 0.2
        player = await h.start_player("spotify", track=1)
        h.start_source()
        await h.wait(track(1))
        h.analyze.gate.clear()
        player.change(metadata=metadata(2))
        late = await h.wait(track(2))
        assert late.palette is None and late.art_url == art(2)
        player.change(status="Paused")  # status changes still flow while we wait
        await h.wait(track(2, "Paused"))
        h.analyze.gate.set()
        media = await h.wait(lambda m: m is not None and m.palette is not None
                             and m.title == "Song 2")
        assert media.palette == palette_for(art(2)) and media.status == "Paused"
    run(private_bus.address, scenario)


def test_failed_analysis_or_missing_art_publishes_without_palette(private_bus):
    async def scenario(h: Harness) -> None:
        player = await h.start_player("spotify", track=1)
        h.start_source()
        await h.wait(track(1))
        h.analyze.fail.add(art(2))
        player.change(metadata=metadata(2))
        failed = await h.wait(track(2))
        assert failed.palette is None and failed.thumbnail is None
        player.change(metadata=metadata(3, art_url=""))
        bare = await h.wait(track(3))
        assert bare.palette is None and bare.art_url == ""
        assert h.analyze.calls == [art(1), art(2)]
    run(private_bus.address, scenario)


def test_reconnects_after_the_bus_restarts(private_bus):
    async def scenario(h: Harness) -> None:
        h.source.reconnect_initial_s = 0.05
        await h.start_player("spotify", track=1)
        h.start_source()
        await h.wait(track(1))
        await asyncio.to_thread(private_bus.restart)  # every connection drops
        await h.wait(lambda m: m is None)
        await h.forget_players()  # their connections died with the old daemon
        await h.start_player("spotify", track=5)
        media = await h.wait(track(5), timeout=5.0)
        assert media.palette == palette_for(art(5))
    run(private_bus.address, scenario)


def test_cancellation_disconnects_cleanly(private_bus):
    async def connection_names(bus: MessageBus) -> set[str]:
        reply = await bus.call(Message(destination="org.freedesktop.DBus",
                                       path="/org/freedesktop/DBus",
                                       interface="org.freedesktop.DBus", member="ListNames"))
        return {n for n in reply.body[0] if n.startswith(":")}

    async def scenario(h: Harness) -> None:
        observer = await MessageBus(bus_address=h.address).connect()
        try:
            await h.start_player("spotify", track=1)
            h.start_source()
            await h.wait(track(1))
            before = await connection_names(observer)
            await asyncio.wait_for(h.stop_source(), 1.0)
            await asyncio.sleep(0.1)
            gone = before - await connection_names(observer)
            assert len(gone) == 1  # exactly the source's own connection
        finally:
            await _close(observer)
    run(private_bus.address, scenario)
