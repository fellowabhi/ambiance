"""MPRIS (D-Bus media player interface): Spotify, browsers, VLC, ...

One-shot helpers (now_playing, player_track, choose) plus MprisSource, the long-running
Source that follows the most relevant player and publishes core.MediaState under
core.MEDIA, complete with the cover-art palette that scenes paint with.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import io
import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from dbus_fast import (AuthError, BusType, DBusError, InvalidAddressError, Message, MessageType,
                       Variant)
from dbus_fast.aio import MessageBus

from ambiance.art import fetch_art
from ambiance.core import MEDIA, MediaState, Source, State
from ambiance.palette import Palette, extract_palette, load_pixels

log = logging.getLogger(__name__)

MPRIS_PREFIX = "org.mpris.MediaPlayer2."
MPRIS_PATH = "/org/mpris/MediaPlayer2"
PLAYER_IFACE = "org.mpris.MediaPlayer2.Player"
PROPERTIES_IFACE = "org.freedesktop.DBus.Properties"
DBUS = ("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus")
SPOTIFY_AD_PREFIXES = ("/com/spotify/ad/", "spotify:ad:")  # current / older Spotify clients

# Signals only reach us through match rules. arg0namespace limits NameOwnerChanged to MPRIS
# names; arg0 limits PropertiesChanged to the Player interface (not the root interface).
NAME_OWNER_RULE = (
    "type='signal',sender='org.freedesktop.DBus',interface='org.freedesktop.DBus',"
    "member='NameOwnerChanged',path='/org/freedesktop/DBus',"
    "arg0namespace='org.mpris.MediaPlayer2'")
PROPERTIES_RULE = (
    f"type='signal',interface='{PROPERTIES_IFACE}',member='PropertiesChanged',"
    f"path='{MPRIS_PATH}',arg0='{PLAYER_IFACE}'")


@dataclass(frozen=True)
class Track:
    player: str  # e.g. "spotify", "chromium.instance123"
    status: str  # Playing | Paused | Stopped
    track_id: str
    title: str
    artists: tuple[str, ...]
    album: str
    art_url: str
    length_us: int

    @property
    def label(self) -> str:
        who = ", ".join(self.artists)
        return f"{self.title} — {who}" if who else self.title


def unwrap(v):
    if isinstance(v, Variant):
        return unwrap(v.value)
    if isinstance(v, dict):
        return {k: unwrap(x) for k, x in v.items()}
    if isinstance(v, list):
        return [unwrap(x) for x in v]
    return v


def track_from_properties(player: str, props: dict) -> Track:
    props = unwrap(props)
    md = props.get("Metadata") or {}
    artists = md.get("xesam:artist") or []
    if isinstance(artists, str):
        artists = [artists]
    return Track(
        player=player,
        status=str(props.get("PlaybackStatus", "Stopped")),
        track_id=str(md.get("mpris:trackid", "")),
        title=str(md.get("xesam:title", "")),
        artists=tuple(str(a) for a in artists),
        album=str(md.get("xesam:album", "")),
        art_url=str(md.get("mpris:artUrl", "")),
        length_us=int(md.get("mpris:length", 0) or 0),
    )


async def call(bus: MessageBus, destination: str, path: str, interface: str, member: str,
               signature: str = "", body: Sequence = ()) -> list:
    reply = await bus.call(Message(destination=destination, path=path, interface=interface,
                                   member=member, signature=signature, body=list(body)))
    if reply.message_type == MessageType.ERROR:
        raise RuntimeError(f"{destination} {member}: {reply.error_name} {reply.body}")
    return reply.body


async def list_players(bus: MessageBus) -> list[str]:
    names = (await call(bus, *DBUS, "ListNames"))[0]
    return sorted(n for n in names if n.startswith(MPRIS_PREFIX))


async def player_track(bus: MessageBus, bus_name: str) -> Track:
    props = (await call(bus, bus_name, MPRIS_PATH, "org.freedesktop.DBus.Properties", "GetAll",
                        "s", [PLAYER_IFACE]))[0]
    return track_from_properties(bus_name.removeprefix(MPRIS_PREFIX), props)


def choose(tracks: Sequence[Track], prefer: Sequence[str] = ("spotify",)) -> Track | None:
    """Pick the track to follow: playing beats paused; preferred players win ties."""
    def rank(t: Track):
        playing = {"Playing": 0, "Paused": 1}.get(t.status, 2)
        pref = next((i for i, p in enumerate(prefer) if t.player.startswith(p)), len(prefer))
        return (playing, pref)
    candidates = [t for t in tracks if t.title or t.art_url]
    return min(candidates, key=rank) if candidates else None


async def now_playing(prefer: Sequence[str] = ("spotify",)) -> Track | None:
    bus = await MessageBus(bus_type=BusType.SESSION).connect()
    try:
        tracks = []
        for name in await list_players(bus):
            try:
                tracks.append(await player_track(bus, name))
            except RuntimeError:
                continue  # player vanished or doesn't implement Player
        return choose(tracks, prefer)
    finally:
        bus.disconnect()


# ---- cover art analysis ---------------------------------------------------------------------

ArtAnalysis = tuple[Palette | None, Any]  # (palette, thumbnail ndarray (H, W, 3) sRGB 0..1)


def is_ad(track: Track) -> bool:
    """Spotify (free tier) announces ads as ordinary MPRIS tracks with an ad track id."""
    return track.track_id.startswith(SPOTIFY_AD_PREFIXES)


def thumbnail_from_image_bytes(data: bytes, size: int = 24) -> np.ndarray:
    """Decode image bytes -> (size, size, 3) sRGB floats 0..1 (centre-cropped to a square)."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as img:
        small = ImageOps.fit(img.convert("RGB"), (size, size), Image.Resampling.BOX)
    return np.asarray(small, dtype=float) / 255.0


def analyze_art(url: str) -> ArtAnalysis:
    """MprisSource's default analyser. Blocking (network/disk + CPU): run it in a thread."""
    data = fetch_art(url)
    palette = extract_palette(load_pixels(data), max_colors=5)
    # 96 px: fine enough that the album-art keyboard mode can pan smoothly across it.
    return palette, thumbnail_from_image_bytes(data, 96)


class ArtAnalyzer:
    """Runs a blocking analyser off the event loop with a small LRU cache.

    Concurrent requests for one URL share a single job, and jobs are never cancelled by
    the callers that wait for them (they await through asyncio.shield) so a result that
    arrives late still lands in the cache. Failures are not cached: a flaky network must
    not leave a track colourless forever, the URL is retried when it becomes current again.
    """

    def __init__(self, analyze: Callable[[str], ArtAnalysis], cache_size: int = 32) -> None:
        self._analyze = analyze
        self._cache_size = cache_size
        self._cache: OrderedDict[str, ArtAnalysis] = OrderedDict()
        self._jobs: dict[str, asyncio.Task[ArtAnalysis]] = {}

    def cached(self, url: str) -> ArtAnalysis | None:
        result = self._cache.get(url)
        if result is not None:
            self._cache.move_to_end(url)
        return result

    def job(self, url: str) -> asyncio.Future[ArtAnalysis]:
        """The shared analysis of url (already resolved when cached). Await it via shield()."""
        if (hit := self.cached(url)) is not None:
            done: asyncio.Future[ArtAnalysis] = asyncio.get_running_loop().create_future()
            done.set_result(hit)
            return done
        task = self._jobs.get(url)
        if task is None:
            task = asyncio.create_task(self._run(url), name=f"art-analysis {url}")
            self._jobs[url] = task
            task.add_done_callback(functools.partial(self._forget, url))
        return task

    async def analyze(self, url: str) -> ArtAnalysis:
        return await asyncio.shield(self.job(url))

    def close(self) -> None:
        """Cancel in-flight jobs (their worker threads finish on their own); keep the cache."""
        for task in list(self._jobs.values()):
            task.cancel()
        self._jobs.clear()

    def _forget(self, url: str, task: asyncio.Task[ArtAnalysis]) -> None:
        if self._jobs.get(url) is task:
            del self._jobs[url]

    async def _run(self, url: str) -> ArtAnalysis:
        try:
            palette, thumbnail = await asyncio.to_thread(self._analyze, url)
        except Exception as e:  # any analyser failure just means "no palette for this art"
            log.warning("cover art analysis failed for %s: %s", url, e)
            return (None, None)
        self._cache[url] = (palette, thumbnail)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return (palette, thumbnail)


# ---- MprisSource ------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Art:
    """One piece of cover art and what was derived from it."""

    url: str
    palette: Palette | None = None
    thumbnail: Any = None


_NO_ART = _Art("")
_UNSET = object()


def _media_key(media: MediaState) -> tuple[Any, ...]:
    return tuple(getattr(media, f.name) for f in dataclasses.fields(media)
                 if f.name != "thumbnail")


def _same_media(a: MediaState | None, b: MediaState | None) -> bool:
    """Equality ignoring the thumbnail (it is derived from art_url, which is compared).

    The dataclass-generated MediaState.__eq__ ends with `thumbnail == thumbnail`, which for
    numpy arrays returns an array, so State.set's truth test raises ValueError whenever all
    other fields are equal. We therefore never hand State.set such a value.
    """
    if a is None or b is None:
        return a is b
    return _media_key(a) == _media_key(b)


class _MediaPublisher:
    """Turns "the followed track" into MediaState publications.

    Anti-flash rule: a track with new, not yet analysed cover art is held back until its
    palette is known (or art_timeout_s passes), then published once with track, palette and
    thumbnail together, so the lights go straight from the old palette to the new one.
    Changes that keep the same art (Playing <-> Paused, same-album tracks) publish at once.
    Spotify ads keep the last real artwork: an ad should not repaint the room.
    """

    def __init__(self, state: State, analyzer: ArtAnalyzer, art_timeout_s: float) -> None:
        self._state = state
        self._analyzer = analyzer
        self._art_timeout_s = art_timeout_s
        self._track: Track | None = None
        self._art = _NO_ART                  # art the published MediaState describes
        self._waiter: asyncio.Task[None] | None = None
        self._waiting_for: str | None = None  # art URL whose analysis we are waiting for
        self._published: Any = _UNSET

    def update(self, track: Track | None) -> None:
        self._track = track
        if track is None or is_ad(track):
            self._stop_waiting()
            self._publish()
            return
        url = track.art_url
        if url == self._waiting_for:
            if self._art.url == url:  # timed out earlier: keep status fresh meanwhile
                self._publish()
            return  # otherwise the pending analysis publishes the newest track
        self._stop_waiting()
        if url != self._art.url:
            analysis = self._analyzer.cached(url) if url else (None, None)
            if analysis is None:
                self._waiting_for = url
                self._waiter = asyncio.create_task(self._await_art(url), name=f"mpris-art {url}")
                return
            self._art = _Art(url, *analysis)
        self._publish()

    def close(self) -> None:
        self._stop_waiting()

    def _stop_waiting(self) -> None:
        if self._waiter is not None:
            self._waiter.cancel()  # only this waiter: the shared analysis job keeps running
        self._waiter = None
        self._waiting_for = None

    async def _await_art(self, url: str) -> None:
        job = self._analyzer.job(url)
        try:
            try:
                analysis = await asyncio.wait_for(asyncio.shield(job), self._art_timeout_s)
            except TimeoutError:
                log.warning("cover art %s not analysed after %.0f s; publishing without palette",
                            url, self._art_timeout_s)
                self._apply(_Art(url))
                analysis = await asyncio.shield(job)  # a late palette still beats none
            self._waiter = None
            self._waiting_for = None
            self._apply(_Art(url, *analysis))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("publishing analysed cover art failed")

    def _apply(self, art: _Art) -> None:
        track = self._track
        if track is None or is_ad(track) or track.art_url != art.url:
            return  # the followed track moved on (update() normally cancels us first)
        self._art = art
        self._publish()

    def _publish(self) -> None:
        track = self._track
        value = None
        if track is not None:
            # For a normal track self._art is that track's art (see update/_apply); for an
            # ad it is deliberately the last real artwork.
            value = MediaState(player=track.player, status=track.status,
                               track_id=track.track_id, title=track.title,
                               artists=track.artists, album=track.album, art_url=self._art.url,
                               palette=self._art.palette, thumbnail=self._art.thumbnail)
        if self._published is not _UNSET and _same_media(self._published, value):
            return
        self._published = value
        self._state.set(MEDIA, value)
        if value is None:
            log.info("media: nothing to follow")
        else:
            log.info("media: %s %s: %s — %s%s", value.player, value.status, value.title,
                     ", ".join(value.artists), "" if value.palette is not None else " (no palette)")


class _OwnerMap:
    """Well-known MPRIS name -> unique name of the connection that owns it.

    NameOwnerChanged is authoritative and arrives in order with everything else the bus
    sends us. A GetNameOwner reply, however, can be stale by the time the awaiting coroutine
    resumes (a NameOwnerChanged read in the same batch is dispatched first), so a queried
    owner is only accepted for names that no NameOwnerChanged has mentioned yet.
    """

    def __init__(self) -> None:
        self._owners: dict[str, str] = {}
        self._signalled: set[str] = set()

    def changed(self, name: str, owner: str) -> None:
        self._signalled.add(name)
        self._set(name, owner)

    def learned(self, name: str, owner: str | None) -> None:
        if name not in self._signalled:
            self._set(name, owner)

    def owner(self, name: str) -> str | None:
        return self._owners.get(name)

    def names_owned_by(self, unique: str | None) -> list[str]:
        return sorted(n for n, o in self._owners.items() if o == unique)

    def _set(self, name: str, owner: str | None) -> None:
        if owner:
            self._owners[name] = owner
        else:
            self._owners.pop(name, None)


_KEEP = object()  # _read() result: transient failure, keep the player's previous Track


class _PlayerWatch:
    """One bus connection's live view of the MPRIS players.

    Signals come from unique names (":1.42"), never from well-known names, hence _OwnerMap.
    Every relevant signal only marks player names dirty; one refresher task coalesces the
    burst (Spotify sends several PropertiesChanged per track change) and re-reads the dirty
    players with GetAll, which is robust against partial updates and invalidated properties.
    """

    def __init__(self, bus: MessageBus, source: MprisSource,
                 on_track: Callable[[Track | None], None]) -> None:
        self._bus = bus
        self._source = source
        self._on_track = on_track
        self._owners = _OwnerMap()
        self._tracks: dict[str, Track] = {}
        self._dirty: set[str] = set()
        self._wakeup = asyncio.Event()

    async def run(self) -> None:
        """Follow the players until the connection drops (returns or raises)."""
        # Handler and match rules first, then list names: no change can fall in between.
        self._bus.add_message_handler(self._on_message)
        for rule in (NAME_OWNER_RULE, PROPERTIES_RULE):
            await self._call(*DBUS, "AddMatch", "s", [rule])
        await self._sync()
        # The refresher never raises (it logs per refresh), so no TaskGroup: that would wrap
        # the disconnect error in an ExceptionGroup and make the reconnect log unreadable.
        refresher = asyncio.create_task(self._refresh_loop(), name="mpris-refresh")
        try:
            await self._bus.wait_for_disconnect()
        finally:
            refresher.cancel()

    async def _call(self, *args: Any) -> list:
        return await asyncio.wait_for(call(self._bus, *args), self._source.call_timeout_s)

    async def _sync(self) -> None:
        names = [n for n in (await self._call(*DBUS, "ListNames"))[0] if self._source.wanted(n)]
        owners = await asyncio.gather(*(self._query_owner(n) for n in names))
        for name, owner in zip(names, owners):
            self._owners.learned(name, owner)
        self._dirty.update(names)
        await self._refresh(self._take_dirty())

    async def _query_owner(self, name: str) -> str | None:
        try:
            return str((await self._call(*DBUS, "GetNameOwner", "s", [name]))[0])
        except RuntimeError:  # NameHasNoOwner: it left while we were listing
            return None

    def _on_message(self, msg: Message) -> None:
        """dbus-fast hook for every incoming message. Must return None: a truthy result
        marks the message handled, which would swallow the replies to our own calls."""
        if msg.message_type is not MessageType.SIGNAL:
            return None
        if (msg.member == "NameOwnerChanged" and msg.interface == DBUS[2]
                and msg.sender == DBUS[0]):
            name, _old, new = msg.body
            if self._source.wanted(name):
                self._owners.changed(name, new)
                self._mark(name)
        elif (msg.member == "PropertiesChanged" and msg.interface == PROPERTIES_IFACE
              and msg.path == MPRIS_PATH and msg.body and msg.body[0] == PLAYER_IFACE):
            for name in self._owners.names_owned_by(msg.sender):
                self._mark(name)
        return None

    def _mark(self, name: str) -> None:
        self._dirty.add(name)
        self._wakeup.set()

    def _take_dirty(self) -> set[str]:
        dirty, self._dirty = self._dirty, set()
        return dirty

    async def _refresh_loop(self) -> None:
        while True:
            await self._wakeup.wait()
            try:
                await self._coalesce()
                await self._refresh(self._take_dirty())
            except Exception:
                log.exception("MPRIS refresh failed")

    async def _coalesce(self) -> None:
        """Wait for a quiet gap of debounce_s so a burst of signals costs one refresh, but
        no longer than max_coalesce_s in total: a chatty player must not stall updates."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._source.max_coalesce_s
        while True:
            self._wakeup.clear()
            quiet = min(self._source.debounce_s, deadline - loop.time())
            if quiet <= 0:
                return
            try:
                await asyncio.wait_for(self._wakeup.wait(), quiet)
            except TimeoutError:
                return

    async def _refresh(self, names: set[str]) -> None:
        ordered = sorted(names)
        results = await asyncio.gather(*(self._read(n) for n in ordered))
        for name, result in zip(ordered, results):
            if result is _KEEP:
                continue
            if result is None or self._owners.owner(name) is None:  # gone meanwhile
                self._tracks.pop(name, None)
            else:
                self._tracks[name] = result
        tracks = [self._tracks[n] for n in sorted(self._tracks)]
        self._on_track(choose(tracks, self._source.prefer))

    async def _read(self, name: str) -> Track | None | object:
        if self._owners.owner(name) is None:
            return None
        try:
            return await asyncio.wait_for(player_track(self._bus, name),
                                          self._source.call_timeout_s)
        except TimeoutError:
            # A busy player is not a gone player. Its next signal triggers another read.
            log.warning("%s did not answer within %.1f s; keeping its last state",
                        name, self._source.call_timeout_s)
            return _KEEP
        except RuntimeError as e:  # error reply: vanished, or no Player interface (yet)
            log.debug("cannot read %s: %s", name, e)
            return None
        except (TypeError, ValueError) as e:  # malformed properties
            log.warning("ignoring %s: unexpected MPRIS properties (%s)", name, e)
            return None
        except Exception as e:  # connection trouble: the session is ending, let run() see it
            log.debug("reading %s failed: %r", name, e)
            return _KEEP


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


# What a dying or unreachable bus raises; anything else reaching run() is logged with traceback.
_BUS_ERRORS = (OSError, EOFError, TimeoutError, RuntimeError, DBusError, AuthError,
               InvalidAddressError)


async def _disconnect(bus: MessageBus, timeout: float = 1.0) -> None:
    """Close the connection and give dbus-fast a moment to finalize (closes the socket)."""
    bus.disconnect()
    try:
        await asyncio.wait_for(bus.wait_for_disconnect(), timeout)
    except Exception:  # already broken, or slow: either way nothing left for us to do
        pass


class MprisSource(Source):
    """Follows the most relevant MPRIS player and publishes state[MEDIA].

    Publishes MediaState (with the cover-art palette and thumbnail once analysed) for the
    track chosen by choose(), or None when no player has anything to show. Survives the
    session bus going away by reconnecting with exponential backoff.

    analyze(art_url) -> (palette, thumbnail) is blocking; it runs in a worker thread and
    results are cached per URL. Default: analyze_art (fetch + 5-colour palette + 24x24).
    """

    name = "mpris"

    # Tunables (instance-overridable, e.g. by tests).
    debounce_s = 0.15        # quiet gap that ends a burst of PropertiesChanged
    max_coalesce_s = 0.6     # upper bound on how long a burst can delay a refresh
    art_timeout_s = 8.0      # how long a new track waits for its palette before publishing
    call_timeout_s = 3.0     # D-Bus method calls (dbus-fast has no default timeout)
    reconnect_initial_s = 0.5
    reconnect_max_s = 30.0
    stable_after_s = 10.0    # a connection that lasted this long resets the backoff

    def __init__(self, prefer: Sequence[str] = ("spotify",), *,
                 analyze: Callable[[str], tuple[Palette | None, Any]] | None = None,
                 bus_address: str | None = None, ignore: Sequence[str] = ()) -> None:
        self.prefer = tuple(p.removeprefix(MPRIS_PREFIX) for p in prefer)
        self.ignore = tuple(p.removeprefix(MPRIS_PREFIX) for p in ignore)
        self._bus_address = bus_address
        self._analyzer = ArtAnalyzer(analyze or analyze_art)

    def wanted(self, bus_name: str) -> bool:
        """Is bus_name an MPRIS player we should look at (not excluded via ignore)?"""
        if not bus_name.startswith(MPRIS_PREFIX):
            return False
        return not bus_name.removeprefix(MPRIS_PREFIX).startswith(self.ignore)

    async def run(self, state: State) -> None:
        publisher = _MediaPublisher(state, self._analyzer, self.art_timeout_s)
        backoff = _Backoff(self.reconnect_initial_s, self.reconnect_max_s)
        loop = asyncio.get_running_loop()
        healthy = True  # log the first failure of a streak loudly, repeats quietly
        try:
            while True:
                started = loop.time()
                error: Exception | None = None
                try:
                    await self._session(publisher)
                except Exception as e:  # refused, auth failure, daemon gone, AddMatch error
                    error = e
                publisher.update(None)  # we can no longer see any player
                if loop.time() - started >= self.stable_after_s:
                    backoff.reset()
                    healthy = True
                delay = backoff.next()
                log.log(logging.WARNING if healthy else logging.DEBUG,
                        "MPRIS: lost the session bus (%s); retrying in %.1f s",
                        "closed" if error is None else f"{type(error).__name__}: {error}", delay,
                        exc_info=None if error is None or isinstance(error, _BUS_ERRORS)
                        else error)  # unexpected: show the traceback, it is probably a bug
                healthy = False
                await asyncio.sleep(delay)
        finally:
            publisher.close()
            self._analyzer.close()

    async def _session(self, publisher: _MediaPublisher) -> None:
        bus = await MessageBus(bus_address=self._bus_address, bus_type=BusType.SESSION).connect()
        log.info("MPRIS: watching media players (bus connection %s)", bus.unique_name)
        try:
            await _PlayerWatch(bus, self, publisher.update).run()
        finally:
            await _disconnect(bus)
