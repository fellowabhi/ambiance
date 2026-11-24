"""Ambiance app (GTK 4 + libadwaita): start/stop the daemon, watch it, adjust the lights.

The app is a thin client of the daemon's D-Bus interface (ambiance.control), so closing
the window never stops the lights. While the daemon is stopped, changes are written to
config.toml directly and apply at the next start.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango, PangoCairo  # noqa: E402

from ambiance.config import load_config, save_config  # noqa: E402
from ambiance.control import BUS_NAME, INTERFACE, OBJECT_PATH, apply_patch  # noqa: E402
from ambiance.layout import LAYOUTS, TKL_ANSI  # noqa: E402
from ambiance.paths import log_path, project_dir  # noqa: E402

APP_ID = "dev.ambiance.App"
BROWSERS = ("chromium", "chrome", "firefox", "brave", "vivaldi", "edge", "opera")
MOTION = [("Still", 0.0), ("Slow", 90.0), ("Medium", 45.0), ("Fast", 20.0)]
CYCLE = [("Slow", 24.0), ("Medium", 12.0), ("Fast", 6.0)]
STYLES = [("Album art", "mosaic"), ("Colour gradient", "gradient")]
DEVICE_TITLES = {"keyboard": ("Keyboard", "input-keyboard-symbolic"),
                 "strip": ("LED strip", "weather-clear-night-symbolic"),
                 "mouse": ("Mouse", "input-mouse-symbolic")}

CSS = b"""
.cover { border-radius: 12px; }
.swatch { min-width: 26px; min-height: 26px; border-radius: 7px; }
.preview { background: #151515; border-radius: 14px; }
.title-big { font-size: 1.25em; font-weight: 700; }
.dim { opacity: 0.65; }
"""


def _rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _rounded(cr, x, y, w, h, r) -> None:
    r = min(r, w / 2, h / 2)
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    cr.close_path()


class Daemon:
    """Session-bus client of dev.ambiance.Daemon, driven by the GLib main loop."""

    def __init__(self, on_status, on_running) -> None:
        self.on_status, self.on_running = on_status, on_running
        self.running = False
        self.bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        Gio.bus_watch_name_on_connection(self.bus, BUS_NAME, Gio.BusNameWatcherFlags.NONE,
                                         self._appeared, self._vanished)
        self.bus.signal_subscribe(BUS_NAME, INTERFACE, "StatusChanged", OBJECT_PATH, None,
                                  Gio.DBusSignalFlags.NONE, self._signal)

    def _appeared(self, *_):
        self.running = True
        self.on_running(True)
        self.call("GetStatus", None, lambda r: self.on_status(json.loads(r)))

    def _vanished(self, *_):
        self.running = False
        self.on_running(False)

    def _signal(self, _conn, _sender, _path, _iface, _name, params):
        self.on_status(json.loads(params.unpack()[0]))

    def call(self, method: str, arg: str | None, done=None) -> None:
        params = GLib.Variant("(s)", (arg,)) if arg is not None else None

        def finish(bus, res):
            try:
                out = bus.call_finish(res).unpack()
            except GLib.Error as e:
                print(f"ambiance: {method} failed: {e.message}", file=sys.stderr)
                return
            if done and out:
                done(out[0])

        self.bus.call(BUS_NAME, OBJECT_PATH, INTERFACE, method, params, None,
                      Gio.DBusCallFlags.NONE, 5000, None, finish)

    def start(self) -> None:
        log = log_path()
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as out:
            subprocess.Popen([sys.executable, "-m", "ambiance", "run"], cwd=project_dir(),
                             stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)

    def stop(self) -> None:
        self.call("Quit", None)


class Preview(Gtk.DrawingArea):
    """Live picture of what every light shows right now."""

    def __init__(self) -> None:
        super().__init__()
        self.add_css_class("preview")
        self.set_content_height(250)
        self.set_hexpand(True)
        self.devices: list[dict] = []
        self.set_draw_func(self._draw)

    def update(self, devices: list[dict]) -> None:
        self.devices = devices
        self.queue_draw()

    def _draw(self, _area, cr, width, height) -> None:
        by_kind = {d["kind"]: d for d in self.devices}
        pad = 16
        strip = by_kind.get("strip")
        if strip:
            c = _rgb(strip["colors"].get("0", "#000000"))
            for i in range(6, 0, -1):  # soft glow
                cr.set_source_rgba(*c, 0.05 * (7 - i))
                _rounded(cr, pad - i * 2, pad - i * 2, width - 2 * pad + i * 4, 12 + i * 4, 10)
                cr.fill()
            cr.set_source_rgb(*c)
            _rounded(cr, pad, pad, width - 2 * pad, 12, 6)
            cr.fill()
        kb = by_kind.get("keyboard")
        mouse = by_kind.get("mouse")
        mouse_w = 70 if mouse else 0
        top = pad + 34
        if kb:
            layout = LAYOUTS.get(kb.get("layout", ""), TKL_ANSI)
            avail_w = width - 2 * pad - (mouse_w + pad if mouse else 0)
            unit = min(avail_w / layout.width, (height - top - pad) / layout.height)
            ox, oy = pad, top
            text = PangoCairo.create_layout(cr)  # Pango: real font fallback (arrow glyphs)
            text.set_font_description(Pango.FontDescription.from_string(
                f"Sans {max(5.5, unit * 0.21):.1f}"))
            for key in layout.keys:
                hx = kb["colors"].get(str(key.zone), "#202020")
                r, g, b = _rgb(hx)
                x, y = ox + key.x * unit + 1.5, oy + key.y * unit + 1.5
                w, h = key.w * unit - 3, key.h * unit - 3
                cr.set_source_rgb(r, g, b)
                _rounded(cr, x, y, w, h, unit * 0.15)
                cr.fill()
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                cr.set_source_rgba(0, 0, 0, 0.7) if lum > 0.55 else cr.set_source_rgba(1, 1, 1, 0.7)
                text.set_text(key.label, -1)
                tw, th = text.get_pixel_size()
                cr.move_to(x + (w - tw) / 2, y + (h - th) / 2)
                PangoCairo.show_layout(cr, text)
        if mouse:
            mx, my, mw, mh = width - pad - mouse_w, top + 10, mouse_w, min(130, height - top - 2 * pad)
            cr.set_source_rgb(0.12, 0.12, 0.12)
            _rounded(cr, mx, my, mw, mh, mw / 2)
            cr.fill()
            logo = _rgb(mouse["colors"].get("2", "#000000"))
            dpi = _rgb(mouse["colors"].get("1", "#000000"))
            cr.set_source_rgb(*logo)
            cr.arc(mx + mw / 2, my + mh * 0.7, mw * 0.17, 0, 2 * math.pi)
            cr.fill()
            for i in range(3):
                cr.set_source_rgb(*dpi)
                cr.arc(mx + 9, my + mh * 0.25 + i * 9, 2.6, 0, 2 * math.pi)
                cr.fill()


class Window(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title="Ambiance", default_width=780, default_height=860)
        self._updating = False
        self._user_touch: dict[str, float] = {}
        self._pending: dict[str, dict] = {}
        self._flush_id = 0
        self.status: dict = {}

        header = Adw.HeaderBar()
        self.run_switch = Gtk.Switch(valign=Gtk.Align.CENTER, tooltip_text="Run ambiance")
        self.run_switch.connect("state-set", self._on_run_switch)
        self.run_label = Gtk.Label(label="Stopped", css_classes=["dim"])
        header.pack_end(self.run_switch)
        header.pack_end(self.run_label)

        self.banner = Adw.Banner(title="Ambiance is stopped. Your settings are saved and apply "
                                       "when it starts.", button_label="Start")
        self.banner.connect("button-clicked", lambda *_: self.daemon.start())

        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18, margin_top=18,
                       margin_bottom=24, margin_start=18, margin_end=18)
        page.append(self._now_playing())
        self.preview = Preview()
        page.append(self.preview)
        page.append(self._lights_group())
        page.append(self._look_group())
        page.append(self._behaviour_group())

        clamp = Adw.Clamp(maximum_size=860, child=page)
        scroll = Gtk.ScrolledWindow(child=clamp, vexpand=True)
        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        body.append(self.banner)
        body.append(scroll)
        view = Adw.ToolbarView(content=body)
        view.add_top_bar(header)
        self.set_content(view)

        self.daemon = Daemon(self._apply_status, self._apply_running)
        self._apply_running(self.daemon.running)
        self._load_offline()

    # -- widgets -----------------------------------------------------------------------------
    def _now_playing(self) -> Gtk.Widget:
        box = Gtk.Box(spacing=16)
        # Gtk.Image (fixed pixel size): a Gtk.Picture would grow towards the 640 px source.
        self.cover = Gtk.Image(pixel_size=110, css_classes=["cover"],
                               overflow=Gtk.Overflow.HIDDEN, valign=Gtk.Align.CENTER,
                               icon_name="audio-x-generic-symbolic")
        info = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, valign=Gtk.Align.CENTER)
        self.title = Gtk.Label(label="Nothing playing", xalign=0, css_classes=["title-big"],
                               ellipsize=Pango.EllipsizeMode.END, max_width_chars=40)
        self.artist = Gtk.Label(label="", xalign=0, css_classes=["dim"],
                                ellipsize=Pango.EllipsizeMode.END)
        self.state_label = Gtk.Label(label="", xalign=0, css_classes=["dim", "caption"])
        self.swatches = Gtk.Box(spacing=6, margin_top=6)
        for w in (self.title, self.artist, self.state_label, self.swatches):
            info.append(w)
        box.append(self.cover)
        box.append(info)
        return box

    def _lights_group(self) -> Gtk.Widget:
        group = Adw.PreferencesGroup(title="Lights",
                                     description="Brightness and movement per light")
        self.device_rows: dict[str, tuple[Adw.ActionRow, Gtk.Scale, Gtk.Switch]] = {}
        for kind, (title, icon) in DEVICE_TITLES.items():
            row = Adw.ActionRow(title=title, subtitle="not found")
            row.add_prefix(Gtk.Image(icon_name=icon))
            scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 100, 5)
            scale.set_size_request(180, -1)
            scale.set_valign(Gtk.Align.CENTER)
            scale.set_draw_value(False)
            scale.set_tooltip_text("Brightness")
            scale.connect("value-changed", self._on_brightness, kind)
            move = Gtk.Switch(valign=Gtk.Align.CENTER, tooltip_text="Move / cycle colours")
            move.connect("state-set", self._on_animate, kind)
            row.add_suffix(scale)
            row.add_suffix(Gtk.Label(label="Move", css_classes=["dim", "caption"]))
            row.add_suffix(move)
            group.add(row)
            self.device_rows[kind] = (row, scale, move)
        detect = Adw.ButtonRow(title="Detect devices", start_icon_name="view-refresh-symbolic")
        detect.connect("activated", self._on_detect)
        group.add(detect)
        return group

    def _look_group(self) -> Gtk.Widget:
        group = Adw.PreferencesGroup(title="Look")
        self.style_row = Adw.ComboRow(title="Keyboard shows",
                                      model=Gtk.StringList.new([s for s, _ in STYLES]))
        self.style_row.connect("notify::selected", self._on_style)
        self.motion_row = Adw.ComboRow(title="Keyboard movement",
                                       subtitle="How fast the picture drifts across the keys",
                                       model=Gtk.StringList.new([m for m, _ in MOTION]))
        self.motion_row.connect("notify::selected", self._on_motion)
        self.cycle_row = Adw.ComboRow(title="Colour cycling",
                                      subtitle="For lights with Move on (strip, mouse)",
                                      model=Gtk.StringList.new([c for c, _ in CYCLE]))
        self.cycle_row.connect("notify::selected", self._on_cycle)
        for r in (self.style_row, self.motion_row, self.cycle_row):
            group.add(r)
        return group

    def _behaviour_group(self) -> Gtk.Widget:
        group = Adw.PreferencesGroup(title="Behaviour")
        self.paused_row = Adw.SpinRow.new_with_range(0, 100, 5)
        self.paused_row.set_title("Brightness when paused (%)")
        self.paused_row.connect("notify::value", self._on_paused)
        self.idle_row = Adw.SpinRow.new_with_range(0, 120, 1)
        self.idle_row.set_title("Turn off after idle (minutes)")
        self.idle_row.set_subtitle("0 = never")
        self.idle_row.connect("notify::value", self._on_idle)
        self.spotify_row = Adw.SwitchRow(title="Only follow Spotify",
                                         subtitle="Ignore videos playing in web browsers")
        self.spotify_row.connect("notify::active", self._on_spotify_only)
        for r in (self.paused_row, self.idle_row, self.spotify_row):
            group.add(r)
        return group

    # -- status -> widgets -----------------------------------------------------------------------
    def _apply_running(self, running: bool) -> None:
        self._updating = True
        self.run_switch.set_state(running)
        self.run_switch.set_active(running)
        self._updating = False
        self.run_label.set_label("Running" if running else "Stopped")
        self.banner.set_revealed(not running)
        if not running:
            self.preview.update([])
            self._load_offline()

    def _apply_status(self, st: dict) -> None:
        self.status = st
        media = st.get("media")
        if media:
            self.title.set_label(media["title"] or "Unknown track")
            self.artist.set_label(", ".join(media["artists"]) + (f"  ·  {media['album']}" if media["album"] else ""))
            self.state_label.set_label(f"{media['status']} on {media['player']}")
            if media.get("art_path") and media["art_path"] != getattr(self, "_art", None):
                self._art = media["art_path"]
                try:
                    self.cover.set_from_paintable(Gdk.Texture.new_from_filename(media["art_path"]))
                except GLib.Error:
                    pass
            self._set_swatches(media.get("palette") or [])
        else:
            self.title.set_label("Nothing playing")
            self.artist.set_label("")
            self.state_label.set_label("")
            self._set_swatches([])
        self.preview.update(st.get("devices", []))
        present = {d["kind"]: d for d in st.get("devices", [])}
        for kind, (row, _scale, _move) in self.device_rows.items():
            d = present.get(kind)
            row.set_visible(d is not None or not self.daemon.running)
            if d:
                where = d.get("detail") or ""
                row.set_subtitle(("connected" if d["connected"] else "searching…")
                                 + (f"  ·  {where}" if where and d["connected"] else ""))
        self._apply_options(st.get("options", {}))

    def _set_swatches(self, colors: list[str]) -> None:
        while (child := self.swatches.get_first_child()) is not None:
            self.swatches.remove(child)
        for hx in colors[:6]:
            self.swatches.append(_swatch(hx))

    def _apply_options(self, o: dict) -> None:
        now = GLib.get_monotonic_time() / 1e6
        recent = {k for k, t in self._user_touch.items() if now - t < 1.5}
        self._updating = True
        try:
            for kind, (_row, scale, move) in self.device_rows.items():
                if kind in o and f"{kind}.brightness" not in recent:
                    scale.set_value(round(o[kind]["brightness"] * 100))
                if kind in o:
                    move.set_active(bool(o[kind]["animate"]))
            album = o.get("album", {})
            if album:
                self.style_row.set_selected(next((i for i, (_, v) in enumerate(STYLES)
                                                  if v == album["style"]), 0))
                self.motion_row.set_selected(_closest(MOTION, album["drift_period_s"]))
                self.cycle_row.set_selected(_closest(CYCLE, album["strip_cycle_s"]))
            beh = o.get("behaviour", {})
            if beh:
                if "paused" not in recent:
                    self.paused_row.set_value(round(beh["paused_level"] * 100))
                if "idle" not in recent:
                    self.idle_row.set_value(round(max(0.0, beh["idle_off_s"]) / 60))
                self.spotify_row.set_active(all(b in beh.get("ignore", []) for b in BROWSERS))
        finally:
            self._updating = False

    def _load_offline(self) -> None:
        """Show config.toml's values while the daemon is stopped."""
        if self.daemon.running:
            return
        try:
            cfg = load_config()
        except Exception:  # a broken config shouldn't stop the app from opening
            return
        self._apply_options({
            "keyboard": {"brightness": cfg.device.brightness, "animate": cfg.device.animate},
            "strip": {"brightness": cfg.strip.brightness, "animate": cfg.strip.animate},
            "mouse": {"brightness": cfg.mouse.brightness, "animate": cfg.mouse.animate},
            "album": {"style": cfg.director.album.style,
                      "drift_period_s": cfg.director.album.drift_period_s,
                      "strip_cycle_s": cfg.director.album.strip_cycle_s},
            "behaviour": {"paused_level": cfg.director.paused_level,
                          "idle_off_s": cfg.director.idle_off_s, "ignore": list(cfg.media.ignore)},
        })

    # -- widgets -> options ----------------------------------------------------------------------
    def _send(self, patch: dict, touch: str | None = None) -> None:
        if self._updating:
            return
        if touch:
            self._user_touch[touch] = GLib.get_monotonic_time() / 1e6
        for k, v in patch.items():
            self._pending.setdefault(k, {}).update(v)
        if not self._flush_id:  # coalesce slider drags into one change every 200 ms
            self._flush_id = GLib.timeout_add(200, self._flush)

    def _flush(self) -> bool:
        self._flush_id = 0
        patch, self._pending = self._pending, {}
        if self.daemon.running:
            self.daemon.call("SetOptions", json.dumps(patch))
        else:
            try:
                save_config(apply_patch(load_config(), patch))
            except Exception as e:
                print(f"ambiance: could not save settings: {e}", file=sys.stderr)
        return False

    def _on_run_switch(self, _sw, state: bool) -> bool:
        if self._updating:
            return False
        if state and not self.daemon.running:
            self.daemon.start()
            self.run_label.set_label("Starting…")
        elif not state and self.daemon.running:
            self.daemon.stop()
            self.run_label.set_label("Stopping…")
        return True  # the switch follows the daemon's real state (_apply_running)

    def _on_brightness(self, scale: Gtk.Scale, kind: str) -> None:
        self._send({kind: {"brightness": scale.get_value() / 100.0}}, f"{kind}.brightness")

    def _on_animate(self, _sw, state: bool, kind: str) -> bool:
        self._send({kind: {"animate": state}})
        return False

    def _on_style(self, row, _p) -> None:
        self._send({"album": {"style": STYLES[row.get_selected()][1]}})

    def _on_motion(self, row, _p) -> None:
        self._send({"album": {"drift_period_s": MOTION[row.get_selected()][1]}})

    def _on_cycle(self, row, _p) -> None:
        self._send({"album": {"strip_cycle_s": CYCLE[row.get_selected()][1]}})

    def _on_paused(self, row, _p) -> None:
        self._send({"behaviour": {"paused_level": row.get_value() / 100.0}}, "paused")

    def _on_idle(self, row, _p) -> None:
        self._send({"behaviour": {"idle_off_s": row.get_value() * 60.0}}, "idle")

    def _on_spotify_only(self, row, _p) -> None:
        current = set((self.status.get("options", {}).get("behaviour", {}) or {}).get("ignore", []))
        ignore = sorted(current | set(BROWSERS)) if row.get_active() else sorted(current - set(BROWSERS))
        self._send({"behaviour": {"ignore": ignore}})

    def _on_detect(self, _row) -> None:
        if self.daemon.running:
            self.daemon.call("Rescan", None)
            self.add_toast("Looking for devices…")
        else:
            self.add_toast(_offline_scan())

    def add_toast(self, text: str) -> None:
        dialog = Adw.AlertDialog(heading="Devices", body=text)
        dialog.add_response("ok", "OK")
        dialog.present(self)


def _swatch(hex_color: str) -> Gtk.Widget:
    area = Gtk.DrawingArea(content_width=26, content_height=26, tooltip_text=hex_color)
    r, g, b = _rgb(hex_color)

    def draw(_area, cr, w, h):
        cr.set_source_rgb(r, g, b)
        _rounded(cr, 0, 0, w, h, 7)
        cr.fill()

    area.set_draw_func(draw)
    return area


def _closest(options: list[tuple[str, float]], value: float) -> int:
    return min(range(len(options)), key=lambda i: abs(options[i][1] - value))


def _offline_scan() -> str:
    """Which lights are reachable right now (daemon stopped): hidraw scan, no BLE."""
    from ambiance.devices.hidraw import enumerate_hidraw

    found = [f"{d.name} ({d.bus_name})" for d in enumerate_hidraw(0x046D) if d.has_hidpp_long()]
    lines = ["Logitech devices:"] + ([f"  • {n}" for n in found] or ["  none found"])
    lines.append("The LED strip is found over Bluetooth when ambiance starts.")
    return "\n".join(lines)


def _snapshot(win: Gtk.Window, path: str) -> None:
    """Debug aid (AMBIANCE_UI_SNAPSHOT=file.png): render the window to a PNG and quit."""
    gi.require_version("Graphene", "1.0")
    from gi.repository import Graphene

    w, h = win.get_width(), win.get_height()
    snap = Gtk.Snapshot()
    Gtk.WidgetPaintable.new(win).snapshot(snap, w, h)
    texture = win.get_native().get_renderer().render_texture(
        snap.to_node(), Graphene.Rect().init(0, 0, w, h))
    texture.save_to_png(path)
    win.get_application().quit()


class App(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)

    def do_activate(self) -> None:
        win = self.props.active_window
        if win is None:
            provider = Gtk.CssProvider()
            provider.load_from_string(CSS.decode())
            Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), provider,
                                                      Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
            win = Window(self)
        win.present()
        import os

        if path := os.environ.get("AMBIANCE_UI_SNAPSHOT"):
            win.set_default_size(780, int(os.environ.get("AMBIANCE_UI_SNAPSHOT_HEIGHT", "860")))
            GLib.timeout_add(2500, lambda: _snapshot(win, path) or False)


def main() -> int:
    return App().run([sys.argv[0]])
