"""Command-line entry point: `python -m ambiance <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from ambiance.color import RGB
from ambiance.devices.hidraw import enumerate_hidraw
from ambiance.devices.logitech.hidpp import DEVICE_INDEX_DIRECT, HidppDevice
from ambiance.devices.logitech.rgb_keyboard import LogitechRgbKeyboard, UnsupportedDevice
from ambiance.layout import GROUPS, LAYOUTS, TKL_ANSI

LOGITECH = 0x046D


def find_keyboard() -> tuple[HidppDevice, LogitechRgbKeyboard]:
    candidates = [d for d in enumerate_hidraw(LOGITECH) if d.has_hidpp_long()]
    errors = []
    for info in candidates:
        try:
            dev = HidppDevice(info.path, DEVICE_INDEX_DIRECT)
        except OSError as e:
            errors.append(f"{info}: {e}")
            continue
        try:
            dev.ping()
            return dev, LogitechRgbKeyboard(dev)
        except (UnsupportedDevice, OSError) as e:
            errors.append(f"{info}: {e}")
            dev.close()
    detail = "\n  ".join(errors) if errors else "no Logitech HID++ hidraw nodes found"
    raise SystemExit(f"No per-key RGB Logitech keyboard found.\n  {detail}")


def cmd_devices(_args) -> int:
    for d in enumerate_hidraw(LOGITECH):
        print(f"{d}  hidpp={'yes' if d.has_hidpp_long() else 'no'}")
    return 0


def cmd_info(_args) -> int:
    dev, kb = find_keyboard()
    with dev:
        major, minor = dev.ping()
        print(f"{dev.path}: {dev.name()!r}  HID++ {major}.{minor}  battery {dev.battery()}%")
        print(json.dumps(kb.describe(), indent=2))
    return 0


def _layout(args):
    return LAYOUTS[args.layout]


def cmd_test_pattern(args) -> int:
    """letters red, digits green, F1-F12 blue, every other zone white, on a purple base."""
    layout = _layout(args)
    dev, kb = find_keyboard()
    with dev:
        t0 = time.monotonic()
        kb.claim()
        kb.set_static(RGB.parse("purple"))
        t1 = time.monotonic()
        colors = {z: RGB(255, 255, 255) for z in kb.zones}  # everything the firmware reports
        for group, color in (("letters", "red"), ("digits", "green"), ("fkeys", "blue")):
            for key in layout.keys:
                if key.name in GROUPS[group]:
                    colors[key.zone] = RGB.parse(color)
        n = kb.paint(colors)
        t2 = time.monotonic()
        print(f"claim+base {1000 * (t1 - t0):.0f} ms; frame of {n} writes "
              f"(sent twice after claim) {1000 * (t2 - t1):.0f} ms")
        print("Keyboard left in software mode. `python -m ambiance release` hands it back.")
    return 0


def cmd_solid(args) -> int:
    color = RGB.parse(args.color)
    dev, kb = find_keyboard()
    with dev:
        kb.claim()
        kb.set_static(color)
        kb.paint({}, fill=color)
    return 0


def _swatch(c: RGB, text: str = "      ") -> str:
    fg = "30" if 0.299 * c.r + 0.587 * c.g + 0.114 * c.b > 140 else "97"
    return f"\x1b[48;2;{c.r};{c.g};{c.b}m\x1b[{fg}m{text}\x1b[0m"


def cmd_now_playing(args) -> int:
    import asyncio

    from ambiance.art import fetch_art
    from ambiance.led import to_led
    from ambiance.palette import palette_from_image_bytes
    from ambiance.scenes.gradient import gradient_frame
    from ambiance.sources.mpris import now_playing

    track = asyncio.run(now_playing())
    if track is None or not track.art_url:
        print("Nothing with cover art is playing.")
        return 1
    print(f"{track.player}: {track.label}  [{track.album}]  ({track.status})")
    palette = palette_from_image_bytes(fetch_art(track.art_url), max_colors=args.colors)
    for s in palette.swatches:
        led = to_led(s.rgb)
        print(f"  {_swatch(s.rgb)} {s.rgb.hex}  share {s.weight:5.1%}  L {s.lightness:.2f} "
              f"C {s.chroma:.3f}  -> LED {_swatch(led)} {tuple(led)}")
    frame = gradient_frame(_layout(args), palette.swatches, steps=args.steps)
    led_cache: dict[RGB, RGB] = {}
    led_frame = {z: led_cache.setdefault(c, to_led(c)) for z, c in frame.items()}
    if args.dry_run:
        return 0
    dev, kb = find_keyboard()
    with dev:
        kb.claim()
        kb.set_static(RGB(0, 0, 0))
        t0 = time.monotonic()
        n = kb.paint(led_frame)
        print(f"painted {len(set(led_frame.values()))} colours in {n} writes "
              f"({1000 * (time.monotonic() - t0):.0f} ms incl. post-claim replay)")
    return 0


def cmd_run(args) -> int:
    from ambiance.app import main_run

    return main_run(args.config, args.verbose)


def cmd_ui(_args) -> int:
    from ambiance.ui import main as ui_main

    return ui_main()


def cmd_install_desktop(_args) -> int:
    """Add "Ambiance" to the Ubuntu app grid (a .desktop launcher for `ambiance ui`)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    entry = Path.home() / ".local" / "share" / "applications" / "dev.ambiance.App.desktop"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(
        "[Desktop Entry]\nType=Application\nName=Ambiance\n"
        "Comment=Ambient lighting that follows your music\n"
        f"Exec={root / '.venv' / 'bin' / 'python'} -m ambiance ui\n"
        f"Path={root}\nIcon={root / 'data' / 'ambiance.svg'}\n"
        "Terminal=false\nCategories=Utility;\nStartupWMClass=dev.ambiance.App\n"
        "Keywords=lights;rgb;keyboard;spotify;ambient;\n")
    print(f"installed {entry}  (find \"Ambiance\" in your app grid; remove the file to uninstall)")
    return 0


def cmd_install_service(_args) -> int:
    import shutil
    import subprocess
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "systemd" / "ambiance.service"
    dst = Path.home() / ".config" / "systemd" / "user" / "ambiance.service"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    print(f"installed {dst}\n"
          "start now and at every login:  systemctl --user enable --now ambiance\n"
          "logs:                          journalctl --user -u ambiance -f")
    return 0


def cmd_release(_args) -> int:
    dev, kb = find_keyboard()
    with dev:
        kb.release()
        print("Lighting handed back to the keyboard's onboard profile.")
    return 0


def cmd_brightness(args) -> int:
    dev, kb = find_keyboard()
    with dev:
        if args.value is not None:
            kb.set_brightness(args.value)
        print(f"brightness: {kb.brightness()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ambiance", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--layout", choices=sorted(LAYOUTS), default=TKL_ANSI.name)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the ambient-lighting daemon (Ctrl+C to stop)")
    r.add_argument("--config", help="config file (default ~/.config/ambiance/config.toml)")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("install-service", help="install the systemd user service"
                   ).set_defaults(fn=cmd_install_service)
    sub.add_parser("ui", help="open the Ambiance app").set_defaults(fn=cmd_ui)
    sub.add_parser("install-desktop", help="add Ambiance to the app grid"
                   ).set_defaults(fn=cmd_install_desktop)
    sub.add_parser("devices", help="list Logitech hidraw nodes").set_defaults(fn=cmd_devices)
    sub.add_parser("info", help="show keyboard lighting capabilities").set_defaults(fn=cmd_info)
    sub.add_parser("test-pattern", help="paint a diagnostic pattern").set_defaults(fn=cmd_test_pattern)
    s = sub.add_parser("solid", help="whole keyboard one colour")
    s.add_argument("color")
    s.set_defaults(fn=cmd_solid)
    n = sub.add_parser("now-playing", help="one-shot: paint the current track's album colours")
    n.add_argument("--colors", type=int, default=4, help="palette size")
    n.add_argument("--steps", type=int, default=10, help="gradient steps (distinct colours)")
    n.add_argument("--dry-run", action="store_true", help="show palette, don't touch the keyboard")
    n.set_defaults(fn=cmd_now_playing)
    sub.add_parser("release", help="hand lighting back to the keyboard").set_defaults(fn=cmd_release)
    b = sub.add_parser("brightness", help="get/set LED brightness (0-100)")
    b.add_argument("value", type=int, nargs="?")
    b.set_defaults(fn=cmd_brightness)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
