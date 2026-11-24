# ambiance

Ambient lighting for your desk on Linux. Your keyboard and room LED strip pick up the colours of
whatever's playing (Spotify album art for now). It's built so new inputs (time of day, screen,
notifications, ...) and new lights (WLED on an ESP32, ...) slot in easily.

**Hardware it drives today**

| Device | Link | Control | Notes |
|---|---|---|---|
| Logitech G515 LS TKL | Bluetooth LE (HID++ 2.0) | per-key RGB, 87 keys | ~21 commands/s, so a full repaint takes ~0.5 s |
| MELK-OA10 strip ("Lotus Lamp X" app) | Bluetooth LE (ELK-BLEDOM) | one colour for the whole strip | ~12 smooth updates/s |

No Logitech G HUB, Solaar or phone app is needed. Everything is plain Python on top of
`/dev/hidraw` and BlueZ.

## Setup

```bash
# 1. let your user talk to Logitech HID devices (once, needs sudo)
sudo install -m 0644 udev/60-ambiance-logitech.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=hidraw --action=change

# 2. Python env. --system-site-packages gives the app Ubuntu's GTK 4 / libadwaita bindings
#    (python3-gi, gir1.2-adw-1) and reuses numpy/Pillow if installed.
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[test]'
```

## Use

```bash
.venv/bin/python -m ambiance install-desktop   # adds "Ambiance" to the app grid (start/stop,
.venv/bin/python -m ambiance ui                # live preview, per-light brightness & movement)
.venv/bin/python -m ambiance run            # the daemon: lights follow your music (Ctrl+C stops)
.venv/bin/python -m ambiance install-service   # then: systemctl --user enable --now ambiance

.venv/bin/python -m ambiance info           # keyboard capabilities (read-only)
.venv/bin/python -m ambiance now-playing    # one-shot: paint the current cover's colours
.venv/bin/python -m ambiance solid warm     # whole keyboard one colour
.venv/bin/python -m ambiance release        # hand the keyboard back to its own lighting
```

Configuration is optional: copy `config.example.toml` to `~/.config/ambiance/config.toml`. Every
option is documented there (LED white balance per device, idle timeout, paused dimming, which
players to follow or ignore, gradient style, ...).

### What the daemon does

- **Track changes:** the keyboard crossfades to a gradient of the cover's palette and the strip
  fades to its main colour. The palette is k-means in OKLab, ranked for "looks good as light".
- **Paused:** the colours dim. **Stopped / no player:** the keyboard returns to its own lighting.
- **Idle or screen locked:** both fade out (GNOME idle monitor + screensaver signals). This matters
  for battery: while software controls the G515's LEDs, its firmware never dims them by itself.
- **Keyboard sleeps or reconnects:** it is rediscovered by ID (the hidraw path changes) and
  re-claimed. A periodic check catches silent reverts to onboard lighting.
- **Shutdown (SIGINT/SIGTERM):** the keyboard is handed back to its onboard profile.

## Architecture

```
Source --writes--> State --snapshot--> Director --picks--> Scene --render--> Frame
                                                                               |
       LightDevice (own I/O thread) <--submit-- LED mapping <-- crossfade/level <--+
```

| Piece | Where | Role |
|---|---|---|
| contracts | `core.py` | `State`, `Source`, `Scene`, `Director`, `Decision`, `LightDevice` |
| sources | `sources/mpris.py`, `sources/presence.py` | MPRIS players (art -> palette), GNOME idle/lock |
| director | `director.py` | pure rules: state snapshot -> which scene, level, transition |
| scenes | `scenes/` | album gradient / mosaic, solid, off. A one-zone layout gets the top colour |
| engine | `engine.py` | crossfades in linear light, per-device LED mapping, skips duplicate frames |
| devices | `devices/logitech/`, `devices/elk_strip.py` | hardware I/O threads with reconnect |
| colour | `colorspace.py`, `palette.py`, `led.py` | OKLab, palette extraction, screen->LED mapping |

**Add an input:** subclass `Source`, publish an immutable value under a new state key, then
teach `DefaultDirector.decide()` (or your own `Director`) what to do with it.
**Add a look:** subclass `Scene`. **Add hardware:** implement `LightDevice` (expose a `Layout`,
and optionally a `led_profile`) and add it in `app.make_devices()`.

## Protocol notes (verified on the hardware)

**G515 over Bluetooth** exposes only HID++ long reports (id `0x11`, device index `0xFF`). Lighting:

1. **Claim:** `0x8101` fn6 `[05]` (host mode), then `0x8071` fn5 `[01 03 04]`.
2. **Static base:** `0x8071` fn1, static effect, RAM-only. Per-key colours only show over static.
3. **Per-key paint:** `0x8081` fn5 (range), fn6 (one colour on 13 zones), fn1 (4 zones).
4. **Commit:** `0x8081` fn7 `[00]`, RAM only.
5. **Release:** `0x8071` fn5 `[01 00 00]`, `0x8101` fn6 `[03]`.

Zones are the USB HID usage minus 3 (letters 1-26), and the modifiers are 104-111. Zones
47, 97, 99-103 and 254 are phantoms. Never use RGB power-mode 3: it can stop the keys from
waking the keyboard. Flash-writing commands are avoided.

**MELK strip:** GATT service `FFF0`, write characteristic `FFF3`. Handshake `7e 07 83`,
`7e 04 04`. Colour `7e 00 05 03 RR GG BB 00 ef`, power `7e 00 04 f0 00 01 ff 00 ef`.
Over BLE it shows one colour; its per-LED animations are firmware presets. An ESP32 running
WLED would give per-LED control (planned).

## Tools

`tools/hidpp_probe.py` (read-only HID++ feature dump), `tools/bench_pipeline.py` (throughput),
`tools/ble_gatt_dump.py`, `tools/strip_probe.py`, `tools/strip_capability.py`.
Reference projects studied while reverse-engineering (Solaar, logirgb, keyboard-lights, a SignalRGB
plugin, OpenRGB) are not part of this repository and are not used at runtime.
