#!/usr/bin/env python3
"""Measure per-key frame throughput at different HID++ pipelining windows.

Paints a moving horizontal rainbow (one colour per key column) and reports the
time per frame for each window size. Run with PYTHONPATH=src.
"""
from __future__ import annotations

import colorsys
import sys
import time

from ambiance.cli import find_keyboard
from ambiance.color import RGB
from ambiance.devices.logitech.hidpp import HidppError, HidppTimeout
from ambiance.layout import TKL_ANSI


def rainbow(shift: float, columns: int = 16) -> dict[int, RGB]:
    out = {}
    for key in TKL_ANSI.keys:
        x, _ = TKL_ANSI.norm(key)
        col = round(x * (columns - 1)) / (columns - 1)  # quantise to columns
        r, g, b = colorsys.hsv_to_rgb((col + shift) % 1.0, 1.0, 1.0)
        out[key.zone] = RGB(int(r * 255), int(g * 255), int(b * 255))
    return out


def main() -> int:
    windows = [int(w) for w in sys.argv[1:]] or [1, 2, 4, 8]
    dev, kb = find_keyboard()
    with dev:
        print("sw_control:", kb.sw_control(), " power timers (idle, sleep):", kb.power_timers())
        kb.claim()
        kb.set_static(RGB(0, 0, 0))
        kb.paint(rainbow(0.0))  # absorbs the post-claim replay
        frame_no = 0
        for w in windows:
            kb.window = w
            times, writes, err = [], 0, None
            for _ in range(12):
                frame_no += 1
                t0 = time.monotonic()
                try:
                    writes = kb.paint(rainbow(frame_no * 0.03))
                except (HidppError, HidppTimeout) as e:
                    err = e
                    break
                times.append(time.monotonic() - t0)
            if times:
                avg = sum(times) / len(times)
                print(f"window={w:2d}: {writes} writes/frame, avg {avg * 1000:6.1f} ms/frame "
                      f"({1 / avg:4.1f} fps), min {min(times) * 1000:.0f} ms, max {max(times) * 1000:.0f} ms"
                      + (f"  ERROR after {len(times)} frames: {err}" if err else ""))
            else:
                print(f"window={w:2d}: ERROR {err}")
        print("power timers after claim (idle, sleep):", kb.power_timers())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
