#!/usr/bin/env python3
"""Strip capability test: update rate (smooth hue sweep), hardware brightness, white tint."""
import asyncio, colorsys, sys, time
from bleak import BleakClient, BleakScanner

WRITE = "0000fff3-0000-1000-8000-00805f9b34fb"
NOTIFY = "0000fff4-0000-1000-8000-00805f9b34fb"

async def main(addr):
    dev = await BleakScanner.find_device_by_address(addr, timeout=15)
    if dev is None:
        print("strip not found"); return 1
    async with BleakClient(dev, timeout=20) as c:
        notes = []
        await c.start_notify(NOTIFY, lambda _, d: notes.append(bytes(d)))
        w = lambda b: c.write_gatt_char(WRITE, bytes(b), response=False)
        await w(bytes.fromhex("7e0783")); await asyncio.sleep(0.5)
        await w(bytes.fromhex("7e0404")); await asyncio.sleep(0.5)
        await w(bytes.fromhex("7e0004f00001ff00ef")); await asyncio.sleep(0.3)
        # 1) smooth hue sweep, ~30 Hz for 6 s
        n, t0 = 0, time.monotonic()
        while (t := time.monotonic() - t0) < 6.0:
            r, g, b = (int(v * 255) for v in colorsys.hsv_to_rgb((t / 6.0) % 1, 1, 1))
            await w([0x7e, 0x00, 0x05, 0x03, r, g, b, 0x00, 0xef]); n += 1
            await asyncio.sleep(max(0, (n / 30) - (time.monotonic() - t0)))
        el = time.monotonic() - t0
        print(f"hue sweep: {n} writes in {el:.2f}s = {n/el:.1f} updates/s")
        # 2) white, then hardware brightness 100 -> 30 -> 5 (variant 7e 04 01 BR 01 ff ff 00 ef)
        await w([0x7e, 0x00, 0x05, 0x03, 255, 255, 255, 0x00, 0xef]); await asyncio.sleep(0.2)
        for level in (100, 30, 5):
            await w([0x7e, 0x04, 0x01, level, 0x01, 0xff, 0xff, 0x00, 0xef])
            print(f"brightness {level}%"); await asyncio.sleep(2.2)
        await w([0x7e, 0x04, 0x01, 100, 0x01, 0xff, 0xff, 0x00, 0xef]); await asyncio.sleep(0.2)
        # 3) end on the keyboard's dominant palette colour (LED-mapped deep red)
        await w([0x7e, 0x00, 0x05, 0x03, 186, 34, 27, 0x00, 0xef]); await asyncio.sleep(0.5)
        if notes:
            print("notifications:", [x.hex(" ") for x in notes[:6]])
    return 0

raise SystemExit(asyncio.run(main(sys.argv[1])))
