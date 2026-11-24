#!/usr/bin/env python3
"""Probe an ELK-BLEDOM/MELK strip: try two common colour-command formats with distinct colours.

Format A (7e 00 05 03 RR GG BB 00 ef): red -> green -> blue
Format B (7e 07 05 03 RR GG BB 10 ef): magenta -> cyan -> yellow
"""
import asyncio, sys, time
from bleak import BleakClient, BleakScanner

WRITE = "0000fff3-0000-1000-8000-00805f9b34fb"
NOTIFY = "0000fff4-0000-1000-8000-00805f9b34fb"

def h(s): return bytes.fromhex(s.replace(" ", ""))

async def main(addr):
    dev = await BleakScanner.find_device_by_address(addr, timeout=15)
    if dev is None:
        print("strip not found"); return 1
    async with BleakClient(dev, timeout=20) as c:
        t0 = time.monotonic()
        def on_note(_, data: bytearray):
            print(f"  notify +{time.monotonic()-t0:5.2f}s: {data.hex(' ')}")
        await c.start_notify(NOTIFY, on_note)
        async def send(label, hexs, pause=0.4):
            print(f"+{time.monotonic()-t0:5.2f}s {label}: {hexs}")
            await c.write_gatt_char(WRITE, h(hexs), response=False)
            await asyncio.sleep(pause)
        # MELK handshake as used by open-source integrations, then power on (both variants)
        await send("init 83", "7e 07 83", 0.6)
        await send("init 04", "7e 04 04", 0.6)
        await send("power on A", "7e 00 04 f0 00 01 ff 00 ef")
        await send("power on B", "7e 04 04 f0 00 01 ff 00 ef")
        for name, rgb in (("RED", "ff 00 00"), ("GREEN", "00 ff 00"), ("BLUE", "00 00 ff")):
            await send(f"format A {name}", f"7e 00 05 03 {rgb} 00 ef", 2.5)
        for name, rgb in (("MAGENTA", "ff 00 ff"), ("CYAN", "00 ff ff"), ("YELLOW", "ff ff 00")):
            await send(f"format B {name}", f"7e 07 05 03 {rgb} 10 ef", 2.5)
        await send("final: format A warm orange", "7e 00 05 03 ff 40 00 00 ef", 1.0)
        await c.stop_notify(NOTIFY)
    return 0

raise SystemExit(asyncio.run(main(sys.argv[1])))
