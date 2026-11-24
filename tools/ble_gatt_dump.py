#!/usr/bin/env python3
"""Connect to a BLE device and list its GATT services/characteristics (read-only)."""
import asyncio, sys
from bleak import BleakClient, BleakScanner

async def main(addr: str):
    dev = await BleakScanner.find_device_by_address(addr, timeout=15)
    if dev is None:
        print("not found (is the phone app connected to it?)"); return 1
    print("found:", dev.address, repr(dev.name))
    async with BleakClient(dev, timeout=20) as c:
        print("connected, MTU", c.mtu_size)
        for s in c.services:
            print(f"service {s.uuid}  {s.description}")
            for ch in s.characteristics:
                val = ""
                if "read" in ch.properties:
                    try:
                        val = " value=" + (await c.read_gatt_char(ch)).hex(" ")
                    except Exception as e:
                        val = f" (read failed: {e})"
                print(f"   char {ch.uuid}  handle={ch.handle}  props={','.join(ch.properties)}{val}  {ch.description}")
    return 0

raise SystemExit(asyncio.run(main(sys.argv[1])))
