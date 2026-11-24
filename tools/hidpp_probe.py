#!/usr/bin/env python3
"""Read-only HID++ 2.0 probe for Logitech devices on Linux hidraw (USB or Bluetooth).

Only issues *query* functions (protocol version, feature table, name, firmware,
battery, lighting/onboard-profile info). It never writes settings.

    python3 tools/hidpp_probe.py            # auto-detect Logitech HID++ devices
    python3 tools/hidpp_probe.py /dev/hidraw1
"""
from __future__ import annotations

import glob
import os
import select
import struct
import sys
import time

REPORT_SHORT, REPORT_LONG = 0x10, 0x11
LONG_LEN = 20
SW_ID = 0x0A  # software id nibble; non-zero so replies are distinguishable from notifications

FEATURES = {
    0x0000: "ROOT", 0x0001: "FEATURE_SET", 0x0002: "FEATURE_INFO", 0x0003: "DEVICE_FW_VERSION",
    0x0004: "DEVICE_UNIT_ID", 0x0005: "DEVICE_NAME", 0x0006: "DEVICE_GROUPS",
    0x0007: "DEVICE_FRIENDLY_NAME", 0x0008: "KEEP_ALIVE", 0x0020: "CONFIG_CHANGE",
    0x0021: "CRYPTO_ID", 0x0080: "TARGET_SOFTWARE", 0x00C0: "DFU_CONTROL",
    0x00C1: "DFU_CONTROL_SIGNED", 0x00C2: "DFU_CONTROL_BOLT", 0x00C3: "DFU_CONTROL_SECURE",
    0x00D0: "DFU", 0x1000: "BATTERY_STATUS", 0x1001: "BATTERY_VOLTAGE",
    0x1004: "UNIFIED_BATTERY", 0x1010: "CHARGING_CONTROL", 0x1300: "LED_CONTROL",
    0x1500: "FORCE_PAIRING", 0x1802: "DEVICE_RESET", 0x1805: "OOBSTATE",
    0x1806: "CONFIG_DEVICE_PROPS", 0x1814: "CHANGE_HOST", 0x1815: "HOSTS_INFO",
    0x1981: "BACKLIGHT", 0x1982: "BACKLIGHT2", 0x1983: "BACKLIGHT3", 0x1990: "ILLUMINATION",
    0x1B00: "KEYBOARD_REPROG_CONTROLS", 0x1B04: "REPROG_CONTROLS_V4",
    0x1BC0: "REPORT_HID_USAGE", 0x1C00: "PERSISTENT_REMAPPABLE_ACTION",
    0x1D4B: "WIRELESS_DEVICE_STATUS", 0x1DF0: "REMAINING_PAIRING",
    0x1F1F: "FIRMWARE_PROPERTIES", 0x1F20: "ADC_MEASUREMENT",
    0x40A0: "FN_INVERSION", 0x40A2: "NEW_FN_INVERSION", 0x40A3: "K375S_FN_INVERSION",
    0x4100: "ENCRYPTION", 0x4220: "LOCK_KEY_STATE", 0x4520: "KEYBOARD_LAYOUT",
    0x4521: "KEYBOARD_DISABLE_KEYS", 0x4522: "KEYBOARD_DISABLE_BY_USAGE",
    0x4530: "DUALPLATFORM", 0x4531: "MULTIPLATFORM", 0x4540: "KEYBOARD_LAYOUT_2",
    0x8010: "GKEY", 0x8020: "MKEYS", 0x8030: "MR", 0x8040: "BRIGHTNESS_CONTROL",
    0x8060: "REPORT_RATE", 0x8061: "EXTENDED_ADJUSTABLE_REPORT_RATE",
    0x8070: "COLOR_LED_EFFECTS", 0x8071: "RGB_EFFECTS", 0x8080: "PER_KEY_LIGHTING",
    0x8081: "PER_KEY_LIGHTING_V2", 0x8090: "MODE_STATUS", 0x8100: "ONBOARD_PROFILES",
    0x8110: "MOUSE_BUTTON_SPY", 0x8111: "LATENCY_MONITORING",
}


class HidppError(Exception):
    pass


def find_logitech_hidraw() -> list[dict]:
    devs = []
    for sys_path in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
        try:
            with open(sys_path + "/device/uevent") as f:
                props = dict(l.split("=", 1) for l in f.read().splitlines() if "=" in l)
            with open(sys_path + "/device/report_descriptor", "rb") as f:
                rd = f.read()
        except OSError:
            continue
        bus, vid, pid = (int(x, 16) for x in props.get("HID_ID", "0:0:0").split(":"))
        if vid != 0x046D:
            continue
        # crude descriptor check: vendor usage page 0xFF43/0xFF00 + report id 0x11
        hidpp = (b"\x06\x43\xff" in rd or b"\x06\x00\xff" in rd) and b"\x85\x11" in rd
        devs.append(dict(node="/dev/" + os.path.basename(sys_path), bus=bus, vid=vid, pid=pid,
                         name=props.get("HID_NAME", "?"), hidpp=hidpp))
    return devs


class Hidpp:
    def __init__(self, node: str, devnum: int):
        self.fd = os.open(node, os.O_RDWR | os.O_NONBLOCK)
        self.devnum = devnum
        self.idx_cache: dict[int, int] = {0x0000: 0}
        self.rtts: list[float] = []

    def close(self):
        os.close(self.fd)

    def _drain(self):
        while select.select([self.fd], [], [], 0)[0]:
            try:
                os.read(self.fd, 64)
            except BlockingIOError:
                break

    def request(self, feat_idx: int, fn: int, *params: int, timeout: float = 2.0) -> bytes:
        fnsw = ((fn & 0x0F) << 4) | SW_ID
        msg = bytes([REPORT_LONG, self.devnum, feat_idx, fnsw, *params]).ljust(LONG_LEN, b"\0")
        self._drain()
        t0 = time.monotonic()
        os.write(self.fd, msg)
        deadline = t0 + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            if not select.select([self.fd], [], [], remaining)[0]:
                continue
            data = os.read(self.fd, 64)
            if len(data) < 5 or data[0] not in (REPORT_SHORT, REPORT_LONG) or data[1] != self.devnum:
                continue  # keyboard/mouse input reports, other devices
            if data[2] in (0xFF, 0x8F) and data[3] == feat_idx and data[4] == fnsw:
                raise HidppError(f"error 0x{data[5]:02X} (feat_idx=0x{feat_idx:02X} fn={fn})")
            if data[2] == feat_idx and data[3] == fnsw:
                self.rtts.append(time.monotonic() - t0)
                return data[4:]
        raise TimeoutError(f"no reply (feat_idx=0x{feat_idx:02X} fn={fn})")

    def feature_index(self, fid: int) -> int | None:
        if fid not in self.idx_cache:
            r = self.request(0x00, 0, fid >> 8, fid & 0xFF)
            self.idx_cache[fid] = r[0] if r[0] else None
        return self.idx_cache[fid]

    def call(self, fid: int, fn: int, *params: int) -> bytes:
        idx = self.feature_index(fid)
        if idx is None:
            raise HidppError(f"feature 0x{fid:04X} not supported")
        return self.request(idx, fn, *params)


def hx(b: bytes) -> str:
    return b.hex(" ")


def probe(node: str, devnum: int = 0xFF):
    d = Hidpp(node, devnum)
    try:
        r = d.request(0x00, 1, 0, 0, 0x5A)
        print(f"HID++ protocol {r[0]}.{r[1]} (ping echo 0x{r[2]:02X}) on {node} devnum=0x{devnum:02X}")

        fs_idx = d.feature_index(0x0001)
        count = d.request(fs_idx, 0)[0]
        print(f"\n{count + 1} features:")
        feats = {}
        for i in range(0, count + 1):
            r = d.request(fs_idx, 1, i)
            fid, ftype, fver = (r[0] << 8) | r[1], r[2], r[3]
            feats[fid] = i
            d.idx_cache[fid] = i
            flags = [n for bit, n in ((7, "obsolete"), (6, "hidden"), (5, "engineering")) if ftype & (1 << bit)]
            print(f"  [0x{i:02X}] 0x{fid:04X} v{fver} {FEATURES.get(fid, '?'):<34} {' '.join(flags)}")

        if 0x0005 in feats:
            n = d.call(0x0005, 0)[0]
            name = b""
            while len(name) < n:
                name += d.call(0x0005, 1, len(name))
            print(f"\nname: {name[:n].decode(errors='replace')!r}")

        if 0x0003 in feats:
            ents = d.call(0x0003, 0)[0]
            for e in range(ents):
                r = d.call(0x0003, 1, e)
                kind = {0: "main-app", 1: "bootloader", 2: "hardware", 3: "touchpad", 4: "optical-sensor",
                        5: "softdevice", 6: "rf-companion"}.get(r[0], f"type{r[0]}")
                prefix = r[1:4].decode(errors="replace").strip("\0")
                print(f"fw[{e}] {kind:<12} {prefix}{r[4]:02X}.{r[5]:02X}.B{r[6]:02X}{r[7]:02X}  raw={hx(r[:16])}")

        if 0x1004 in feats:
            r = d.call(0x1004, 1)
            print(f"battery: {r[0]}%  level_flags=0x{r[1]:02X} charging_status={r[2]} external_power={r[3]}")

        if 0x8100 in feats:
            r = d.call(0x8100, 0)
            print(f"\nONBOARD_PROFILES info raw: {hx(r[:16])}")
            try:
                mode = d.call(0x8100, 2)[0]
                print(f"  onboard mode: {mode} ({ {1: 'onboard', 2: 'host'}.get(mode, '?') })")
            except HidppError as e:
                print(f"  getMode: {e}")

        if 0x8071 in feats:
            r = d.call(0x8071, 0, 0xFF, 0xFF, 0x00)
            print(f"\nRGB_EFFECTS getInfo(0xFF,0xFF,0x00) raw: {hx(r[:16])}")

        if 0x8070 in feats:
            r = d.call(0x8070, 0)
            print(f"\nCOLOR_LED_EFFECTS getInfo raw: {hx(r[:16])}")

        if 0x8081 in feats:
            print("\nPER_KEY_LIGHTING_V2 getInfo pages:")
            bitmap = b""
            for page in range(3):
                try:
                    r = d.call(0x8081, 0, 0x00, page)
                    print(f"  page {page}: {hx(r[:16])}")
                    bitmap += r[2:16]
                except HidppError as e:
                    print(f"  page {page}: {e}")
            zones = [i for i in range(len(bitmap) * 8) if (bitmap[i // 8] >> (i % 8)) & 1]
            print(f"  {len(zones)} zone bits set (flat-bitmap interpretation): {zones}")

        if d.rtts:
            rt = sorted(d.rtts)
            print(f"\nround-trip over {len(rt)} requests: min {rt[0]*1000:.1f} ms, "
                  f"median {rt[len(rt)//2]*1000:.1f} ms, max {rt[-1]*1000:.1f} ms")
    finally:
        d.close()


def main():
    if len(sys.argv) > 1:
        probe(sys.argv[1], int(sys.argv[2], 0) if len(sys.argv) > 2 else 0xFF)
        return
    devs = find_logitech_hidraw()
    for dv in devs:
        print(f"{dv['node']}: bus=0x{dv['bus']:02X} {dv['vid']:04X}:{dv['pid']:04X} {dv['name']!r} hidpp={dv['hidpp']}")
    for dv in devs:
        if dv["hidpp"] and dv["bus"] == 0x05:
            print()
            probe(dv["node"], 0xFF)


if __name__ == "__main__":
    main()
