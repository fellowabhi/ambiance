"""Linux hidraw discovery via sysfs, plus a tiny HID report-descriptor parser.

Bluetooth HID devices are re-created on every reconnect (new /dev/hidrawN,
new sysfs node), so callers should always discover by IDs, never by path.
"""

from __future__ import annotations

import glob
import os
import threading
from dataclasses import dataclass, field

BUS_USB = 0x03
BUS_BLUETOOTH = 0x05
BUS_NAMES = {BUS_USB: "usb", BUS_BLUETOOTH: "bluetooth", 0x18: "i2c"}


@dataclass(frozen=True)
class ReportInfo:
    """Per report-ID sizes (in bits) and the usage page active when declared."""

    input_bits: dict[int, int] = field(default_factory=dict)
    output_bits: dict[int, int] = field(default_factory=dict)
    usage_pages: dict[int, int] = field(default_factory=dict)


def parse_report_descriptor(rd: bytes) -> ReportInfo:
    """Minimal HID report descriptor walk: enough to find report IDs and sizes."""
    info = ReportInfo()
    usage_page = report_size = report_count = report_id = 0
    stack: list[tuple[int, int, int, int]] = []
    i = 0
    while i < len(rd):
        prefix = rd[i]
        if prefix == 0xFE:  # long item: 0xFE, size, tag, data
            if i + 1 >= len(rd):
                break
            i += 3 + rd[i + 1]
            continue
        size = (0, 1, 2, 4)[prefix & 0x03]
        kind = (prefix >> 2) & 0x03
        tag = prefix >> 4
        data = int.from_bytes(rd[i + 1 : i + 1 + size], "little")
        i += 1 + size
        if kind == 1:  # global
            if tag == 0x0:
                usage_page = data
            elif tag == 0x7:
                report_size = data
            elif tag == 0x8:
                report_id = data
            elif tag == 0x9:
                report_count = data
            elif tag == 0xA:  # push
                stack.append((usage_page, report_size, report_count, report_id))
            elif tag == 0xB and stack:  # pop
                usage_page, report_size, report_count, report_id = stack.pop()
        elif kind == 0:  # main
            bits = report_size * report_count
            if tag == 0x8:  # input
                info.input_bits[report_id] = info.input_bits.get(report_id, 0) + bits
                info.usage_pages.setdefault(report_id, usage_page)
            elif tag == 0x9:  # output
                info.output_bits[report_id] = info.output_bits.get(report_id, 0) + bits
                info.usage_pages.setdefault(report_id, usage_page)
    return info


@dataclass(frozen=True)
class HidrawDevice:
    path: str  # /dev/hidrawN
    sys_path: str  # /sys/class/hidraw/hidrawN
    bus: int
    vendor_id: int
    product_id: int
    name: str
    phys: str
    uniq: str
    report_descriptor: bytes

    @property
    def bus_name(self) -> str:
        return BUS_NAMES.get(self.bus, f"bus{self.bus:#x}")

    @property
    def reports(self) -> ReportInfo:
        return parse_report_descriptor(self.report_descriptor)

    def has_hidpp_long(self) -> bool:
        """True if this node carries Logitech HID++ long reports (id 0x11, 19 bytes)."""
        rep = self.reports
        return (
            rep.input_bits.get(0x11) == 19 * 8
            and rep.output_bits.get(0x11) == 19 * 8
            and rep.usage_pages.get(0x11, 0) >= 0xFF00
        )

    def __str__(self) -> str:
        return (f"{self.path} [{self.bus_name} {self.vendor_id:04x}:{self.product_id:04x}] "
                f"{self.name!r}")


def enumerate_hidraw(vendor_id: int | None = None) -> list[HidrawDevice]:
    devices = []
    for sys_path in sorted(glob.glob("/sys/class/hidraw/hidraw*"), key=_natural):
        try:
            with open(os.path.join(sys_path, "device", "uevent")) as f:
                props = dict(line.split("=", 1) for line in f.read().splitlines() if "=" in line)
            with open(os.path.join(sys_path, "device", "report_descriptor"), "rb") as f:
                rd = f.read()
            bus, vid, pid = (int(x, 16) for x in props["HID_ID"].split(":"))
        except (OSError, KeyError, ValueError):
            continue  # node vanished mid-scan or is malformed
        if vendor_id is not None and vid != vendor_id:
            continue
        devices.append(HidrawDevice(
            path="/dev/" + os.path.basename(sys_path), sys_path=sys_path, bus=bus,
            vendor_id=vid, product_id=pid, name=props.get("HID_NAME", ""),
            phys=props.get("HID_PHYS", ""), uniq=props.get("HID_UNIQ", ""), report_descriptor=rd,
        ))
    return devices


def _natural(path: str) -> int:
    digits = "".join(c for c in os.path.basename(path) if c.isdigit())
    return int(digits) if digits else 0


# ---- process-wide ownership of hidraw nodes -----------------------------------------------------
#
# hidraw delivers every input report to every open file descriptor, and HID++ replies are
# matched on (feature index, function, software id). Two drivers talking to one node at
# the same time (say the mouse driver probing the keyboard's node while the keyboard
# driver claims it) can therefore read each other's replies. Drivers claim a node before
# opening it and skip nodes another driver holds.

_claims_lock = threading.Lock()
_claims: dict[str, str] = {}


def try_claim(path: str, owner: str) -> bool:
    """Claim `path` for `owner`; False if another owner holds it (re-claiming is fine)."""
    with _claims_lock:
        holder = _claims.get(path)
        if holder is not None and holder != owner:
            return False
        _claims[path] = owner
        return True


def release_claim(path: str, owner: str) -> None:
    with _claims_lock:
        if _claims.get(path) == owner:
            del _claims[path]
