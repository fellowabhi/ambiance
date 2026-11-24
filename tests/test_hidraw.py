"""Tests for hidraw discovery and the minimal HID report-descriptor parser.

The parser decides which hidraw node carries HID++ (cli.find_keyboard keeps only nodes
where has_hidpp_long() is True), so it is checked against the real report descriptor
of the G515 LS TKL over Bluetooth. Discovery runs against a fake sysfs tree in tmp_path:
no /dev/hidraw* node is ever opened.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from ambiance.devices import hidraw
from ambiance.devices.hidraw import (BUS_BLUETOOTH, BUS_USB, HidrawDevice, ReportInfo,
                                     enumerate_hidraw, parse_report_descriptor)

# /sys/class/hidraw/hidrawN/device/report_descriptor of a G515 LS TKL (046d:b389) on BT.
G515_BT_DESCRIPTOR = bytes.fromhex(
    "05010906a101851895087501150025010507"
    "19e029e7810295057501150025010508190129059102950391039570750105071904297381029505198729"
    "8b81029503199029928102c005010902a10185150901a10095107501150025010509190129108102950275"
    "1016008026ff7f0501093009318106950175081580257f09388106050c0a38028106c0c0050c0901a10185"
    "0395027510150126ff0219012aff028100c00647ff0901a10185f095047508150026ff000901b102c00643"
    "ff0a0202a101851195137508150026ff000902810009029100c0"
)

# HID 1.11 spec, appendix B.1: boot keyboard. No report IDs, no vendor page.
BOOT_KEYBOARD_DESCRIPTOR = bytes.fromhex(
    "05010906a101050719e029e71500250175019508810295017508810195057501050819012905"
    "910295017503910195067508150025650507190029658100c0"
)

HIDPP_LONG_BITS = 19 * 8  # report 0x11: 1 id byte + 19 payload bytes = 20-byte message


def _vendor_collection(page: int = 0xFF00, report_id: int = 0x11, count: int = 19,
                       with_input: bool = True, with_output: bool = True) -> bytes:
    """One application collection with an 8-bit x `count` report, like Logitech's."""
    rd = bytes((0x06, page & 0xFF, page >> 8,  # Usage Page (2-byte data)
                0x09, 0x01, 0xA1, 0x01,       # Usage, Collection (Application)
                0x85, report_id,              # Report ID
                0x95, count, 0x75, 0x08,      # Report Count, Report Size 8
                0x15, 0x00, 0x26, 0xFF, 0x00))  # Logical Min 0, Max 255
    if with_input:
        rd += bytes((0x09, 0x01, 0x81, 0x00))
    if with_output:
        rd += bytes((0x09, 0x01, 0x91, 0x00))
    return rd + b"\xc0"


def _device(rd: bytes, *, bus: int = BUS_BLUETOOTH, vid: int = 0x046D,
            pid: int = 0xB389) -> HidrawDevice:
    return HidrawDevice(path="/dev/hidraw-not-opened", sys_path="/sys/class/hidraw/x",
                        bus=bus, vendor_id=vid, product_id=pid, name="G515 LS TKL",
                        phys="", uniq="", report_descriptor=rd)


# -- real G515 descriptor ------------------------------------------------------------------------

def test_g515_hidpp_long_report_sizes_and_vendor_page():
    info = parse_report_descriptor(G515_BT_DESCRIPTOR)
    assert info.input_bits[0x11] == HIDPP_LONG_BITS
    assert info.output_bits[0x11] == HIDPP_LONG_BITS
    assert info.usage_pages[0x11] == 0xFF43


def test_g515_other_reports_are_parsed_too():
    """Keyboard (0x18), mouse (0x15) and consumer (0x03) reports share the HID++ node."""
    info = parse_report_descriptor(G515_BT_DESCRIPTOR)
    assert info.input_bits[0x18] == 8 + 112 + 5 + 3   # modifiers + key bitmap: 16 bytes
    assert info.output_bits[0x18] == 5 + 3            # keyboard LEDs + padding
    assert info.input_bits[0x15] == 16 + 2 * 16 + 8 + 8
    assert info.input_bits[0x03] == 2 * 16
    assert 0x10 not in info.input_bits                # BLE exposes no short HID++ report
    # 0xF0 is a feature report only; feature items are not input/output
    assert 0xF0 not in info.input_bits and 0xF0 not in info.output_bits


def test_g515_has_hidpp_long_via_constructed_device():
    dev = _device(G515_BT_DESCRIPTOR)
    assert dev.has_hidpp_long() is True
    assert dev.bus_name == "bluetooth"
    assert "046d:b389" in str(dev) and "bluetooth" in str(dev)


def test_truncated_g515_descriptor_loses_hidpp():
    """A cut-off descriptor must not raise, and must not claim HID++ it never declared."""
    rd = G515_BT_DESCRIPTOR[:-6]  # drops the 0x11 Output item (and End Collection)
    info = parse_report_descriptor(rd)
    assert info.input_bits[0x11] == HIDPP_LONG_BITS
    assert 0x11 not in info.output_bits
    assert _device(rd).has_hidpp_long() is False


# -- has_hidpp_long negatives ---------------------------------------------------------------------

def test_boot_keyboard_without_hidpp():
    info = parse_report_descriptor(BOOT_KEYBOARD_DESCRIPTOR)
    assert info.input_bits == {0: 64}
    assert info.output_bits == {0: 8}
    assert _device(BOOT_KEYBOARD_DESCRIPTOR, bus=BUS_USB).has_hidpp_long() is False


@pytest.mark.parametrize(("rd", "expected"), [
    (_vendor_collection(), True),
    (_vendor_collection(page=0xFF43), True),
    (_vendor_collection(page=0x000C), False),          # report 0x11 on a standard page
    (_vendor_collection(count=7), False),              # wrong size
    (_vendor_collection(with_output=False), False),    # host could not send requests
    (_vendor_collection(with_input=False), False),     # host could not read replies
    (_vendor_collection(report_id=0x12), False),
    (b"", False),
], ids=["ff00", "ff43", "consumer-page", "7-bytes", "input-only", "output-only",
        "other-id", "empty"])
def test_has_hidpp_long_requires_vendor_page_size_and_both_directions(rd, expected):
    assert _device(rd).has_hidpp_long() is expected


# -- parser mechanics ------------------------------------------------------------------------------

def test_long_item_is_skipped_exactly():
    """Long items (0xFE size tag data...) are opaque: their data must not be parsed.

    The payload mimics a report 0x22 declaration and ends with 0x95 (Report Count,
    one data byte), so skipping one byte too few or too many garbles report 0x11.
    """
    payload = bytes((0x85, 0x22, 0x75, 0x08, 0x95, 0x05, 0x81, 0x95))
    long_item = bytes((0xFE, len(payload), 0xF0)) + payload
    empty_long_item = bytes((0xFE, 0x00, 0xF1))
    rd = (bytes((0x06, 0x00, 0xFF)) + long_item + empty_long_item
          + _vendor_collection()[3:])  # collection without its own usage page item
    info = parse_report_descriptor(rd)
    assert 0x22 not in info.input_bits
    assert info.input_bits == {0x11: HIDPP_LONG_BITS}
    assert info.output_bits == {0x11: HIDPP_LONG_BITS}
    assert info.usage_pages[0x11] == 0xFF00


@pytest.mark.parametrize("rd", [b"\xfe", b"\xfe\x10", b"\xfe\x10\xf0\x01", b"\x26\xff",
                                b"\x27\x01\x02", b"\x85"])
def test_truncated_items_do_not_raise(rd):
    assert isinstance(parse_report_descriptor(rd), ReportInfo)


def test_push_pop_restores_global_state():
    rd = bytes((
        0x05, 0x01,              # Usage Page (Generic Desktop)
        0x85, 0x01,              # Report ID 1
        0x75, 0x08, 0x95, 0x01,  # 1 x 8 bits
        0xA4,                    # Push
        0x06, 0x43, 0xFF,        # Usage Page 0xFF43
        0x85, 0x11,              # Report ID 0x11
        0x95, 0x13,              # 19 x 8 bits
        0x81, 0x00, 0x91, 0x00,  # Input, Output -> report 0x11
        0xB4,                    # Pop: back to page 1, report 1, 1 x 8 bits
        0x81, 0x02,              # Input -> report 1
    ))
    info = parse_report_descriptor(rd)
    assert info.input_bits == {0x11: HIDPP_LONG_BITS, 0x01: 8}
    assert info.output_bits == {0x11: HIDPP_LONG_BITS}
    assert info.usage_pages == {0x11: 0xFF43, 0x01: 0x01}


def test_pop_without_push_is_ignored():
    rd = bytes((0xB4,)) + _vendor_collection()
    assert parse_report_descriptor(rd).input_bits == {0x11: HIDPP_LONG_BITS}


def test_four_byte_items_are_decoded():
    """Item size code 3 means 4 data bytes (e.g. a 32-bit Usage Page)."""
    rd = bytes((0x07, 0x43, 0xFF, 0x00, 0x00,                      # Usage Page, 4 bytes
                0x27, 0xFF, 0x00, 0x00, 0x00,                      # Logical Max, 4 bytes
                0x85, 0x11, 0x77, 0x08, 0x00, 0x00, 0x00,          # Report Size 8, 4 bytes
                0x95, 0x13, 0x81, 0x00, 0x91, 0x00))
    info = parse_report_descriptor(rd)
    assert info.usage_pages[0x11] == 0xFF43
    assert info.input_bits[0x11] == info.output_bits[0x11] == HIDPP_LONG_BITS


def test_reports_without_report_id_use_id_0():
    rd = bytes((0x06, 0x00, 0xFF, 0x75, 0x08, 0x95, 0x04, 0x81, 0x00))
    info = parse_report_descriptor(rd)
    assert info.input_bits == {0: 32}


# -- discovery over a fake sysfs ---------------------------------------------------------------------

def _fake_node(root: Path, name: str, uevent: str | None,
               descriptor: bytes | None = G515_BT_DESCRIPTOR) -> Path:
    node = root / name
    (node / "device").mkdir(parents=True)
    if uevent is not None:
        (node / "device" / "uevent").write_text(uevent)
    if descriptor is not None:
        (node / "device" / "report_descriptor").write_bytes(descriptor)
    return node


@pytest.fixture
def fake_sysfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point enumerate_hidraw's glob at tmp_path; the real sysfs and /dev are never read."""
    root = tmp_path / "sys" / "class" / "hidraw"
    root.mkdir(parents=True)

    def fake_glob(pattern: str) -> list[str]:
        assert pattern == "/sys/class/hidraw/hidraw*"
        return [str(p) for p in root.iterdir()]

    monkeypatch.setattr(hidraw, "glob", types.SimpleNamespace(glob=fake_glob))
    return root


def test_enumerate_hidraw_filters_sorts_and_skips_broken_nodes(fake_sysfs: Path):
    g515 = ("DRIVER=hid-generic\nHID_ID=0005:0000046D:0000B389\nHID_NAME=G515 LS TKL\n"
            "HID_PHYS=aa:bb:cc:dd:ee:ff\nHID_UNIQ=11:22:33:44:55:66\n")
    _fake_node(fake_sysfs, "hidraw10", g515)
    _fake_node(fake_sysfs, "hidraw2",
               "HID_ID=0003:0000046D:0000C547\nHID_NAME=USB Receiver\n",
               _vendor_collection())
    _fake_node(fake_sysfs, "hidraw3", "HID_ID=0003:00001234:00005678\nHID_NAME=Other\n",
               BOOT_KEYBOARD_DESCRIPTOR)
    _fake_node(fake_sysfs, "hidraw4", "HID_NAME=no id line\n")             # malformed
    _fake_node(fake_sysfs, "hidraw5", "HID_ID=0005:zz:0000B389\n")         # bad hex
    _fake_node(fake_sysfs, "hidraw6", None)                                # no uevent
    _fake_node(fake_sysfs, "hidraw7", "HID_ID=0005:0000046D:0000B389\n", None)  # no descriptor

    logitech = enumerate_hidraw(0x046D)
    # natural order: hidraw2 before hidraw10 (a plain sort would invert them)
    assert [d.path for d in logitech] == ["/dev/hidraw2", "/dev/hidraw10"]
    kb = logitech[1]
    assert kb.sys_path == str(fake_sysfs / "hidraw10")
    assert (kb.bus, kb.vendor_id, kb.product_id) == (BUS_BLUETOOTH, 0x046D, 0xB389)
    assert (kb.name, kb.phys, kb.uniq) == ("G515 LS TKL", "aa:bb:cc:dd:ee:ff",
                                            "11:22:33:44:55:66")
    assert kb.report_descriptor == G515_BT_DESCRIPTOR
    assert [d.has_hidpp_long() for d in logitech] == [True, True]

    everything = enumerate_hidraw()
    assert [d.path for d in everything] == ["/dev/hidraw2", "/dev/hidraw3", "/dev/hidraw10"]
    assert everything[1].has_hidpp_long() is False
