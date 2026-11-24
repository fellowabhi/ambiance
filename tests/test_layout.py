"""Tests for the physical keyboard layouts (zone ids and key geometry).

The zone ids are checked against the zone set the G515 LS TKL firmware reports
(PER_KEY_LIGHTING_V2 getInfo bitmap, read from the hardware). Some reported zones have
no key on an ANSI board ("phantoms"). Every other reported zone must belong to exactly
one key, or that key would never light.
"""

from __future__ import annotations

import itertools

import pytest

from ambiance.layout import GROUPS, LAYOUTS, TKL_ANSI, TKL_ISO, Key, Layout

G515_REPORTED_ZONES = frozenset(range(1, 80)) | frozenset(range(97, 112)) | {254}
PHANTOM_ZONES = frozenset({47, 97, 99, 100, 101, 102, 103, 254})  # no key on the ANSI board
ISO_ONLY_ZONES = frozenset({47, 97})  # Non-US '#' (HID 0x32) and Non-US '\' (HID 0x64)
ANSI_ONLY_ZONES = frozenset({46})     # ANSI '\' (HID 0x31)

BOTH = pytest.mark.parametrize("layout", [TKL_ANSI, TKL_ISO], ids=lambda lay: lay.name)


def test_key_counts():
    assert len(TKL_ANSI.keys) == 87
    assert len(TKL_ISO.keys) == 88


@BOTH
def test_zones_and_names_are_unique(layout: Layout):
    assert len(layout.zones) == len(layout.keys)
    assert len(layout.by_name) == len(layout.keys)


def test_ansi_uses_exactly_the_non_phantom_reported_zones():
    assert TKL_ANSI.zones <= G515_REPORTED_ZONES - PHANTOM_ZONES
    assert TKL_ANSI.zones == G515_REPORTED_ZONES - PHANTOM_ZONES  # no lightable zone left out
    assert not TKL_ANSI.zones & ISO_ONLY_ZONES


def test_iso_adds_hash_and_iso_backslash_instead_of_ansi_backslash():
    assert ISO_ONLY_ZONES <= TKL_ISO.zones
    assert TKL_ISO.zones <= G515_REPORTED_ZONES - (PHANTOM_ZONES - ISO_ONLY_ZONES)
    assert TKL_ISO.zones == (TKL_ANSI.zones - ANSI_ONLY_ZONES) | ISO_ONLY_ZONES
    assert TKL_ISO.by_name["hash"].zone == 47
    assert TKL_ISO.by_name["iso_backslash"].zone == 97


@pytest.mark.parametrize(("name", "zone"), [
    ("a", 1), ("z", 26), ("1", 27), ("0", 36), ("enter", 37), ("esc", 38), ("space", 41),
    ("f1", 55), ("f12", 66), ("right", 76), ("up", 79), ("menu", 98), ("lctrl", 104),
    ("rshift", 109),
])
def test_zone_numbering_follows_hid_usage(name, zone):
    """Ordinary keys: HID usage - 3 (A = 0x04 -> 1); modifiers: usage - 0x78 (LCtrl -> 104)."""
    assert TKL_ANSI.by_name[name].zone == zone


def _overlap(a: Key, b: Key) -> bool:
    return a.x < b.x + b.w and b.x < a.x + a.w and a.y < b.y + b.h and b.y < a.y + a.h


@BOTH
def test_key_rectangles_do_not_overlap(layout: Layout):
    clashes = [(a.name, b.name) for a, b in itertools.combinations(layout.keys, 2)
               if _overlap(a, b)]
    assert clashes == []


@BOTH
def test_keys_have_positive_size_and_lie_inside_the_board(layout: Layout):
    for k in layout.keys:
        assert k.w > 0 and k.h > 0
        assert 0 <= k.x and k.x + k.w <= layout.width
        assert 0 <= k.y and k.y + k.h <= layout.height


@BOTH
def test_norm_is_within_unit_square(layout: Layout):
    points = [layout.norm(k) for k in layout.keys]
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)
    xs, ys = zip(*points)
    assert min(xs) < 0.1 and max(xs) > 0.9   # the board spans the unit square
    assert min(ys) < 0.15 and max(ys) > 0.85


def test_geometry_matches_a_tkl_board():
    assert TKL_ANSI.width == pytest.approx(18.25)   # 15u alpha block + gap + 3u nav cluster
    assert TKL_ANSI.height == pytest.approx(6.25)   # 6 rows + gap below the F row
    esc, right = TKL_ANSI.by_name["esc"], TKL_ANSI.by_name["right"]
    assert TKL_ANSI.norm(esc)[0] < TKL_ANSI.norm(right)[0]
    assert TKL_ANSI.norm(esc)[1] < TKL_ANSI.norm(right)[1]
    assert TKL_ANSI.by_name["space"].w == pytest.approx(6.25)


@BOTH
def test_rows_are_ordered_top_to_bottom(layout: Layout):
    rows = [["esc", "f1", "f12"], ["grave", "1", "backspace"], ["tab", "q", "p"],
            ["caps", "a", "enter"], ["lshift", "z", "rshift"], ["lctrl", "space", "rctrl"]]
    ys = [{layout.by_name[name].y for name in row} for row in rows]
    assert all(len(y) == 1 for y in ys)
    flat = [y.pop() for y in ys]
    assert flat == sorted(flat) and len(set(flat)) == len(flat)


@BOTH
@pytest.mark.parametrize(("group", "count"), [("letters", 26), ("digits", 10), ("fkeys", 12),
                                              ("modifiers", 9), ("nav", 9), ("arrows", 4)])
def test_groups_resolve_to_keys(layout: Layout, group: str, count: int):
    keys = layout.group(group)
    assert len(keys) == count
    assert {k.name for k in keys} == GROUPS[group]


def test_layout_registry():
    assert LAYOUTS == {"tkl-ansi": TKL_ANSI, "tkl-iso": TKL_ISO}
