"""Tests for encode_frame: zone colours -> minimal PER_KEY_LIGHTING_V2 (0x8081) writes.

Each HID++ write costs ~47 ms over BLE and pipelining does not help, so the encoder must
use as few writes as possible without changing the picture. `apply_writes` below is a
reference model of what the keyboard does with fn5 (ranges), fn6 (one colour, many
zones) and fn1 (individual zones). The property tests check that random frames decode
back to exactly the requested colours.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping

import pytest

from ambiance.color import RGB
from ambiance.devices.logitech.rgb_keyboard import (FN_FRAME_END, FN_SET_INDIVIDUAL,
                                                    FN_SET_RANGE, FN_SET_SINGLE_VALUE,
                                                    ZONE_MAX, ZONE_MIN, encode_frame)
from ambiance.layout import TKL_ANSI

PARAMS_LEN = 16  # parameter bytes in an HID++ long report; HidppDevice zero-pads to this
ALL_ZONES = range(ZONE_MIN, ZONE_MAX + 1)
RED, GREEN, BLUE, WHITE = RGB(255, 0, 0), RGB(0, 255, 0), RGB(0, 0, 255), RGB(255, 255, 255)

Writes = list[tuple[int, bytes]]
State = dict[int, RGB | None]


def apply_writes(writes: Iterable[tuple[int, bytes]], state: Mapping[int, RGB | None]) -> State:
    """Reference model of the keyboard: apply writes, as they appear on the wire, to a state.

    The model is strict on purpose. An unterminated short fn1 list fails, because the
    firmware would read the zero padding as "zone 0 = black". So does a range or zone
    outside 1..254.
    """
    out = dict(state)

    def paint(zone: int, color: RGB) -> None:
        assert ZONE_MIN <= zone <= ZONE_MAX, f"write touches invalid zone {zone}"
        out[zone] = color

    for fn, params in writes:
        assert len(params) <= PARAMS_LEN, f"fn{fn} params too long: {params.hex()}"
        wire = params.ljust(PARAMS_LEN, b"\x00")
        if fn == FN_SET_RANGE:          # [first last R G B] x3, first == 0 ends the list
            for off in range(0, 15, 5):
                first, last, r, g, b = wire[off:off + 5]
                if first == 0:
                    break
                assert first <= last, f"inverted range {first}..{last}"
                for zone in range(first, last + 1):
                    paint(zone, RGB(r, g, b))
        elif fn == FN_SET_SINGLE_VALUE:  # [R G B z1..z13], zone 0 ends the list
            color = RGB(*wire[:3])
            for zone in wire[3:]:
                if zone == 0:
                    break
                paint(zone, color)
        elif fn == FN_SET_INDIVIDUAL:    # [z R G B] x4, zone 0xFF ends a short list
            for off in range(0, PARAMS_LEN, 4):
                zone = wire[off]
                if zone == 0xFF:
                    break
                assert zone != 0, "short fn1 list without 0xFF terminator"
                paint(zone, RGB(*wire[off + 1:off + 4]))
        else:
            raise AssertionError(f"unexpected 0x8081 function {fn} (frameEnd must not be here)")
    return out


def untouched() -> State:
    return dict.fromkeys(ALL_ZONES)


def check_round_trip(frame: Mapping[int, tuple[int, int, int]], fill: RGB | None = None, *,
                     auto_fill: bool = False) -> Writes:
    """Encode, decode with the reference model and assert the picture is exactly right."""
    writes = encode_frame(frame, fill, auto_fill=auto_fill)
    state = apply_writes(writes, untouched())
    for zone, color in frame.items():
        assert state[zone] == RGB(*color), f"zone {zone}"
    ranges = [params for fn, params in writes if fn == FN_SET_RANGE]
    assert len(ranges) <= 1, "at most one fill range per frame"
    if ranges:
        assert writes[0][0] == FN_SET_RANGE, "the fill must come first or it would overpaint"
    base = RGB(*ranges[0][2:5]) if ranges else None
    if fill is not None:
        assert base == RGB(*fill)
    # Zones absent from the frame get the fill colour, or are left alone if there is no fill.
    for zone in ALL_ZONES:
        if zone not in frame:
            assert state[zone] == base, f"absent zone {zone}"
    # Never worse than naive packing (4 zones per fn1 write) plus one fill write.
    assert len(writes) <= 1 + math.ceil(len(frame) / 4)
    return writes


def fns(writes: Writes) -> list[int]:
    return [fn for fn, _ in writes]


# -- basic shapes ------------------------------------------------------------------------------

def test_empty_frame_needs_no_writes():
    assert encode_frame({}) == []


def test_single_colour_frame_is_one_range_write():
    frame = {z: RGB(10, 20, 30) for z in TKL_ANSI.zones}
    assert encode_frame(frame, auto_fill=True) == [(FN_SET_RANGE, bytes((1, 254, 10, 20, 30)))]


def test_lone_zone_is_not_promoted_to_a_fill():
    """One zone must not trigger a 1..254 range write (that would repaint everything)."""
    assert encode_frame({5: RED}) == [(FN_SET_INDIVIDUAL, bytes((5, 255, 0, 0, 0xFF)))]


def test_most_common_colour_becomes_the_fill():
    zones = sorted(TKL_ANSI.zones)
    frame = {z: BLUE for z in zones[:60]}
    frame |= {z: RED for z in zones[60:80]}
    frame |= {z: GREEN for z in zones[80:]}           # 7 zones
    writes = check_round_trip(frame, auto_fill=True)
    assert writes[0] == (FN_SET_RANGE, bytes((1, 254, *BLUE)))
    assert all(BLUE != RGB(*p[:3]) for fn, p in writes[1:] if fn == FN_SET_SINGLE_VALUE)
    assert fns(writes) == [FN_SET_RANGE, FN_SET_SINGLE_VALUE, FN_SET_SINGLE_VALUE,
                           FN_SET_SINGLE_VALUE]       # red: 13 + 7, green: 7


def test_groups_of_four_or_more_use_fn6_chunks_of_at_most_13():
    zones = list(range(1, 101))
    frame = {z: WHITE for z in zones[:50]}            # fill
    frame |= {z: RED for z in zones[50:80]}           # 30 zones -> 13 + 13 + 4
    writes = encode_frame(frame, auto_fill=True)
    chunks = [p for fn, p in writes if fn == FN_SET_SINGLE_VALUE]
    assert [len(p) - 3 for p in chunks] == [13, 13, 4]
    assert all(p[:3] == bytes(RED) for p in chunks)
    painted = [z for p in chunks for z in p[3:]]
    assert painted == list(range(51, 81))             # ascending, each zone exactly once
    assert len(writes) == 4
    check_round_trip(frame, auto_fill=True)


def test_group_remainder_below_four_is_packed_individually():
    frame = {z: WHITE for z in range(1, 51)}
    frame |= {z: RED for z in range(51, 78)}          # 27 = 13 + 13 + 1 leftover
    writes = check_round_trip(frame, auto_fill=True)
    assert [len(p) - 3 for fn, p in writes if fn == FN_SET_SINGLE_VALUE] == [13, 13]
    assert [p for fn, p in writes if fn == FN_SET_INDIVIDUAL] == [bytes((77, *RED, 0xFF))]


def test_group_of_three_goes_to_individual_writes():
    frame = {z: WHITE for z in range(1, 20)} | {30: RED, 31: RED, 32: RED}
    writes = check_round_trip(frame, auto_fill=True)
    assert fns(writes) == [FN_SET_RANGE, FN_SET_INDIVIDUAL]
    assert writes[1][1] == bytes((30, *RED, 31, *RED, 32, *RED, 0xFF))


def test_leftovers_packed_four_per_fn1_with_terminator_only_when_short():
    colors = [RGB(i, 2 * i, 3 * i) for i in range(1, 7)]
    frame = {z: c for z, c in zip((60, 10, 50, 20, 40, 30), colors)}   # 6 distinct colours
    writes = check_round_trip(frame)
    assert fns(writes) == [FN_SET_INDIVIDUAL, FN_SET_INDIVIDUAL]       # no fill: no repeats
    first, second = writes[0][1], writes[1][1]
    assert len(first) == 16 and 0xFF not in first[0::4]                # full: no terminator
    assert list(first[0::4]) == [10, 20, 30, 40]                       # sorted by zone
    assert second == bytes((50, *frame[50], 60, *frame[60], 0xFF))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 9])
def test_fn1_termination_for_every_length(n):
    frame = {z: RGB(z, 0, 255 - z) for z in range(1, n + 1)}          # all distinct
    writes = check_round_trip(frame)
    assert len(writes) == math.ceil(n / 4)
    for fn, params in writes:
        assert fn == FN_SET_INDIVIDUAL
        assert len(params) == 16 or (len(params) % 4 == 1 and params[-1] == 0xFF)


@pytest.mark.parametrize("zone", [0, -1, 255, 256, 1000])
def test_zone_out_of_range_raises(zone):
    with pytest.raises(ValueError, match="out of range"):
        encode_frame({1: RED, zone: RED})


def test_boundary_zones_are_valid():
    check_round_trip({ZONE_MIN: RED, ZONE_MAX: GREEN})


def test_frame_end_is_never_included():
    """paint() appends frameEnd itself (persistence byte 0); the encoder must not emit it."""
    rng = random.Random(1)
    for _ in range(50):
        frame = {z: RGB(rng.randrange(3), 0, 0) for z in rng.sample(ALL_ZONES, 40)}
        assert FN_FRAME_END not in fns(encode_frame(frame))


# -- explicit fill -----------------------------------------------------------------------------

def test_explicit_fill_on_empty_frame_paints_everything():
    assert encode_frame({}, fill=RGB(1, 2, 3)) == [(FN_SET_RANGE, bytes((1, 254, 1, 2, 3)))]


def test_explicit_fill_drops_zones_already_that_colour():
    writes = check_round_trip({5: BLUE, 6: RED}, fill=BLUE)
    assert writes == [(FN_SET_RANGE, bytes((1, 254, *BLUE))),
                      (FN_SET_INDIVIDUAL, bytes((6, *RED, 0xFF)))]


def test_explicit_fill_overrides_the_most_common_colour():
    frame = {z: RED for z in range(1, 41)} | {50: GREEN}   # red: 13 + 13 + 13 + 1 leftover
    writes = check_round_trip(frame, fill=BLUE)
    assert writes[0] == (FN_SET_RANGE, bytes((1, 254, *BLUE)))
    assert fns(writes) == [FN_SET_RANGE, FN_SET_SINGLE_VALUE, FN_SET_SINGLE_VALUE,
                           FN_SET_SINGLE_VALUE, FN_SET_INDIVIDUAL]
    assert writes[-1][1] == bytes((40, *RED, 50, *GREEN, 0xFF))


def test_plain_tuples_are_accepted_and_grouped_with_rgb():
    writes = check_round_trip({1: (9, 8, 7), 2: RGB(9, 8, 7), 3: (9, 8, 7)}, fill=(0, 0, 0))
    assert fns(writes) == [FN_SET_RANGE, FN_SET_INDIVIDUAL]
    assert writes[1][1] == bytes((1, 9, 8, 7, 2, 9, 8, 7, 3, 9, 8, 7, 0xFF))


def test_partial_frame_leaves_other_zones_alone():
    """Diffs (only the changed zones) must not repaint anything else."""
    writes = encode_frame({1: RED, 2: RED})
    state = apply_writes(writes, dict.fromkeys(ALL_ZONES, GREEN))   # keyboard was green
    assert state[80] == GREEN and state[1] == state[2] == RED


def test_auto_fill_is_opt_in_and_may_repaint_absent_zones():
    """auto_fill=True is for complete frames: the fill repaints zones absent from it."""
    frame = {z: RED for z in range(1, 30)}
    state = apply_writes(encode_frame(frame, auto_fill=True), dict.fromkeys(ALL_ZONES, GREEN))
    assert state[80] == RED


def test_no_fill_unless_asked():
    # Partial frames must never be range-filled: that would repaint every other zone.
    writes = encode_frame({1: RED, 2: RED, 3: BLUE})
    assert len(writes) == 1 and writes[0][0] != FN_SET_RANGE


def test_auto_fill_only_when_it_saves_writes():
    # As individual writes this frame is a single fn1 message (3 entries + terminator).
    assert len(encode_frame({1: RED, 2: RED, 3: BLUE}, auto_fill=True)) == 1
    # Forty red zones plus one blue: fill + one fn1 beats four fn6 writes.
    frame = {**{z: RED for z in range(1, 41)}, 50: BLUE}
    writes = encode_frame(frame, auto_fill=True)
    assert writes[0] == (FN_SET_RANGE, bytes((ZONE_MIN, ZONE_MAX, *RED))) and len(writes) == 2


# -- property-style round trips ----------------------------------------------------------------------

def _random_zones(rng: random.Random) -> list[int]:
    style = rng.choice(("layout", "full", "sparse", "runs"))
    if style == "layout":
        return sorted(TKL_ANSI.zones)
    if style == "full":
        return list(ALL_ZONES)
    if style == "sparse":
        return rng.sample(ALL_ZONES, rng.randint(1, 60))
    zones: set[int] = set()
    for _ in range(rng.randint(1, 6)):
        start = rng.randint(ZONE_MIN, ZONE_MAX)
        zones.update(range(start, min(ZONE_MAX, start + rng.randint(1, 40)) + 1))
    return sorted(zones)


def _random_frame(rng: random.Random) -> tuple[dict[int, RGB], RGB | None]:
    zones = _random_zones(rng)
    n_colors = rng.choice((1, 2, 3, 5, 10, 30, 300))
    palette = [RGB(rng.randrange(256), rng.randrange(256), rng.randrange(256))
               for _ in range(n_colors)]
    weights = [1.0 / (i + 1) ** rng.uniform(0.0, 2.0) for i in range(n_colors)]  # skewed
    frame = {z: rng.choices(palette, weights)[0] for z in zones}
    fill = rng.choice((None, None, rng.choice(palette),
                       RGB(rng.randrange(256), rng.randrange(256), rng.randrange(256))))
    return frame, fill


@pytest.mark.parametrize("seed", range(60))
def test_random_frames_decode_to_requested_colours(seed):
    rng = random.Random(seed)
    for _ in range(8):
        frame, fill = _random_frame(rng)
        check_round_trip(frame, fill)


@pytest.mark.parametrize("seed", range(20))
def test_explicit_fill_is_first_and_the_rest_paints_only_given_zones(seed):
    """With a fill colour no zone uses, writes[0] is the fill range and writes[1:] paint
    exactly the frame's zones (nothing else), still grouped by colour."""
    rng = random.Random(500 + seed)
    for _ in range(5):
        frame, _ = _random_frame(rng)
        spare = next(c for c in (RGB(0, 0, b) for b in range(256)) if c not in frame.values())
        writes = encode_frame(frame, fill=spare)
        assert writes[0] == (FN_SET_RANGE, bytes((ZONE_MIN, ZONE_MAX, *spare)))
        assert FN_SET_RANGE not in fns(writes[1:])
        painted = {z: c for z, c in apply_writes(writes[1:], untouched()).items() if c is not None}
        assert painted == {z: RGB(*c) for z, c in frame.items()}


@pytest.mark.parametrize("seed", range(10))
def test_random_frames_survive_repainting_over_previous_frame(seed):
    """Frames are painted over whatever was there; full frames must fully replace it."""
    rng = random.Random(1000 + seed)
    state = dict.fromkeys(ALL_ZONES, WHITE)
    for _ in range(10):
        frame = {z: rng.choice((RED, GREEN, BLUE, RGB(rng.randrange(256), 0, 0)))
                 for z in TKL_ANSI.zones}
        state = apply_writes(encode_frame(frame), state)
        assert {z: state[z] for z in TKL_ANSI.zones} == frame


def test_full_frame_of_distinct_colours_costs_ceil_n_over_4():
    rng = random.Random(7)
    colors = rng.sample([RGB(r, g, 0) for r in range(256) for g in range(0, 256, 16)], 87)
    frame = dict(zip(sorted(TKL_ANSI.zones), colors))
    writes = check_round_trip(frame)
    assert fns(writes) == [FN_SET_INDIVIDUAL] * math.ceil(87 / 4)
