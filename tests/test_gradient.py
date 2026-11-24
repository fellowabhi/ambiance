"""Tests for the palette -> keyboard gradient scene helpers (order_stops, gradient_frame)."""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from ambiance.color import RGB
from ambiance.colorspace import delta_e, srgb_to_oklab
from ambiance.layout import TKL_ANSI, TKL_ISO, Layout
from ambiance.palette import Swatch
from ambiance.scenes.gradient import gradient_frame, order_stops

COLOURS = [(230, 20, 20), (20, 200, 40), (30, 60, 220), (240, 220, 30), (200, 30, 200),
           (30, 200, 210), (250, 140, 20), (255, 255, 255), (10, 10, 10)]


def swatch(rgb: tuple[int, int, int], weight: float = 0.2) -> Swatch:
    lab = tuple(float(v) for v in srgb_to_oklab(np.array(rgb, dtype=float) / 255.0))
    return Swatch(RGB(*rgb), lab, weight, 1.0)


PALETTE5 = [swatch(c, w) for c, w in zip(COLOURS, (0.4, 0.3, 0.15, 0.1, 0.05))]
BOTH = pytest.mark.parametrize("layout", [TKL_ANSI, TKL_ISO], ids=lambda lay: lay.name)


def path_cost(stops) -> float:
    return sum(float(delta_e(a.oklab, b.oklab)) for a, b in zip(stops, stops[1:]))


def near(a: RGB, b: RGB, tol: int = 1) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# -- gradient_frame -------------------------------------------------------------------------------

@BOTH
@pytest.mark.parametrize("n", range(1, 9))
def test_gradient_covers_every_key(layout: Layout, n: int):
    frame = gradient_frame(layout, [swatch(c) for c in COLOURS[:n]])
    assert set(frame) == layout.zones
    assert all(isinstance(c, RGB) and all(0 <= v <= 255 for v in c) for c in frame.values())


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 6, 10, 16])
def test_steps_caps_the_number_of_distinct_colours(steps: int):
    """Fewer distinct colours = fewer fn6 groups = fewer ~47 ms HID++ writes."""
    for phase in (0.0, 0.3):
        frame = gradient_frame(TKL_ANSI, PALETTE5, steps=steps, phase=phase)
        assert set(frame) == TKL_ANSI.zones
        assert len(set(frame.values())) <= steps


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_few_steps_use_every_level(steps: int):
    assert len(set(gradient_frame(TKL_ANSI, PALETTE5, steps=steps).values())) == steps


def test_empty_palette_and_single_swatch():
    assert gradient_frame(TKL_ANSI, []) == {}
    only = swatch((12, 34, 56))
    assert set(gradient_frame(TKL_ANSI, [only]).values()) == {RGB(12, 34, 56)}


def test_gradient_runs_from_first_stop_to_last_stop():
    stops = order_stops(PALETTE5)
    frame = gradient_frame(TKL_ANSI, PALETTE5)
    esc, right = TKL_ANSI.by_name["esc"], TKL_ANSI.by_name["right"]  # extreme projections
    assert near(frame[esc.zone], stops[0].rgb)
    assert near(frame[right.zone], stops[-1].rgb)


def test_vertical_angle_paints_whole_rows_alike():
    frame = gradient_frame(TKL_ANSI, PALETTE5, steps=6, angle_deg=90.0)
    rows: dict[float, set[RGB]] = {}
    for key in TKL_ANSI.keys:
        rows.setdefault(key.cy, set()).add(frame[key.zone])
    assert all(len(colors) == 1 for colors in rows.values())
    assert len({next(iter(c)) for c in rows.values()}) > 1


def test_dominant_colour_covers_more_keys():
    """Stops get room proportional to sqrt(weight)."""
    dark, light = (40, 20, 120), (250, 200, 60)

    def share_of_first(w_dark: float) -> float:
        a, b = swatch(dark, w_dark), swatch(light, 1.0 - w_dark)
        frame = gradient_frame(TKL_ANSI, [a, b], steps=32)
        lab = {c: srgb_to_oklab(np.array(c) / 255.0) for c in set(frame.values())}
        nearer_a = sum(delta_e(lab[c], a.oklab) < delta_e(lab[c], b.oklab) for c in frame.values())
        return nearer_a / len(frame)

    assert share_of_first(0.9) > 0.55
    assert share_of_first(0.1) < 0.45


def test_phase_shifts_the_gradient_but_keeps_coverage():
    base = gradient_frame(TKL_ANSI, PALETTE5, steps=10)
    shifted = gradient_frame(TKL_ANSI, PALETTE5, steps=10, phase=0.37)
    assert set(shifted) == TKL_ANSI.zones
    assert shifted != base
    assert len(set(shifted.values())) <= 10


# -- order_stops --------------------------------------------------------------------------------

@pytest.mark.parametrize("n", range(3, 10))
def test_order_stops_runs_dark_to_light(n: int):
    rng = random.Random(n)
    for _ in range(5):
        stops = [swatch(c) for c in rng.sample(COLOURS, n)]
        ordered = order_stops(stops)
        assert sorted(map(id, ordered)) == sorted(map(id, stops))  # a permutation
        assert ordered[0].lightness <= ordered[-1].lightness


@pytest.mark.parametrize("n", range(3, 7))
def test_order_stops_finds_the_shortest_open_path(n: int):
    rng = random.Random(100 + n)
    stops = [swatch(c) for c in rng.sample(COLOURS, n)]
    best = min(path_cost(p) for p in itertools.permutations(stops))
    assert path_cost(order_stops(stops)) == pytest.approx(best)


def test_order_stops_falls_back_to_lightness_sort_for_many_stops():
    stops = [swatch(c) for c in COLOURS]  # 9 > 7: permutations would be too slow
    ordered = order_stops(stops)
    assert [s.lightness for s in ordered] == sorted(s.lightness for s in stops)


def test_order_stops_small_inputs_are_returned_as_is():
    assert order_stops([]) == []
    one = [swatch((1, 2, 3))]
    assert order_stops(one) == one


def test_order_stops_two_swatches_dark_to_light():
    light, dark = swatch((250, 240, 200)), swatch((20, 10, 60))
    ordered = order_stops([light, dark])
    assert ordered[0].lightness <= ordered[-1].lightness
