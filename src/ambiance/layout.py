"""Physical keyboard layouts: every key's lighting zone and position.

Zone IDs follow Logitech's PER_KEY_LIGHTING_V2 numbering, which is the USB HID usage
minus 3 for ordinary keys (A=1 … Z=26, 1…0 = 27…36) and usage minus 0x78 for the
modifiers (LCtrl=104 … RGui=111). Positions are in key units (1u = 19.05 mm), origin top
left, so effects can work spatially (gradients, waves) independent of zone numbering.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class Key:
    name: str     # stable identifier, e.g. "a", "f1", "lshift"
    label: str    # what's printed on the keycap
    zone: int     # lighting zone id
    x: float      # left edge (u)
    y: float      # top edge (u)
    w: float = 1.0
    h: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass(frozen=True)
class Layout:
    name: str
    keys: tuple[Key, ...]

    @cached_property
    def width(self) -> float:
        return max(k.x + k.w for k in self.keys)

    @cached_property
    def height(self) -> float:
        return max(k.y + k.h for k in self.keys)

    @cached_property
    def by_name(self) -> dict[str, Key]:
        return {k.name: k for k in self.keys}

    @cached_property
    def zones(self) -> frozenset[int]:
        return frozenset(k.zone for k in self.keys)

    def norm(self, key: Key) -> tuple[float, float]:
        """Key centre normalised to 0..1 in both axes."""
        return key.cx / self.width, key.cy / self.height

    def group(self, name: str) -> tuple[Key, ...]:
        return tuple(k for k in self.keys if k.name in GROUPS[name])


def _row(y: float, x0: float, items) -> list[Key]:
    """items: (name, label, zone[, width]) tuples, or a float meaning a gap of that width."""
    keys, x = [], x0
    for it in items:
        if isinstance(it, (int, float)):
            x += it
            continue
        name, label, zone, *w = it
        width = w[0] if w else 1.0
        keys.append(Key(name, label, zone, x, y, width))
        x += width
    return keys


_LETTERS = {chr(ord("a") + i): 1 + i for i in range(26)}


def _l(c: str):
    return (c, c.upper(), _LETTERS[c])


_NAV_X = 15.25
_F_ROW = [("esc", "Esc", 38), 1.0, ("f1", "F1", 55), ("f2", "F2", 56), ("f3", "F3", 57),
          ("f4", "F4", 58), 0.5, ("f5", "F5", 59), ("f6", "F6", 60), ("f7", "F7", 61),
          ("f8", "F8", 62), 0.5, ("f9", "F9", 63), ("f10", "F10", 64), ("f11", "F11", 65),
          ("f12", "F12", 66)]
_F_NAV = [("print", "PrtSc", 67), ("scroll", "ScrLk", 68), ("pause", "Pause", 69)]
_NUM_ROW = [("grave", "`", 50), ("1", "1", 27), ("2", "2", 28), ("3", "3", 29), ("4", "4", 30),
            ("5", "5", 31), ("6", "6", 32), ("7", "7", 33), ("8", "8", 34), ("9", "9", 35),
            ("0", "0", 36), ("minus", "-", 42), ("equal", "=", 43), ("backspace", "Bksp", 39, 2.0)]
_NUM_NAV = [("insert", "Ins", 70), ("home", "Home", 71), ("pageup", "PgUp", 72)]
_TOP_NAV = [("delete", "Del", 73), ("end", "End", 74), ("pagedown", "PgDn", 75)]
_BOTTOM = [("lctrl", "Ctrl", 104, 1.25), ("lwin", "Win", 107, 1.25), ("lalt", "Alt", 106, 1.25),
           ("space", "Space", 41, 6.25), ("ralt", "Alt", 110, 1.25), ("fn", "Fn", 111, 1.25),
           ("menu", "Menu", 98, 1.25), ("rctrl", "Ctrl", 108, 1.25)]
_ARROWS = [("left", "←", 77), ("down", "↓", 78), ("right", "→", 76)]


def _tkl(name: str, iso: bool) -> Layout:
    k: list[Key] = []
    k += _row(0.0, 0.0, _F_ROW) + _row(0.0, _NAV_X, _F_NAV)
    k += _row(1.25, 0.0, _NUM_ROW) + _row(1.25, _NAV_X, _NUM_NAV)
    top = [("tab", "Tab", 40, 1.5)] + [_l(c) for c in "qwertyuiop"] + \
          [("lbracket", "[", 44), ("rbracket", "]", 45)]
    if not iso:
        top.append(("backslash", "\\", 46, 1.5))
    k += _row(2.25, 0.0, top) + _row(2.25, _NAV_X, _TOP_NAV)
    home = [("caps", "Caps", 54, 1.75)] + [_l(c) for c in "asdfghjkl"] + \
           [("semicolon", ";", 48), ("quote", "'", 49)]
    home += [("hash", "#", 47), ("enter", "Enter", 37, 1.25)] if iso else [("enter", "Enter", 37, 2.25)]
    k += _row(3.25, 0.0, home)
    if iso:
        bottom_alpha = [("lshift", "Shift", 105, 1.25), ("iso_backslash", "\\", 97)]
    else:
        bottom_alpha = [("lshift", "Shift", 105, 2.25)]
    bottom_alpha += [_l(c) for c in "zxcvbnm"] + [("comma", ",", 51), ("period", ".", 52),
                                                 ("slash", "/", 53), ("rshift", "Shift", 109, 2.75)]
    k += _row(4.25, 0.0, bottom_alpha) + [Key("up", "↑", 79, _NAV_X + 1, 4.25)]
    k += _row(5.25, 0.0, _BOTTOM) + _row(5.25, _NAV_X, _ARROWS)
    return Layout(name, tuple(k))


TKL_ANSI = _tkl("tkl-ansi", iso=False)
TKL_ISO = _tkl("tkl-iso", iso=True)
LAYOUTS = {lay.name: lay for lay in (TKL_ANSI, TKL_ISO)}

GROUPS: dict[str, frozenset[str]] = {
    "letters": frozenset(_LETTERS),
    "digits": frozenset("1234567890"),
    "fkeys": frozenset(f"f{i}" for i in range(1, 13)),
    "modifiers": frozenset({"lctrl", "lwin", "lalt", "ralt", "fn", "menu", "rctrl",
                            "lshift", "rshift"}),
    "nav": frozenset({"print", "scroll", "pause", "insert", "home", "pageup", "delete",
                      "end", "pagedown"}),
    "arrows": frozenset({"up", "down", "left", "right"}),
}
