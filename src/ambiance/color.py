"""Colour primitives shared by sources, scenes and devices."""

from __future__ import annotations

from typing import NamedTuple

_NAMED = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 200, 0),
    "orange": (255, 90, 0),
    "purple": (160, 0, 255),
    "magenta": (255, 0, 255),
    "pink": (255, 60, 140),
    "cyan": (0, 255, 255),
    "teal": (0, 160, 140),
    "warm": (255, 150, 60),
    "off": (0, 0, 0),
}


class RGB(NamedTuple):
    """8-bit sRGB colour. A tuple, so it is hashable and cheap to group by."""

    r: int
    g: int
    b: int

    @classmethod
    def parse(cls, text: str) -> "RGB":
        """Parse 'red', '#ff8800', 'ff8800' or '255,136,0'."""
        s = text.strip().lower()
        if s in _NAMED:
            return cls(*_NAMED[s])
        if "," in s:
            parts = [int(p) for p in s.split(",")]
            if len(parts) != 3:
                raise ValueError(f"expected r,g,b — got {text!r}")
            return cls(*parts).clamped()
        s = s.removeprefix("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        if len(s) != 6:
            raise ValueError(f"cannot parse colour {text!r}")
        return cls(int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))

    def clamped(self) -> "RGB":
        return RGB(*(max(0, min(255, int(round(c)))) for c in self))

    def scaled(self, k: float) -> "RGB":
        return RGB(*(max(0, min(255, int(round(c * k)))) for c in self))

    def lerp(self, other: "RGB", t: float) -> "RGB":
        return RGB(*(int(round(a + (b - a) * t)) for a, b in zip(self, other)))

    @property
    def hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


BLACK = RGB(0, 0, 0)
WHITE = RGB(255, 255, 255)
