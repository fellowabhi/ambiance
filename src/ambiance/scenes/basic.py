"""Trivial scenes: one colour everywhere, or everything off."""

from __future__ import annotations

from typing import Any

from ambiance.color import BLACK, RGB
from ambiance.core import Frame, RenderContext, Scene


class SolidScene(Scene):
    """Every key of the layout in one colour."""

    name = "solid"

    def __init__(self, color: RGB) -> None:
        self.color = RGB(*color)

    def render(self, ctx: RenderContext) -> Frame:
        return {key.zone: self.color for key in ctx.layout.keys}

    def key(self) -> Any:
        return ("SolidScene", self.color)

    def __repr__(self) -> str:
        return f"SolidScene({self.color.hex})"


class OffScene(Scene):
    """All keys black.

    "Off" is painted, never requested through power-mode commands: black keeps the host in
    control (so the firmware can't wake the LEDs into its own effect) and costs a single
    HID++ range write.
    """

    name = "off"

    def render(self, ctx: RenderContext) -> Frame:
        return {key.zone: BLACK for key in ctx.layout.keys}

    def key(self) -> Any:
        return ("OffScene",)

    def __repr__(self) -> str:
        return "OffScene()"
