"""User configuration: an optional TOML file merged over built-in defaults.

The file lives at ``${XDG_CONFIG_HOME:-~/.config}/ambiance/config.toml``; every key is
optional and ``config.example.toml`` (repo root) documents them all. Sections map 1:1 onto
the dataclasses below, so adding an option = adding a dataclass field (plus a line in the
example). Validation errors name the file and the dotted key, e.g.
``config.toml: unknown key 'director.idle_of_s' (did you mean 'idle_off_s'?)``.
"""

from __future__ import annotations

import dataclasses
import difflib
import logging
import math
import os
import tomllib
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ambiance.color import RGB
from ambiance.director import DirectorConfig
from ambiance.layout import LAYOUTS, TKL_ANSI
from ambiance.led import DEFAULT_PROFILE, LedProfile

log = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Invalid configuration; the message names the file and the offending key."""


@dataclass(frozen=True)
class DeviceConfig:
    layout: str = TKL_ANSI.name        # physical layout: "tkl-ansi" | "tkl-iso"
    rescan_interval: float = 2.0       # s between looks for the keyboard while it's absent
    verify_interval: float = 30.0      # s between checks that we still own the lighting
    release_on_exit: bool = True       # hand lighting back to the firmware on shutdown
    brightness: float = 1.0            # 0..1 keyboard light level
    animate: bool = True               # let scenes move on the keyboard (album pan/drift)

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {', '.join(map(repr, sorted(LAYOUTS)))}, "
                             f"got {self.layout!r}")
        for name in ("rescan_interval", "verify_interval"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")
        _check_unit("brightness", self.brightness)


@dataclass(frozen=True)
class LedConfig:
    """See ambiance.led: how screen colours become LED drive values."""

    white_balance: tuple[float, float, float] = DEFAULT_PROFILE.white_balance
    vibrance: float = DEFAULT_PROFILE.vibrance
    min_level: float = DEFAULT_PROFILE.min_level
    max_level: float = DEFAULT_PROFILE.max_level

    def __post_init__(self) -> None:
        if len(self.white_balance) != 3 or not all(v > 0 for v in self.white_balance):
            raise ValueError(f"white_balance must be three numbers > 0, got {self.white_balance!r}")
        if not self.vibrance >= 0:
            raise ValueError(f"vibrance must be >= 0, got {self.vibrance!r}")
        if not 0.0 <= self.min_level <= 1.0:
            raise ValueError(f"min_level must be within 0..1, got {self.min_level!r}")
        if not self.min_level <= self.max_level <= 1.0:
            raise ValueError(f"max_level must be within min_level..1, got {self.max_level!r}")

    def profile(self) -> LedProfile:
        return LedProfile(white_balance=self.white_balance, vibrance=self.vibrance,
                          min_level=self.min_level, max_level=self.max_level)


@dataclass(frozen=True)
class StripConfig:
    """An ELK-BLEDOM / MELK Bluetooth LED strip ("Lotus Lamp X" controllers)."""

    enabled: bool = True                     # drive the strip alongside the keyboard
    address: str = ""                        # Bluetooth address; "" = first name-prefix match
    name_prefixes: tuple[str, ...] = ("MELK-", "ELK-", "LEDBLE")
    max_rate_hz: float = 12.0                # colour writes per second (BLE bursts above ~15)
    smoothing_s: float = 0.3                 # fade between the engine's keyframes
    release_color: str = ""                  # shown when ambiance lets go; "" = keep last
    power_off_on_exit: bool = False
    hw_brightness: int = 100                 # controller brightness set on connect; 0 = leave
    brightness: float = 0.6                  # 0..1 light level (a room light overpowers keys)
    animate: bool = False                    # cycle through the cover's colours (else static)
    # A room light should stay bright: dark cover colours keep their hue at >= 75% level.
    led: LedConfig = field(default_factory=lambda: LedConfig(min_level=0.75))

    def __post_init__(self) -> None:
        _check_unit("brightness", self.brightness)
        if not 0 <= self.hw_brightness <= 100:
            raise ValueError(f"hw_brightness must be within 0..100, got {self.hw_brightness!r}")
        if not 1.0 <= self.max_rate_hz <= 30.0:
            raise ValueError(f"max_rate_hz must be within 1..30, got {self.max_rate_hz!r}")
        if not 0.0 <= self.smoothing_s <= 5.0:
            raise ValueError(f"smoothing_s must be within 0..5, got {self.smoothing_s!r}")
        if self.release_color:
            RGB.parse(self.release_color)  # raises ValueError with a readable message

    def release_rgb(self) -> RGB | None:
        return RGB.parse(self.release_color) if self.release_color else None


@dataclass(frozen=True)
class MouseConfig:
    """A Logitech mouse with zone lighting (COLOR_LED_EFFECTS), e.g. G502 HERO."""

    enabled: bool = True
    brightness: float = 1.0                  # 0..1 light level
    animate: bool = False                    # cycle the logo through the cover's colours

    def __post_init__(self) -> None:
        _check_unit("brightness", self.brightness)


def _check_unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within 0..1, got {value!r}")


@dataclass(frozen=True)
class MediaConfig:
    prefer: tuple[str, ...] = ("spotify",)  # MPRIS player name prefixes, most preferred first
    ignore: tuple[str, ...] = ()            # player name prefixes never followed


@dataclass(frozen=True)
class EngineConfig:
    tick_hz: float = 4.0  # frame rate during transitions/animation

    def __post_init__(self) -> None:
        if not 0 < self.tick_hz <= 60:
            raise ValueError(f"tick_hz must be within (0, 60], got {self.tick_hz!r}")


@dataclass(frozen=True)
class Config:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    led: LedConfig = field(default_factory=LedConfig)
    strip: StripConfig = field(default_factory=StripConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    director: DirectorConfig = field(default_factory=DirectorConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)


def default_path() -> Path:
    from ambiance.paths import config_path

    return config_path()


def load_config(path: Path | str | None = None) -> Config:
    """Load ``path`` (or the default location, if that file exists) over the defaults.

    An explicitly given path must exist; the default one is optional.
    """
    explicit = path is not None
    file = Path(path).expanduser() if explicit else default_path()
    if not file.exists():
        if explicit:
            raise ConfigError(f"config file not found: {file}")
        log.debug("no config file at %s; using defaults", file)
        return Config()
    try:
        with file.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{file}: invalid TOML: {e}") from None
    except OSError as e:
        raise ConfigError(f"{file}: cannot read: {e.strerror or e}") from None
    try:
        cfg = config_from_dict(data)
    except ConfigError as e:
        raise ConfigError(f"{file}: {e}") from None
    log.info("loaded config from %s", file)
    return cfg


def config_from_dict(data: Mapping[str, Any]) -> Config:
    """Validate a parsed TOML document and merge it over the defaults."""
    return _build(Config, data, "")


# ---- generic dataclass <- TOML conversion ------------------------------------------------------

def _type_name(value: Any) -> str:
    return {bool: "a boolean", int: "an integer", float: "a number", str: "a string",
            list: "an array", dict: "a table"}.get(type(value), type(value).__name__)


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"'{where}' must be a table, got {_type_name(data)}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{where}.{key}" if where else key
        if key not in names:
            close = difflib.get_close_matches(key, names, n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            valid = f"valid keys in [{where}]" if where else "valid sections"
            raise ConfigError(f"unknown key '{dotted}'{hint}; {valid}: {', '.join(names)}")
        kwargs[key] = _coerce(hints[key], value, dotted)
    try:
        return cls(**kwargs)
    except ValueError as e:  # semantic checks in __post_init__; messages start with the field
        raise ConfigError(f"{where}.{e}" if where else str(e)) from None


def _coerce(hint: Any, value: Any, where: str) -> Any:
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, where)
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"'{where}' must be true or false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{where}' must be an integer, got {_type_name(value)} {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"'{where}' must be a number, got {_type_name(value)} {value!r}")
        if not math.isfinite(value):
            raise ConfigError(f"'{where}' must be a finite number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"'{where}' must be a string, got {_type_name(value)} {value!r}")
        return value
    if hint is RGB:
        return _parse_color(value, where)
    if typing.get_origin(hint) is tuple:
        args = typing.get_args(hint)
        if not isinstance(value, list):
            raise ConfigError(f"'{where}' must be an array, got {_type_name(value)} {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"'{where}' must have {len(args)} items, got {len(value)}")
        return tuple(_coerce(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    raise TypeError(f"unsupported config field type {hint!r} at {where}")  # programming error


def _parse_color(value: Any, where: str) -> RGB:
    if isinstance(value, str):
        try:
            return RGB.parse(value)
        except ValueError as e:
            raise ConfigError(f"'{where}': cannot parse colour {value!r} ({e}); use a name "
                              f"like 'warm', '#ff9a3c' or '255,154,60'") from None
    if (isinstance(value, list) and len(value) == 3
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 255
                    for v in value)):
        return RGB(*value)
    raise ConfigError(f"'{where}' must be a colour string or [r, g, b] with 0-255 values, "
                      f"got {value!r}")


# ---- saving ----------------------------------------------------------------------------------------

_HEADER = ("# ambiance settings, saved by the ambiance app. Only values that differ from the\n"
           "# defaults are written; config.example.toml documents every option.\n")


def save_config(cfg: Config, path: Path | str | None = None) -> Path:
    """Write the options that differ from the defaults as TOML (atomically)."""
    file = Path(path).expanduser() if path is not None else default_path()
    body = _toml_tables(_diff(cfg, Config()), [])
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(file.suffix + ".tmp")
    tmp.write_text(_HEADER + body)
    tmp.replace(file)
    return file


def _diff(value: Any, default: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in dataclasses.fields(value):
        v, d = getattr(value, f.name), getattr(default, f.name)
        if dataclasses.is_dataclass(v):
            sub = _diff(v, d)
            if sub:
                out[f.name] = sub
        elif v != d:
            out[f.name] = v
    return out


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, RGB):
        return f'"{v.hex}"'
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (tuple, list)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {v!r} as TOML")


def _toml_tables(tree: Mapping[str, Any], prefix: list[str]) -> str:
    scalars = {k: v for k, v in tree.items() if not isinstance(v, Mapping)}
    tables = {k: v for k, v in tree.items() if isinstance(v, Mapping)}
    text = ""
    if scalars and prefix:
        text += f"\n[{'.'.join(prefix)}]\n"
    text += "".join(f"{k} = {_toml_value(v)}\n" for k, v in scalars.items())
    for k, v in tables.items():
        text += _toml_tables(v, [*prefix, k])
    return text
