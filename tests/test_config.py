"""Config loading: defaults, merging, the example file, and validation messages."""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

import pytest

from ambiance.color import RGB
from ambiance.config import (Config, ConfigError, config_from_dict, default_path, load_config)
from ambiance.director import AlbumStyle, DirectorConfig
from ambiance.led import LedProfile

EXAMPLE = Path(__file__).resolve().parent.parent / "config.example.toml"


@pytest.fixture(autouse=True)
def isolated_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never read the real config: not the project's config.toml, not ~/.config."""
    import ambiance.paths

    home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    monkeypatch.delenv("AMBIANCE_CONFIG", raising=False)
    monkeypatch.setattr(ambiance.paths, "project_dir", lambda: None)  # as if installed
    return home


def write(tmp_path: Path, text: str, name: str = "config.toml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def load_text(tmp_path: Path, text: str) -> Config:
    return load_config(write(tmp_path, text))


# ---- locations & defaults ----------------------------------------------------------------------

def test_defaults_when_there_is_no_config_file() -> None:
    assert load_config() == Config()


def test_default_location_follows_xdg_config_home(isolated_xdg: Path) -> None:
    assert default_path() == isolated_xdg / "ambiance" / "config.toml"
    default_path().parent.mkdir(parents=True)
    default_path().write_text("[engine]\ntick_hz = 8\n")
    assert load_config().engine.tick_hz == 8.0


def test_empty_xdg_config_home_means_home_dot_config(monkeypatch: pytest.MonkeyPatch,
                                                     tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_path() == tmp_path / ".config" / "ambiance" / "config.toml"


def test_source_checkout_uses_project_config(monkeypatch: pytest.MonkeyPatch,
                                             tmp_path: Path) -> None:
    import ambiance.paths

    monkeypatch.setattr(ambiance.paths, "project_dir", lambda: tmp_path)
    assert default_path() == tmp_path / "config.toml"


def test_env_var_overrides_config_location(monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: Path) -> None:
    monkeypatch.setenv("AMBIANCE_CONFIG", str(tmp_path / "mine.toml"))
    assert default_path() == tmp_path / "mine.toml"


def test_explicit_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


# ---- the example file --------------------------------------------------------------------------

def _missing_keys(cls: type, data: dict, prefix: str = "") -> list[str]:
    missing = []
    for f in dataclasses.fields(cls):
        dotted = f"{prefix}{f.name}"
        if f.name not in data:
            missing.append(dotted)
        elif isinstance(data[f.name], dict):
            sub = type(getattr(cls(), f.name))
            missing += _missing_keys(sub, data[f.name], dotted + ".")
    return missing


def test_example_documents_every_option() -> None:
    data = tomllib.loads(EXAMPLE.read_text())
    assert _missing_keys(Config, data) == []


def test_example_values_are_the_defaults() -> None:
    assert load_config(EXAMPLE) == Config()


# ---- merging -----------------------------------------------------------------------------------

def test_partial_file_merges_over_defaults(tmp_path: Path) -> None:
    cfg = load_text(tmp_path, """
        [director]
        paused_level = 0.2
        idle_off_s = 120          # integers are fine for float options
        fallback_color = "#102030"

        [director.album]
        style = "mosaic"

        [media]
        prefer = ["firefox", "spotify"]
    """)
    defaults = Config()
    assert cfg.director.paused_level == 0.2
    assert cfg.director.idle_off_s == 120.0 and isinstance(cfg.director.idle_off_s, float)
    assert cfg.director.fallback_color == RGB(0x10, 0x20, 0x30)
    assert cfg.director.album == AlbumStyle(style="mosaic")
    assert cfg.director.playing_level == defaults.director.playing_level
    assert cfg.media.prefer == ("firefox", "spotify") and cfg.media.ignore == ()
    assert cfg.device == defaults.device and cfg.led == defaults.led
    assert cfg.engine == defaults.engine


def test_led_section_builds_a_profile(tmp_path: Path) -> None:
    cfg = load_text(tmp_path, "[led]\nwhite_balance = [1, 0.9, 0.7]\nmin_level = 0.1\n")
    assert cfg.led.profile() == LedProfile(white_balance=(1.0, 0.9, 0.7), vibrance=1.3,
                                           min_level=0.1, max_level=1.0)


@pytest.mark.parametrize("text, expected", [
    ('fallback_color = "teal"', RGB(0, 160, 140)),
    ('fallback_color = "255,136,0"', RGB(255, 136, 0)),
    ('fallback_color = "#f80"', RGB(255, 136, 0)),
    ("fallback_color = [10, 20, 30]", RGB(10, 20, 30)),
])
def test_colour_formats(tmp_path: Path, text: str, expected: RGB) -> None:
    assert load_text(tmp_path, f"[director]\n{text}\n").director.fallback_color == expected


def test_config_from_dict_without_a_file() -> None:
    cfg = config_from_dict({"device": {"layout": "tkl-iso", "release_on_exit": False}})
    assert cfg.device.layout == "tkl-iso" and cfg.device.release_on_exit is False
    assert cfg.director == DirectorConfig()


# ---- validation --------------------------------------------------------------------------------

@pytest.mark.parametrize("text, needles", [
    ("[director]\nidle_of_s = 3", ["director.idle_of_s", "did you mean 'idle_off_s'"]),
    ("[devices]\nlayout = 'tkl-iso'", ["'devices'", "did you mean 'device'"]),
    ("[director.album]\ncolour = 1", ["director.album.colour"]),
    ("tick_hz = 4", ["unknown key 'tick_hz'"]),
])
def test_unknown_keys_are_named(tmp_path: Path, text: str, needles: list[str]) -> None:
    with pytest.raises(ConfigError) as err:
        load_text(tmp_path, text)
    message = str(err.value)
    assert "config.toml" in message
    for needle in needles:
        assert needle in message


@pytest.mark.parametrize("value", ['"not-a-colour"', '"#12345"', '"1,2"', '"gg0000"',
                                   "[1, 2, 300]", "[1, 2]", "7"])
def test_bad_colours_are_rejected_with_the_key(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError, match=r"director\.fallback_color"):
        load_text(tmp_path, f"[director]\nfallback_color = {value}\n")


def test_unparseable_colour_message_explains_formats(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot parse colour 'not-a-colour'"):
        load_text(tmp_path, '[director]\nfallback_color = "not-a-colour"\n')


@pytest.mark.parametrize("text, needle", [
    ('[engine]\ntick_hz = "fast"', "'engine.tick_hz' must be a number"),
    ("[device]\nrelease_on_exit = 1", "'device.release_on_exit' must be true or false"),
    ("[director.album]\nsteps = 2.5", "'director.album.steps' must be an integer"),
    ("[director.album]\nsteps = true", "'director.album.steps' must be an integer"),
    ('[media]\nprefer = "spotify"', "'media.prefer' must be an array"),
    ("[media]\nprefer = [1]", "'media.prefer[0]' must be a string"),
    ("[led]\nwhite_balance = [1, 1]", "'led.white_balance' must have 3 items"),
    ("device = 3", "'device' must be a table"),
    ("[director]\nidle_off_s = nan", "finite"),
])
def test_type_errors_name_the_key(tmp_path: Path, text: str, needle: str) -> None:
    with pytest.raises(ConfigError) as err:
        load_text(tmp_path, text)
    assert needle in str(err.value)


@pytest.mark.parametrize("text, needle", [
    ("[director]\npaused_level = 1.5", "director.paused_level must be within 0..1"),
    ('[director]\nwhen_idle = "sleep"', "director.when_idle must be one of"),
    ('[director.album]\nstyle = "spiral"', "director.album.style must be one of"),
    ("[director.album]\nband = 0", "director.album.band must be within (0, 1]"),
    ('[device]\nlayout = "full-size"', "device.layout must be one of 'tkl-ansi', 'tkl-iso'"),
    ("[device]\nrescan_interval = 0", "device.rescan_interval must be > 0"),
    ("[led]\nmin_level = 0.8\nmax_level = 0.5", "led.max_level must be within min_level..1"),
    ("[led]\nwhite_balance = [1, 0, 1]", "led.white_balance must be three numbers > 0"),
    ("[engine]\ntick_hz = 0", "engine.tick_hz must be within (0, 60]"),
])
def test_out_of_range_values_name_the_key(tmp_path: Path, text: str, needle: str) -> None:
    with pytest.raises(ConfigError) as err:
        load_text(tmp_path, text)
    assert needle in str(err.value)


def test_invalid_toml_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"config\.toml: invalid TOML"):
        load_text(tmp_path, "[director\npaused_level = ")


def test_config_error_is_a_value_error() -> None:
    assert issubclass(ConfigError, ValueError)
