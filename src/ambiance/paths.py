"""Where ambiance keeps its files.

Running from the source tree (the normal case here), everything lives inside the project
directory: ``config.toml`` at its root and runtime data under ``.ambiance/``. An installed
copy falls back to the XDG locations. ``AMBIANCE_CONFIG`` / ``AMBIANCE_DATA`` override both.
"""

from __future__ import annotations

import os
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]  # <project>/src/ambiance/paths.py -> <project>


def project_dir() -> Path | None:
    """The source checkout this package runs from, if any."""
    return _SRC_ROOT if (_SRC_ROOT / "pyproject.toml").exists() else None


def config_path() -> Path:
    if env := os.environ.get("AMBIANCE_CONFIG"):
        return Path(env).expanduser()
    if (proj := project_dir()) is not None:
        return proj / "config.toml"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "ambiance" / "config.toml"


def data_dir() -> Path:
    if env := os.environ.get("AMBIANCE_DATA"):
        return Path(env).expanduser()
    if (proj := project_dir()) is not None:
        return proj / ".ambiance"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "ambiance"


def art_cache_dir() -> Path:
    return data_dir() / "art"


def log_path() -> Path:
    return data_dir() / "daemon.log"
