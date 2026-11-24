"""Fetch cover art (https:// or file://) with a small on-disk cache."""

from __future__ import annotations

import hashlib
import os
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "ambiance/0.1 (+local ambient lighting)"
MAX_BYTES = 8 * 1024 * 1024


def cache_dir() -> Path:
    from ambiance.paths import art_cache_dir

    return art_cache_dir()


def fetch_art(url: str, timeout: float = 10.0) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        return Path(urllib.parse.unquote(parsed.path)).read_bytes()
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported art URL: {url!r}")
    path = cache_dir() / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".img")
    if path.exists():
        return path.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"art too large: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    _trim(path.parent)
    return data


def _trim(directory: Path, keep: int = 300) -> None:
    files = sorted(directory.glob("*.img"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        old.unlink(missing_ok=True)
