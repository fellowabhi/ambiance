"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# No <servicedir>/<standard_session_servicedirs/>: nothing can be D-Bus-activated on the
# private bus (the stock session config would e.g. spawn GNOME's real org.gnome.ScreenSaver
# helper when a test probes for a missing service).
_BUS_CONFIG = """<!DOCTYPE busconfig PUBLIC "-//freedesktop//DTD D-Bus Bus Configuration 1.0//EN"
 "http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">
<busconfig>
  <type>session</type>
  <listen>unix:path={socket}</listen>
  <auth>EXTERNAL</auth>
  <policy context="default">
    <allow send_destination="*" eavesdrop="true"/>
    <allow eavesdrop="true"/>
    <allow own="*"/>
  </policy>
</busconfig>
"""


class PrivateBus:
    """A throwaway dbus-daemon on a fixed socket path (so restart() simulates the bus
    going away and coming back at the same address)."""

    def __init__(self, executable: str, workdir: Path) -> None:
        self._executable = executable
        self._socket = workdir / "bus"
        self._config = workdir / "bus.conf"
        self._config.write_text(_BUS_CONFIG.format(socket=self._socket))
        self._process: subprocess.Popen[str] | None = None
        self.address = f"unix:path={self._socket}"

    def start(self) -> None:
        self._socket.unlink(missing_ok=True)
        self._process = subprocess.Popen(
            [self._executable, f"--config-file={self._config}", "--nofork", "--print-address"],
            stdout=subprocess.PIPE, text=True)
        assert self._process.stdout is not None
        if not self._process.stdout.readline():  # printed once the socket accepts clients
            self._process.wait(timeout=5)
            raise RuntimeError("dbus-daemon failed to start")

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        self._process.wait(timeout=5)
        assert self._process.stdout is not None
        self._process.stdout.close()
        self._process = None

    def restart(self) -> None:
        self.stop()
        self.start()


@pytest.fixture(scope="module")
def private_bus() -> Iterator[PrivateBus]:
    """A private session-type bus per test module; tests connect via private_bus.address."""
    executable = shutil.which("dbus-daemon")
    if executable is None:
        pytest.skip("dbus-daemon is not installed")
    # mkdtemp rather than tmp_path_factory: AF_UNIX socket paths must stay under ~100 bytes.
    workdir = Path(tempfile.mkdtemp(prefix="ambiance-dbus-"))
    bus = PrivateBus(executable, workdir)
    try:
        bus.start()
        yield bus
    finally:
        bus.stop()
        shutil.rmtree(workdir, ignore_errors=True)
