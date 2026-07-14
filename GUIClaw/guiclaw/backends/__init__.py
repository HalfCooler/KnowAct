"""guiclaw.backends — Device backend implementations.

Import individual backends explicitly to avoid pulling platform-specific
dependencies for users who only need a subset::

    from guiclaw.backends.adb import AdbBackend
    from guiclaw.backends.dry_run import DryRunBackend
    from guiclaw.backends.desktop import LocalDesktopBackend
"""

from __future__ import annotations

import struct
from pathlib import Path


def read_png_size(path: Path) -> tuple[int, int] | None:
    """Return PNG image width/height from file header without external deps."""
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) < 24:
        return None
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if header[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        return None
    return width, height
