"""Virtual display abstractions for background GUI execution.

Defines :class:`DisplayInfo` (immutable display metadata) and
:class:`VirtualDisplayManager` (async lifecycle protocol).
"""

from __future__ import annotations

import dataclasses
import typing


@dataclasses.dataclass(frozen=True)
class DisplayInfo:
    """Immutable metadata for a virtual display."""

    display_id: str  # e.g. ":99" for Xvfb, desktop handle for Windows
    width: int
    height: int
    offset_x: int = 0  # global coordinate offset for the isolated surface
    offset_y: int = 0
    monitor_index: int = 1  # mss monitor index


@typing.runtime_checkable
class VirtualDisplayManager(typing.Protocol):
    """Async lifecycle protocol for virtual display backends."""

    async def start(self) -> DisplayInfo: ...

    async def stop(self) -> None: ...
