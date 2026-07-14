"""Phase 9 Plan 01 tests — VirtualDisplayManager protocol and DisplayInfo.

Requirements covered:
  VDISP-01: VirtualDisplayManager protocol (importable, isinstance, async methods)
  VDISP-02: DisplayInfo frozen dataclass (fields, defaults, immutability)
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from guiclaw.backends.virtual_display import DisplayInfo, VirtualDisplayManager


class _FakeDisplayManager:
    async def start(self) -> DisplayInfo:
        return DisplayInfo(display_id="fake", width=800, height=600)

    async def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
# VDISP-01: protocol importability and async shape
# ---------------------------------------------------------------------------


async def test_protocol_importable() -> None:
    """VirtualDisplayManager is importable and supports local test doubles."""
    mgr = _FakeDisplayManager()
    assert isinstance(mgr, VirtualDisplayManager)


async def test_protocol_methods_are_async() -> None:
    """VirtualDisplayManager.start() and stop() are coroutine functions."""
    mgr = _FakeDisplayManager()
    assert inspect.iscoroutinefunction(mgr.start)
    assert inspect.iscoroutinefunction(mgr.stop)


# ---------------------------------------------------------------------------
# VDISP-02: DisplayInfo frozen dataclass
# ---------------------------------------------------------------------------


async def test_display_info_frozen() -> None:
    """DisplayInfo is immutable — assigning any attribute raises FrozenInstanceError."""
    di = DisplayInfo(display_id=":99", width=1920, height=1080)
    with pytest.raises(dataclasses.FrozenInstanceError):
        di.width = 100  # type: ignore[misc]


async def test_display_info_field_names() -> None:
    """DisplayInfo declares the exact fields required by the locked protocol decision."""
    field_names = [f.name for f in dataclasses.fields(DisplayInfo)]
    assert field_names == [
        "display_id",
        "width",
        "height",
        "offset_x",
        "offset_y",
        "monitor_index",
    ]


async def test_display_info_defaults() -> None:
    """Optional fields default to the locked values: offset_x=0, offset_y=0, monitor_index=1."""
    di = DisplayInfo(display_id=":99", width=1920, height=1080)
    assert di.offset_x == 0
    assert di.offset_y == 0
    assert di.monitor_index == 1
