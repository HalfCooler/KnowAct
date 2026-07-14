"""Background display target-routing contract coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from guiclaw.action import Action
from guiclaw.backends.background import BackgroundDesktopBackend
from guiclaw.backends.virtual_display import DisplayInfo
from guiclaw.observation import Observation


@pytest.mark.asyncio
async def test_background_wrapper_configures_target_display_before_preflight() -> None:
    display_info = DisplayInfo(
        display_id=":42",
        width=1440,
        height=900,
        offset_x=200,
        offset_y=120,
        monitor_index=2,
    )
    call_order: list[str] = []

    manager = AsyncMock()
    manager.start = AsyncMock(return_value=display_info)
    manager.stop = AsyncMock(side_effect=lambda: call_order.append("stop"))

    inner = AsyncMock()
    type(inner).platform = PropertyMock(return_value="linux")
    inner.configure_target_display = MagicMock(
        side_effect=lambda info: call_order.append(
            f"configure:{'set' if info is display_info else 'clear'}"
        )
    )
    inner.preflight = AsyncMock(side_effect=lambda: call_order.append("preflight"))
    inner.observe = AsyncMock()
    inner.execute = AsyncMock(return_value="ok")

    backend = BackgroundDesktopBackend(inner, manager)
    await backend.preflight()

    assert call_order == ["configure:set", "preflight"]
    inner.configure_target_display.assert_called_once_with(display_info)

    await backend.shutdown()

    assert call_order[-2:] == ["stop", "configure:clear"]
    assert inner.configure_target_display.call_args_list[-1].args == (None,)


@pytest.mark.asyncio
async def test_background_wrapper_preflight_is_idempotent() -> None:
    display_info = DisplayInfo(display_id=":42", width=1440, height=900)

    manager = AsyncMock()
    manager.start = AsyncMock(return_value=display_info)
    manager.stop = AsyncMock()

    inner = AsyncMock()
    type(inner).platform = PropertyMock(return_value="linux")
    inner.configure_target_display = MagicMock()
    inner.preflight = AsyncMock()
    inner.observe = AsyncMock()
    inner.execute = AsyncMock(return_value="ok")

    backend = BackgroundDesktopBackend(inner, manager)
    await backend.preflight()
    await backend.preflight()

    assert manager.start.await_count == 1
    assert inner.preflight.await_count == 1

    await backend.shutdown()


@pytest.mark.asyncio
async def test_target_surface_routing_keeps_observe_and_execute_aligned(
    tmp_path: Path,
) -> None:
    display_info = DisplayInfo(
        display_id=":42",
        width=1440,
        height=900,
        offset_x=200,
        offset_y=120,
        monitor_index=2,
    )
    configured_displays: list[DisplayInfo | None] = []

    manager = AsyncMock()
    manager.start = AsyncMock(return_value=display_info)
    manager.stop = AsyncMock()

    inner = AsyncMock()
    type(inner).platform = PropertyMock(return_value="linux")
    inner.configure_target_display = MagicMock(side_effect=configured_displays.append)
    inner.preflight = AsyncMock()
    inner.observe = AsyncMock(
        side_effect=lambda screenshot_path, timeout=5.0: Observation(
            screenshot_path=str(screenshot_path),
            screen_width=display_info.width,
            screen_height=display_info.height,
            foreground_app="test-app",
            platform="linux",
        )
    )
    inner.execute = AsyncMock(return_value="ok")

    backend = BackgroundDesktopBackend(inner, manager)
    await backend.preflight()

    observation = await backend.observe(tmp_path / "surface.png")
    result = await backend.execute(Action(action_type="tap", x=100.0, y=200.0))

    assert configured_displays == [display_info]
    assert observation.screen_width == 1440
    assert observation.screen_height == 900
    passed_action = inner.execute.call_args.args[0]
    assert passed_action.x == 300.0
    assert passed_action.y == 320.0
    assert result == "ok"

    await backend.shutdown()
