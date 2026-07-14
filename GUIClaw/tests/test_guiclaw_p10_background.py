"""Phase 10 Plan 01 tests — BackgroundDesktopBackend decorator pattern.

Requirements covered:
  BGND-01: Decorator pattern and lifecycle (isinstance, preflight order, pre-flight guards, context manager)
  BGND-02: DISPLAY environment variable set/restored based on virtual display ID
  BGND-03: Coordinate offset application (absolute shifted, relative skipped, zero-offset passthrough)
  BGND-04: Idempotent and error-suppressing shutdown
"""
from __future__ import annotations

import os
import pathlib
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from guiclaw.action import Action
from guiclaw.backends.background import BackgroundDesktopBackend
from guiclaw.backends.virtual_display import DisplayInfo
from guiclaw.interfaces import DeviceBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_manager(
    display_id: str = ":99",
    offset_x: int = 0,
    offset_y: int = 0,
) -> AsyncMock:
    """Return an AsyncMock that satisfies VirtualDisplayManager.

    .start() resolves to a DisplayInfo; .stop() is a plain AsyncMock.
    """
    mgr = AsyncMock()
    mgr.start = AsyncMock(
        return_value=DisplayInfo(
            display_id=display_id,
            width=1920,
            height=1080,
            offset_x=offset_x,
            offset_y=offset_y,
        )
    )
    mgr.stop = AsyncMock()
    return mgr


def _make_mock_inner(platform: str = "linux") -> AsyncMock:
    """Return an AsyncMock that satisfies DeviceBackend.

    .platform is a PropertyMock; .observe/.execute/.preflight/.list_apps are AsyncMocks.
    """
    inner = AsyncMock()
    type(inner).platform = PropertyMock(return_value=platform)
    inner.observe = AsyncMock(return_value=MagicMock())
    inner.execute = AsyncMock(return_value="ok")
    inner.preflight = AsyncMock()
    inner.list_apps = AsyncMock(return_value=[])
    return inner


# ---------------------------------------------------------------------------
# BGND-01: Decorator pattern and lifecycle
# ---------------------------------------------------------------------------


async def test_isinstance_device_backend() -> None:
    """BackgroundDesktopBackend satisfies the DeviceBackend protocol at runtime."""
    backend = BackgroundDesktopBackend(_make_mock_inner(), _make_mock_manager())
    assert isinstance(backend, DeviceBackend)


async def test_preflight_calls_start_and_inner_preflight() -> None:
    """preflight() must call display_manager.start() before inner.preflight(), in that order."""
    call_order: list[str] = []

    mgr = _make_mock_manager()
    mgr.start = AsyncMock(
        side_effect=lambda: call_order.append("start")
        or DisplayInfo(display_id=":99", width=1920, height=1080),
    )

    inner = _make_mock_inner()
    inner.preflight = AsyncMock(side_effect=lambda: call_order.append("inner_preflight"))

    backend = BackgroundDesktopBackend(inner, mgr)
    await backend.preflight()

    assert call_order == ["start", "inner_preflight"], (
        f"Expected ['start', 'inner_preflight'], got {call_order}"
    )
    mgr.start.assert_called_once()
    inner.preflight.assert_called_once()


async def test_observe_before_preflight_raises() -> None:
    """observe() before preflight() must raise RuntimeError with the lifecycle message."""
    backend = BackgroundDesktopBackend(_make_mock_inner(), _make_mock_manager())
    with pytest.raises(RuntimeError, match="call preflight\\(\\) or use async with before observe/execute"):
        await backend.observe(pathlib.Path("/tmp/shot.png"))


async def test_execute_before_preflight_raises() -> None:
    """execute() before preflight() must raise RuntimeError with the lifecycle message."""
    backend = BackgroundDesktopBackend(_make_mock_inner(), _make_mock_manager())
    action = Action(action_type="tap", x=100.0, y=200.0)
    with pytest.raises(RuntimeError, match="call preflight\\(\\) or use async with before observe/execute"):
        await backend.execute(action)


async def test_async_context_manager() -> None:
    """__aenter__ must call preflight(); __aexit__ must call shutdown()."""
    inner = _make_mock_inner()
    mgr = _make_mock_manager()

    async with BackgroundDesktopBackend(inner, mgr) as backend:
        # preflight() should have been called — display_manager.start() is the proxy
        mgr.start.assert_called_once()
        inner.preflight.assert_called_once()
        assert backend is not None

    # After exiting the context, shutdown() should have been called (mgr.stop once)
    mgr.stop.assert_called_once()


# ---------------------------------------------------------------------------
# BGND-02: DISPLAY environment variable management
# ---------------------------------------------------------------------------


async def test_display_env_set_after_preflight() -> None:
    """After preflight() with display_id=':99', os.environ['DISPLAY'] must equal ':99'."""
    original = os.environ.get("DISPLAY")
    try:
        backend = BackgroundDesktopBackend(_make_mock_inner(), _make_mock_manager(display_id=":99"))
        await backend.preflight()
        assert os.environ.get("DISPLAY") == ":99"
    finally:
        # Restore original DISPLAY to avoid test pollution
        if original is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = original
        await backend.shutdown()


async def test_display_env_restored_after_shutdown() -> None:
    """After shutdown(), DISPLAY is restored to its pre-preflight value (or removed if unset)."""
    original = os.environ.get("DISPLAY")
    try:
        backend = BackgroundDesktopBackend(_make_mock_inner(), _make_mock_manager(display_id=":99"))
        await backend.preflight()
        # DISPLAY is now ":99"; after shutdown it should revert
        await backend.shutdown()
        if original is None:
            assert "DISPLAY" not in os.environ, (
                "DISPLAY was not set before preflight — it must be removed after shutdown"
            )
        else:
            assert os.environ["DISPLAY"] == original, (
                f"DISPLAY should be restored to {original!r}, got {os.environ.get('DISPLAY')!r}"
            )
    finally:
        if original is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = original


async def test_non_x11_display_does_not_set_display_env() -> None:
    """A display ID without an X11 prefix must not change DISPLAY."""
    original = os.environ.get("DISPLAY")
    try:
        # Remove DISPLAY so we can detect a spurious write
        os.environ.pop("DISPLAY", None)
        backend = BackgroundDesktopBackend(_make_mock_inner(), _make_mock_manager(display_id="desktop"))
        await backend.preflight()
        assert "DISPLAY" not in os.environ, (
            "DISPLAY must not be set when display_id does not start with ':'"
        )
    finally:
        if original is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = original
        await backend.shutdown()


# ---------------------------------------------------------------------------
# BGND-03: Coordinate offset application
# ---------------------------------------------------------------------------


async def test_zero_offset_passthrough() -> None:
    """Action coordinates must pass through unchanged when display offset is (0, 0)."""
    inner = _make_mock_inner()
    backend = BackgroundDesktopBackend(inner, _make_mock_manager(offset_x=0, offset_y=0))
    await backend.preflight()

    action = Action(action_type="tap", x=100.0, y=200.0)
    await backend.execute(action)

    passed = inner.execute.call_args[0][0]
    assert passed.x == 100.0, f"Expected x=100.0, got {passed.x}"
    assert passed.y == 200.0, f"Expected y=200.0, got {passed.y}"


async def test_nonzero_offset_applied() -> None:
    """Non-zero display offset must shift absolute coordinates in both tap and swipe actions."""
    inner = _make_mock_inner()
    backend = BackgroundDesktopBackend(inner, _make_mock_manager(offset_x=50, offset_y=30))
    await backend.preflight()

    # Tap: only x/y
    tap = Action(action_type="tap", x=100.0, y=200.0)
    await backend.execute(tap)
    passed_tap = inner.execute.call_args[0][0]
    assert passed_tap.x == 150.0, f"Expected x=150.0, got {passed_tap.x}"
    assert passed_tap.y == 230.0, f"Expected y=230.0, got {passed_tap.y}"

    # Swipe: x/y AND x2/y2 must both be shifted
    swipe = Action(action_type="swipe", x=100.0, y=200.0, x2=300.0, y2=400.0)
    await backend.execute(swipe)
    passed_swipe = inner.execute.call_args[0][0]
    assert passed_swipe.x == 150.0, f"Swipe x: expected 150.0, got {passed_swipe.x}"
    assert passed_swipe.y == 230.0, f"Swipe y: expected 230.0, got {passed_swipe.y}"
    assert passed_swipe.x2 == 350.0, f"Swipe x2: expected 350.0, got {passed_swipe.x2}"
    assert passed_swipe.y2 == 430.0, f"Swipe y2: expected 430.0, got {passed_swipe.y2}"


async def test_relative_action_offset_skipped() -> None:
    """Actions with relative=True must NOT have the display offset applied."""
    inner = _make_mock_inner()
    backend = BackgroundDesktopBackend(inner, _make_mock_manager(offset_x=50, offset_y=30))
    await backend.preflight()

    action = Action(action_type="tap", x=500.0, y=500.0, relative=True)
    await backend.execute(action)

    passed = inner.execute.call_args[0][0]
    assert passed.x == 500.0, f"Relative action x should be unchanged: expected 500.0, got {passed.x}"
    assert passed.y == 500.0, f"Relative action y should be unchanged: expected 500.0, got {passed.y}"
    assert passed.relative is True


# ---------------------------------------------------------------------------
# BGND-04: Idempotent and error-suppressing shutdown
# ---------------------------------------------------------------------------


async def test_shutdown_stops_manager() -> None:
    """The first shutdown() call must invoke mgr.stop() exactly once."""
    mgr = _make_mock_manager()
    backend = BackgroundDesktopBackend(_make_mock_inner(), mgr)
    await backend.preflight()
    await backend.shutdown()
    mgr.stop.assert_called_once()


async def test_shutdown_idempotent() -> None:
    """A second shutdown() must NOT call mgr.stop() again — stop is invoked exactly once total."""
    mgr = _make_mock_manager()
    backend = BackgroundDesktopBackend(_make_mock_inner(), mgr)
    await backend.preflight()

    await backend.shutdown()
    await backend.shutdown()  # second call — must be a no-op

    assert mgr.stop.call_count == 1, (
        f"mgr.stop() should be called exactly once, was called {mgr.stop.call_count} times"
    )


async def test_shutdown_suppresses_stop_error() -> None:
    """If mgr.stop() raises RuntimeError, shutdown() must absorb it and complete normally."""
    mgr = _make_mock_manager()
    mgr.stop = AsyncMock(side_effect=RuntimeError("boom"))

    backend = BackgroundDesktopBackend(_make_mock_inner(), mgr)
    await backend.preflight()

    # Must not propagate the RuntimeError
    await backend.shutdown()


async def test_background_backend_exposes_handoff_target_metadata() -> None:
    inner = _make_mock_inner(platform="linux")
    backend = BackgroundDesktopBackend(
        inner,
        _make_mock_manager(display_id=":77"),
    )

    await backend.preflight()
    try:
        assert backend.get_intervention_target() == {
            "display_id": ":77",
            "monitor_index": 1,
            "width": 1920,
            "height": 1080,
            "platform": "linux",
        }
    finally:
        await backend.shutdown()
