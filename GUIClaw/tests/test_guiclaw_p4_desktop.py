"""Phase 4 desktop backend unit tests.

All pyautogui, mss, and pyperclip calls are fully mocked so tests run in CI
environments that have no display.

Each test corresponds to exactly one behaviour from the BACK-03 specification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from guiclaw.backends.virtual_display import DisplayInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(platform_name: str = "macos") -> Any:
    """Return a LocalDesktopBackend with a pre-set _platform."""
    from guiclaw.backends.desktop import LocalDesktopBackend

    backend = LocalDesktopBackend.__new__(LocalDesktopBackend)
    # Bypass __init__ so we control _platform without calling platform.system()
    # and without triggering lazy pyautogui import.
    backend._screen_width = 1440
    backend._screen_height = 900
    backend._monitor_left = 0
    backend._monitor_top = 0
    backend._platform = platform_name
    backend._target_display = None
    return backend


# ---------------------------------------------------------------------------
# platform property
# ---------------------------------------------------------------------------


def test_desktop_module_import_does_not_load_pyautogui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the backend must not access a real display."""
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pyautogui":
            raise AssertionError("desktop backend imported pyautogui eagerly")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "guiclaw.backends.desktop", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("guiclaw.backends.desktop")

    assert module.pyautogui is None


def test_platform_property_macos() -> None:
    with patch("platform.system", return_value="Darwin"):
        from guiclaw.backends.desktop import LocalDesktopBackend

        backend = LocalDesktopBackend()
    assert backend.platform == "macos"


def test_platform_property_linux() -> None:
    with patch("platform.system", return_value="Linux"):
        from guiclaw.backends.desktop import LocalDesktopBackend

        backend = LocalDesktopBackend()
    assert backend.platform == "linux"


def test_platform_property_windows() -> None:
    with patch("platform.system", return_value="Windows"):
        from guiclaw.backends.desktop import LocalDesktopBackend

        backend = LocalDesktopBackend()
    assert backend.platform == "windows"


# ---------------------------------------------------------------------------
# observe()
# ---------------------------------------------------------------------------


def _build_mss_mock(
    physical_w: int,
    physical_h: int,
    logical_w: int,
    logical_h: int,
    *,
    additional_monitors: list[dict[str, int]] | None = None,
) -> MagicMock:
    """Build a mock mss context manager with the given dimensions."""
    mock_img = MagicMock()
    mock_img.size = (physical_w, physical_h)
    mock_img.bgra = b"\x00" * (physical_w * physical_h * 4)

    monitors = [
        {},  # index 0 is "all monitors"
        {"width": logical_w, "height": logical_h},
    ]
    if additional_monitors:
        monitors.extend(additional_monitors)

    mock_sct = MagicMock()
    mock_sct.monitors = monitors
    mock_sct.grab.return_value = mock_img
    mock_sct.__enter__ = MagicMock(return_value=mock_sct)
    mock_sct.__exit__ = MagicMock(return_value=False)

    mock_mss = MagicMock()
    mock_mss.return_value = mock_sct
    return mock_mss


@pytest.mark.asyncio
async def test_observe_returns_observation(tmp_path: Path) -> None:
    screenshot_path = tmp_path / "screens" / "shot.png"
    backend = _make_backend("macos")

    mock_mss = _build_mss_mock(1440, 900, 1440, 900)
    mock_pil_img = MagicMock()
    mock_pil_img.resize.return_value = mock_pil_img

    with (
        patch("guiclaw.backends.desktop.mss.mss", mock_mss),
        patch("guiclaw.backends.desktop.Image.frombytes", return_value=mock_pil_img),
        patch.object(backend, "_query_foreground_app", AsyncMock(return_value="Finder")),
    ):
        obs = await backend.observe(screenshot_path)

    assert obs.screenshot_path == str(screenshot_path)
    assert obs.screen_width == 1440
    assert obs.screen_height == 900
    assert obs.foreground_app == "Finder"
    assert obs.platform == "macos"


@pytest.mark.asyncio
async def test_observe_writes_png(tmp_path: Path) -> None:
    screenshot_path = tmp_path / "deep" / "nested" / "shot.png"
    backend = _make_backend("macos")

    mock_mss = _build_mss_mock(1440, 900, 1440, 900)
    mock_pil_img = MagicMock()
    mock_pil_img.resize.return_value = mock_pil_img

    with (
        patch("guiclaw.backends.desktop.mss.mss", mock_mss),
        patch("guiclaw.backends.desktop.Image.frombytes", return_value=mock_pil_img),
        patch.object(backend, "_query_foreground_app", AsyncMock(return_value="unknown")),
    ):
        await backend.observe(screenshot_path)

    # Parent directories must exist and save() must have been called with the path
    assert screenshot_path.parent.exists()
    mock_pil_img.save.assert_called_once_with(str(screenshot_path), "PNG")


@pytest.mark.asyncio
async def test_observe_hidpi_downscale(tmp_path: Path) -> None:
    """Physical 2880x1800 captured from mss; logical size is 1440x900 (2× HiDPI).

    The saved image must be resized to 1440x900 and the Observation must report
    the logical resolution.
    """
    screenshot_path = tmp_path / "shot.png"
    backend = _make_backend("macos")

    mock_mss = _build_mss_mock(2880, 1800, 1440, 900)
    mock_pil_img = MagicMock()
    mock_resized = MagicMock()
    mock_pil_img.resize.return_value = mock_resized

    with (
        patch("guiclaw.backends.desktop.mss.mss", mock_mss),
        patch(
            "guiclaw.backends.desktop.Image.frombytes", return_value=mock_pil_img
        ) as frombytes_mock,
        patch.object(backend, "_query_foreground_app", AsyncMock(return_value="unknown")),
    ):
        obs = await backend.observe(screenshot_path)

    # frombytes called with physical dimensions
    frombytes_mock.assert_called_once()
    call_args = frombytes_mock.call_args
    assert call_args[0][1] == (2880, 1800)

    # resize called because physical != logical
    mock_pil_img.resize.assert_called_once()
    resize_args = mock_pil_img.resize.call_args
    assert resize_args[0][0] == (1440, 900)

    # observation reports logical resolution
    assert obs.screen_width == 1440
    assert obs.screen_height == 900

    # resized image was saved, not the original
    mock_resized.save.assert_called_once_with(str(screenshot_path), "PNG")


@pytest.mark.asyncio
async def test_observe_keeps_observation_orientation_in_sync_with_saved_image(
    tmp_path: Path,
) -> None:
    screenshot_path = tmp_path / "orientation-sync.png"
    backend = _make_backend("macos")

    # Simulate a rotated capture where mss physical pixels are portrait while
    # monitor logical dimensions are landscape. observe() should still normalize
    # the saved image to logical dimensions and report matching observation dims.
    mock_mss = _build_mss_mock(900, 1600, 1600, 900)
    mock_pil_img = MagicMock()
    mock_resized = MagicMock()
    mock_pil_img.resize.return_value = mock_resized

    with (
        patch("guiclaw.backends.desktop.mss.mss", mock_mss),
        patch("guiclaw.backends.desktop.Image.frombytes", return_value=mock_pil_img),
        patch.object(backend, "_query_foreground_app", AsyncMock(return_value="Finder")),
    ):
        obs = await backend.observe(screenshot_path)

    mock_pil_img.resize.assert_called_once_with((1600, 900), ANY)
    mock_resized.save.assert_called_once_with(str(screenshot_path), "PNG")
    assert obs.screen_width == 1600
    assert obs.screen_height == 900


@pytest.mark.asyncio
async def test_observe_uses_configured_monitor_index(tmp_path: Path) -> None:
    screenshot_path = tmp_path / "monitor-two.png"
    backend = _make_backend("macos")
    backend.configure_target_display(
        DisplayInfo(
            display_id="macos:42",
            width=1440,
            height=900,
            offset_x=200,
            offset_y=120,
            monitor_index=2,
        )
    )

    second_monitor = {"width": 1600, "height": 1000}
    mock_mss = _build_mss_mock(
        1600,
        1000,
        1440,
        900,
        additional_monitors=[second_monitor],
    )
    mock_pil_img = MagicMock()
    mock_pil_img.resize.return_value = mock_pil_img

    with (
        patch("guiclaw.backends.desktop.mss.mss", mock_mss),
        patch("guiclaw.backends.desktop.Image.frombytes", return_value=mock_pil_img),
        patch.object(backend, "_query_foreground_app", AsyncMock(return_value="Finder")),
    ):
        obs = await backend.observe(screenshot_path)

    mock_mss.return_value.grab.assert_called_once_with(second_monitor)
    assert obs.screen_width == 1600
    assert obs.screen_height == 1000


@pytest.mark.asyncio
async def test_observe_defaults_to_primary_monitor_when_target_display_missing(
    tmp_path: Path,
) -> None:
    screenshot_path = tmp_path / "primary.png"
    backend = _make_backend("linux")

    primary_monitor = {"width": 1440, "height": 900}
    second_monitor = {"width": 1600, "height": 1000}
    mock_mss = _build_mss_mock(
        1440,
        900,
        1440,
        900,
        additional_monitors=[second_monitor],
    )
    mock_pil_img = MagicMock()
    mock_pil_img.resize.return_value = mock_pil_img

    with (
        patch("guiclaw.backends.desktop.mss.mss", mock_mss),
        patch("guiclaw.backends.desktop.Image.frombytes", return_value=mock_pil_img),
        patch.object(backend, "_query_foreground_app", AsyncMock(return_value="unknown")),
    ):
        obs = await backend.observe(screenshot_path)

    mock_mss.return_value.grab.assert_called_once_with(primary_monitor)
    assert obs.screen_width == 1440
    assert obs.screen_height == 900


@pytest.mark.asyncio
async def test_observe_falls_back_to_primary_monitor_when_configured_index_unavailable(
    tmp_path: Path,
) -> None:
    screenshot_path = tmp_path / "monitor-fallback.png"
    backend = _make_backend("macos")
    backend.configure_target_display(
        DisplayInfo(
            display_id="macos:42",
            width=1440,
            height=900,
            offset_x=200,
            offset_y=120,
            monitor_index=3,
        )
    )

    primary_monitor = {"width": 1440, "height": 900}
    second_monitor = {"width": 1600, "height": 1000}
    mock_mss = _build_mss_mock(
        1440,
        900,
        1440,
        900,
        additional_monitors=[second_monitor],
    )
    mock_pil_img = MagicMock()
    mock_pil_img.resize.return_value = mock_pil_img

    with (
        patch("guiclaw.backends.desktop.mss.mss", mock_mss),
        patch("guiclaw.backends.desktop.Image.frombytes", return_value=mock_pil_img),
        patch.object(backend, "_query_foreground_app", AsyncMock(return_value="Finder")),
    ):
        obs = await backend.observe(screenshot_path)

    mock_mss.return_value.grab.assert_called_once_with(primary_monitor)
    assert obs.screen_width == 1440
    assert obs.screen_height == 900


# ---------------------------------------------------------------------------
# execute() — coordinate actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tap() -> None:
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="tap", x=500.0, y=500.0)

    with patch("guiclaw.backends.desktop.pyautogui") as mock_pa:
        result = await backend.execute(action)

    mock_pa.click.assert_called_once_with(500, 500)
    assert result  # describe_action returns a non-empty string


@pytest.mark.asyncio
async def test_execute_double_tap() -> None:
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="double_tap", x=300.0, y=400.0)

    with patch("guiclaw.backends.desktop.pyautogui") as mock_pa:
        await backend.execute(action)

    mock_pa.doubleClick.assert_called_once_with(300, 400)


@pytest.mark.asyncio
async def test_execute_long_press() -> None:
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="long_press", x=200.0, y=300.0)

    with patch("guiclaw.backends.desktop.pyautogui") as mock_pa:
        await backend.execute(action)

    mock_pa.rightClick.assert_called_once_with(200, 300)


@pytest.mark.asyncio
async def test_execute_swipe() -> None:
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="swipe", x=100.0, y=200.0, x2=400.0, y2=600.0, duration_ms=500)

    with patch("guiclaw.backends.desktop.pyautogui") as mock_pa:
        await backend.execute(action)

    mock_pa.mouseDown.assert_called_once_with(100, 200, button="left")
    mock_pa.moveTo.assert_called_once_with(400, 600, duration=0.5)
    mock_pa.mouseUp.assert_called_once_with(400, 600, button="left")


@pytest.mark.asyncio
async def test_execute_scroll_down() -> None:
    """scroll down 240px → 240//120 = 2 clicks, negative (down)."""
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="scroll", text="down", pixels=240, x=500.0, y=500.0)

    with patch("guiclaw.backends.desktop.pyautogui") as mock_pa:
        await backend.execute(action)

    mock_pa.scroll.assert_called_once_with(-2)


@pytest.mark.asyncio
async def test_execute_scroll_small_pixels() -> None:
    """scroll down 60px → 60//120 = 0, clamped to 1 minimum click."""
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="scroll", text="down", pixels=60, x=500.0, y=500.0)

    with patch("guiclaw.backends.desktop.pyautogui") as mock_pa:
        await backend.execute(action)

    mock_pa.scroll.assert_called_once_with(-1)


@pytest.mark.asyncio
async def test_execute_scroll_up() -> None:
    """scroll up → positive clicks."""
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="scroll", text="up", pixels=240, x=500.0, y=500.0)

    with patch("guiclaw.backends.desktop.pyautogui") as mock_pa:
        await backend.execute(action)

    mock_pa.scroll.assert_called_once_with(2)


@pytest.mark.asyncio
async def test_execute_input_text_uses_clipboard_on_macos() -> None:
    """input_text on macOS: pyperclip.copy() then pyautogui.hotkey('command', 'v')."""
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="input_text", text="Hello World")

    with (
        patch("guiclaw.backends.desktop.pyautogui") as mock_pa,
        patch("guiclaw.backends.desktop.pyperclip") as mock_pyperclip,
    ):
        await backend.execute(action)

    mock_pyperclip.copy.assert_called_once_with("Hello World")
    mock_pa.hotkey.assert_called_once_with("command", "v")
    mock_pa.press.assert_not_called()


@pytest.mark.asyncio
async def test_execute_input_text_uses_clipboard_on_linux() -> None:
    """input_text on Linux: pyperclip.copy() then pyautogui.hotkey('ctrl', 'v')."""
    from guiclaw.action import Action

    backend = _make_backend("linux")
    action = Action(action_type="input_text", text="Hello Linux")

    with (
        patch("guiclaw.backends.desktop.pyautogui") as mock_pa,
        patch("guiclaw.backends.desktop.pyperclip") as mock_pyperclip,
    ):
        await backend.execute(action)

    mock_pyperclip.copy.assert_called_once_with("Hello Linux")
    mock_pa.hotkey.assert_called_once_with("ctrl", "v")
    mock_pa.press.assert_not_called()


@pytest.mark.asyncio
async def test_execute_hotkey_normalizes_modifiers() -> None:
    """'cmd' → 'command' on macOS; 'shift' and 'c' pass through unchanged."""
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="hotkey", key=["cmd", "shift", "c"])

    with patch("guiclaw.backends.desktop.pyautogui") as mock_pa:
        with patch("sys.platform", "darwin"):
            await backend.execute(action)

    mock_pa.hotkey.assert_called_once_with("command", "shift", "c")


@pytest.mark.asyncio
async def test_execute_wait() -> None:
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="wait", duration_ms=500)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await backend.execute(action)

    mock_sleep.assert_called_once_with(0.5)


@pytest.mark.asyncio
async def test_execute_done() -> None:
    """done: no side effects, returns describe_action string."""
    from guiclaw.action import Action, describe_action

    backend = _make_backend("macos")
    action = Action(action_type="done", status="success")

    result = await backend.execute(action)

    assert result == describe_action(action)


@pytest.mark.asyncio
async def test_execute_back_macos() -> None:
    """back on macOS → pyautogui.hotkey('command', '[')."""
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="back")

    with patch("guiclaw.backends.desktop.pyautogui") as mock_pa:
        await backend.execute(action)

    mock_pa.hotkey.assert_called_once_with("command", "[")


@pytest.mark.asyncio
async def test_execute_home_macos() -> None:
    """home on macOS → pyautogui.hotkey('command', 'shift', 'h') (Show Desktop)."""
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="home")

    with patch("guiclaw.backends.desktop.pyautogui") as mock_pa:
        await backend.execute(action)

    mock_pa.hotkey.assert_called_once_with("command", "shift", "h")


@pytest.mark.asyncio
async def test_execute_open_app_macos() -> None:
    """open_app on macOS runs: open -a "Safari"."""
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="open_app", text="Safari")

    with patch.object(backend, "_run_cmd", AsyncMock(return_value="")) as mock_run:
        await backend.execute(action)

    mock_run.assert_called_once_with("open", "-a", "Safari", timeout=5.0)


@pytest.mark.asyncio
async def test_execute_close_app_macos() -> None:
    """close_app on macOS: osascript quit, then pkill fallback."""
    from guiclaw.action import Action

    backend = _make_backend("macos")
    action = Action(action_type="close_app", text="Safari")

    with patch.object(backend, "_run_cmd", AsyncMock(return_value="")) as mock_run:
        await backend.execute(action)

    calls = mock_run.call_args_list
    # Should have tried osascript quit first
    assert any("osascript" in str(c) for c in calls)
    # Should also have tried pkill as fallback
    assert any("pkill" in str(c) for c in calls)


# ---------------------------------------------------------------------------
# preflight()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_success() -> None:
    """When pyautogui.position() succeeds, preflight() returns None."""
    backend = _make_backend("macos")

    with patch("guiclaw.backends.desktop.pyautogui") as mock_pa:
        mock_pa.position.return_value = (0, 0)
        result = await backend.preflight()

    assert result is None


@pytest.mark.asyncio
async def test_preflight_raises_on_permission_error() -> None:
    """When pyautogui.position() raises, preflight() raises RuntimeError mentioning Accessibility."""
    backend = _make_backend("macos")

    with patch("guiclaw.backends.desktop.pyautogui") as mock_pa:
        mock_pa.position.side_effect = Exception("cannot connect to display")
        with pytest.raises(RuntimeError, match="Accessibility"):
            await backend.preflight()


# ---------------------------------------------------------------------------
# Relative coordinates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_relative_coordinates() -> None:
    """Action with relative=True and x=500,y=500 resolves to screen center."""
    from guiclaw.action import Action

    backend = _make_backend("macos")
    backend._screen_width = 1440
    backend._screen_height = 900
    # relative=True, x=500, y=500 → 500/999 * (1440-1) ≈ 720, 500/999 * (900-1) ≈ 449
    action = Action(action_type="tap", x=500.0, y=500.0, relative=True)

    with patch("guiclaw.backends.desktop.pyautogui") as mock_pa:
        await backend.execute(action)

    mock_pa.click.assert_called_once()
    x_called, y_called = mock_pa.click.call_args[0]
    # Should be roughly center of 1440x900
    assert 700 <= x_called <= 740
    assert 440 <= y_called <= 460


# ---------------------------------------------------------------------------
# Foreground app (_query_foreground_app)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_foreground_app_macos_returns_app_name() -> None:
    """_query_foreground_app on macOS runs osascript and returns the app name."""
    backend = _make_backend("macos")

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"Safari\n", b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        name = await backend._query_foreground_app()

    assert name == "Safari"


@pytest.mark.asyncio
async def test_foreground_app_returns_unknown_on_error() -> None:
    """_query_foreground_app returns 'unknown' on any subprocess error."""
    backend = _make_backend("macos")

    with patch("asyncio.create_subprocess_exec", side_effect=OSError("no display")):
        name = await backend._query_foreground_app()

    assert name == "unknown"


# ---------------------------------------------------------------------------
# GuiSubagentTool wiring (Task 2 integration test added here for co-location)
# ---------------------------------------------------------------------------


def test_gui_tool_builds_local_backend(tmp_path: Path) -> None:
    """GuiSubagentTool._build_backend('local') returns a LocalDesktopBackend instance."""
    from guiclaw.backends.desktop import LocalDesktopBackend
    from nanobot.agent.tools.gui import GuiSubagentTool
    from nanobot.config.schema import Config

    # Build tool with dry-run so __init__ doesn't fail, then test _build_backend directly.
    class _FakeProvider:
        def get_default_model(self) -> str:
            return "test"

        async def chat(self, messages, **kw): ...
        async def chat_with_retry(self, messages, **kw): ...

    (tmp_path / "gui_runs").mkdir()
    (tmp_path / "gui_skills").mkdir()
    provider = _FakeProvider()
    tool = GuiSubagentTool(
        gui_config=Config(gui={"backend": "dry-run"}).gui,
        provider=provider,
        model="test",
        workspace=tmp_path,
    )

    result = tool._build_backend("local")

    assert isinstance(result, LocalDesktopBackend)
