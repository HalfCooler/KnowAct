"""Phase 11 integration tests: GuiConfig background fields and GuiSubagentTool wrapping.

Requirements covered:
  INTG-02: GuiConfig schema validates background + backend constraint at config load time
  INTG-04: GuiSubagentTool.execute() wraps backend in BackgroundDesktopBackend on Linux
  TEST-V11-01: All new tests pass without a real Xvfb binary
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from pydantic import ValidationError

from nanobot.config.schema import GuiConfig
from guiclaw.backends.dry_run import _TINY_PNG

# ---------------------------------------------------------------------------
# GuiConfig schema tests
# ---------------------------------------------------------------------------


def test_guiconfig_background_fields_defaults() -> None:
    """GuiConfig() has background=False, display_num=None, display_width=1280, display_height=720."""
    cfg = GuiConfig()
    assert cfg.background is False
    assert cfg.display_num is None
    assert cfg.display_width == 1280
    assert cfg.display_height == 720


def test_guiconfig_background_with_local() -> None:
    """GuiConfig(backend='local', background=True) succeeds and background is True."""
    cfg = GuiConfig(backend="local", background=True)
    assert cfg.background is True


def test_guiconfig_background_requires_local() -> None:
    """GuiConfig(backend='adb', background=True) raises ValidationError mentioning 'local'."""
    with pytest.raises(ValidationError, match="background mode requires backend='local'"):
        GuiConfig(backend="adb", background=True)


def test_guiconfig_background_rejects_dryrun() -> None:
    """GuiConfig(backend='dry-run', background=True) raises ValidationError."""
    with pytest.raises(ValidationError, match="background mode requires backend='local'"):
        GuiConfig(backend="dry-run", background=True)


def test_guiconfig_camel_case_aliases() -> None:
    """GuiConfig accepts camelCase aliases for background display fields."""
    cfg = GuiConfig(
        backend="local", background=True, displayWidth=1920, displayHeight=1080, displayNum=42
    )
    assert cfg.display_width == 1920
    assert cfg.display_height == 1080
    assert cfg.display_num == 42


def test_guiconfig_skill_extraction_defaults_to_false() -> None:
    """GuiConfig() keeps skill extraction off unless explicitly enabled."""
    cfg = GuiConfig()
    assert cfg.enable_skill_extraction is False


# ---------------------------------------------------------------------------
# GuiSubagentTool.execute() wrapping tests
# ---------------------------------------------------------------------------


def _make_gui_tool(background: bool = False, **config_kwargs: Any) -> Any:
    """Build a GuiSubagentTool with mocked internals for execute() wrapping tests.

    Patches _build_backend and _get_skill_library so no real guiclaw backend is
    instantiated. The returned tool has gui_config.background set to the given value.
    """
    from nanobot.agent.tools.gui import GuiSubagentTool

    gui_config = GuiConfig(backend="local", background=background, **config_kwargs)

    mock_provider = MagicMock()
    mock_provider.api_key = "fake-key"
    mock_provider.api_base = None
    mock_provider.extra_headers = None

    mock_backend = MagicMock()
    mock_backend.platform = "linux"

    mock_skill_lib = MagicMock()

    with (
        patch.object(GuiSubagentTool, "_build_backend", return_value=mock_backend),
        patch.object(GuiSubagentTool, "_get_skill_library", return_value=mock_skill_lib),
    ):
        tool = GuiSubagentTool(
            gui_config=gui_config,
            provider=mock_provider,
            model="test-model",
            workspace=Path("/tmp/test_workspace"),
        )

    return tool


class _ScriptedNanobotProvider:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Any = None,
        tool_choice: Any = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        del tools, tool_choice, model, max_tokens
        if _is_workflow_planner_call(messages):
            from guiclaw.interfaces import LLMResponse

            return LLMResponse(content='{"mode":"single","subtasks":[]}')
        if not self._responses:
            raise AssertionError("No scripted nanobot responses left.")
        return _profile_response(self._responses.pop(0))


def _is_workflow_planner_call(messages: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    first_content = messages[0].get("content", "")
    return isinstance(first_content, str) and "narrow GUI task router" in first_content


def _profile_response(response: Any) -> Any:
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return response
    action = dict(tool_calls[0].arguments)
    action_type = action.get("action_type")
    if action_type == "done":
        goal_status = "complete" if action.get("status", "success") == "success" else "failed"
        profile_action = {"action_type": "status", "goal_status": goal_status}
    elif action_type == "wait":
        profile_action = {"action_type": "wait"}
    elif action_type == "request_intervention":
        profile_action = {"action_type": "request_intervention", "text": action.get("text", "")}
    elif action_type == "input_text":
        profile_action = {"action_type": "input_text", "text": action.get("text", "")}
    else:
        profile_action = action
    return replace(
        response,
        content=f"Thought: {response.content}\nAction: {json.dumps(profile_action, ensure_ascii=False)}",
    )


class _InterventionBackend:
    platform = "linux"

    def __init__(self) -> None:
        from guiclaw.observation import Observation

        self._Observation = Observation
        self.preflight = AsyncMock()
        self.execute = AsyncMock(return_value="execute should not run")
        self.list_apps = AsyncMock(return_value=[])
        self._observations = [
            {
                "foreground_app": "Payroll Login",
                "extra": {"display_id": ":88", "session_token": "secret-session-token"},
            },
            {
                "foreground_app": "Payroll Home",
                "extra": {"display_id": ":88"},
            },
        ]

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Any:
        del timeout
        if not self._observations:
            raise AssertionError("No scripted observations left.")
        payload = self._observations.pop(0)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_path.write_bytes(_TINY_PNG)
        return self._Observation(
            screenshot_path=str(screenshot_path),
            screen_width=1280,
            screen_height=720,
            foreground_app=payload["foreground_app"],
            platform=self.platform,
            extra=payload["extra"],
        )

    def get_intervention_target(self) -> dict[str, Any]:
        return {
            "display_id": ":88",
            "monitor_index": 1,
            "width": 1280,
            "height": 720,
            "platform": "linux",
            "session_token": "secret-session-token",
        }


@pytest.mark.asyncio
async def test_gui_tool_execute_background_wraps_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Linux, execute() with gui_config.background=True wraps backend in BackgroundDesktopBackend."""
    # Force platform to linux
    monkeypatch.setattr(sys, "platform", "linux")

    tool = _make_gui_tool(background=True)
    inner_backend = tool._backend

    # Build mocks for BackgroundDesktopBackend and XvfbDisplayManager
    mock_xvfb_cls = MagicMock()
    mock_xvfb_instance = MagicMock()
    mock_xvfb_cls.return_value = mock_xvfb_instance

    mock_bg_instance = MagicMock()
    mock_bg_instance.platform = "linux"
    mock_bg_instance.shutdown = AsyncMock()

    mock_bg_cls = MagicMock(return_value=mock_bg_instance)

    canned_result = json.dumps(
        {"success": True, "summary": "done", "trace_path": None, "steps_taken": 1, "error": None}
    )

    with (
        patch(
            "nanobot.agent.tools.gui.GuiSubagentTool._run_task",
            new=AsyncMock(return_value=canned_result),
        ) as mock_run_task,
        patch(
            "guiclaw.backends.background_runtime.probe_isolated_background_support",
            return_value=MagicMock(
                supported=True,
                reason_code="xvfb_available",
                backend_name="xvfb",
            ),
        ),
        patch("guiclaw.backends.background.BackgroundDesktopBackend", mock_bg_cls),
        patch("guiclaw.backends.displays.xvfb.XvfbDisplayManager", mock_xvfb_cls),
    ):
        result = await tool.execute("test task")

    assert result == canned_result
    # BackgroundDesktopBackend must have been constructed with inner backend and xvfb manager
    mock_bg_cls.assert_called_once_with(
        inner_backend,
        mock_xvfb_instance,
        run_metadata={"owner": "nanobot", "task": "test task", "model": "test-model"},
    )
    mock_bg_instance.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_gui_tool_execute_background_nonlinux_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fallback acknowledgement keeps the raw backend path and logs the resolved mode."""
    monkeypatch.setattr(sys, "platform", "darwin")

    tool = _make_gui_tool(background=True)
    raw_backend = tool._backend

    canned_result = json.dumps(
        {"success": True, "summary": "done", "trace_path": None, "steps_taken": 1, "error": None}
    )

    with (
        patch(
            "nanobot.agent.tools.gui.GuiSubagentTool._run_task",
            new=AsyncMock(return_value=canned_result),
        ) as mock_run_task,
        patch(
            "guiclaw.backends.background_runtime.probe_isolated_background_support",
            return_value=MagicMock(
                supported=False,
                reason_code="platform_unsupported",
                retryable=False,
                host_platform="macos",
                backend_name=None,
                sys_platform="darwin",
            ),
        ),
        caplog.at_level(logging.WARNING, logger="nanobot.agent.tools.gui"),
    ):
        result = await tool.execute("test task", acknowledge_background_fallback=True)

    assert result == canned_result
    called_backend = mock_run_task.call_args[0][0]
    assert called_backend is raw_backend
    assert "background runtime resolved:" in caplog.text
    assert "mode=fallback" in caplog.text


@pytest.mark.asyncio
async def test_gui_tool_execute_no_background() -> None:
    """With gui_config.background=False, execute() calls _run_task with the raw backend."""
    tool = _make_gui_tool(background=False)
    raw_backend = tool._backend

    canned_result = json.dumps(
        {"success": True, "summary": "ok", "trace_path": None, "steps_taken": 0, "error": None}
    )

    with patch(
        "nanobot.agent.tools.gui.GuiSubagentTool._run_task",
        new=AsyncMock(return_value=canned_result),
    ) as mock_run_task:
        result = await tool.execute("test task")

    assert result == canned_result
    # _run_task must be called with the unwrapped raw backend
    mock_run_task.assert_called_once()
    called_backend = mock_run_task.call_args[0][0]
    assert called_backend is raw_backend


@pytest.mark.asyncio
async def test_gui_tool_requires_ack_for_background_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tool = _make_gui_tool(background=True)
    raw_backend = tool._backend
    monkeypatch.setattr(sys, "platform", "darwin")

    with patch(
        "guiclaw.backends.background_runtime.probe_isolated_background_support",
        return_value=MagicMock(
            supported=False,
            reason_code="platform_unsupported",
            retryable=False,
            host_platform="macos",
            backend_name=None,
            sys_platform="darwin",
        ),
    ):
        with patch(
            "nanobot.agent.tools.gui.GuiSubagentTool._run_task", new=AsyncMock()
        ) as mock_run_task:
            with caplog.at_level(logging.WARNING, logger="nanobot.agent.tools.gui"):
                payload = json.loads(await tool.execute("open settings"))

            assert payload["success"] is False
            assert "resolved to fallback" in payload["summary"]
            assert "platform_unsupported" in payload["summary"]
            assert (
                "Run without background isolation on this host until a supported isolated backend exists."
                in payload["summary"]
            )
            assert "acknowledge_background_fallback=true" in payload["summary"]
            assert "background_mode" not in payload
            mock_run_task.assert_not_awaited()
            assert "mode=fallback" in caplog.text
            assert "reason=platform_unsupported" in caplog.text

        with patch(
            "nanobot.agent.tools.gui.GuiSubagentTool._run_task",
            new=AsyncMock(
                return_value=json.dumps(
                    {
                        "success": True,
                        "summary": "done",
                        "trace_path": None,
                        "steps_taken": 1,
                        "error": None,
                    }
                )
            ),
        ) as mock_run_task:
            await tool.execute("open settings", acknowledge_background_fallback=True)
            assert mock_run_task.await_count == 1
            assert mock_run_task.await_args.args[0] is raw_backend


@pytest.mark.asyncio
async def test_gui_tool_reports_busy_waiting_metadata_for_serialized_background_runs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import guiclaw.backends.background_runtime as runtime
    from guiclaw.backends.virtual_display import DisplayInfo

    tool = _make_gui_tool(background=True)
    inner_backend = AsyncMock()
    type(inner_backend).platform = PropertyMock(return_value="linux")
    inner_backend.preflight = AsyncMock()
    inner_backend.observe = AsyncMock()
    inner_backend.execute = AsyncMock(return_value="ok")
    inner_backend.list_apps = AsyncMock(return_value=[])
    tool._backend = inner_backend

    fresh_coordinator = runtime.BackgroundRuntimeCoordinator()
    monkeypatch.setattr(runtime, "GLOBAL_BACKGROUND_RUNTIME_COORDINATOR", fresh_coordinator)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        runtime,
        "probe_isolated_background_support",
        lambda **_: runtime.IsolationProbeResult(
            supported=True,
            reason_code="xvfb_available",
            retryable=False,
            host_platform="linux",
            backend_name="xvfb",
            sys_platform="linux",
        ),
    )

    class FakeXvfbDisplayManager:
        def __init__(self, display_num: int = 99, width: int = 1280, height: int = 720) -> None:
            self.display_num = display_num
            self.width = width
            self.height = height

        async def start(self) -> DisplayInfo:
            return DisplayInfo(display_id=":99", width=self.width, height=self.height)

        async def stop(self) -> None:
            return None

    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def fake_run_task(active_backend: Any, task: str, **kwargs: Any) -> str:
        await active_backend.preflight()
        if task == "first":
            first_started.set()
            await release_first.wait()
        else:
            await first_started.wait()
        return json.dumps(
            {"success": True, "summary": task, "trace_path": None, "steps_taken": 1, "error": None}
        )

    with (
        patch("guiclaw.backends.displays.xvfb.XvfbDisplayManager", FakeXvfbDisplayManager),
        patch(
            "nanobot.agent.tools.gui.GuiSubagentTool._run_task",
            new=AsyncMock(side_effect=fake_run_task),
        ),
        caplog.at_level(logging.WARNING, logger="guiclaw.backends.background_runtime"),
    ):
        first_task = asyncio.create_task(
            tool.execute("first", acknowledge_background_fallback=True)
        )
        await first_started.wait()
        second_task = asyncio.create_task(
            tool.execute("second", acknowledge_background_fallback=True)
        )
        await asyncio.sleep(0.05)
        release_first.set()
        await asyncio.gather(first_task, second_task)

    busy_messages = [
        record.message
        for record in caplog.records
        if record.message.startswith("background runtime busy:")
    ]
    assert busy_messages
    assert "waiting_owner=nanobot" in busy_messages[0]
    assert "active_owner=nanobot" in busy_messages[0]
    assert "active_task=first" in busy_messages[0]


@pytest.mark.asyncio
async def test_gui_tool_uses_windows_isolated_desktop_backend_for_windows_isolated_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import guiclaw.backends.background_runtime as runtime
    from nanobot.agent.tools.gui import GuiSubagentTool

    tool = _make_gui_tool(background=True, display_width=1600, display_height=900)
    inner_backend = tool._backend
    inner_backend.platform = "windows"

    manager_init_kwargs: list[dict[str, Any]] = []
    wrapped_backend_ref: list[Any] = []
    canned_result = json.dumps(
        {"success": True, "summary": "done", "trace_path": None, "steps_taken": 1, "error": None}
    )

    class FakeWin32DesktopManager:
        def __init__(self, width: int = 1280, height: int = 720) -> None:
            manager_init_kwargs.append({"width": width, "height": height})

    class FakeWindowsIsolatedBackend:
        def __init__(
            self, inner: Any, manager: Any, run_metadata: dict[str, str] | None = None
        ) -> None:
            self._inner = inner
            self._manager = manager
            self._run_metadata = run_metadata
            self.shutdown = AsyncMock()
            self.platform = "windows"
            wrapped_backend_ref.append(self)

    with (
        patch(
            "nanobot.agent.tools.gui.GuiSubagentTool._run_task",
            new=AsyncMock(return_value=canned_result),
        ),
        patch(
            "guiclaw.backends.background_runtime.probe_isolated_background_support",
            return_value=runtime.IsolationProbeResult(
                supported=True,
                reason_code="windows_isolated_desktop_available",
                retryable=False,
                host_platform="windows",
                backend_name="windows_isolated_desktop",
                sys_platform="win32",
            ),
        ),
        patch(
            "guiclaw.backends.displays.win32desktop.Win32DesktopManager", FakeWin32DesktopManager
        ),
        patch(
            "nanobot.agent.tools.gui.WindowsIsolatedBackend",
            FakeWindowsIsolatedBackend,
            create=True,
        ),
    ):
        monkeypatch.setattr(sys, "platform", "win32")
        result = await tool.execute("test task")

    assert result == canned_result
    assert manager_init_kwargs == [{"width": 1600, "height": 900}]
    assert len(wrapped_backend_ref) == 1
    assert wrapped_backend_ref[0]._inner is inner_backend
    assert wrapped_backend_ref[0]._run_metadata == {
        "owner": "nanobot",
        "task": "test task",
        "model": "test-model",
    }
    wrapped_backend_ref[0].shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_gui_tool_passes_target_app_class_to_windows_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _make_gui_tool(background=True)
    canned_result = json.dumps(
        {"success": True, "summary": "done", "trace_path": None, "steps_taken": 1, "error": None}
    )
    probe_calls: list[dict[str, Any]] = []

    def fake_probe(**kwargs: Any) -> Any:
        probe_calls.append(kwargs)
        return MagicMock(
            supported=False,
            reason_code="platform_unsupported",
            retryable=False,
            host_platform="windows",
            backend_name="windows_isolated_desktop",
            sys_platform="win32",
        )

    monkeypatch.setattr(sys, "platform", "win32")

    with (
        patch(
            "guiclaw.backends.background_runtime.probe_isolated_background_support",
            side_effect=fake_probe,
        ),
        patch(
            "nanobot.agent.tools.gui.GuiSubagentTool._run_task",
            new=AsyncMock(return_value=canned_result),
        ) as mock_run_task,
    ):
        assert (
            await tool.execute(
                "test task", target_app_class="uwp", acknowledge_background_fallback=True
            )
            == canned_result
        )
        assert (
            await tool.execute("test task", acknowledge_background_fallback=True) == canned_result
        )

    assert mock_run_task.await_count == 2
    assert probe_calls[0] == {"sys_platform": "win32", "target_app_class": "uwp"}
    assert probe_calls[1] == {"sys_platform": "win32", "target_app_class": "classic-win32"}


@pytest.mark.asyncio
async def test_gui_tool_blocks_windows_non_interactive_isolation_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _make_gui_tool(background=True)
    monkeypatch.setattr(sys, "platform", "win32")

    with patch(
        "guiclaw.backends.background_runtime.probe_isolated_background_support",
        return_value=MagicMock(
            supported=False,
            reason_code="windows_non_interactive_session",
            retryable=False,
            host_platform="windows",
            backend_name="windows_isolated_desktop",
            sys_platform="win32",
        ),
    ):
        with patch(
            "nanobot.agent.tools.gui.GuiSubagentTool._run_task", new=AsyncMock()
        ) as mock_run_task:
            payload = json.loads(await tool.execute("open settings"))

    assert payload["success"] is False
    assert payload["trace_path"] is None
    assert payload["steps_taken"] == 0
    assert "windows_non_interactive_session" in payload["summary"]
    assert "Session 0 and service contexts" in payload["summary"]
    mock_run_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_gui_tool_blocks_windows_unsupported_app_class_before_agent_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _make_gui_tool(background=True)
    monkeypatch.setattr(sys, "platform", "win32")

    with patch(
        "guiclaw.backends.background_runtime.probe_isolated_background_support",
        return_value=MagicMock(
            supported=False,
            reason_code="windows_app_class_unsupported",
            retryable=False,
            host_platform="windows",
            backend_name="windows_isolated_desktop",
            sys_platform="win32",
        ),
    ):
        with patch(
            "nanobot.agent.tools.gui.GuiSubagentTool._run_task",
            new=AsyncMock(side_effect=AssertionError("agent execution should not start")),
        ) as mock_run_task:
            payload = json.loads(await tool.execute("open settings"))

    assert payload["success"] is False
    assert payload["trace_path"] is None
    assert payload["steps_taken"] == 0
    assert "windows_app_class_unsupported" in payload["summary"]
    mock_run_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_gui_tool_reports_windows_cleanup_reason_codes_in_failure_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import guiclaw.backends.background_runtime as runtime
    from nanobot.agent.tools.gui import GuiSubagentTool

    tool = _make_gui_tool(background=True)
    inner_backend = tool._backend
    inner_backend.platform = "windows"

    class FakeWin32DesktopManager:
        def __init__(self, width: int = 1280, height: int = 720) -> None:
            self.width = width
            self.height = height

    class FakeWindowsIsolatedBackend:
        def __init__(
            self, inner: Any, manager: Any, run_metadata: dict[str, str] | None = None
        ) -> None:
            self._inner = inner
            self._manager = manager
            self._run_metadata = run_metadata
            self.shutdown = AsyncMock()
            self.platform = "windows"

    failure_message = (
        "worker startup failed cleanup_reason=startup_failed "
        "display_id=windows_isolated_desktop:GUIClaw-Background-1"
    )

    with (
        patch(
            "guiclaw.backends.background_runtime.probe_isolated_background_support",
            return_value=runtime.IsolationProbeResult(
                supported=True,
                reason_code="windows_isolated_desktop_available",
                retryable=False,
                host_platform="windows",
                backend_name="windows_isolated_desktop",
                sys_platform="win32",
            ),
        ),
        patch(
            "guiclaw.backends.displays.win32desktop.Win32DesktopManager", FakeWin32DesktopManager
        ),
        patch(
            "nanobot.agent.tools.gui.WindowsIsolatedBackend",
            FakeWindowsIsolatedBackend,
            create=True,
        ),
        patch(
            "nanobot.agent.tools.gui.GuiSubagentTool._run_task",
            new=AsyncMock(side_effect=RuntimeError(failure_message)),
        ),
    ):
        monkeypatch.setattr(sys, "platform", "win32")
        payload = json.loads(await tool.execute("test task"))

    assert payload["success"] is False
    assert "cleanup_reason=startup_failed" in payload["summary"]
    assert "display_id=windows_isolated_desktop:GUIClaw-Background-1" in payload["summary"]


@pytest.mark.asyncio
async def test_gui_tool_background_decision_tokens_stay_consistent_across_supported_hosts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import guiclaw.backends.background_runtime as runtime

    class _SimpleBackgroundWrapper:
        def __init__(
            self, inner: Any, manager: Any, run_metadata: dict[str, str] | None = None
        ) -> None:
            self._inner = inner

        @property
        def platform(self) -> str:
            return self._inner.platform

        async def shutdown(self) -> None:
            return None

    async def _supported_case() -> str:
        tool = _make_gui_tool(background=True)
        monkeypatch.setattr(sys, "platform", "win32")
        with (
            patch(
                "guiclaw.backends.background_runtime.probe_isolated_background_support",
                return_value=runtime.IsolationProbeResult(
                    supported=True,
                    reason_code="windows_isolated_desktop_available",
                    retryable=False,
                    host_platform="windows",
                    backend_name="windows_isolated_desktop",
                    sys_platform="win32",
                ),
            ),
            patch(
                "guiclaw.backends.displays.win32desktop.Win32DesktopManager",
                lambda *args, **kwargs: object(),
            ),
            patch(
                "nanobot.agent.tools.gui.WindowsIsolatedBackend",
                _SimpleBackgroundWrapper,
                create=True,
            ),
            patch(
                "nanobot.agent.tools.gui.GuiSubagentTool._run_task",
                new=AsyncMock(
                    return_value=json.dumps(
                        {
                            "success": True,
                            "summary": "done",
                            "trace_path": None,
                            "steps_taken": 1,
                            "error": None,
                        }
                    )
                ),
            ),
        ):
            return await tool.execute("test task", acknowledge_background_fallback=True)

    async def _fallback_case() -> str:
        tool = _make_gui_tool(background=True)
        monkeypatch.setattr(sys, "platform", "linux")
        with patch(
            "guiclaw.backends.background_runtime.probe_isolated_background_support",
            return_value=runtime.IsolationProbeResult(
                supported=False,
                reason_code="xvfb_missing",
                retryable=True,
                host_platform="linux",
                backend_name="xvfb",
                sys_platform="linux",
            ),
        ):
            return await tool.execute("test task")

    async def _blocked_case() -> str:
        tool = _make_gui_tool(background=True)
        monkeypatch.setattr(sys, "platform", "win32")
        with patch(
            "guiclaw.backends.background_runtime.probe_isolated_background_support",
            return_value=runtime.IsolationProbeResult(
                supported=False,
                reason_code="windows_app_class_unsupported",
                retryable=False,
                host_platform="windows",
                backend_name="windows_isolated_desktop",
                sys_platform="win32",
            ),
        ):
            return await tool.execute("test task")

    with caplog.at_level(logging.INFO, logger="nanobot.agent.tools.gui"):
        supported_payload = json.loads(await _supported_case())
        fallback_payload = json.loads(await _fallback_case())
        blocked_payload = json.loads(await _blocked_case())

    assert supported_payload["success"] is True
    assert fallback_payload["success"] is False
    assert "acknowledge_background_fallback=true" in fallback_payload["summary"]
    assert blocked_payload["success"] is False
    assert "windows_app_class_unsupported" in blocked_payload["summary"]
    assert any(
        "owner=nanobot" in record.message
        and "mode=isolated" in record.message
        and "reason=windows_isolated_desktop_available" in record.message
        for record in caplog.records
    )
    assert any(
        "owner=nanobot" in record.message
        and "mode=fallback" in record.message
        and "reason=xvfb_missing" in record.message
        for record in caplog.records
    )
    assert any(
        "owner=nanobot" in record.message
        and "mode=fallback" in record.message
        and "reason=windows_app_class_unsupported" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_gui_tool_preserves_cleanup_and_intervention_tokens_in_structured_payloads(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from guiclaw.interfaces import InterventionRequest
    import guiclaw.backends.background_runtime as runtime
    from nanobot.agent.tools.gui import _GuiToolInterventionHandler

    tool = _make_gui_tool(background=True)
    inner_backend = tool._backend
    inner_backend.platform = "windows"

    class FakeWin32DesktopManager:
        def __init__(self, width: int = 1280, height: int = 720) -> None:
            self.width = width
            self.height = height

    class FakeWindowsIsolatedBackend:
        def __init__(
            self, inner: Any, manager: Any, run_metadata: dict[str, str] | None = None
        ) -> None:
            self._inner = inner
            self._manager = manager
            self._run_metadata = run_metadata
            self.shutdown = AsyncMock()
            self.platform = "windows"

        def get_intervention_target(self) -> dict[str, Any]:
            return {
                "display_id": "windows_isolated_desktop:GUIClaw-Background-1",
                "desktop_name": "GUIClaw-Background-1",
                "session_token": "top-secret",
            }

    failure_message = (
        "worker startup failed cleanup_reason=startup_failed "
        "display_id=windows_isolated_desktop:GUIClaw-Background-1 "
        "desktop_name=GUIClaw-Background-1"
    )

    with (
        patch(
            "guiclaw.backends.background_runtime.probe_isolated_background_support",
            return_value=runtime.IsolationProbeResult(
                supported=True,
                reason_code="windows_isolated_desktop_available",
                retryable=False,
                host_platform="windows",
                backend_name="windows_isolated_desktop",
                sys_platform="win32",
            ),
        ),
        patch(
            "guiclaw.backends.displays.win32desktop.Win32DesktopManager", FakeWin32DesktopManager
        ),
        patch(
            "nanobot.agent.tools.gui.WindowsIsolatedBackend",
            FakeWindowsIsolatedBackend,
            create=True,
        ),
        patch(
            "nanobot.agent.tools.gui.GuiSubagentTool._run_task",
            new=AsyncMock(side_effect=RuntimeError(failure_message)),
        ),
    ):
        monkeypatch.setattr(sys, "platform", "win32")
        payload = json.loads(await tool.execute("test task"))

    with caplog.at_level(logging.WARNING, logger="nanobot.agent.tools.gui"):
        resolution = await _GuiToolInterventionHandler(
            active_backend=FakeWindowsIsolatedBackend(inner_backend, object()),
            task="Handle payroll login",
        ).request_intervention(
            InterventionRequest(
                task="Handle payroll login",
                reason="Need the user to enter OTP 123456 for the payroll login.",
                step_index=1,
                platform="windows",
                foreground_app=None,
                target={
                    "display_id": "windows_isolated_desktop:GUIClaw-Background-1",
                    "desktop_name": "GUIClaw-Background-1",
                    "session_token": "top-secret",
                },
            )
        )

    assert payload["success"] is False
    assert "cleanup_reason=startup_failed" in payload["summary"]
    assert "display_id=windows_isolated_desktop:GUIClaw-Background-1" in payload["summary"]
    assert "desktop_name=GUIClaw-Background-1" in payload["summary"]
    assert resolution.resume_confirmed is False
    assert any("<redacted:intervention_reason>" in record.message for record in caplog.records)
    assert not any("OTP 123456" in record.message for record in caplog.records)
    assert not any("top-secret" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_gui_tool_intervention_flow_returns_structured_resume_result(
    tmp_path: Path,
) -> None:
    from guiclaw.interfaces import InterventionResolution, LLMResponse, ToolCall

    tool = _make_gui_tool(background=False)
    tool._workspace = tmp_path
    tool._backend = _InterventionBackend()
    tool._llm_adapter = _ScriptedNanobotProvider(
        [
            LLMResponse(
                content="request intervention",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="computer_use",
                        arguments={
                            "action_type": "request_intervention",
                            "text": "Need the user to enter OTP 123456 for the payroll login.",
                        },
                    )
                ],
            ),
            LLMResponse(
                content="done",
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="computer_use",
                        arguments={"action_type": "done", "status": "success"},
                    )
                ],
            ),
        ]
    )

    class _ResumeHandler:
        async def request_intervention(self, request: Any) -> InterventionResolution:
            assert request.target["display_id"] == ":88"
            return InterventionResolution(resume_confirmed=True, note="operator resumed")

    with (
        patch.object(
            type(tool), "_build_intervention_handler", return_value=_ResumeHandler(), create=True
        ),
        patch.object(tool._postprocessor, "_summarize_trajectory", new=AsyncMock(return_value="")),
        patch.object(tool._postprocessor, "_extract_skill", new=AsyncMock(return_value=None)),
    ):
        payload = json.loads(await tool.execute("Handle payroll login"))
        await tool._wait_for_pending_postprocessing()

    assert payload["success"] is True
    assert payload["summary"].startswith("Status: completed")
    assert payload["error"] is None
    assert payload["steps_taken"] == 2
    assert payload["trace_path"] is not None


@pytest.mark.asyncio
async def test_gui_tool_intervention_trace_payload_is_scrubbed(
    tmp_path: Path,
) -> None:
    from guiclaw.interfaces import InterventionResolution, LLMResponse, ToolCall

    reason = "Need the user to enter OTP 123456 for the payroll login."
    note = "operator typed password=Sup3rSecret and otp=123456"
    tool = _make_gui_tool(background=False)
    tool._workspace = tmp_path
    tool._backend = _InterventionBackend()
    tool._llm_adapter = _ScriptedNanobotProvider(
        [
            LLMResponse(
                content="request intervention",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="computer_use",
                        arguments={"action_type": "request_intervention", "text": reason},
                    )
                ],
            )
        ]
    )

    class _CancelHandler:
        async def request_intervention(self, request: Any) -> InterventionResolution:
            assert request.target["session_token"] == "secret-session-token"
            return InterventionResolution(resume_confirmed=False, note=note)

    with (
        patch.object(
            type(tool), "_build_intervention_handler", return_value=_CancelHandler(), create=True
        ),
        patch.object(tool._postprocessor, "_summarize_trajectory", new=AsyncMock(return_value="")),
        patch.object(tool._postprocessor, "_extract_skill", new=AsyncMock(return_value=None)),
    ):
        payload = json.loads(await tool.execute("Handle payroll login"))
        await tool._wait_for_pending_postprocessing()

    trace_text = Path(payload["trace_path"]).read_text(encoding="utf-8")
    serialized_payload = json.dumps(payload)

    assert payload["success"] is False
    assert "intervention_cancelled" in payload["error"]
    assert reason not in serialized_payload
    assert note not in serialized_payload
    assert "secret-session-token" not in serialized_payload
    assert "<redacted:intervention_reason>" in trace_text
    assert reason not in trace_text
    assert "secret-session-token" not in trace_text


@pytest.mark.asyncio
async def test_gui_tool_returns_before_background_postprocessing_finishes(tmp_path: Path) -> None:
    from guiclaw.agent import AgentResult

    tool = _make_gui_tool(background=False)
    tool._workspace = tmp_path

    async def fake_run(
        self, task: str, *, max_retries: int = 3, app_hint: str | None = None
    ) -> AgentResult:
        del max_retries, app_hint
        self._trajectory_recorder.start()
        self._trajectory_recorder.record_step(action={"action_type": "wait"}, model_output="wait")
        trace_path = self._trajectory_recorder.finish(success=True)
        return AgentResult(
            success=True,
            summary=f"completed {task}",
            model_summary=None,
            trace_path=str(trace_path),
            steps_taken=1,
            error=None,
        )

    release_postprocess = asyncio.Event()
    postprocess_started = asyncio.Event()

    async def fake_promote(
        trace_path: Path, is_success: bool, platform: str, **kwargs: Any
    ) -> None:
        del trace_path, is_success, platform, kwargs
        postprocess_started.set()
        await release_postprocess.wait()

    with (
        patch("guiclaw.agent.GuiAgent.run", new=fake_run),
        patch.object(
            tool._postprocessor, "_extract_skill", new=AsyncMock(side_effect=fake_promote)
        ),
        patch.object(
            tool._postprocessor,
            "_summarize_trajectory",
            new=AsyncMock(return_value="background summary"),
        ),
        patch.object(tool._postprocessor, "_run_evaluation", new=AsyncMock(return_value=None)),
    ):
        payload = json.loads(await tool.execute("test"))
        await asyncio.wait_for(postprocess_started.wait(), timeout=1.0)
        assert payload["success"] is True
        assert payload["summary"] == "completed test"
        assert payload["steps_taken"] == 1
        assert payload["trace_path"] is not None
        assert tool._postprocessor._pending
        release_postprocess.set()
        await tool._wait_for_pending_postprocessing()


@pytest.mark.asyncio
async def test_gui_tool_promotion_failure_is_non_fatal(tmp_path: Path) -> None:
    from guiclaw.agent import AgentResult

    tool = _make_gui_tool(background=False)
    tool._workspace = tmp_path

    async def fake_run(
        self, task: str, *, max_retries: int = 3, app_hint: str | None = None
    ) -> AgentResult:
        del max_retries, app_hint
        self._trajectory_recorder.start()
        self._trajectory_recorder.record_step(action={"action_type": "wait"}, model_output="wait")
        trace_path = self._trajectory_recorder.finish(success=True)
        return AgentResult(
            success=True,
            summary=f"completed {task}",
            model_summary=None,
            trace_path=str(trace_path),
            steps_taken=1,
            error=None,
        )

    with (
        patch("guiclaw.agent.GuiAgent.run", new=fake_run),
        patch.object(
            tool._postprocessor,
            "_extract_skill",
            new=AsyncMock(side_effect=RuntimeError("extraction exploded")),
        ),
        patch.object(
            tool._postprocessor,
            "_summarize_trajectory",
            new=AsyncMock(return_value="background summary"),
        ),
        patch.object(tool._postprocessor, "_run_evaluation", new=AsyncMock(return_value=None)),
    ):
        payload = json.loads(await tool.execute("test"))
        await tool._wait_for_pending_postprocessing()

    assert payload["success"] is True
    assert payload["summary"] == "completed test"
    assert payload["steps_taken"] == 1
    assert payload["trace_path"] is not None


@pytest.mark.asyncio
async def test_gui_tool_skips_skill_extraction_when_disabled(tmp_path: Path) -> None:
    tool = _make_gui_tool(background=False, enable_skill_extraction=False)
    tool._workspace = tmp_path

    trace_path = tmp_path / "traj.json"
    trace_path.write_text('{"instruction":"test","steps":[]}', encoding="utf-8")

    result = await tool._postprocessor._extract_skill(trace_path, is_success=True, platform="linux")

    assert result is None
