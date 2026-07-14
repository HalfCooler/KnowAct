"""Phase 16 host-integration parity tests."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.config.schema import GuiConfig
from guiclaw.agent import AgentResult
from guiclaw.interfaces import InterventionRequest, InterventionResolution


def _make_gui_tool(background: bool = True, **config_kwargs: Any) -> Any:
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
        return GuiSubagentTool(
            gui_config=gui_config,
            provider=mock_provider,
            model="test-model",
            workspace=Path("/tmp/test_workspace"),
        )


def test_cli_and_gui_tool_share_windows_default_app_class_contract() -> None:
    import guiclaw.cli as cli

    cli_default = cli.resolve_target_app_class(
        cli.parse_args(["--background", "--task", "Open Settings"]),
        sys_platform="win32",
    )
    cli_override = cli.resolve_target_app_class(
        cli.parse_args(["--background", "--target-app-class", "uwp", "--task", "Open Settings"]),
        sys_platform="win32",
    )

    tool = _make_gui_tool(background=True)

    assert cli_default == "classic-win32"
    assert cli_override == "uwp"
    assert tool._resolve_probe_target_app_class(None, None, sys_platform="win32") == "classic-win32"
    assert tool._resolve_probe_target_app_class(None, "uwp", sys_platform="win32") == "uwp"


@pytest.mark.asyncio
async def test_gui_task_shuts_down_android_backend_after_foreground_run() -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool

    gui_config = GuiConfig(backend="adb", background=False)
    mock_provider = MagicMock()
    mock_provider.api_key = "fake-key"
    mock_provider.api_base = None
    mock_provider.extra_headers = None
    mock_backend = MagicMock()
    mock_backend.platform = "android"
    mock_backend.shutdown = AsyncMock()

    with (
        patch.object(GuiSubagentTool, "_build_backend", return_value=mock_backend),
        patch.object(GuiSubagentTool, "_run_task", new=AsyncMock(return_value='{"success": true}')),
    ):
        tool = GuiSubagentTool(
            gui_config=gui_config,
            provider=mock_provider,
            model="test-model",
            workspace=Path("/tmp/test_workspace"),
        )
        result = await tool.execute("Open Settings")

    assert result == '{"success": true}'
    mock_backend.shutdown.assert_awaited_once()


def test_cli_and_gui_tool_share_reason_codes_and_remediation_copy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import guiclaw.cli as cli
    import guiclaw.backends.background_runtime as runtime

    expected = runtime.resolve_run_mode(
        runtime.IsolationProbeResult(
            supported=False,
            reason_code="windows_app_class_unsupported",
            retryable=False,
            host_platform="windows",
            backend_name="windows_isolated_desktop",
            sys_platform="win32",
        ),
        require_isolation=False,
        require_acknowledgement_for_fallback=False,
    )
    expected_with_ack = runtime.resolve_run_mode(
        runtime.IsolationProbeResult(
            supported=False,
            reason_code="windows_app_class_unsupported",
            retryable=False,
            host_platform="windows",
            backend_name="windows_isolated_desktop",
            sys_platform="win32",
        ),
        require_isolation=False,
        require_acknowledgement_for_fallback=True,
    )

    config = cli.CliConfig(
        provider=cli.ProviderConfig(
            base_url="http://localhost:1234/v1",
            model="qwen-gui",
            api_key="test-key",
        )
    )

    class FakeProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FakeGuiAgent:
        def __init__(self, **kwargs: Any) -> None:
            self._backend = kwargs["backend"]

        async def run(self, task: str, **_: Any) -> AgentResult:
            return AgentResult(
                success=True,
                summary=f"done {task}",
                model_summary=None,
                trace_path=None,
                steps_taken=1,
                error=None,
            )

    async def fake_build_optional_components(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        return None, None, None

    tool = _make_gui_tool(background=True)

    with caplog.at_level(logging.INFO):
        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(cli, "load_config", lambda path=None: config)
            monkeypatch.setattr(cli, "OpenAICompatibleLLMProvider", FakeProvider)
            monkeypatch.setattr(cli, "build_backend", lambda backend_name, cfg: MagicMock(platform="windows"))
            monkeypatch.setattr(cli, "GuiAgent", FakeGuiAgent)
            monkeypatch.setattr(cli, "build_optional_components", fake_build_optional_components)
            monkeypatch.setattr(
                cli,
                "probe_isolated_background_support",
                lambda **_: runtime.IsolationProbeResult(
                    supported=False,
                    reason_code="windows_app_class_unsupported",
                    retryable=False,
                    host_platform="windows",
                    backend_name="windows_isolated_desktop",
                    sys_platform="win32",
                ),
            )
            monkeypatch.setattr(cli, "TrajectoryRecorder", lambda *args, **kwargs: MagicMock(path=None))
            monkeypatch.setattr(sys, "platform", "win32")

            asyncio.run(cli.run_cli(cli.parse_args(["--background", "--task", "Open Settings"])))
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
                payload = json.loads(asyncio.run(tool.execute("Open Settings")))
        finally:
            monkeypatch.undo()

    cli_logs = [record.message for record in caplog.records if record.name == "guiclaw.cli"]
    nanobot_logs = [record.message for record in caplog.records if record.name == "nanobot.agent.tools.gui"]

    assert expected.reason_code == "windows_app_class_unsupported"
    assert "classic Win32/GDI" in expected.message
    assert expected_with_ack.message == expected.message
    assert expected_with_ack.requires_acknowledgement is True
    assert any(expected.message in message and "owner=cli" in message for message in cli_logs)
    assert any(expected_with_ack.message in message and "owner=nanobot" in message for message in nanobot_logs)
    assert payload["success"] is False
    assert "windows_app_class_unsupported" in payload["summary"]


def test_phase16_host_matrix_preserves_cleanup_and_intervention_tokens(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    import guiclaw.cli as cli
    from nanobot.agent.tools.gui import GuiSubagentTool, _GuiToolInterventionHandler

    class _FakeBackend:
        def get_intervention_target(self) -> dict[str, Any]:
            return {
                "display_id": "windows_isolated_desktop:GUIClaw-Background-1",
                "desktop_name": "GUIClaw-Background-1",
                "session_token": "top-secret",
            }

    request = InterventionRequest(
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

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr("builtins.input", lambda prompt="": "cancel")
        cli_resolution = asyncio.run(cli._CliInterventionHandler(_FakeBackend()).request_intervention(request))
    finally:
        monkeypatch.undo()

    with caplog.at_level(logging.WARNING, logger="nanobot.agent.tools.gui"):
        nanobot_resolution = asyncio.run(
            _GuiToolInterventionHandler(active_backend=_FakeBackend(), task="Handle payroll login").request_intervention(
                request
            )
        )

    cli_target = cli._resolve_intervention_target(_FakeBackend(), request)
    nanobot_target = _GuiToolInterventionHandler(active_backend=_FakeBackend(), task="t")._resolve_target(request)
    cleanup_message = (
        "worker startup failed cleanup_reason=startup_failed "
        "display_id=windows_isolated_desktop:GUIClaw-Background-1"
    )
    cli._print_human_result(
        AgentResult(
            success=False,
            summary="CLI execution failed.",
            model_summary=None,
            trace_path=None,
            steps_taken=0,
            error=f"RuntimeError: {cleanup_message}",
        )
    )
    nanobot_payload = json.loads(GuiSubagentTool._background_json_failure(cleanup_message))

    captured = capsys.readouterr()

    assert cli_resolution.resume_confirmed is False
    assert nanobot_resolution.resume_confirmed is False
    assert cli_target == nanobot_target
    assert "session_token" not in cli_target
    assert "<redacted:intervention_reason>" in captured.out
    assert "OTP 123456" not in captured.out
    assert "cleanup_reason=startup_failed" in captured.out
    assert nanobot_payload["summary"] == cleanup_message
    assert any("<redacted:intervention_reason>" in record.message for record in caplog.records)
    assert not any("OTP 123456" in record.message for record in caplog.records)
    assert not any("top-secret" in record.message for record in caplog.records)
