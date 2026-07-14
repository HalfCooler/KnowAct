"""Phase 19 Plan 01 runtime-status tests for the TUI backend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

try:
    from fastapi.testclient import TestClient

    from nanobot.tui.app import create_app
    from nanobot.tui.contracts import RuntimeInspectionContract, SessionContract
    from nanobot.tui.services import OperationsRegistry
    from nanobot.tui.dependencies import get_runtime_service
    from nanobot.tui.services import RuntimeService

    _IMPORTS_OK = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - Wave 0 guard until Task 1 lands
    TestClient = None
    create_app = None
    RuntimeInspectionContract = None
    SessionContract = None
    OperationsRegistry = None
    get_runtime_service = None
    RuntimeService = None
    _IMPORTS_OK = False
    _IMPORT_ERROR = exc


def _require_imports() -> None:
    if not _IMPORTS_OK:
        pytest.fail(f"phase 19 runtime modules are not importable yet: {_IMPORT_ERROR}")


def test_runtime_endpoint_reports_sessions_runs_and_recent_failures() -> None:
    _require_imports()

    app = create_app(include_runtime_routes=True)
    app.dependency_overrides[get_runtime_service] = lambda: RuntimeService(
        RuntimeInspectionContract(
            inspect_runtime=lambda: {
                "status": "idle",
                "channel_runtime_booted": False,
                "agent_loop_booted": False,
                "task_launch_available": True,
                "session_stats": {
                    "total": 2,
                    "active": 1,
                    "most_recent_session_id": "cli:direct",
                },
                "active_runs": [
                    {
                        "run_id": "run-active-001",
                        "task_kind": "nanobot_open_url",
                        "status": "running",
                        "summary": "Opening local docs",
                        "steps_taken": 3,
                        "started_at": "2026-03-21T10:00:00Z",
                        "finished_at": None,
                    }
                ],
                "recent_failures": [
                    {
                        "run_id": "run-failed-001",
                        "task_kind": "guiclaw_launch_app",
                        "status": "failed",
                        "summary": "Calculator did not open",
                        "steps_taken": 2,
                        "started_at": "2026-03-21T09:55:00Z",
                        "finished_at": "2026-03-21T09:56:00Z",
                    }
                ],
            }
        )
    )

    client = TestClient(app)
    response = client.get("/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "idle"
    assert payload["task_launch_available"] is True
    assert payload["session_stats"] == {
        "total": 2,
        "active": 1,
        "most_recent_session_id": "cli:direct",
    }
    assert payload["active_runs"][0]["run_id"] == "run-active-001"
    assert payload["active_runs"][0]["status"] == "running"
    assert payload["recent_failures"][0]["run_id"] == "run-failed-001"
    assert payload["recent_failures"][0]["status"] == "failed"


def test_runtime_recent_failures_are_filtered_to_browser_safe_fields() -> None:
    _require_imports()

    app = create_app(include_runtime_routes=True)
    app.dependency_overrides[get_runtime_service] = lambda: RuntimeService(
        RuntimeInspectionContract(
            inspect_runtime=lambda: {
                "status": "degraded",
                "channel_runtime_booted": False,
                "agent_loop_booted": False,
                "task_launch_available": True,
                "session_stats": {
                    "total": 0,
                    "active": 0,
                    "most_recent_session_id": None,
                },
                "active_runs": [],
                "recent_failures": [
                    {
                        "run_id": "run-failed-002",
                        "task_kind": "nanobot_open_settings",
                        "status": "failed",
                        "summary": "Display settings panel timed out",
                        "steps_taken": 5,
                        "started_at": "2026-03-21T09:40:00Z",
                        "finished_at": "2026-03-21T09:42:00Z",
                        "trace_path": str(Path("/tmp/secret-trace.jsonl")),
                        "raw_event": {"prompt": "do not leak"},
                    }
                ],
            }
        )
    )

    client = TestClient(app)
    response = client.get("/runtime")

    assert response.status_code == 200
    failure = response.json()["recent_failures"][0]
    assert failure == {
        "run_id": "run-failed-002",
        "task_kind": "nanobot_open_settings",
        "status": "failed",
        "summary": "Display settings panel timed out",
        "steps_taken": 5,
        "started_at": "2026-03-21T09:40:00Z",
        "finished_at": "2026-03-21T09:42:00Z",
    }


def test_runtime_route_stays_import_safe_and_does_not_boot_cli_runtime(tmp_path: Path) -> None:
    _require_imports()

    with (
        patch("nanobot.cli.commands.sync_workspace_templates") as sync_templates,
        patch("nanobot.agent.loop.ChannelManager", create=True) as channel_manager,
        patch("nanobot.agent.loop.AgentLoop") as agent_loop,
    ):
        app = create_app(include_runtime_routes=True)
        app.dependency_overrides[get_runtime_service] = lambda: RuntimeService(
            SessionContract(
                workspace_path=tmp_path,
                list_sessions=lambda: [],
            ),
            OperationsRegistry(),
            artifacts_root=tmp_path / "gui_runs",
            task_launch_available=False,
        )
        client = TestClient(app)
        response = client.get("/runtime")

    assert response.status_code == 200
    assert response.json()["channel_runtime_booted"] is False
    assert response.json()["agent_loop_booted"] is False
    assert response.json()["session_stats"] == {
        "total": 0,
        "active": 0,
        "most_recent_session_id": None,
    }
    assert response.json()["active_runs"] == []
    assert response.json()["recent_failures"] == []
    sync_templates.assert_not_called()
    channel_manager.assert_not_called()
    agent_loop.assert_not_called()
