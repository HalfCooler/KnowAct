"""Phase 17 Plan 01 runtime-boundary tests for the isolated TUI backend."""

from __future__ import annotations

from pathlib import Path
from typing import get_type_hints
from unittest.mock import patch

import pytest

try:
    from nanobot.tui.app import create_app
    from nanobot.tui.contracts import (
        RuntimeInspectionContract,
        SessionContract,
        TaskLaunchContract,
    )
    from nanobot.tui.dependencies import (
        get_runtime_inspection_contract,
        get_session_contract,
        get_task_launch_contract,
    )

    _IMPORTS_OK = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - Wave 0 guard until Task 2 lands
    create_app = None
    get_runtime_inspection_contract = None
    get_session_contract = None
    get_task_launch_contract = None
    RuntimeInspectionContract = None
    SessionContract = None
    TaskLaunchContract = None
    _IMPORTS_OK = False
    _IMPORT_ERROR = exc


def _require_imports() -> None:
    if not _IMPORTS_OK:
        pytest.fail(f"nanobot.tui runtime boundary is not importable yet: {_IMPORT_ERROR}")


def test_create_app_builds_isolated_tui_routes() -> None:
    _require_imports()

    with (
        patch("nanobot.cli.commands.sync_workspace_templates") as sync_templates,
        patch("nanobot.agent.loop.ChannelManager", create=True) as channel_manager,
        patch("nanobot.agent.loop.AgentLoop") as agent_loop,
    ):
        app = create_app()

    paths = set(app.openapi()["paths"])

    assert "/health" in paths
    assert "/sessions" not in paths
    assert "/runtime" not in paths
    assert all(not path.startswith("/tasks") for path in paths)
    sync_templates.assert_not_called()
    channel_manager.assert_not_called()
    agent_loop.assert_not_called()


def test_tui_service_contracts_are_lazy_and_typed() -> None:
    _require_imports()

    session_contract = get_session_contract()
    runtime_contract = get_runtime_inspection_contract()

    assert isinstance(session_contract, SessionContract)
    assert isinstance(runtime_contract, RuntimeInspectionContract)
    assert callable(session_contract.list_sessions)
    assert callable(runtime_contract.inspect_runtime)

    session_hints = get_type_hints(type(session_contract))
    runtime_hints = get_type_hints(type(runtime_contract))

    assert "list_sessions" in session_hints
    assert "inspect_runtime" in runtime_hints
    session_contract_repr = repr(session_hints["list_sessions"])
    assert "list[dict[str," in session_contract_repr
    assert "Any" in session_contract_repr


def test_task_launch_contract_is_declared_without_mutating_routes() -> None:
    _require_imports()

    task_contract = get_task_launch_contract()
    app = create_app()

    assert isinstance(task_contract, TaskLaunchContract)
    assert callable(task_contract.describe_capability)
    assert task_contract.describe_capability()["mutable"] is False
    assert task_contract.launch_task is None

    route_methods = {
        path: {method.upper() for method in operations}
        for path, operations in app.openapi()["paths"].items()
    }
    assert "/tasks" not in route_methods
    assert all("POST" not in methods for methods in route_methods.values())


def test_tui_routes_expose_read_only_session_runtime_and_task_contracts() -> None:
    _require_imports()

    from fastapi.testclient import TestClient

    from nanobot.tui.dependencies import (
        get_runtime_service,
        get_session_service,
        get_task_launch_service,
    )
    from nanobot.tui.services import RuntimeService, SessionService, TaskLaunchService

    app = create_app(include_runtime_routes=True)
    app.dependency_overrides[get_session_service] = lambda: SessionService(
        SessionContract(
            workspace_path=Path("/tmp/nanobot-workspace"),
            list_sessions=lambda: [
                {
                    "key": "cli:direct",
                    "created_at": "2026-03-21T10:00:00",
                    "updated_at": "2026-03-21T10:30:00",
                    "path": "/tmp/nanobot-workspace/sessions/cli_direct.jsonl",
                }
            ],
        )
    )
    app.dependency_overrides[get_runtime_service] = lambda: RuntimeService(
        RuntimeInspectionContract(
            inspect_runtime=lambda: {
                "status": "idle",
                "channel_runtime_booted": False,
                "agent_loop_booted": False,
                "task_launch_available": False,
            }
        )
    )
    app.dependency_overrides[get_task_launch_service] = lambda: TaskLaunchService(
        TaskLaunchContract(
            describe_capability=lambda: {
                "name": "task-launch",
                "mutable": False,
                "phase": 17,
                "status": "read-only-capability",
            },
            launch_task=None,
        )
    )

    client = TestClient(app)

    sessions = client.get("/sessions")
    runtime = client.get("/runtime")
    tasks = client.get("/tasks")

    assert sessions.status_code == 200
    assert sessions.json()["workspace_path"] == "/tmp/nanobot-workspace"
    assert sessions.json()["items"][0]["key"] == "cli:direct"

    assert runtime.status_code == 200
    assert runtime.json()["status"] == "idle"
    assert runtime.json()["task_launch_available"] is False

    assert tasks.status_code == 200
    assert tasks.json()["mutable"] is False
    assert tasks.json()["status"] == "read-only-capability"
