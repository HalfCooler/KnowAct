"""Phase 19 Plan 02 launch-contract tests for the TUI backend."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, get_type_hints
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nanobot.tui.app import create_app
from nanobot.tui.contracts import SessionContract, TaskLaunchContract
from nanobot.tui.dependencies import get_runtime_service, get_task_launch_service
from nanobot.tui.schemas.tasks import LaunchRunResponse, TaskLaunchRequest
from nanobot.tui.services import OperationsRegistry, RuntimeService, TaskLaunchService


def _make_contract_test_app() -> FastAPI:
    app = FastAPI()

    @app.post("/tasks/runs", response_model=LaunchRunResponse)
    def launch_task(payload: TaskLaunchRequest) -> LaunchRunResponse:
        return LaunchRunResponse(
            run_id="run-contract-001",
            status="queued",
            accepted_at="2026-03-21T15:00:00Z",
        )

    return app


def test_launch_endpoint_accepts_only_supported_task_kinds() -> None:
    client = TestClient(_make_contract_test_app())

    accepted_payloads = [
        {
            "kind": "nanobot_open_url",
            "url": "https://example.com/docs",
            "require_background_isolation": True,
            "acknowledge_background_fallback": False,
        },
        {
            "kind": "nanobot_open_settings",
            "panel": "privacy",
            "require_background_isolation": False,
            "acknowledge_background_fallback": True,
        },
        {
            "kind": "guiclaw_launch_app",
            "app_id": "calculator",
            "backend": "dry-run",
        },
        {
            "kind": "guiclaw_open_settings",
            "panel": "network",
            "backend": "local",
        },
    ]

    run_ids: list[str] = []
    for payload in accepted_payloads:
        response = client.post("/tasks/runs", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == "run-contract-001"
        assert body["status"] == "queued"
        assert body["accepted_at"] == "2026-03-21T15:00:00Z"
        run_ids.append(body["run_id"])

    assert run_ids == ["run-contract-001"] * 4


@pytest.mark.parametrize(
    ("payload", "missing_field"),
    [
        ({"kind": "unsupported_operation"}, "kind"),
        ({"kind": "nanobot_open_url", "task": "open docs"}, "url"),
        ({"kind": "guiclaw_launch_app", "prompt": "open calculator"}, "app_id"),
        (
            {"kind": "guiclaw_open_settings", "panel": "network", "params": {"backend": "local"}},
            "backend",
        ),
        (
            {
                "kind": "nanobot_open_settings",
                "panel": "network",
                "command": ["python", "-m", "guiclaw.cli"],
            },
            "command",
        ),
    ],
)
def test_launch_endpoint_rejects_untyped_or_unsafe_parameters(
    payload: dict[str, Any],
    missing_field: str,
) -> None:
    client = TestClient(_make_contract_test_app())

    response = client.post("/tasks/runs", json=payload)

    assert response.status_code == 422
    assert missing_field in response.text


def test_launch_request_contract_exposes_only_typed_requests() -> None:
    contract_hints = get_type_hints(TaskLaunchContract)

    assert "launch_task" in contract_hints
    launch_repr = repr(contract_hints["launch_task"])
    assert "NanobotOpenUrlLaunchRequest" in launch_repr
    assert "GUIClawOpenSettingsRequest" in launch_repr
    assert "LaunchRunResponse" in launch_repr
    assert "dict[str, Any]" not in launch_repr


def test_tui_app_keeps_mutating_launch_routes_opt_in_only() -> None:
    with (
        patch("nanobot.cli.commands.sync_workspace_templates") as sync_templates,
        patch("nanobot.agent.loop.ChannelManager", create=True) as channel_manager,
        patch("nanobot.agent.loop.AgentLoop") as agent_loop,
    ):
        app = create_app()

    paths = set(app.openapi()["paths"])

    assert "/tasks/runs" not in paths
    sync_templates.assert_not_called()
    channel_manager.assert_not_called()
    agent_loop.assert_not_called()


def _awaitable_result(
    *,
    summary: str,
    steps_taken: int = 1,
    trace_ref: str | None = None,
) -> Callable[[Any], Awaitable[dict[str, Any]]]:
    async def _runner(_: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        return {
            "summary": summary,
            "steps_taken": steps_taken,
            "trace_ref": trace_ref,
        }

    return _runner


def _make_runtime_launch_app(
    service: TaskLaunchService,
    registry: OperationsRegistry,
) -> TestClient:
    app = create_app(include_runtime_routes=True)
    app.dependency_overrides[get_task_launch_service] = lambda: service
    app.dependency_overrides[get_runtime_service] = lambda: RuntimeService(
        SessionContract(workspace_path=Path("/tmp/nanobot-workspace"), list_sessions=lambda: []),
        registry=registry,
        artifacts_root=Path("/tmp/nanobot-workspace/gui_runs"),
        task_launch_available=True,
    )
    return TestClient(app)


def test_launch_endpoint_returns_registry_backed_run_id_immediately() -> None:
    registry = OperationsRegistry()
    service = TaskLaunchService(
        registry=registry,
        nanobot_runner=_awaitable_result(summary="Opened docs", trace_ref="gui/run-001"),
        guiclaw_runner=_awaitable_result(summary="Opened calculator"),
    )
    client = _make_runtime_launch_app(service, registry)

    response = client.post(
        "/tasks/runs",
        json={"kind": "guiclaw_launch_app", "app_id": "calculator", "backend": "dry-run"},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["run_id"].startswith("run-")
    assert payload["status"] == "queued"
    assert payload["accepted_at"].endswith("Z")

    run = registry.get_run(payload["run_id"])
    assert run is not None
    assert run.task_kind == "guiclaw_launch_app"
    assert run.status in {"queued", "running", "succeeded"}


def test_launch_runs_transition_through_registry_states() -> None:
    states: list[str] = []

    class TrackingRegistry(OperationsRegistry):
        def start_run(self, **kwargs: Any):  # type: ignore[override]
            snapshot = super().start_run(**kwargs)
            states.append(snapshot.status)
            return snapshot

        def update_run(self, run_id: str, **kwargs: Any):  # type: ignore[override]
            snapshot = super().update_run(run_id, **kwargs)
            if snapshot is not None:
                states.append(snapshot.status)
            return snapshot

    tracking_registry = TrackingRegistry()
    service = TaskLaunchService(
        registry=tracking_registry,
        nanobot_runner=_awaitable_result(summary="Opened privacy settings", steps_taken=4),
        guiclaw_runner=_awaitable_result(summary="Opened calculator"),
    )
    client = _make_runtime_launch_app(service, tracking_registry)
    response = client.post(
        "/tasks/runs",
        json={
            "kind": "nanobot_open_settings",
            "panel": "privacy",
            "acknowledge_background_fallback": True,
        },
    )
    assert response.status_code == 202
    payload = response.json()

    snapshot = tracking_registry.get_run(payload["run_id"])
    assert snapshot is not None
    assert snapshot.status == "succeeded"
    assert snapshot.summary == "Opened privacy settings"
    assert snapshot.steps_taken == 4
    assert states[0] == "queued"
    assert "running" in states
    assert states[-1] == "succeeded"


def test_launch_failures_surface_in_runtime_inspection() -> None:
    registry = OperationsRegistry()

    async def _failing_runner(_: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        raise RuntimeError("Settings panel timed out")

    service = TaskLaunchService(
        registry=registry,
        nanobot_runner=_awaitable_result(summary="Opened docs"),
        guiclaw_runner=_failing_runner,
    )
    client = _make_runtime_launch_app(service, registry)

    response = client.post(
        "/tasks/runs",
        json={
            "kind": "guiclaw_open_settings",
            "panel": "network",
            "backend": "dry-run",
        },
    )
    assert response.status_code == 202

    failure = registry.get_run(response.json()["run_id"])
    assert failure is not None
    assert failure.status == "failed"
    assert "timed out" in (failure.summary or "")
