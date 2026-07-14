"""Phase 20 static-serving and startup regression coverage."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from nanobot.tui.app import create_app
from nanobot.tui.contracts import RuntimeInspectionContract, TaskLaunchContract
from nanobot.tui.dependencies import (
    get_chat_workspace_service,
    get_runtime_service,
    get_task_launch_service,
)
from nanobot.tui.services import RuntimeService, TaskLaunchService
from nanobot.tui.static import frontend_dist_path


def _write_frontend_bundle(dist_path: Path) -> None:
    (dist_path / "assets").mkdir(parents=True, exist_ok=True)
    (dist_path / "assets" / "index.js").write_text("console.log('bundle');", encoding="utf-8")
    (dist_path / "index.html").write_text(
        "<!doctype html><html><body><div id='root' data-app='nanobot-tui'></div></body></html>",
        encoding="utf-8",
    )


def _make_static_test_client(monkeypatch, tmp_path: Path, *, with_bundle: bool) -> TestClient:
    dist_path = tmp_path / "dist"
    if with_bundle:
        _write_frontend_bundle(dist_path)
    else:
        dist_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("nanobot.tui.static.frontend_dist_path", lambda: dist_path)

    app = create_app(include_runtime_routes=True, serve_frontend=True)

    class _FakeChatWorkspaceService:
        def create_session(self) -> dict[str, object]:
            return {
                "session": {
                    "session_id": "demo-session",
                    "session_key": "tui:demo-session",
                    "created_at": "2026-03-22T00:00:00Z",
                    "updated_at": "2026-03-22T00:00:00Z",
                    "metadata": {},
                    "message_count": 0,
                },
                "messages": [],
            }

        def get_session(self, session_id: str) -> dict[str, object]:
            return {
                "session": {
                    "session_id": session_id,
                    "session_key": f"tui:{session_id}",
                    "created_at": "2026-03-22T00:00:00Z",
                    "updated_at": "2026-03-22T00:00:00Z",
                    "metadata": {},
                    "message_count": 0,
                },
                "messages": [],
            }

    app.dependency_overrides[get_chat_workspace_service] = _FakeChatWorkspaceService
    app.dependency_overrides[get_runtime_service] = lambda: RuntimeService(
        RuntimeInspectionContract(
            inspect_runtime=lambda: {
                "status": "idle",
                "channel_runtime_booted": False,
                "agent_loop_booted": False,
                "task_launch_available": True,
                "session_stats": {
                    "total": 1,
                    "active": 1,
                    "most_recent_session_id": "demo-session",
                },
                "active_runs": [],
                "recent_failures": [],
            }
        )
    )
    app.dependency_overrides[get_task_launch_service] = lambda: TaskLaunchService(
        TaskLaunchContract(
            describe_capability=lambda: {
                "name": "task-launch",
                "mutable": False,
                "phase": 20,
                "status": "browser-ready",
            },
            launch_task=None,
        )
    )
    return TestClient(app)


def test_root_serves_built_index_when_frontend_assets_are_present(monkeypatch, tmp_path: Path) -> None:
    client = _make_static_test_client(monkeypatch, tmp_path, with_bundle=True)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "data-app='nanobot-tui'" in response.text


def test_frontend_dist_path_points_at_packaged_web_dist_directory() -> None:
    dist_path = frontend_dist_path()

    assert dist_path.name == "dist"
    assert dist_path.parent.name == "web"


def test_chat_and_operations_deep_links_return_the_spa_shell(monkeypatch, tmp_path: Path) -> None:
    client = _make_static_test_client(monkeypatch, tmp_path, with_bundle=True)

    chat_root_response = client.get("/chat")
    chat_response = client.get("/chat/demo-session")
    operations_response = client.get("/operations?runId=run-123")

    assert chat_root_response.status_code == 200
    assert chat_response.status_code == 200
    assert operations_response.status_code == 200
    assert "data-app='nanobot-tui'" in chat_root_response.text
    assert "data-app='nanobot-tui'" in chat_response.text
    assert "data-app='nanobot-tui'" in operations_response.text


def test_existing_api_routes_bypass_the_spa_shell(monkeypatch, tmp_path: Path) -> None:
    client = _make_static_test_client(monkeypatch, tmp_path, with_bundle=True)

    health_response = client.get("/health")
    chat_response = client.get("/chat/sessions/demo-session")
    runtime_response = client.get("/runtime")
    tasks_response = client.get("/tasks")

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert chat_response.status_code == 200
    assert chat_response.json()["session"]["session_id"] == "demo-session"
    assert runtime_response.status_code == 200
    assert runtime_response.json()["status"] == "idle"
    assert tasks_response.status_code == 200
    assert tasks_response.json()["status"] == "browser-ready"
    assert "data-app='nanobot-tui'" not in chat_response.text


def test_missing_build_assets_return_startup_guidance(monkeypatch, tmp_path: Path) -> None:
    client = _make_static_test_client(monkeypatch, tmp_path, with_bundle=False)

    response = client.get("/")

    assert response.status_code == 503
    assert "Nanobot web frontend build missing" in response.text
    assert "npm --prefix nanobot/tui/web run build" in response.text
    assert "npm --prefix nanobot/tui/web run dev" in response.text
