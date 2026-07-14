"""Phase 18 Plan 01 browser chat workspace tests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi.testclient import TestClient
from nanobot.config.schema import Config

from nanobot.session.manager import SessionManager
from nanobot.tui.app import create_app
from nanobot.tui.dependencies import get_chat_workspace_service
from nanobot.tui.services import ChatWorkspaceService, EventStreamBroker


class FakeAgentLoop:
    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager
        self.calls: list[dict[str, object]] = []
        self.closed = False

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        self.calls.append(
            {
                "content": content,
                "session_key": session_key,
                "channel": channel,
                "chat_id": chat_id,
                "on_progress": on_progress,
            }
        )
        if on_progress is not None:
            await on_progress("thinking")
        session = self._session_manager.get_or_create(session_key)
        session.add_message("user", content)
        session.add_message("assistant", f"assistant:{content}")
        self._session_manager.save(session)
        return f"assistant:{content}"

    async def close_mcp(self) -> None:
        self.closed = True


def _make_client(tmp_path: Path) -> tuple[TestClient, FakeAgentLoop]:
    config = Config.model_validate(
        {
            "agents": {"defaults": {"workspace": str(tmp_path)}},
        }
    )
    app = create_app(config=config, include_runtime_routes=True)
    session_manager = SessionManager(tmp_path)
    runtime = FakeAgentLoop(session_manager)
    broker = EventStreamBroker()
    app.state.chat_event_broker = broker
    app.dependency_overrides[get_chat_workspace_service] = lambda: ChatWorkspaceService(
        session_manager=session_manager,
        event_broker=broker,
        runtime_factory=lambda: runtime,
    )
    return TestClient(app), runtime


def test_chat_routes_create_and_reuse_session_backed_conversations(tmp_path: Path) -> None:
    client, runtime = _make_client(tmp_path)

    create_response = client.post("/chat/sessions")

    assert create_response.status_code == 200
    created = create_response.json()
    session_id = created["session"]["session_id"]
    assert session_id
    assert created["session"]["session_key"] == f"tui:{session_id}"
    assert created["session"]["message_count"] == 0
    assert created["messages"] == []

    send_response = client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"content": "hello from browser"},
    )

    assert send_response.status_code == 200
    sent = send_response.json()
    assert sent["session"]["session_id"] == session_id
    assert sent["session"]["session_key"] == f"tui:{session_id}"
    assert sent["session"]["session_key"] != "cli:direct"
    assert sent["reply"]["role"] == "assistant"
    assert sent["reply"]["content"] == "assistant:hello from browser"
    assert runtime.calls == [
        {
            "content": "hello from browser",
            "session_key": f"tui:{session_id}",
            "channel": "tui",
            "chat_id": session_id,
            "on_progress": runtime.calls[0]["on_progress"],
        }
    ]
    assert callable(runtime.calls[0]["on_progress"])
    assert runtime.closed is True

    history = client.get(f"/chat/sessions/{session_id}")
    assert history.status_code == 200
    assert [message["content"] for message in history.json()["messages"]] == [
        "hello from browser",
        "assistant:hello from browser",
    ]


def test_chat_history_route_reads_persisted_session_state(tmp_path: Path) -> None:
    session_id = "existing-chat"
    session_key = f"tui:{session_id}"
    manager = SessionManager(tmp_path)
    session = manager.get_or_create(session_key)
    session.metadata["origin"] = "browser"
    session.add_message("user", "persisted hello")
    session.add_message("assistant", "persisted reply")
    manager.save(session)

    client, _runtime = _make_client(tmp_path)

    response = client.get(f"/chat/sessions/{session_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session"]["session_id"] == session_id
    assert payload["session"]["session_key"] == session_key
    assert payload["session"]["metadata"]["origin"] == "browser"
    assert [message["content"] for message in payload["messages"]] == [
        "persisted hello",
        "persisted reply",
    ]


def test_reconnect_recovers_recent_session_history_from_persisted_state(tmp_path: Path) -> None:
    client, _runtime = _make_client(tmp_path)

    created = client.post("/chat/sessions")
    session_id = created.json()["session"]["session_id"]

    submission = client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"content": "recover me"},
    )

    assert submission.status_code == 200

    refreshed = client.get(f"/chat/sessions/{session_id}")

    assert refreshed.status_code == 200
    payload = refreshed.json()
    assert payload["session"]["session_id"] == session_id
    assert payload["session"]["message_count"] == 2
    assert [message["content"] for message in payload["messages"]] == [
        "recover me",
        "assistant:recover me",
    ]
