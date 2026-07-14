"""Phase 18 Plan 02 streaming transport tests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nanobot.config.schema import Config
from nanobot.session.manager import SessionManager
from nanobot.tui.app import create_app
from nanobot.tui.dependencies import get_chat_workspace_service
from nanobot.tui.schemas import ChatEvent
from nanobot.tui.services import ChatWorkspaceService, EventStreamBroker


class FakeStreamingAgentLoop:
    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        if on_progress is not None:
            await on_progress("thinking")
            await on_progress("calling tool")
        session = self._session_manager.get_or_create(session_key)
        session.add_message("user", content)
        session.add_message("assistant", f"assistant:{content}")
        self._session_manager.save(session)
        return f"assistant:{content}"

    async def close_mcp(self) -> None:
        return None


def _make_app(tmp_path: Path) -> FastAPI:
    config = Config.model_validate(
        {
            "agents": {"defaults": {"workspace": str(tmp_path)}},
        }
    )
    app = create_app(config=config, include_runtime_routes=True)
    session_manager = SessionManager(tmp_path)
    runtime = FakeStreamingAgentLoop(session_manager)
    broker = EventStreamBroker()
    app.state.chat_event_broker = broker
    app.dependency_overrides[get_chat_workspace_service] = lambda: ChatWorkspaceService(
        session_manager=session_manager,
        event_broker=broker,
        runtime_factory=lambda: runtime,
    )
    return app


def test_sse_stream_emits_progress_and_final_reply_events(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    events: list[dict[str, object]] = []

    with TestClient(app) as setup_client:
        created = setup_client.post("/chat/sessions")
        session_id = created.json()["session"]["session_id"]

    with TestClient(app) as stream_client:
        submission = stream_client.post(
            f"/chat/sessions/{session_id}/messages",
            json={"content": "hello from browser"},
        )
        assert submission.status_code == 200
        response = stream_client.get(f"/chat/sessions/{session_id}/events")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.text.splitlines():
            if not line.startswith("data: "):
                continue
            event = ChatEvent.model_validate_json(line.removeprefix("data: "))
            events.append(event.model_dump())
            if event.type == "complete":
                break

    assert [event["type"] for event in events] == [
        "message.accepted",
        "progress",
        "progress",
        "assistant.final",
        "complete",
    ]
    assert events[0]["session_id"] == session_id
    assert events[1]["payload"] == {"content": "thinking"}
    assert events[2]["payload"] == {"content": "calling tool"}
    assert events[3]["payload"] == {"content": "assistant:hello from browser"}


def test_stream_transport_preserves_event_order_for_progress_and_final_message(tmp_path: Path) -> None:
    app = _make_app(tmp_path)

    with TestClient(app) as client:
        created = client.post("/chat/sessions")
        session_id = created.json()["session"]["session_id"]

        submission = client.post(
            f"/chat/sessions/{session_id}/messages",
            json={"content": "transport contract"},
        )
        assert submission.status_code == 200

        response = client.get(f"/chat/sessions/{session_id}/events")
        assert response.request.method == "GET"
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        history = client.get(f"/chat/sessions/{session_id}")
        assert history.status_code == 200
        assert [message["content"] for message in history.json()["messages"]] == [
            "transport contract",
            "assistant:transport contract",
        ]

    progress_event = ChatEvent(
        id="evt-progress",
        type="progress",
        session_id=session_id,
        run_id="run-1",
        payload={"content": "thinking"},
    )
    final_event = ChatEvent(
        id="evt-final",
        type="assistant.final",
        session_id=session_id,
        run_id="run-1",
        payload={"content": "assistant:transport contract"},
    )

    assert progress_event.type == "progress"
    assert final_event.type == "assistant.final"
    assert progress_event.model_dump()["payload"]["content"] == "thinking"
    assert final_event.model_dump()["payload"]["content"] == "assistant:transport contract"


def test_sse_reconnect_replays_events_after_last_event_id_only(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    events: list[ChatEvent] = []

    with TestClient(app) as client:
        created = client.post("/chat/sessions")
        session_id = created.json()["session"]["session_id"]

        submission = client.post(
            f"/chat/sessions/{session_id}/messages",
            json={"content": "resume stream"},
        )
        assert submission.status_code == 200

        initial_response = client.get(f"/chat/sessions/{session_id}/events")
        assert initial_response.status_code == 200
        for line in initial_response.text.splitlines():
            if not line.startswith("data: "):
                continue
            events.append(ChatEvent.model_validate_json(line.removeprefix("data: ")))

        assert [event.type for event in events] == [
            "message.accepted",
            "progress",
            "progress",
            "assistant.final",
            "complete",
        ]

        reconnect = client.get(
            f"/chat/sessions/{session_id}/events",
            headers={"Last-Event-ID": events[1].id},
        )
        assert reconnect.status_code == 200

    resumed: list[ChatEvent] = []
    for line in reconnect.text.splitlines():
        if not line.startswith("data: "):
            continue
        resumed.append(ChatEvent.model_validate_json(line.removeprefix("data: ")))

    assert [event.id for event in resumed] == [event.id for event in events[2:]]
    assert [event.type for event in resumed] == [
        "progress",
        "assistant.final",
        "complete",
    ]
