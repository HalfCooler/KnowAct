"""Read-only session services for the TUI backend."""

from __future__ import annotations

from nanobot.tui.contracts import SessionContract
from nanobot.tui.schemas import SessionListResponse, SessionSummary


class SessionService:
    """Adapter-backed session reader for browser-facing routes."""

    def __init__(self, contract: SessionContract):
        self._contract = contract

    def list_sessions(self) -> SessionListResponse:
        return SessionListResponse(
            workspace_path=str(self._contract.workspace_path),
            items=[SessionSummary.model_validate(item) for item in self._contract.list_sessions()],
        )
