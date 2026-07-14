"""Session response schemas for the TUI backend."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionSummary(BaseModel):
    """Single persisted session entry exposed to the browser."""

    key: str
    created_at: str | None = None
    updated_at: str | None = None
    path: str


class SessionListResponse(BaseModel):
    """Read-only session list payload for the TUI frontend."""

    workspace_path: str
    items: list[SessionSummary] = Field(default_factory=list)
