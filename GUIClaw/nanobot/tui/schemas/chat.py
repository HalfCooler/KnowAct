"""Chat workspace schemas for the TUI backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single persisted chat message exposed to the browser."""

    role: str
    content: str
    timestamp: str | None = None


class ChatSessionSummary(BaseModel):
    """Browser-scoped session identity and metadata."""

    session_id: str
    session_key: str
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0


class ChatSessionResponse(BaseModel):
    """Transcript payload returned for a browser chat session."""

    session: ChatSessionSummary
    messages: list[ChatMessage] = Field(default_factory=list)


class ChatCreateSessionResponse(BaseModel):
    """Response payload for browser chat session creation."""

    session: ChatSessionSummary
    messages: list[ChatMessage] = Field(default_factory=list)


class ChatMessageRequest(BaseModel):
    """Browser chat message submission payload."""

    content: str


class ChatMessageResponse(BaseModel):
    """Response payload for browser chat message submission."""

    session: ChatSessionSummary
    reply: ChatMessage


class ChatEvent(BaseModel):
    """Typed chat event envelope published over SSE."""

    id: str
    type: Literal["message.accepted", "progress", "assistant.final", "error", "complete"]
    session_id: str
    run_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
