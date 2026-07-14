"""
guiclaw.memory.gui_memory_item
==============================
Minimal memory item model for GUI agent JSONL storage.

Inspired by reasoning-bank's title-description-content schema, this replaces
the heavier ``MemoryEntry`` type when the caller only needs lightweight,
LLM-inducible guidance items without platform/type/tag accounting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

_VALID_STATUSES = frozenset({"success", "failure"})


@dataclass(frozen=True)
class GuiMemoryItem:
    """A concise GUI memory item inspired by reasoning-bank entries.

    Each item captures a single piece of actionable guidance derived from
    a successful or failed GUI trajectory.  Items are serialised as one
    JSON object per line (JSONL) for simple append-only storage.
    """

    title: str
    description: str
    content: str
    status: str = "success"
    app: str | None = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_VALID_STATUSES)!r}, got {self.status!r}"
            )

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize this item for JSONL storage."""
        return {
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "status": self.status,
            "app": self.app,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuiMemoryItem:
        """Deserialize a memory item from a JSONL record."""
        return cls(
            title=data["title"],
            description=data["description"],
            content=data["content"],
            status=data.get("status", "success"),
            app=data.get("app"),
            created_at=data.get("created_at", time.time()),
        )

    # -- helpers -------------------------------------------------------------

    def __repr__(self) -> str:
        return f"GuiMemoryItem(title={self.title!r}, status={self.status!r})"
