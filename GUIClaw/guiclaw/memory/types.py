"""
guiclaw.memory.types
====================
Memory data models for the GUI agent's knowledge store.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryType(Enum):
    OS_GUIDE = "os"
    APP_GUIDE = "app"
    ICON_GUIDE = "icon"
    POLICY = "policy"


@dataclass(frozen=True)
class MemoryEntry:
    """A single knowledge entry in the GUI agent's memory store."""

    entry_id: str
    memory_type: MemoryType
    platform: str
    content: str
    app: str | None = None
    tags: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    access_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "memory_type": self.memory_type.value,
            "platform": self.platform,
            "content": self.content,
            "app": self.app,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            memory_type=MemoryType(data["memory_type"]),
            platform=data.get("platform", "unknown"),
            content=data["content"],
            app=data.get("app"),
            tags=tuple(data.get("tags", ())),
            created_at=data.get("created_at", time.time()),
            access_count=data.get("access_count", 0),
        )
