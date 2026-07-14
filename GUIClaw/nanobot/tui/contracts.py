"""Typed lazy contracts for future browser-facing seams."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.tui.schemas.tasks import LaunchRunResponse, TaskLaunchRequest
from nanobot.tui.schemas.traces import LogInspectionResponse, TraceInspectionResponse


@dataclass(slots=True)
class SessionContract:
    """Lazy contract for reading persisted session metadata."""

    workspace_path: Path
    list_sessions: Callable[[], list[dict[str, Any]]]


@dataclass(slots=True)
class RuntimeInspectionContract:
    """Lazy contract for non-mutating runtime inspection."""

    inspect_runtime: Callable[[], dict[str, Any]]


@dataclass(slots=True)
class RuntimeRunLookupContract:
    """Lazy contract for reading one browser-safe run summary by run id."""

    get_run: Callable[[str], dict[str, Any] | None]


@dataclass(slots=True)
class TaskLaunchContract:
    """Future-facing task launch contract kept non-executing in Phase 17."""

    describe_capability: Callable[[], dict[str, Any]]
    launch_task: Callable[[TaskLaunchRequest], LaunchRunResponse] | None = None


@dataclass(slots=True)
class TraceInspectionContract:
    """Lazy contract for browser-safe run trace and log inspection."""

    inspect_trace: Callable[[str], TraceInspectionResponse]
    inspect_logs: Callable[[str], LogInspectionResponse]


@dataclass(slots=True)
class ChatWorkspaceContract:
    """Future-facing browser chat contract for persisted session workflows."""

    create_session: Callable[[], dict[str, Any]]
    get_session: Callable[[str], dict[str, Any]]
    send_message: Callable[[str, str], Any]


@dataclass(slots=True)
class ChatEventStreamContract:
    """Transient event-stream contract for browser chat SSE subscriptions."""

    subscribe: Callable[[str], AsyncIterator[dict[str, Any]]]
