"""Browser-safe trace and log inspection services for the TUI backend."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from guiclaw.trajectory.recorder import load_trajectory_events
from nanobot.tui.schemas import (
    LogInspectionResponse,
    TraceEventSummary,
    TraceInspectionResponse,
    TraceLogLine,
)
from nanobot.tui.services.operations_registry import OperationsRegistry

ALLOWED_TRACE_EVENT_TYPES = (
    "metadata",
    "attempt_start",
    "attempt_result",
    "attempt_exception",
    "retry",
    "step",
    "result",
)

TRUNCATE_TEXT_AT = 240

_PATH_PATTERN = re.compile(
    r"(?P<path>(?:[A-Za-z]:\\[^\s]+)|(?:/(?:[^/\s]+/?)+))"
)
_PROMPT_PATTERN = re.compile(r"\bprompt\b", re.IGNORECASE)


class TraceInspectionService:
    """Parse persisted run artifacts into browser-safe inspection DTOs."""

    def __init__(
        self,
        *,
        registry: OperationsRegistry | None = None,
        artifacts_root: Path | None = None,
    ) -> None:
        self._registry = registry or OperationsRegistry()
        self._artifacts_root = artifacts_root

    def inspect_trace(self, run_id: str) -> TraceInspectionResponse:
        run_dir = self._resolve_run_dir(run_id)
        if run_dir is None:
            return TraceInspectionResponse(run_id=run_id, status="not_found", events=[])

        trace_path = self._resolve_trace_path(run_dir)
        if trace_path is None:
            return TraceInspectionResponse(run_id=run_id, status="empty", events=[])

        events = self._load_trace_events(trace_path)
        return TraceInspectionResponse(
            run_id=run_id,
            status="ok" if events else "empty",
            events=events,
        )

    def inspect_logs(self, run_id: str) -> LogInspectionResponse:
        run_dir = self._resolve_run_dir(run_id)
        if run_dir is None:
            return LogInspectionResponse(run_id=run_id, status="not_found", lines=[])

        lines = self._load_log_lines(run_dir)
        return LogInspectionResponse(
            run_id=run_id,
            status="ok" if lines else "empty",
            lines=lines,
        )

    def _resolve_run_dir(self, run_id: str) -> Path | None:
        registry_entry = self._registry.get_run(run_id)
        trace_ref = getattr(registry_entry, "trace_ref", None)
        if isinstance(trace_ref, str) and trace_ref:
            trace_path = Path(trace_ref)
            run_dir = trace_path if trace_path.is_dir() else trace_path.parent
            if run_dir.exists():
                return run_dir

        if self._artifacts_root is None or not self._artifacts_root.exists():
            return None

        direct = self._artifacts_root / run_id
        if direct.exists():
            return direct

        matches = list(self._artifacts_root.glob(f"**/{run_id}"))
        for match in matches:
            if match.is_dir():
                return match
        return None

    @staticmethod
    def _resolve_trace_path(run_dir: Path) -> Path | None:
        path = run_dir / "traj.json"
        return path if path.is_file() else None

    def _load_trace_events(self, trace_path: Path) -> list[TraceEventSummary]:
        events: list[TraceEventSummary] = []
        try:
            for raw_event in load_trajectory_events(trace_path, subtask_index=None):
                event = self._filter_trace_event(raw_event)
                if event is not None:
                    events.append(event)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return []
        return events

    def _load_log_lines(self, run_dir: Path) -> list[TraceLogLine]:
        log_path = run_dir / "log.jsonl"
        if log_path.exists():
            lines = self._read_log_file(log_path)
            if lines:
                return lines

        trace_path = self._resolve_trace_path(run_dir)
        if trace_path is None:
            return []
        return self._build_log_lines_from_trace(trace_path)

    def _read_log_file(self, log_path: Path) -> list[TraceLogLine]:
        lines: list[TraceLogLine] = []
        try:
            with open(log_path, encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    raw_event = json.loads(line)
                    if not isinstance(raw_event, dict):
                        continue
                    filtered = self._filter_log_line(raw_event)
                    if filtered is not None:
                        lines.append(filtered)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return []
        return lines

    def _build_log_lines_from_trace(self, trace_path: Path) -> list[TraceLogLine]:
        lines: list[TraceLogLine] = []
        try:
            for raw_event in load_trajectory_events(trace_path, subtask_index=None):
                log_line = self._trace_event_to_log_line(raw_event)
                if log_line is not None:
                    lines.append(log_line)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return []
        return lines

    def _filter_trace_event(self, raw_event: dict[str, Any]) -> TraceEventSummary | None:
        event_type = self._event_type(raw_event)
        if event_type not in ALLOWED_TRACE_EVENT_TYPES:
            return None

        summary = self._trace_summary(raw_event, event_type)
        return TraceEventSummary(
            event_id=self._optional_str(raw_event.get("event_id")),
            event_type=event_type,
            timestamp=self._timestamp(raw_event),
            step_index=self._optional_int(raw_event.get("step_index") or raw_event.get("at_step")),
            status=self._trace_status(raw_event, event_type),
            summary=summary,
            success=self._optional_bool(raw_event.get("success")),
            done=self._optional_bool(raw_event.get("done")),
            retry_count=self._retry_count(raw_event, event_type),
        )

    def _filter_log_line(self, raw_event: dict[str, Any]) -> TraceLogLine | None:
        message = self._sanitize_text(raw_event.get("message"))
        if message is None:
            return None
        return TraceLogLine(
            timestamp=self._optional_str(raw_event.get("timestamp")) or self._timestamp(raw_event),
            level=self._optional_str(raw_event.get("level")),
            code=self._optional_str(raw_event.get("code")),
            message=message,
        )

    def _trace_event_to_log_line(self, raw_event: dict[str, Any]) -> TraceLogLine | None:
        event_type = self._event_type(raw_event)
        if event_type not in ALLOWED_TRACE_EVENT_TYPES:
            return None

        level = "INFO"
        if event_type in {"attempt_exception"}:
            level = "ERROR"
        elif event_type in {"retry"}:
            level = "WARNING"
        elif event_type in {"attempt_result", "result"} and raw_event.get("success") is False:
            level = "ERROR"

        message = self._trace_summary(raw_event, event_type)
        if message is None:
            return None
        return TraceLogLine(
            timestamp=self._timestamp(raw_event),
            level=level,
            code=event_type.upper(),
            message=message,
        )

    @staticmethod
    def _event_type(raw_event: dict[str, Any]) -> str:
        value = raw_event.get("type") or raw_event.get("event")
        return value if isinstance(value, str) else ""

    def _trace_summary(self, raw_event: dict[str, Any], event_type: str) -> str | None:
        candidates: list[Any] = []
        if event_type == "retry":
            candidates.extend([raw_event.get("reason"), raw_event.get("summary")])
        elif event_type == "attempt_exception":
            candidates.extend([raw_event.get("error_message"), raw_event.get("summary")])
        elif event_type in {"attempt_result", "result"}:
            candidates.extend([raw_event.get("summary"), raw_event.get("error")])
        else:
            candidates.append(raw_event.get("summary"))

        for candidate in candidates:
            cleaned = self._sanitize_text(candidate)
            if cleaned:
                return cleaned
        return None

    @staticmethod
    def _trace_status(raw_event: dict[str, Any], event_type: str) -> str | None:
        status = raw_event.get("status")
        if isinstance(status, str) and status:
            return status
        success = raw_event.get("success")
        if event_type in {"attempt_result", "result"} and isinstance(success, bool):
            return "succeeded" if success else "failed"
        if event_type == "attempt_exception":
            return "failed"
        return None

    @staticmethod
    def _retry_count(raw_event: dict[str, Any], event_type: str) -> int | None:
        if event_type != "retry":
            return None
        retry_count = raw_event.get("retry_count")
        if isinstance(retry_count, int):
            return retry_count
        attempt = raw_event.get("attempt")
        if isinstance(attempt, int):
            return attempt + 1
        next_attempt = raw_event.get("next_attempt")
        if isinstance(next_attempt, int):
            return next_attempt
        return None

    @staticmethod
    def _timestamp(raw_event: dict[str, Any]) -> str | None:
        explicit = raw_event.get("timestamp_iso")
        if isinstance(explicit, str) and explicit:
            if explicit.endswith("+0000"):
                return explicit[:-5] + "Z"
            return explicit
        raw = raw_event.get("timestamp")
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(raw, tz=UTC).isoformat().replace("+00:00", "Z")
        if isinstance(raw, str) and raw:
            return raw
        return None

    def _sanitize_text(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        cleaned = " ".join(value.split())
        if not cleaned:
            return None
        cleaned = _PATH_PATTERN.sub("[redacted-path]", cleaned)
        cleaned = _PROMPT_PATTERN.sub("request", cleaned)
        return cleaned[:TRUNCATE_TEXT_AT]

    @staticmethod
    def _optional_bool(value: Any) -> bool | None:
        return value if isinstance(value, bool) else None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        return None

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        return value if isinstance(value, str) and value else None
