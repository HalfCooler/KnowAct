"""Process-local registry for browser-visible operations status."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class OperationStatusSnapshot:
    """Browser-safe summary of one tracked run."""

    run_id: str
    task_kind: str
    status: str
    summary: str | None = None
    steps_taken: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    trace_ref: str | None = None

    def to_public_dict(self) -> dict[str, str | int | None]:
        payload = asdict(self)
        payload.pop("trace_ref", None)
        return payload


class OperationsRegistry:
    """Track active and recently completed browser-launched runs in-process."""

    def __init__(self, *, recent_limit: int = 20) -> None:
        self._entries: dict[str, OperationStatusSnapshot] = {}
        self._recent_limit = recent_limit

    def start_run(
        self,
        *,
        run_id: str,
        task_kind: str,
        status: str = "queued",
        summary: str | None = None,
        steps_taken: int = 0,
        started_at: str | None = None,
        trace_ref: str | None = None,
    ) -> OperationStatusSnapshot:
        snapshot = OperationStatusSnapshot(
            run_id=run_id,
            task_kind=task_kind,
            status=status,
            summary=summary,
            steps_taken=steps_taken,
            started_at=started_at or _utc_now(),
            finished_at=None,
            trace_ref=trace_ref,
        )
        self._entries[run_id] = snapshot
        self._trim()
        return snapshot

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        summary: str | None = None,
        steps_taken: int | None = None,
        finished_at: str | None = None,
        trace_ref: str | None = None,
    ) -> OperationStatusSnapshot | None:
        current = self._entries.get(run_id)
        if current is None:
            return None
        updated = replace(
            current,
            status=status or current.status,
            summary=summary if summary is not None else current.summary,
            steps_taken=steps_taken if steps_taken is not None else current.steps_taken,
            finished_at=finished_at if finished_at is not None else current.finished_at,
            trace_ref=trace_ref if trace_ref is not None else current.trace_ref,
        )
        self._entries[run_id] = updated
        self._trim()
        return updated

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        summary: str | None = None,
        steps_taken: int | None = None,
        finished_at: str | None = None,
        trace_ref: str | None = None,
    ) -> OperationStatusSnapshot | None:
        terminal_time = finished_at or _utc_now()
        return self.update_run(
            run_id,
            status=status,
            summary=summary,
            steps_taken=steps_taken,
            finished_at=terminal_time,
            trace_ref=trace_ref,
        )

    def get_run(self, run_id: str) -> OperationStatusSnapshot | None:
        return self._entries.get(run_id)

    def list_active_runs(self) -> list[OperationStatusSnapshot]:
        return sorted(
            (
                entry
                for entry in self._entries.values()
                if entry.status in {"queued", "running"}
            ),
            key=lambda entry: entry.started_at or "",
            reverse=True,
        )

    def list_recent_failures(self) -> list[OperationStatusSnapshot]:
        return sorted(
            (
                entry
                for entry in self._entries.values()
                if entry.status in {"failed", "cancelled"}
            ),
            key=lambda entry: entry.finished_at or entry.started_at or "",
            reverse=True,
        )

    def _trim(self) -> None:
        if len(self._entries) <= self._recent_limit:
            return
        ordered = sorted(
            self._entries.values(),
            key=lambda entry: entry.finished_at or entry.started_at or "",
            reverse=True,
        )
        keep_ids = {entry.run_id for entry in ordered[: self._recent_limit]}
        self._entries = {
            run_id: entry
            for run_id, entry in self._entries.items()
            if run_id in keep_ids
        }
