"""Read-only runtime services for the TUI backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guiclaw.trajectory.recorder import load_trajectory_events
from nanobot.tui.contracts import RuntimeInspectionContract, SessionContract
from nanobot.tui.schemas import (
    RuntimeFailureSummary,
    RuntimeInspectionResponse,
    RuntimeRunSummary,
    RuntimeSessionStats,
)
from nanobot.tui.services.operations_registry import OperationsRegistry


class RuntimeService:
    """Adapter-backed runtime inspector for browser-facing routes."""

    def __init__(
        self,
        session_contract: SessionContract | RuntimeInspectionContract,
        registry: OperationsRegistry | None = None,
        *,
        artifacts_root: Path | None = None,
        task_launch_available: bool = False,
    ) -> None:
        self._legacy_contract = (
            session_contract if isinstance(session_contract, RuntimeInspectionContract) else None
        )
        self._session_contract = (
            session_contract if isinstance(session_contract, SessionContract) else None
        )
        self._registry = registry or OperationsRegistry()
        self._artifacts_root = artifacts_root
        self._task_launch_available = task_launch_available

    def inspect_runtime(self) -> RuntimeInspectionResponse:
        if self._legacy_contract is not None:
            payload = dict(self._legacy_contract.inspect_runtime())
            payload.setdefault(
                "session_stats",
                {
                    "total": 0,
                    "active": 0,
                    "most_recent_session_id": None,
                },
            )
            payload.setdefault("active_runs", [])
            payload.setdefault("recent_failures", [])
            return RuntimeInspectionResponse.model_validate(payload)

        assert self._session_contract is not None
        sessions = self._session_contract.list_sessions()
        active_runs = [
            RuntimeRunSummary.model_validate(snapshot.to_public_dict())
            for snapshot in self._registry.list_active_runs()
        ]
        recent_failures = self._build_recent_failures()
        return RuntimeInspectionResponse(
            status=self._derive_status(active_runs=active_runs, recent_failures=recent_failures),
            channel_runtime_booted=False,
            agent_loop_booted=False,
            task_launch_available=self._task_launch_available,
            session_stats=self._build_session_stats(sessions),
            active_runs=active_runs,
            recent_failures=recent_failures,
        )

    def get_run(self, run_id: str) -> RuntimeRunSummary | None:
        if self._legacy_contract is not None:
            return None
        registry_entry = self._registry.get_run(run_id)
        if registry_entry is not None:
            return RuntimeRunSummary.model_validate(registry_entry.to_public_dict())

        for summary in self._iter_historical_runs():
            if summary.run_id == run_id:
                return summary
        return None

    @staticmethod
    def _build_session_stats(sessions: list[dict[str, Any]]) -> RuntimeSessionStats:
        most_recent = sessions[0].get("key") if sessions else None
        return RuntimeSessionStats(
            total=len(sessions),
            active=len(sessions),
            most_recent_session_id=most_recent if isinstance(most_recent, str) else None,
        )

    @staticmethod
    def _derive_status(
        *,
        active_runs: list[RuntimeRunSummary],
        recent_failures: list[RuntimeFailureSummary],
    ) -> str:
        if active_runs:
            return "running"
        if recent_failures:
            return "degraded"
        return "idle"

    def _build_recent_failures(self) -> list[RuntimeFailureSummary]:
        failures: dict[str, RuntimeFailureSummary] = {}
        for snapshot in self._registry.list_recent_failures():
            failures[snapshot.run_id] = RuntimeFailureSummary.model_validate(
                snapshot.to_public_dict()
            )
        for failure in self._iter_historical_failures():
            failures.setdefault(failure.run_id, failure)
        return sorted(
            failures.values(),
            key=lambda failure: failure.finished_at or failure.started_at or "",
            reverse=True,
        )[:10]

    def _iter_historical_failures(self) -> list[RuntimeFailureSummary]:
        failures: list[RuntimeFailureSummary] = []
        for run in self._iter_historical_runs():
            if run.status in {"failed", "cancelled"}:
                failures.append(
                    RuntimeFailureSummary(
                        run_id=run.run_id,
                        task_kind=run.task_kind,
                        status=run.status,
                        summary=run.summary,
                        steps_taken=run.steps_taken,
                        started_at=run.started_at,
                        finished_at=run.finished_at,
                    )
                )
        return failures

    def _iter_historical_runs(self) -> list[RuntimeRunSummary]:
        artifacts_root = self._artifacts_root
        if artifacts_root is None or not artifacts_root.exists():
            return []

        summaries: dict[str, RuntimeRunSummary] = {}
        for trace_path in sorted(
            artifacts_root.glob("**/traj.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            summary = self._summary_from_trace(trace_path)
            if summary is not None and summary.run_id not in summaries:
                summaries[summary.run_id] = summary
        return list(summaries.values())

    def _summary_from_trace(self, trace_path: Path) -> RuntimeRunSummary | None:
        task_kind = "unknown"
        run_id = trace_path.parent.name
        status = "running"
        summary: str | None = None
        steps_taken = 0

        try:
            for event in load_trajectory_events(trace_path, subtask_index=None):
                event_name = str(event.get("type") or "")
                if event_name == "metadata":
                    task_kind = self._infer_task_kind(event)
                elif event_name == "result":
                    status = "succeeded" if bool(event.get("success")) else "failed"
                    summary = _clean_summary(event.get("summary") or event.get("error"))
                    steps_taken = _coerce_int(event.get("total_steps"), default=steps_taken)
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            return None

        return RuntimeRunSummary(
            run_id=run_id,
            task_kind=task_kind,
            status=status,
            summary=summary,
            steps_taken=steps_taken,
            started_at=None,
            finished_at=None,
        )

    @staticmethod
    def _infer_task_kind(event: dict[str, Any], fallback: str = "unknown") -> str:
        raw_task = event.get("task")
        if not isinstance(raw_task, str) or not raw_task.strip():
            return fallback
        prefix = raw_task.strip().split(maxsplit=1)[0].lower()
        return prefix.replace(":", "_")[:80] or fallback


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_summary(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    if not cleaned:
        return None
    return cleaned[:240]
