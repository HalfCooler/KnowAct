"""Runtime inspection schemas for the TUI backend."""

from typing import Literal

from pydantic import BaseModel


class RuntimeSessionStats(BaseModel):
    """Summary of persisted browser-visible session state."""

    total: int
    active: int
    most_recent_session_id: str | None = None


class RuntimeRunSummary(BaseModel):
    """Browser-safe summary of an active or completed run."""

    run_id: str
    task_kind: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    summary: str | None = None
    steps_taken: int = 0
    started_at: str | None = None
    finished_at: str | None = None


class RuntimeFailureSummary(BaseModel):
    """Allowlisted failure summary for browser diagnostics."""

    run_id: str
    task_kind: str
    status: Literal["failed", "cancelled"]
    summary: str | None = None
    steps_taken: int = 0
    started_at: str | None = None
    finished_at: str | None = None


class RuntimeInspectionResponse(BaseModel):
    """Read-only runtime state returned to the browser."""

    status: str
    channel_runtime_booted: bool
    agent_loop_booted: bool
    task_launch_available: bool
    session_stats: RuntimeSessionStats
    active_runs: list[RuntimeRunSummary]
    recent_failures: list[RuntimeFailureSummary]
