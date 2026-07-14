"""Trace and log inspection routes for the TUI backend."""

from fastapi import APIRouter, Depends

from nanobot.tui.dependencies import get_trace_inspection_service
from nanobot.tui.schemas import LogInspectionResponse, TraceInspectionResponse
from nanobot.tui.services import TraceInspectionService

router = APIRouter()


@router.get("/runtime/runs/{run_id}/trace", response_model=TraceInspectionResponse)
def inspect_trace(
    run_id: str,
    service: TraceInspectionService = Depends(get_trace_inspection_service),
) -> TraceInspectionResponse:
    """Expose filtered trace events for one run id."""

    return service.inspect_trace(run_id)


@router.get("/runtime/runs/{run_id}/logs", response_model=LogInspectionResponse)
def inspect_logs(
    run_id: str,
    service: TraceInspectionService = Depends(get_trace_inspection_service),
) -> LogInspectionResponse:
    """Expose filtered log lines for one run id."""

    return service.inspect_logs(run_id)
