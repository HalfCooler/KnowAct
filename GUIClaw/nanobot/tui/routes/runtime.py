"""Read-only runtime routes for the TUI backend."""

from fastapi import APIRouter, Depends, HTTPException

from nanobot.tui.dependencies import get_runtime_service
from nanobot.tui.schemas import RuntimeInspectionResponse, RuntimeRunSummary
from nanobot.tui.services import RuntimeService

router = APIRouter()


@router.get("/runtime", response_model=RuntimeInspectionResponse)
def inspect_runtime(
    service: RuntimeService = Depends(get_runtime_service),
) -> RuntimeInspectionResponse:
    """Expose lightweight runtime state without booting agent orchestration."""

    return service.inspect_runtime()


@router.get("/runtime/runs/{run_id}", response_model=RuntimeRunSummary)
def inspect_runtime_run(
    run_id: str,
    service: RuntimeService = Depends(get_runtime_service),
) -> RuntimeRunSummary:
    """Expose one browser-safe run summary without leaking artifact paths."""

    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
