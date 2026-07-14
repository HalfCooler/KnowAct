"""Read-only session routes for the TUI backend."""

from fastapi import APIRouter, Depends

from nanobot.tui.dependencies import get_session_service
from nanobot.tui.schemas import SessionListResponse
from nanobot.tui.services import SessionService

router = APIRouter()


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(service: SessionService = Depends(get_session_service)) -> SessionListResponse:
    """Expose persisted session metadata without mutating runtime state."""

    return service.list_sessions()
