"""Health route for the isolated TUI backend."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Return a simple health payload for local startup checks."""

    return {
        "status": "ok",
        "service": "nanobot-tui",
    }
