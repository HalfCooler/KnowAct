"""Route modules for the TUI backend."""

from nanobot.tui.routes.chat import router as chat_router
from nanobot.tui.routes.health import router as health_router
from nanobot.tui.routes.runtime import router as runtime_router
from nanobot.tui.routes.sessions import router as sessions_router
from nanobot.tui.routes.tasks import router as tasks_router
from nanobot.tui.routes.traces import router as traces_router

__all__ = [
    "chat_router",
    "health_router",
    "runtime_router",
    "sessions_router",
    "tasks_router",
    "traces_router",
]
