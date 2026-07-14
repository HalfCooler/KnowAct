"""FastAPI app factory for the isolated TUI backend."""

from __future__ import annotations

from fastapi import FastAPI

from nanobot.config.schema import Config
from nanobot.tui.routes import (
    chat_router,
    health_router,
    runtime_router,
    sessions_router,
    tasks_router,
    traces_router,
)
from nanobot.tui.static import install_frontend_routes


def create_app(
    *,
    config: Config | None = None,
    include_runtime_routes: bool = False,
    serve_frontend: bool = False,
) -> FastAPI:
    """Create the Phase 17 TUI backend app without booting the CLI runtime."""

    app = FastAPI(
        title="nanobot tui",
        version="0.1.0",
    )
    if config is not None:
        app.state.nanobot_config = config
    app.include_router(health_router)
    if include_runtime_routes:
        app.include_router(chat_router)
        app.include_router(sessions_router)
        app.include_router(runtime_router)
        app.include_router(tasks_router)
        app.include_router(traces_router)
        if serve_frontend:
            install_frontend_routes(app)
    return app
