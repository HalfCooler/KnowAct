"""Module entrypoint for the local-first TUI backend."""

from __future__ import annotations

from pathlib import Path

import uvicorn

from nanobot.config.loader import load_config, set_config_path
from nanobot.tui.app import create_app
from nanobot.tui.config import resolve_tui_runtime_config


def main(config_path: str | Path | None = None) -> None:
    """Start the TUI backend using the isolated app factory."""

    resolved_config_path: Path | None = None
    if config_path is not None:
        resolved_config_path = Path(config_path).expanduser().resolve()
        set_config_path(resolved_config_path)

    config = load_config(resolved_config_path)
    runtime = resolve_tui_runtime_config(config)
    app = create_app(
        config=config,
        include_runtime_routes=True,
        serve_frontend=True,
    )
    uvicorn.run(
        app,
        host=runtime.host,
        port=runtime.port,
        reload=runtime.reload,
        log_level=runtime.log_level,
    )


if __name__ == "__main__":
    main()
