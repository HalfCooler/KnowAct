"""Runtime configuration helpers for the isolated TUI backend."""

from __future__ import annotations

from dataclasses import dataclass

from nanobot.config.loader import load_config
from nanobot.config.schema import Config


@dataclass(frozen=True, slots=True)
class TuiRuntimeConfig:
    """Normalized runtime settings for the local-first TUI server."""

    host: str
    port: int
    reload: bool
    log_level: str


def resolve_tui_runtime_config(config: Config | None = None) -> TuiRuntimeConfig:
    """Resolve local-first runtime settings without reusing gateway defaults."""

    loaded = config or load_config()
    return TuiRuntimeConfig(
        host=loaded.tui.host,
        port=loaded.tui.port,
        reload=loaded.tui.reload,
        log_level=loaded.tui.log_level,
    )
