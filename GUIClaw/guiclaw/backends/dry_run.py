"""
guiclaw.backends.dry_run
========================
No-op backend for testing and CI environments.
"""

from __future__ import annotations

import base64
from pathlib import Path

from guiclaw.action import Action, describe_action
from guiclaw.observation import Observation

_TINY_PNG: bytes = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class DryRunBackend:
    """No-op backend. Returns a 1x1 transparent PNG and acknowledges actions."""

    def __init__(self, screen_width: int = 1080, screen_height: int = 1920) -> None:
        self._width = screen_width
        self._height = screen_height

    @property
    def platform(self) -> str:
        return "dry-run"

    async def preflight(self) -> None:
        pass

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_path.write_bytes(_TINY_PNG)
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=self._width,
            screen_height=self._height,
            foreground_app="DryRun",
            platform=self.platform,
        )

    async def list_apps(self) -> list[str]:
        return []

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        return f"[dry-run] {describe_action(action)}"
