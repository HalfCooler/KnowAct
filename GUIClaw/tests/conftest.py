"""Shared pytest isolation for GUIClaw-owned runtime data."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_guiclaw_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import guiclaw.cli as cli
    import guiclaw.paths as paths
    import nanobot.agent.tools.gui as gui_tool

    guiclaw_home = tmp_path / ".guiclaw"
    monkeypatch.setattr(paths, "DEFAULT_GUICLAW_HOME", guiclaw_home)
    monkeypatch.setattr(cli, "DEFAULT_MEMORY_DIR", guiclaw_home / "memory")
    monkeypatch.setattr(gui_tool, "DEFAULT_GUICLAW_MEMORY_DIR", guiclaw_home / "memory")
