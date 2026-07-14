"""Default filesystem locations owned by GUIClaw."""

from pathlib import Path

DEFAULT_GUICLAW_HOME = Path.home() / ".guiclaw"
DEFAULT_GUI_RUNS_DIR = DEFAULT_GUICLAW_HOME / "gui_runs"
DEFAULT_SHORTCUT_CACHE_DIR = DEFAULT_GUICLAW_HOME / "shortcut_cache"


def resolve_guiclaw_data_dir(path: Path | str) -> Path:
    """Resolve relative GUIClaw data directories under ``~/.guiclaw``."""
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return DEFAULT_GUICLAW_HOME / resolved
