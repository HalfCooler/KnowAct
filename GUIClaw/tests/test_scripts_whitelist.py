from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXPECTED_SCRIPTS = {
    "induce_compact_skills.py",
    "induce_gui_memory.py",
    "install.ps1",
    "install.sh",
}


def test_repository_scripts_match_supported_whitelist() -> None:
    actual = {path.name for path in SCRIPTS.iterdir() if path.is_file()}
    assert actual == EXPECTED_SCRIPTS


@pytest.mark.parametrize(
    "script_name",
    ("induce_compact_skills.py", "induce_gui_memory.py"),
)
def test_retained_python_script_help(script_name: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / script_name), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
