"""HuggingFace-facing CLI progress format for standalone ``guiclaw``.

The original CLI printer emits one line per step:

    GUI step <N/M>: <action>

This module replaces that line at the CLI print site with task difficulty
and per-step model output. Generic progress callbacks keep the original
one-line format.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO


def print_difficulty(task_difficulty: str | None) -> str:
    difficulty = str(task_difficulty or "").strip()
    if not difficulty:
        return ""
    return f"Task difficulty: {difficulty}"


def print_step(
    step_index: int,
    total_steps: int,
    action: str,
    model_output: str = "",
    switched: bool = False,
) -> str:
    switched_tag = " [Switched]" if switched else ""
    header = f"GUI Step {step_index}/{total_steps}{switched_tag}: {action}"
    output = str(model_output or "").strip()
    if not output:
        return header
    if "\n" in output:
        return f"{header}\nModel Output:\n{output}"
    return f"{header}\nModel Output: {output}"


class HfCliProgressPrinter:
    """CLI progress sink used by ``guiclaw.cli._make_progress_printer``."""

    def __init__(
        self,
        *,
        json_output: bool = False,
        scrub: Callable[[str], str] | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self._json_output = json_output
        self._scrub = scrub or (lambda text: text)
        self._stream = stream
        self._difficulty_emitted = False

    def _target(self) -> TextIO:
        if self._stream is not None:
            return self._stream
        return sys.stderr if self._json_output else sys.stdout

    def _emit(self, text: str) -> None:
        body = self._scrub(str(text or "")).rstrip()
        if not body:
            return
        print(body, file=self._target(), flush=True)

    async def __call__(self, message: str) -> None:
        self._emit(message)

    async def emit_difficulty(self, task_difficulty: str | None) -> None:
        if self._difficulty_emitted:
            return
        line = print_difficulty(task_difficulty)
        if not line:
            return
        self._emit(line)
        self._difficulty_emitted = True

    async def emit_step(
        self,
        *,
        step_index: int,
        total_steps: int,
        action: str,
        model_output: str = "",
        switched: bool = False,
    ) -> None:
        self._emit(
            print_step(
                step_index,
                total_steps,
                action,
                model_output=model_output,
                switched=switched,
            )
        )
