from __future__ import annotations

from io import StringIO

import pytest

from guiclaw.hf_cli import HfCliProgressPrinter, print_difficulty, print_step


def test_print_difficulty() -> None:
    assert print_difficulty("easy") == "Task difficulty: easy"
    assert print_difficulty("  HARD  ") == "Task difficulty: HARD"
    assert print_difficulty("") == ""
    assert print_difficulty(None) == ""


def test_print_step_default_and_switched() -> None:
    assert print_step(1, 20, "tap at (10, 20)") == "GUI Step 1/20: tap at (10, 20)"
    assert print_step(
        3,
        20,
        "tap at (80, 90)",
        model_output="click the result row",
        switched=True,
    ) == (
        "GUI Step 3/20 [Switched]: tap at (80, 90)\n"
        "Model Output: click the result row"
    )


def test_print_step_multiline_model_output() -> None:
    assert print_step(
        1,
        15,
        "task done – success",
        model_output="Thought: done\nAction: done",
    ) == (
        "GUI Step 1/15: task done – success\n"
        "Model Output:\n"
        "Thought: done\n"
        "Action: done"
    )


@pytest.mark.asyncio
async def test_hf_cli_progress_printer_emits_difficulty_once_then_steps() -> None:
    stream = StringIO()
    printer = HfCliProgressPrinter(stream=stream, scrub=lambda text: text.replace("secret", "<redacted>"))

    await printer.emit_difficulty("medium")
    await printer.emit_difficulty("hard")
    await printer.emit_step(
        step_index=1,
        total_steps=20,
        action="type \"secret\"",
        model_output="type secret into the box",
    )
    await printer("GUI step leftover")

    assert stream.getvalue() == (
        "Task difficulty: medium\n"
        "GUI Step 1/20: type \"<redacted>\"\n"
        "Model Output: type <redacted> into the box\n"
        "GUI step leftover\n"
    )
