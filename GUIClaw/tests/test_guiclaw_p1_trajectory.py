"""Unit tests for compact GUIClaw trajectory artifacts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from guiclaw.interfaces import LLMResponse
from guiclaw.trajectory.recorder import (
    ExecutionPhase,
    TrajectoryRecorder,
    load_trajectory_events,
)
from guiclaw.trajectory.summarizer import TrajectorySummarizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    """Minimal LLMProvider fake: pops canned string responses in order."""

    def __init__(self, *responses: str) -> None:
        self._queue = list(responses)

    async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        if not self._queue:
            raise AssertionError("No scripted responses left.")
        return LLMResponse(content=self._queue.pop(0))


# ---------------------------------------------------------------------------
# TrajectoryRecorder tests
# ---------------------------------------------------------------------------


def test_trajectory_recorder_writes_compact_json_artifacts(tmp_path: Path) -> None:
    rec = TrajectoryRecorder(
        output_dir=tmp_path,
        task="search video",
        platform="android",
        instruction="Find and play a video",
        subtask_index=1,
        app_hint="tv.danmaku.bili",
    )
    path = rec.start()
    rec.set_attempt(1)
    initial = tmp_path / "subtasks/01_bilibili/attempt_01/screenshots/000_initial.png"
    initial.parent.mkdir(parents=True)
    initial.write_bytes(b"png")
    rec.record_screenshot(initial, kind="initial")
    screenshot = initial.with_name("001_tap.png")
    screenshot.write_bytes(b"png")
    rec.record_step(
        action={"action_type": "tap", "x": 100, "y": 200},
        model_output="Thought: tap the result\nAction: tap",
        screenshot_path=str(screenshot),
        foreground_app="tv.danmaku.bili",
        interaction_target={"selector": {"text": "result"}},
        token_usage={"prompt_tokens": 10, "completion_tokens": 2},
        inference_time_s=1.2345,
    )
    rec.finish(success=True, summary="video is playing")

    assert path == tmp_path / "traj.json"
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    assert trajectory["instruction"] == "Find and play a video"
    assert trajectory["platform"] == "android"
    assert trajectory["subtasks"] == [
        {
            "subtask": 1,
            "task": "search video",
            "app_hint": "tv.danmaku.bili",
        }
    ]
    assert trajectory["screenshots"] == [
        {
            "subtask": 1,
            "attempt": 1,
            "sequence": 0,
            "kind": "initial",
            "file": "subtasks/01_bilibili/attempt_01/screenshots/000_initial.png",
        },
        {
            "subtask": 1,
            "attempt": 1,
            "sequence": 1,
            "kind": "tap",
            "file": "subtasks/01_bilibili/attempt_01/screenshots/001_tap.png",
        },
    ]
    assert trajectory["steps"] == [
        {
            "step": 1,
            "subtask": 1,
            "attempt": 1,
            "phase": "agent",
            "model_output": "Thought: tap the result\nAction: tap",
            "action": {"action_type": "tap", "x": 100, "y": 200},
            "screenshot": "subtasks/01_bilibili/attempt_01/screenshots/001_tap.png",
            "app": "tv.danmaku.bili",
            "interaction_target": {"selector": {"text": "result"}},
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 2},
            "inference_time_s": 1.234,
        }
    ]
    assert "duration_s" not in trajectory["steps"][0]
    assert "chat_latency_s" not in trajectory["steps"][0]
    assert "ttft_s" not in trajectory["steps"][0]
    assert not list(tmp_path.glob("*.jsonl"))
    assert not (tmp_path / "gui_metrics.json").exists()

    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["run"]["success"] is True
    assert result["run"]["summary"] == "video is playing"
    assert result["run"]["steps_taken"] == 1
    assert result["run"]["token_usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
    }
    assert "total_token_usage" not in result["run"]


def test_trajectory_recorder_preserves_structured_model_output(tmp_path: Path) -> None:
    rec = TrajectoryRecorder(output_dir=tmp_path, task="open settings", platform="android")
    path = rec.start()
    model_output = {
        "content": "Open Settings",
        "reasoning_content": "The Settings icon is visible.",
        "thinking_blocks": [{"type": "thinking", "thinking": "tap the icon"}],
        "tool_calls": [
            {
                "id": "call-1",
                "name": "computer_use",
                "arguments": {"action_type": "tap", "x": 100, "y": 200},
            }
        ],
        "finish_reason": "tool_calls",
    }

    rec.record_step(action={"action_type": "tap", "x": 100, "y": 200}, model_output=model_output)

    trajectory = json.loads(path.read_text(encoding="utf-8"))
    assert trajectory["steps"][0]["model_output"] == model_output


def test_trajectory_recorder_keeps_global_steps_across_subtasks(tmp_path: Path) -> None:
    first = TrajectoryRecorder(
        output_dir=tmp_path,
        instruction="Compare two apps",
        task="Check app A",
        subtask_index=1,
    )
    first.start()
    first.record_step(action={"action_type": "tap"}, token_usage={"prompt_tokens": 3})
    first.finish(success=True)

    second = TrajectoryRecorder(
        output_dir=tmp_path,
        instruction="Compare two apps",
        task="Check app B",
        subtask_index=2,
    )
    second.start()
    second.set_attempt(2)
    second.record_step(action={"action_type": "done"}, token_usage={"completion_tokens": 4})
    second.finish(success=True)

    trajectory = json.loads((tmp_path / "traj.json").read_text(encoding="utf-8"))
    assert [step["step"] for step in trajectory["steps"]] == [1, 2]
    assert [step["subtask"] for step in trajectory["steps"]] == [1, 2]
    assert [step["attempt"] for step in trajectory["steps"]] == [1, 2]
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["run"]["token_usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
    }
    assert sorted(path.name for path in tmp_path.glob("*.json")) == [
        "result.json",
        "traj.json",
    ]
    assert not list(tmp_path.rglob("*.jsonl"))


def test_trajectory_recorder_set_phase_changes_compact_step(tmp_path: Path) -> None:
    rec = TrajectoryRecorder(output_dir=tmp_path, task="test task", platform="macos")
    path = rec.start()
    rec.record_step(action={"action_type": "tap"})
    rec.set_phase(ExecutionPhase.SKILL, reason="matched skill")
    rec.record_step(action={"action_type": "done"})
    rec.finish(success=True)

    trajectory = json.loads(path.read_text(encoding="utf-8"))
    assert [step["phase"] for step in trajectory["steps"]] == ["agent", "skill"]


def test_trajectory_recorder_embeds_minimal_failed_skill_data_in_step(tmp_path: Path) -> None:
    rec = TrajectoryRecorder(output_dir=tmp_path, task="Open messages", platform="android")
    path = rec.start()
    rec.record_event("skill_execution_start", skill_id="skill-1", skill_name="open_messages")
    rec.record_event(
        "skill_step",
        skill_id="skill-1",
        skill_name="open_messages",
        step_index=0,
        target="Messages",
        valid_state="Messages tab visible",
        state_contract={"anchor": {"app_package": "com.example.app"}},
        valid_state_check=False,
        observation={"foreground_app": "com.example.app", "platform": "android"},
        screenshot_path="subtasks/01_messages/attempt_01/screenshots/001_use_skill.png",
        error="popup visible",
        duration_s=12.3,
    )
    rec.record_event(
        "skill_execution_result",
        skill_id="skill-1",
        skill_name="open_messages",
        state="failed",
        error="Step 0 valid_state not reached",
    )
    rec.record_step(
        action={"action_type": "use_skill", "text": "skill-1"},
        model_output="Use the saved Messages skill",
    )
    rec.finish(success=False, error="skill failed")

    trajectory = json.loads(path.read_text(encoding="utf-8"))
    assert "skill_events" not in trajectory
    assert trajectory["steps"][0]["skill"] == {
        "skill_id": "skill-1",
        "skill_name": "open_messages",
        "state": "failed",
        "error": "Step 0 valid_state not reached",
        "failed_step": {
            "step_index": 0,
            "target": "Messages",
            "valid_state": "Messages tab visible",
            "state_contract": {"anchor": {"app_package": "com.example.app"}},
            "observation": {"foreground_app": "com.example.app", "platform": "android"},
            "screenshot_path": "subtasks/01_messages/attempt_01/screenshots/001_use_skill.png",
            "error": "popup visible",
        },
    }
    events = load_trajectory_events(path, subtask_index=1)
    assert [event["type"] for event in events] == [
        "metadata",
        "skill_step",
        "skill_execution_result",
        "step",
        "result",
    ]


def test_trajectory_recorder_not_started_raises(tmp_path: Path) -> None:
    """Calling record_step() before start() raises RuntimeError."""
    rec = TrajectoryRecorder(output_dir=tmp_path, task="test", platform="android")

    with pytest.raises(RuntimeError, match="Recorder not started"):
        rec.record_step(action={"action_type": "tap", "x": 0, "y": 0})


def test_trajectory_recorder_finish_failure(tmp_path: Path) -> None:
    rec = TrajectoryRecorder(output_dir=tmp_path, task="failing task", platform="android")
    rec.start()
    rec.record_step(action={"action_type": "wait"}, model_output="waiting")
    rec.finish(success=False, error="timeout", summary="timed out")

    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["run"]["success"] is False
    assert result["run"]["error"] == "timeout"
    assert result["run"]["steps_taken"] == 1


# ---------------------------------------------------------------------------
# TrajectorySummarizer tests
# ---------------------------------------------------------------------------


async def test_trajectory_summarizer_returns_string() -> None:
    """summarize_events() returns the strict GUI state note from the LLM."""
    canned = (
        "Status: completed\n"
        "Done: Opened settings and finished the flow.\n"
        "Remaining: none\n"
        "Current: Settings screen\n"
        "Resume: No further action needed."
    )
    llm = _ScriptedLLM(canned)
    summarizer = TrajectorySummarizer(llm)

    events = [
        {"type": "metadata", "task": "open settings", "platform": "android"},
        {
            "type": "step",
            "step_index": 0,
            "action": {"action_type": "tap"},
            "model_output": "tap settings icon",
        },
        {"type": "result", "success": True, "duration_s": 1.2, "error": None},
    ]

    summary = await summarizer.summarize_events(events)

    assert isinstance(summary, str)
    assert summary == canned
    assert summary.startswith("Status: completed")


async def test_trajectory_summarizer_compacts_structured_model_output() -> None:
    canned = (
        "Status: completed\n"
        "Done: Opened settings.\n"
        "Remaining: none\n"
        "Current: Settings screen\n"
        "Resume: No further action needed."
    )
    llm = _ScriptedLLM(canned)
    summarizer = TrajectorySummarizer(llm)
    events = [
        {"type": "metadata", "task": "open settings", "platform": "android"},
        {
            "type": "step",
            "step_index": 1,
            "action": {"action_type": "tap"},
            "model_output": {
                "content": "Open Settings",
                "reasoning_content": "The Settings icon is visible.",
                "tool_calls": [],
            },
        },
        {"type": "result", "success": True, "duration_s": 1.2, "error": None},
    ]

    summary = await summarizer.summarize_events(events)

    assert summary == canned


async def test_trajectory_summarizer_accepts_fenced_state_note() -> None:
    state_note = (
        "Status: completed\n"
        "Done: Opened Contacts and viewed John Steven.\n"
        "Remaining: none\n"
        "Current: Contact details screen\n"
        "Resume: No further action needed."
    )
    llm = _ScriptedLLM(f"Here is the state note:\n```text\n{state_note}\n```")
    summarizer = TrajectorySummarizer(llm)
    events = [
        {"type": "metadata", "task": "view John", "platform": "android"},
        {"type": "result", "success": True, "duration_s": 1.2, "error": None},
    ]

    summary = await summarizer.summarize_events(events)

    assert summary == state_note


async def test_trajectory_summarizer_empty_events_returns_empty_string() -> None:
    """summarize_events([]) returns '' without calling the LLM."""
    llm = _ScriptedLLM()  # no responses — would raise if called
    summarizer = TrajectorySummarizer(llm)

    result = await summarizer.summarize_events([])

    assert result == ""


async def test_postprocessing_sections_share_one_result_without_lost_subtasks(
    tmp_path: Path,
) -> None:
    from guiclaw.postprocessing import PostRunProcessor

    recorder = TrajectoryRecorder(output_dir=tmp_path, task="first")
    trace_path = recorder.start()
    recorder.finish(success=True)
    processor = PostRunProcessor(llm=_ScriptedLLM())

    await asyncio.gather(
        processor._write_result_section(
            trace_path,
            "evaluation",
            1,
            {"success": True, "reason": "done", "token_usage": {"total_tokens": 7}},
        ),
        processor._write_result_section(
            trace_path,
            "evaluation",
            2,
            {"success": False, "reason": "missing evidence"},
        ),
        processor._write_result_section(
            trace_path,
            "extraction",
            1,
            {
                "status": "no_candidate",
                "trace": str(trace_path),
                "detail": {"reason": "empty", "token_usage": {"total_tokens": 5}},
            },
        ),
    )

    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["evaluation"] == {
        "1": {"success": True, "reason": "done"},
        "2": {"success": False, "reason": "missing evidence"},
    }
    assert result["extraction"] == {"1": {"status": "no_candidate", "detail": {"reason": "empty"}}}
    assert not list(tmp_path.glob("evaluation*.json"))
    assert not list(tmp_path.glob("extraction*.json"))
