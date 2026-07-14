"""Tests for scripts/induce_gui_memory.py trace-derived app/outcome + status.

Covers the three induction-side fixes:
  #5  parse_memory_items applies status at construction (validated by __post_init__)
  #3  get_trace_outcome reads the trajectory's own ``result`` event (per-run truth)
  #2  _resolve_trace_app takes the app from the trace, not a task-name guess
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.induce_gui_memory import (
    GuiMemoryItem,
    _guess_app,
    _is_abnormal_termination,
    _resolve_trace_app,
    _select_task_items,
    append_to_memory_bank,
    get_trace_outcome,
    load_memory_bank,
    parse_memory_items,
    trace_is_abnormal,
)

_ITEM_TEXT = (
    "# Memory Item 1\n"
    "## Title Compose flow\n"
    "## Description When sending a new message.\n"
    "## Content Tap compose, fill recipient, then the body field.\n"
)


def _write_trace(path: Path, events: list[dict]) -> Path:
    run_dir = path.parent / path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = next((event for event in events if event.get("type") == "metadata"), {})
    steps = []
    for event in events:
        if event.get("type") != "step":
            continue
        observation = event.get("observation") or {}
        step = {
            "step": len(steps) + 1,
            "subtask": 1,
            "attempt": 1,
            "phase": "agent",
            "model_output": event.get("model_output") or "",
            "action": event.get("action") or {},
        }
        app = observation.get("foreground_app") or observation.get("app")
        if app:
            step["app"] = app
        steps.append(step)
    trace_path = run_dir / "traj.json"
    trace_path.write_text(
        json.dumps(
            {
                "instruction": metadata.get("task") or "test task",
                "platform": "android",
                "subtasks": [{"subtask": 1, "task": metadata.get("task") or "test task"}],
                "steps": steps,
                "screenshots": [],
            }
        ),
        encoding="utf-8",
    )
    result_event = next(
        (
            event
            for event in reversed(events)
            if event.get("type") == "result" and "success" in event
        ),
        None,
    )
    if result_event is not None:
        subtask_result = {
            "subtask": 1,
            "task": metadata.get("task") or "test task",
            "success": result_event["success"],
            "error": result_event.get("error"),
            "steps_taken": result_event.get("total_steps", len(steps)),
        }
        (run_dir / "result.json").write_text(
            json.dumps({"run": subtask_result, "subtasks": [subtask_result]}),
            encoding="utf-8",
        )
    return trace_path


def _step(app: str | None, *, key: str = "foreground_app") -> dict:
    obs: dict = {}
    if app is not None:
        obs[key] = app
    return {"type": "step", "step_index": 0, "action": {"action_type": "tap"}, "observation": obs}


def _result(success: bool, error: str | None = None) -> dict:
    event: dict = {"type": "result", "success": success, "total_steps": 3, "final_phase": "task"}
    if error is not None:
        event["error"] = error
    return event


# ---------------------------------------------------------------------------
# #5 — parse_memory_items status
# ---------------------------------------------------------------------------


class TestParseMemoryItemsStatus:
    def test_status_applied_at_construction(self):
        items = parse_memory_items(_ITEM_TEXT, app="com.x", status="failure")
        assert len(items) == 1
        assert items[0].status == "failure"
        assert items[0].app == "com.x"

    def test_default_status_success(self):
        items = parse_memory_items(_ITEM_TEXT)
        assert items[0].status == "success"

    def test_invalid_status_is_rejected_by_post_init(self):
        # A post-hoc object.__setattr__ would have bypassed this; constructing with
        # the status routes through GuiMemoryItem.__post_init__ validation.
        with pytest.raises(ValueError):
            parse_memory_items(_ITEM_TEXT, status="bogus")


class TestPackagedMemoryInduction:
    @pytest.mark.asyncio
    async def test_uses_guiclaw_llm_provider(self) -> None:
        from guiclaw.memory.induction import induce_memory_items

        class LLM:
            def __init__(self) -> None:
                self.messages: list[dict] | None = None

            async def chat(
                self,
                messages: list[dict],
                tools: list[dict] | None = None,
                tool_choice: str | None = None,
                model: str | None = None,
                max_tokens: int | None = None,
            ) -> SimpleNamespace:
                del tools, tool_choice, model
                assert max_tokens == 2048
                self.messages = messages
                return SimpleNamespace(content=_ITEM_TEXT)

        llm = LLM()
        items = await induce_memory_items(
            llm=llm,
            trajectory_text="Task: compose a message",
            task_outcome="success",
            app="com.gmailclone",
        )

        assert llm.messages is not None
        assert llm.messages[0]["role"] == "system"
        assert llm.messages[1] == {"role": "user", "content": "Task: compose a message"}
        assert len(items) == 1
        assert items[0].app == "com.gmailclone"
        assert items[0].status == "success"


# ---------------------------------------------------------------------------
# #3 — get_trace_outcome (per-trace result event)
# ---------------------------------------------------------------------------


class TestGetTraceOutcome:
    def test_success_result(self, tmp_path: Path):
        trace = _write_trace(tmp_path / "t.jsonl", [_step("com.a"), _result(True)])
        assert get_trace_outcome(trace) == ("success", "")

    def test_failure_result_with_error(self, tmp_path: Path):
        trace = _write_trace(tmp_path / "t.jsonl", [_step("com.a"), _result(False, "deeplink boom")])
        assert get_trace_outcome(trace) == ("failure", "deeplink boom")

    def test_failure_result_without_error(self, tmp_path: Path):
        trace = _write_trace(tmp_path / "t.jsonl", [_step("com.a"), _result(False)])
        outcome, note = get_trace_outcome(trace)
        assert outcome == "failure"
        assert note  # non-empty default note

    def test_no_result_event_is_indeterminate(self, tmp_path: Path):
        # None lets the caller fall back to the task-level outcome.
        trace = _write_trace(tmp_path / "t.jsonl", [_step("com.a")])
        assert get_trace_outcome(trace) is None

    def test_result_without_success_field_is_indeterminate(self, tmp_path: Path):
        trace = _write_trace(tmp_path / "t.jsonl", [{"type": "result", "error": "x"}])
        assert get_trace_outcome(trace) is None


# ---------------------------------------------------------------------------
# #2 — _resolve_trace_app (app from trace, not task name)
# ---------------------------------------------------------------------------


class TestResolveTraceApp:
    def test_dominant_app_wins(self, tmp_path: Path):
        trace = _write_trace(
            tmp_path / "t.jsonl",
            [_step("com.a"), _step("com.a"), _step("com.b"), _result(True)],
        )
        assert _resolve_trace_app(trace) == "com.a"

    def test_launchers_and_system_are_ignored(self, tmp_path: Path):
        trace = _write_trace(
            tmp_path / "t.jsonl",
            [
                _step("com.google.android.apps.nexuslauncher"),
                _step("com.android.systemui"),
                _step("org.fossify.calendar"),
                _step("org.fossify.calendar"),
            ],
        )
        assert _resolve_trace_app(trace) == "org.fossify.calendar"

    def test_app_key_fallback_when_no_foreground_app(self, tmp_path: Path):
        trace = _write_trace(tmp_path / "t.jsonl", [_step("com.real", key="app")])
        assert _resolve_trace_app(trace) == "com.real"

    def test_all_launchers_returns_none(self, tmp_path: Path):
        trace = _write_trace(
            tmp_path / "t.jsonl",
            [_step("com.android.launcher"), _step("com.google.android.apps.nexuslauncher")],
        )
        assert _resolve_trace_app(trace) is None

    def test_no_steps_returns_none(self, tmp_path: Path):
        trace = _write_trace(tmp_path / "t.jsonl", [{"type": "metadata", "task": "x"}, _result(True)])
        assert _resolve_trace_app(trace) is None

    def test_display_name_launchers_ignored(self, tmp_path: Path):
        # MobileWorld reports labels, not packages: "桌面" (launcher) must not win.
        trace = _write_trace(
            tmp_path / "t.jsonl",
            [_step("桌面"), _step("桌面"), _step("Settings"), _step("Settings")],
        )
        assert _resolve_trace_app(trace) == "Settings"

    def test_all_display_name_launchers_returns_none(self, tmp_path: Path):
        trace = _write_trace(tmp_path / "t.jsonl", [_step("桌面"), _step("System UI")])
        assert _resolve_trace_app(trace) is None

    def test_guess_app_remains_the_documented_fallback(self, tmp_path: Path):
        # When the trace yields nothing, the caller falls back to the name guess.
        trace = _write_trace(tmp_path / "t.jsonl", [_step("com.android.systemui")])
        assert _resolve_trace_app(trace) is None
        assert _guess_app("send an email to bob") == "com.gmailclone"


# ---------------------------------------------------------------------------
# near-duplicate dedup on write
# ---------------------------------------------------------------------------


def _item(title: str, content: str, *, app: str = "Settings", status: str = "success") -> GuiMemoryItem:
    return GuiMemoryItem(title=title, description="d", content=content, status=status, app=app)


class TestNearDuplicateDedup:
    def test_exact_repeat_is_skipped(self, tmp_path: Path):
        bank = tmp_path / "bank.jsonl"
        item = _item("Open network page", "Open settings then tap network and internet for wifi.")
        assert append_to_memory_bank([item], bank) == 1
        assert append_to_memory_bank([item], bank) == 0
        assert len(load_memory_bank(bank)) == 1

    def test_reworded_same_app_is_skipped(self, tmp_path: Path):
        bank = tmp_path / "bank.jsonl"
        a = _item("Network nav", "navigate settings open network internet check wifi connection status")
        # Same lesson, one word changed: Jaccard ~0.8 >= 0.6 -> near-duplicate.
        b = _item("Network nav", "navigate settings open network internet check wifi connection state")
        added = append_to_memory_bank([a, b], bank)
        assert added == 1
        assert len(load_memory_bank(bank)) == 1

    def test_distinct_facets_same_app_are_kept(self, tmp_path: Path):
        bank = tmp_path / "bank.jsonl"
        a = _item("Network nav", "navigate settings open network internet check wifi connection status")
        c = _item("Date picker", "tap year header date picker jump directly target year selection")
        assert append_to_memory_bank([a, c], bank) == 2

    def test_cross_app_never_collides(self, tmp_path: Path):
        bank = tmp_path / "bank.jsonl"
        text = "navigate settings open network internet check wifi connection status"
        a = _item("Network nav", text, app="Settings")
        b = _item("Network nav", text, app="Chrome")
        assert append_to_memory_bank([a, b], bank) == 2

    def test_threshold_is_tunable(self, tmp_path: Path):
        bank = tmp_path / "bank.jsonl"
        a = _item("Network nav", "navigate settings open network internet check wifi connection status")
        c = _item("Date picker", "tap year header date picker jump directly target year selection")
        # An impossibly low threshold collapses even unrelated items.
        assert append_to_memory_bank([a, c], bank, similarity_threshold=0.0) == 1


# ---------------------------------------------------------------------------
# per-task item budget
# ---------------------------------------------------------------------------


class TestSelectTaskItems:
    def test_under_budget_keeps_all_unchanged(self):
        items = [_item(f"t{i}", f"c{i}") for i in range(3)]
        assert _select_task_items(items, 5) == items

    def test_zero_budget_is_unlimited(self):
        items = [_item(f"t{i}", f"c{i}") for i in range(8)]
        assert _select_task_items(items, 0) == items

    def test_truncates_to_budget(self):
        items = [_item(f"t{i}", f"c{i}", app="Settings") for i in range(8)]
        assert len(_select_task_items(items, 3)) == 3

    def test_failures_are_preferred(self):
        # 5 successes + 1 failure, budget 2 → the scarce failure must survive.
        items = [_item(f"s{i}", f"c{i}", app="Settings") for i in range(5)]
        items.insert(3, _item("fail", "cf", app="Settings", status="failure"))
        picked = _select_task_items(items, 2)
        assert len(picked) == 2
        assert any(p.status == "failure" for p in picked)

    def test_round_robin_across_apps(self):
        # Budget 2 over two apps → one from each, not two from the first.
        items = [
            _item("a1", "c", app="Settings"),
            _item("a2", "c", app="Settings"),
            _item("b1", "c", app="Chrome"),
        ]
        picked = _select_task_items(items, 2)
        assert {p.app for p in picked} == {"Settings", "Chrome"}


# ---------------------------------------------------------------------------
# abnormal-termination filter (intervention / stagnation / timeout)
# ---------------------------------------------------------------------------


def _result_ev(*, error: str | None = None, total_steps: int = 3, success: bool = False) -> dict:
    ev: dict = {"type": "result", "success": success, "total_steps": total_steps}
    if error is not None:
        ev["error"] = error
    return ev


class TestAbnormalTermination:
    def test_intervention_cancelled_always_abnormal(self):
        assert _is_abnormal_termination(_result_ev(error="intervention_cancelled", total_steps=10))
        assert _is_abnormal_termination(_result_ev(error="intervention_cancelled:user", total_steps=10))
        assert _is_abnormal_termination(_result_ev(error="intervention_cancelled", total_steps=0))

    def test_stagnation_or_timeout_kept_when_progress_made(self):
        # Detector fired after real steps → still a learnable failure, NOT abnormal.
        assert not _is_abnormal_termination(_result_ev(error="stagnation_detected", total_steps=12))
        assert not _is_abnormal_termination(_result_ev(error="step_timeout:30s", total_steps=5))

    def test_stagnation_or_timeout_abnormal_when_no_progress(self):
        assert _is_abnormal_termination(_result_ev(error="stagnation_detected", total_steps=0))
        assert _is_abnormal_termination(_result_ev(error="step_timeout", total_steps=0))

    def test_normal_failure_is_not_abnormal(self):
        assert not _is_abnormal_termination(_result_ev(error="SMS content mismatch", total_steps=8))

    def test_success_and_missing_are_not_abnormal(self):
        assert not _is_abnormal_termination(_result_ev(total_steps=8, success=True))
        assert not _is_abnormal_termination(None)
        assert not _is_abnormal_termination({})

    def test_trace_is_abnormal_reads_result_event(self, tmp_path: Path):
        good = _write_trace(
            tmp_path / "g.jsonl",
            [_step("com.a"), _result_ev(error="step_timeout", total_steps=9)],
        )
        bad = _write_trace(
            tmp_path / "b.jsonl",
            [_step("com.a"), _result_ev(error="intervention_cancelled", total_steps=9)],
        )
        assert trace_is_abnormal(good) is False
        assert trace_is_abnormal(bad) is True
