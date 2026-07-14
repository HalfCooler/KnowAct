""" "Tests for gui_memory_item.py and induce_gui_memory.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# GuiMemoryItem
# ---------------------------------------------------------------------------


class TestGuiMemoryItem:
    def test_creation_defaults(self):
        from guiclaw.memory.gui_memory_item import GuiMemoryItem

        item = GuiMemoryItem(title="T", description="D", content="C")
        assert item.title == "T"
        assert item.description == "D"
        assert item.content == "C"
        assert item.status == "success"
        assert item.app is None
        assert isinstance(item.created_at, float)

    def test_creation_explicit_status(self):
        from guiclaw.memory.gui_memory_item import GuiMemoryItem

        item = GuiMemoryItem(title="T", description="D", content="C", status="failure")
        assert item.status == "failure"

    def test_creation_invalid_status_raises(self):
        from guiclaw.memory.gui_memory_item import GuiMemoryItem

        with pytest.raises(ValueError, match="status must be one of"):
            GuiMemoryItem(title="T", description="D", content="C", status="invalid")

    def test_repr(self):
        from guiclaw.memory.gui_memory_item import GuiMemoryItem

        item = GuiMemoryItem(title="Fix Alarm", description="D", content="C", status="failure")
        r = repr(item)
        assert "Fix Alarm" in r
        assert "failure" in r

    def test_to_dict_roundtrip(self):
        from guiclaw.memory.gui_memory_item import GuiMemoryItem

        item = GuiMemoryItem(
            title="Navigate Settings",
            description="Use when changing wallpaper",
            content="Open Settings then Wallpaper & style",
            status="success",
            app="com.android.settings",
        )
        d = item.to_dict()
        assert d["title"] == "Navigate Settings"
        assert d["app"] == "com.android.settings"
        assert d["status"] == "success"

        restored = GuiMemoryItem.from_dict(d)
        assert restored.title == item.title
        assert restored.description == item.description
        assert restored.content == item.content
        assert restored.status == item.status
        assert restored.app == item.app

    def test_from_dict_defaults(self):
        from guiclaw.memory.gui_memory_item import GuiMemoryItem

        item = GuiMemoryItem.from_dict(
            {
                "title": "Minimal",
                "description": "No extra fields",
                "content": "Just content",
            }
        )
        assert item.status == "success"
        assert item.app is None


# ---------------------------------------------------------------------------
# Fallback GuiMemoryItem (standalone)
# ---------------------------------------------------------------------------


class TestFallbackGuiMemoryItem:
    def test_fallback_to_dict_from_dict(self):
        """Verify the ImportError fallback defines serialization helpers."""
        import scripts.induce_gui_memory as im

        item = im.GuiMemoryItem(
            title="Fallback Title",
            description="Fallback Desc",
            content="Fallback Content",
            status="failure",
            app="com.example",
        )
        d = item.to_dict()
        assert d["title"] == "Fallback Title"
        assert d["status"] == "failure"

        restored = im.GuiMemoryItem.from_dict(d)
        assert restored.title == item.title
        assert restored.content == item.content
        assert restored.app == item.app


# ---------------------------------------------------------------------------
# parse_memory_items
# ---------------------------------------------------------------------------


class TestParseMemoryItems:
    def test_single_item(self):
        from scripts.induce_gui_memory import parse_memory_items

        text = (
            "# Memory Item 1\n"
            "## Title Fix Alarm\n"
            "## Description When setting alarm time\n"
            "## Content Always verify AM/PM before confirming\n"
        )
        items = parse_memory_items(text)
        assert len(items) == 1
        assert items[0].title == "Fix Alarm"
        assert items[0].description == "When setting alarm time"
        assert "AM/PM" in items[0].content
        assert items[0].status == "success"  # caller overrides later

    def test_single_item_no_trailing_newline(self):
        """Regression: old regex required \n before EOF."""
        from scripts.induce_gui_memory import parse_memory_items

        text = "# Memory Item 1\n## Title T\n## Description D\n## Content C"
        items = parse_memory_items(text)
        assert len(items) >= 1
        assert items[0].title == "T"

    def test_multiple_items(self):
        from scripts.induce_gui_memory import parse_memory_items

        text = (
            "# Memory Item 1\n## Title T1\n## Description D1\n## Content C1\n"
            "# Memory Item 2\n## Title T2\n## Description D2\n## Content C2\n"
        )
        items = parse_memory_items(text)
        assert len(items) == 2
        assert items[0].title == "T1"
        assert items[1].title == "T2"

    def test_app_passed_to_items(self):
        from scripts.induce_gui_memory import parse_memory_items

        text = "# Memory Item 1\n## Title T\n## Description D\n## Content C\n"
        items = parse_memory_items(text, app="com.gmailclone")
        assert items[0].app == "com.gmailclone"

    def test_empty_input(self):
        from scripts.induce_gui_memory import parse_memory_items

        assert parse_memory_items("") == []
        assert parse_memory_items("No memory items here") == []

    def test_partial_item_missing_content(self):
        """Items missing required fields are skipped."""
        from scripts.induce_gui_memory import parse_memory_items

        text = "# Memory Item 1\n## Title T\n## Description D\n"
        items = parse_memory_items(text)
        assert items == []  # No "## Content"


# ---------------------------------------------------------------------------
# format_trajectory_compact
# ---------------------------------------------------------------------------


def _write_compact_trajectory(path: Path, events: list[dict]) -> None:
    metadata = next((event for event in events if event.get("type") == "metadata"), {})
    steps: list[dict] = []
    for event in events:
        if event.get("type") == "step":
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
        elif event.get("type") == "skill_execution_result" and steps:
            steps[-1]["skill"] = {
                "skill_id": event.get("skill_id"),
                "skill_name": event.get("skill_name"),
                "state": event.get("state"),
                "error": event.get("error"),
            }
    path.write_text(
        json.dumps(
            {
                "instruction": metadata.get("task") or "test task",
                "platform": "android",
                "subtasks": [{"subtask": 1, "task": metadata.get("task") or "test task"}],
                "steps": steps,
                "screenshots": [],
            },
            ensure_ascii=False,
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
        subtask = {
            "subtask": 1,
            "task": metadata.get("task") or "test task",
            "success": result_event["success"],
            "error": result_event.get("error"),
            "steps_taken": len(steps),
        }
        (path.parent / "result.json").write_text(
            json.dumps({"run": subtask, "subtasks": [subtask]}),
            encoding="utf-8",
        )


class TestFormatTrajectoryCompact:
    def test_basic_format(self, tmp_path):
        from scripts.induce_gui_memory import format_trajectory_compact

        trace = tmp_path / "traj.json"
        _write_compact_trajectory(
            trace,
            [
                {"type": "metadata", "task": "Open Clock and set alarm"},
                {
                    "type": "step",
                    "step_index": 0,
                    "action": {"action_type": "tap"},
                    "observation": {"foreground_app": "桌面"},
                    "model_output": 'Thought: I need to open the Clock app. Action: {"action_type": "click"}',
                },
                {
                    "type": "step",
                    "step_index": 1,
                    "action": {"action_type": "input_text"},
                    "observation": {"foreground_app": "Clock"},
                    "model_output": 'Thought: Type 8:25 AM. Action: {"action_type": "input_text"}',
                },
                {"type": "result", "error": None},
            ],
        )

        text = format_trajectory_compact(trace)
        assert text is not None
        assert "Open Clock and set alarm" in text
        assert "Step 0 [tap] (桌面)" in text
        assert "Step 1 [input_text] (Clock)" in text
        assert "I need to open the Clock app" in text

    def test_omits_verbose_ui_dump(self, tmp_path):
        from scripts.induce_gui_memory import format_trajectory_compact

        trace = tmp_path / "traj.json"
        _write_compact_trajectory(
            trace,
            [
                {"type": "metadata", "task": "Rename files"},
                {
                    "type": "step",
                    "step_index": 0,
                    "action": {"action_type": "tap"},
                    "observation": {
                        "foreground_app": "Files",
                        "extra": {
                            "visible_text": [
                                "Sort by...",
                                "Select all",
                                "Copy to…",
                                "Move to…",
                                "Compress",
                            ],
                            "clickable_text": ["Sort by...", "Select all"],
                        },
                    },
                    "model_output": "Thought: Open the menu. Action: {}",
                },
            ],
        )

        text = format_trajectory_compact(trace)
        assert text is not None
        assert "| UI:" not in text
        assert "Sort by..." not in text

    def test_truncates_long_trajectories(self, tmp_path):
        from scripts.induce_gui_memory import format_trajectory_compact

        events = [{"type": "metadata", "task": "Long task"}]
        for i in range(40):
            events.append(
                {
                    "type": "step",
                    "step_index": i,
                    "action": {"action_type": "tap"},
                    "observation": {"foreground_app": "App"},
                    "model_output": f"Thought: Step {i} thought. Action: {{}}",
                }
            )
        events.append({"type": "result", "error": None})

        trace = tmp_path / "traj.json"
        _write_compact_trajectory(trace, events)

        text = format_trajectory_compact(trace)
        assert text is not None
        # Should show first 3 + truncation marker + last 10
        assert "Step 0" in text
        assert "Step 1" in text
        assert "Step 2" in text
        assert "steps omitted" in text
        assert "Step 39" in text

    def test_includes_skill_failures(self, tmp_path):
        from scripts.induce_gui_memory import format_trajectory_compact

        trace = tmp_path / "traj.json"
        _write_compact_trajectory(
            trace,
            [
                {"type": "metadata", "task": "Send email"},
                {
                    "type": "step",
                    "step_index": 0,
                    "action": {"action_type": "use_skill"},
                    "observation": {"foreground_app": "Mail"},
                    "model_output": 'Thought: Use adb skill. Action: {"action_type": "use_skill"}',
                },
                {
                    "type": "skill_execution_result",
                    "skill_name": "adb_read_sent_email",
                    "state": "failed",
                    "error": "adb failed: No such file",
                },
                {"type": "result", "error": "max_steps_exceeded"},
            ],
        )

        text = format_trajectory_compact(trace)
        assert "Skill execution failures:" in text
        assert "adb_read_sent_email" in text

    def test_empty_trace_returns_none(self, tmp_path):
        from scripts.induce_gui_memory import format_trajectory_compact

        trace = tmp_path / "traj.json"
        _write_compact_trajectory(
            trace,
            [
                {"type": "metadata", "task": "No steps"},
                {"type": "result", "error": None},
            ],
        )
        assert format_trajectory_compact(trace) is None


# ---------------------------------------------------------------------------
# GUI task trace discovery
# ---------------------------------------------------------------------------


class TestGuiTaskTraceDiscovery:
    def test_discovers_one_trace_per_gui_task_run_and_counts_steps(self, tmp_path):
        from scripts.induce_gui_memory import _find_gui_task_traces, _trace_step_count

        task_dir = tmp_path / "BidFileRenameTask"
        runs_root = task_dir / "nanobot_gui_task_runs"
        run_a = runs_root / "2026-01-01_000000"
        run_b = runs_root / "2026-01-01_000001"
        run_a.mkdir(parents=True)
        run_b.mkdir(parents=True)

        trace_a = run_a / "traj.json"
        trace_b = run_b / "traj.json"
        _write_compact_trajectory(
            trace_a,
            [
                {"type": "metadata", "task": "Long GUI task"},
                {"type": "step", "step_index": 0, "action": {"action_type": "tap"}},
                {"type": "step", "step_index": 1, "action": {"action_type": "scroll"}},
                {"type": "step", "step_index": 2, "action": {"action_type": "tap"}},
                {"type": "result", "success": False},
            ],
        )
        _write_compact_trajectory(
            trace_b,
            [
                {"type": "metadata", "task": "Short GUI task"},
                {"type": "step", "step_index": 0, "action": {"action_type": "done"}},
                {"type": "result", "success": True},
            ],
        )

        traces = _find_gui_task_traces(task_dir)
        assert traces == [trace_a, trace_b]
        assert _trace_step_count(trace_a) == 3
        assert _trace_step_count(trace_b) == 1

    def test_one_run_exposes_each_router_subtask_independently(self, tmp_path):
        from guiclaw.trajectory.recorder import trajectory_subtask_indices
        from scripts.induce_gui_memory import (
            _resolve_trace_app,
            _trace_step_count,
            format_trajectory_compact,
        )

        run_dir = tmp_path / "gui_runs" / "run-1"
        run_dir.mkdir(parents=True)
        trace = run_dir / "traj.json"
        trace.write_text(
            json.dumps(
                {
                    "instruction": "Read mail, then search maps",
                    "platform": "android",
                    "subtasks": [
                        {"subtask": 1, "task": "Read mail"},
                        {"subtask": 2, "task": "Search maps"},
                    ],
                    "steps": [
                        {
                            "step": 1,
                            "subtask": 1,
                            "attempt": 1,
                            "phase": "agent",
                            "model_output": "Read the address",
                            "action": {"action_type": "tap"},
                            "app": "Mail",
                        },
                        {
                            "step": 2,
                            "subtask": 2,
                            "attempt": 1,
                            "phase": "agent",
                            "model_output": "Search the address",
                            "action": {"action_type": "input_text", "text": "address"},
                            "app": "Maps",
                        },
                    ],
                    "screenshots": [],
                }
            ),
            encoding="utf-8",
        )

        assert trajectory_subtask_indices(trace) == (1, 2)
        assert _trace_step_count(trace, subtask_index=1) == 1
        assert _trace_step_count(trace, subtask_index=2) == 1
        assert _resolve_trace_app(trace, subtask_index=2) == "Maps"
        assert "Search maps" in format_trajectory_compact(trace, subtask_index=2)


# ---------------------------------------------------------------------------
# get_task_outcome
# ---------------------------------------------------------------------------


class TestGetTaskOutcome:
    def test_success(self, tmp_path):
        from scripts.induce_gui_memory import get_task_outcome

        task_dir = tmp_path / "SuccessTask"
        task_dir.mkdir()
        (task_dir / "result.txt").write_text("score: 1.0\n", encoding="utf-8")
        outcome, error = get_task_outcome(task_dir)
        assert outcome == "success"
        assert error == ""

    def test_failure_zero(self, tmp_path):
        from scripts.induce_gui_memory import get_task_outcome

        task_dir = tmp_path / "FailTask"
        task_dir.mkdir()
        (task_dir / "result.txt").write_text("score: 0.0\n", encoding="utf-8")
        outcome, error = get_task_outcome(task_dir)
        assert outcome == "failure"
        assert "0.0" in error

    def test_failure_partial(self, tmp_path):
        from scripts.induce_gui_memory import get_task_outcome

        task_dir = tmp_path / "PartialTask"
        task_dir.mkdir()
        (task_dir / "result.txt").write_text("score: 0.5\n", encoding="utf-8")
        outcome, _ = get_task_outcome(task_dir)
        assert outcome == "failure"

    def test_missing_result(self, tmp_path):
        from scripts.induce_gui_memory import get_task_outcome

        task_dir = tmp_path / "NoResultTask"
        task_dir.mkdir()
        outcome, error = get_task_outcome(task_dir)
        assert outcome == "failure"
        assert "no result.txt" in error


# ---------------------------------------------------------------------------
# _guess_app ordering
# ---------------------------------------------------------------------------


class TestGuessApp:
    def test_calendar_before_mail(self):
        """Calendar tasks with 'event' or 'meeting' must not return Mail."""
        from scripts.induce_gui_memory import _guess_app

        # "CalendarMultiMemosTask" contains "calendar" → explicit
        assert _guess_app("CalendarMultiMemosTask") == "org.fossify.calendar"
        # "CheckConferenceDurationTask" contains "conference" → explicit
        assert _guess_app("CheckConferenceDurationTask") == "org.fossify.calendar"

    def test_mattermost_before_generic(self):
        from scripts.induce_gui_memory import _guess_app

        assert _guess_app("MattermostProjectStatusReportTask") == "com.mattermost.rnbeta"
        assert _guess_app("MattermostEmailTask") == "com.mattermost.rnbeta"

    def test_mastodon_before_generic(self):
        from scripts.induce_gui_memory import _guess_app

        assert _guess_app("MastodonNewFilterTask") == "org.joinmastodon.android.mastodon"

    def test_mail_generic_match(self):
        from scripts.induce_gui_memory import _guess_app

        # Only mail/email keywords should map to Mail
        assert _guess_app("SendInterviewEmailTask") == "com.gmailclone"
        assert _guess_app("CVEmailTask") == "com.gmailclone"

    def test_no_match(self):
        from scripts.induce_gui_memory import _guess_app

        assert _guess_app("UnknownTaskName") is None


# ---------------------------------------------------------------------------
# Memory bank I/O
# ---------------------------------------------------------------------------


class TestMemoryBankIO:
    def test_append_and_load(self, tmp_path):
        from scripts.induce_gui_memory import (
            GuiMemoryItem,
            append_to_memory_bank,
            load_memory_bank,
        )

        bank = tmp_path / "test_bank.jsonl"

        items = [
            GuiMemoryItem(title=f"Item {i}", description=f"D{i}", content=f"C{i}", status="success")
            for i in range(3)
        ]
        append_to_memory_bank(items, bank)
        assert bank.exists()

        loaded = load_memory_bank(bank)
        assert len(loaded) == 3
        assert loaded[0].title == "Item 0"

    def test_dedup_on_append(self, tmp_path):
        from scripts.induce_gui_memory import (
            GuiMemoryItem,
            append_to_memory_bank,
            load_memory_bank,
        )

        bank = tmp_path / "dedup_bank.jsonl"

        item1 = GuiMemoryItem(title="Dup", description="D", content="C", status="success")
        append_to_memory_bank([item1], bank)
        assert len(load_memory_bank(bank)) == 1

        # Append same item again — should be skipped
        append_to_memory_bank([item1], bank)
        assert len(load_memory_bank(bank)) == 1

    def test_load_empty_bank(self, tmp_path):
        from scripts.induce_gui_memory import load_memory_bank

        bank = tmp_path / "nonexistent.jsonl"
        assert load_memory_bank(bank) == []


# ---------------------------------------------------------------------------
# GUIClaw memory store + router retrieval
# ---------------------------------------------------------------------------


class TestMemoryBankJSONL:
    def test_append_to_memory_bank_jsonl_dedups(self, tmp_path):
        from scripts.induce_gui_memory import GuiMemoryItem, append_to_memory_bank, load_memory_bank

        bank_path = tmp_path / "gui_memory_bank.jsonl"
        item = GuiMemoryItem(
            title="Generate invite links from server controls",
            description="Use when creating constrained invite links.",
            content="Use server settings or invite controls rather than profile sharing.",
            status="failure",
            app="org.joinmastodon.android.mastodon",
        )

        append_to_memory_bank([item], bank_path)
        assert bank_path.exists()
        loaded = load_memory_bank(bank_path)
        assert len(loaded) == 1
        assert loaded[0].title == item.title
        assert loaded[0].app == "org.joinmastodon.android.mastodon"

        # Dedup: same title+content should be skipped
        append_to_memory_bank([item], bank_path)
        assert len(load_memory_bank(bank_path)) == 1

    def test_append_to_memory_bank_jsonl_preserves_fields(self, tmp_path):
        from scripts.induce_gui_memory import GuiMemoryItem, append_to_memory_bank, load_memory_bank

        bank_path = tmp_path / "gui_memory_bank.jsonl"
        item = GuiMemoryItem(
            title="Use search over scroll",
            description="When looking for files by prefix.",
            content="Always use the search toolbar control.",
            status="failure",
            app="com.google.android.documentsui",
        )
        append_to_memory_bank([item], bank_path)
        loaded = load_memory_bank(bank_path)
        assert len(loaded) == 1
        assert loaded[0].title == "Use search over scroll"
        assert loaded[0].status == "failure"
        assert loaded[0].app == "com.google.android.documentsui"


class TestGuiRouterMemoryRetriever:
    def test_router_reads_jsonl_memory_bank(self, tmp_path, monkeypatch):
        import nanobot.agent.tools.gui as gui_tools
        from nanobot.agent.tools.gui import GuiRouterMemoryRetriever
        from scripts.induce_gui_memory import GuiMemoryItem, append_to_memory_bank

        bank_dir = tmp_path / "guiclaw_memory"
        bank_dir.mkdir()
        bank_path = bank_dir / "gui_memory_bank.jsonl"
        append_to_memory_bank(
            [
                GuiMemoryItem(
                    title="Differentiate profile sharing from invite generation",
                    description="Use when generating one-person expiring invite links.",
                    content=(
                        "Create invite links from server settings or invite controls; "
                        "profile share links do not configure expiry, usage limits, or auto-follow."
                    ),
                    status="failure",
                    app="org.joinmastodon.android.mastodon",
                )
            ],
            bank_path,
        )
        monkeypatch.setattr(gui_tools, "DEFAULT_GUICLAW_MEMORY_DIR", bank_dir)

        retriever = GuiRouterMemoryRetriever(tmp_path / "workspace")
        context = retriever.retrieve(
            "Generate a one-person invite link that expires in one day and auto-follows the user.",
            platform="android",
        )

        assert any("invite" in item.text.casefold() for item in context.evidence)
        assert any(
            item.source.startswith("guiclaw/gui_memory:org.joinmastodon.android.mastodon")
            for item in context.evidence
        )

    def test_router_filters_unrelated_policy_without_app_candidate(self, tmp_path, monkeypatch):
        from nanobot.agent.tools.gui import GuiRouterMemoryRetriever

        retriever = GuiRouterMemoryRetriever(tmp_path / "workspace")
        chunks = [
            (
                "guiclaw/policy:com.google.android.documentsui:mw_file_management_fuzzy_search_first",
                "Generate invite link automation policy for Files search before rename tasks.",
            ),
            (
                "guiclaw/policy:mw_adb_shortcuts_when_allowed",
                "For GUI automation tasks, adb shell commands can generate results when allowed.",
            ),
            (
                "guiclaw/app:org.joinmastodon.android.mastodon:gui_memory_invite",
                "Title: Generate invite links\nGuidance: Use server invite controls to generate invite links.",
            ),
        ]
        monkeypatch.setattr(retriever, "_iter_chunks", lambda: chunks)

        context = retriever.retrieve(
            "Generate a one-person invite link that expires in one day and auto-follows the user.",
            platform="android",
        )

        sources = [item.source for item in context.evidence]
        assert sources == ["guiclaw/app:org.joinmastodon.android.mastodon:gui_memory_invite"]
