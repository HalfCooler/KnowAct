"""Shared GUI-memory induction for online post-processing and offline scripts."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any

from guiclaw.interfaces import LLMProvider
from guiclaw.memory.gui_memory_item import GuiMemoryItem
from guiclaw.trajectory.recorder import load_trajectory_events

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_BANK_PATH = Path.home() / ".guiclaw" / "memory" / "gui_memory_bank.jsonl"
DEFAULT_DEDUP_THRESHOLD = 0.6

SUCCESS_SYSTEM_PROMPT = """\
You are an expert in GUI automation. You will be given a user task and a
successful trajectory. Extract at most 3 distinct, reusable memory items.

Prefer concrete navigation, form-filling, UI recovery, and verification lessons.
Do not embed literal user data or task-specific entities. Ground each lesson in
the visible controls and actions from the trajectory.

Output exactly:
# Memory Item i
## Title <title>
## Description <one sentence describing when to use it>
## Content <1-3 actionable sentences>
"""

FAILURE_SYSTEM_PROMPT = """\
You are an expert in GUI automation. You will be given a user task and a failed
trajectory. Extract at most 3 distinct, reusable failure-avoidance memory items.

Prefer concrete recovery procedures for navigation mistakes, form errors,
premature completion, wrong state assumptions, and blocking UI patterns. Do not
embed literal user data or task-specific entities.

Output exactly:
# Memory Item i
## Title <title>
## Description <one sentence describing when to avoid the failed approach>
## Content <1-3 actionable sentences>
"""

MEMORY_ITEM_RE = re.compile(
    r"# Memory Item \d+\s*\n"
    r"## Title\s*(.+?)\s*\n"
    r"## Description\s*(.+?)\s*\n"
    r"## Content\s*(.+?)(?=\n# Memory Item|\n?\Z)",
    re.DOTALL,
)

_ABNORMAL_TERMINATION_PREFIXES = (
    "stagnation_detected",
    "step_timeout",
    "intervention_cancelled",
)
_DEDUP_STOPWORDS = frozenset({
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "be",
    "this",
    "that",
    "these",
    "those",
    "when",
    "use",
    "using",
    "your",
    "you",
    "it",
    "its",
    "as",
    "at",
    "by",
    "from",
    "then",
    "if",
    "which",
    "while",
    "after",
    "before",
    "into",
    "any",
    "all",
    "not",
    "do",
    "does",
    "will",
    "can",
    "should",
    "must",
    "may",
    "each",
    "via",
    "than",
    "such",
})
_DEDUP_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿㐀-䶿]")
_BANK_LOCKS: dict[Path, Lock] = {}
_LAUNCHER_OR_SYSTEM_PACKAGES = frozenset({
    "",
    "?",
    "android",
    "com.android.systemui",
    "com.android.launcher",
    "com.android.launcher3",
    "com.sec.android.app.launcher",
    "com.google.android.apps.nexuslauncher",
    "com.google.android.apps.pixellauncher",
    "桌面",
    "主屏幕",
    "launcher",
    "desktop",
    "home",
    "system ui",
    "系统界面",
    "android system",
})


def format_trajectory_compact(
    trace_path: Path,
    *,
    subtask_index: int = 1,
    max_steps_full: int = 30,
    keep_first: int = 3,
    keep_last: int = 10,
    thought_max_chars: int = 200,
    ui_hint_max_chars: int = 240,
) -> str | None:
    """Convert one compact GUIClaw subtask into concise LLM-readable text."""
    events = load_trajectory_events(trace_path, subtask_index=subtask_index)
    if not events:
        return None

    task_goal = _find_task_goal(events) or trace_path.stem
    steps = [event for event in events if _event_type(event) == "step"]
    step_lines: list[str] = []
    for step in steps:
        action = step.get("action") or {}
        action_type = action.get("action_type", "?")
        observation = step.get("observation") or {}
        app = observation.get("foreground_app", "") or step.get("app") or "?"
        thought = _extract_thought(step.get("model_output") or "")
        extra = observation.get("extra") or {}
        ui_hint = _step_ui_hint(extra, max_chars=ui_hint_max_chars)
        if len(thought) > thought_max_chars:
            thought = thought[: thought_max_chars - 3] + "..."
        index = step.get("step_index", step.get("step", len(step_lines)))
        line = f"  Step {index} [{action_type}] ({app}): {thought}"
        if ui_hint:
            line += f" | UI: {ui_hint}"
        step_lines.append(line)

    if not step_lines:
        return None
    if len(step_lines) > max_steps_full:
        omitted = len(step_lines) - keep_first - keep_last
        step_lines = (
            step_lines[:keep_first]
            + [f"  ... ({omitted} steps omitted) ..."]
            + step_lines[-keep_last:]
        )

    parts = [f"Task: {task_goal}", "", "Action sequence:", *step_lines]
    skill_failures = [
        event
        for event in events
        if _event_type(event) == "skill_execution_result" and event.get("state") == "failed"
    ]
    if skill_failures:
        parts.extend(["", "Skill execution failures:"])
        for failure in skill_failures:
            name = failure.get("skill_name", "?")
            error = failure.get("error", "") or "unknown"
            parts.append(f"  - {name}: {_truncate(error, 200)}")

    result = _find_result(events)
    if result and result.get("error"):
        parts.extend(["", f"Final error: {_truncate(str(result['error']), 200)}"])
    return "\n".join(parts).rstrip()


def parse_memory_items(
    text: str,
    *,
    app: str | None = None,
    status: str = "success",
) -> list[GuiMemoryItem]:
    """Parse validated GUI memory items from an LLM response."""
    items: list[GuiMemoryItem] = []
    for match in MEMORY_ITEM_RE.finditer(text):
        title = match.group(1).strip()
        description = match.group(2).strip()
        content = match.group(3).strip()
        if title and content:
            items.append(
                GuiMemoryItem(
                    title=title,
                    description=description,
                    content=content,
                    status=status,
                    app=app,
                )
            )
    return items


async def induce_memory_items(
    *,
    llm: LLMProvider,
    trajectory_text: str,
    task_outcome: str,
    app: str | None = None,
    max_tokens: int = 2048,
) -> list[GuiMemoryItem]:
    """Induce at most three memory items through the configured GUI LLM."""
    system_prompt = (
        SUCCESS_SYSTEM_PROMPT if task_outcome == "success" else FAILURE_SYSTEM_PROMPT
    )
    response = await llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": trajectory_text},
        ],
        max_tokens=max_tokens,
    )
    return parse_memory_items(
        response.content,
        app=app,
        status=task_outcome,
    )[:3]


def get_trace_outcome(
    trace_path: Path,
    *,
    subtask_index: int = 1,
) -> tuple[str, str] | None:
    """Return one subtask's outcome from its compact result event."""
    result = _find_result(load_trajectory_events(trace_path, subtask_index=subtask_index))
    if result is None or "success" not in result:
        return None
    if result.get("success"):
        return "success", ""
    error = str(result.get("error") or "").strip()
    return "failure", error or "trace result.success is false"


def is_abnormal_termination(result_event: dict[str, Any] | None) -> bool:
    """Return whether a trajectory ended without a learnable agent outcome."""
    if not result_event:
        return False
    error = result_event.get("error")
    total_steps = result_event.get("total_steps") or 0
    if total_steps == 0 and error:
        return True
    if not isinstance(error, str):
        return False
    if total_steps > 0 and (
        error == "stagnation_detected"
        or error.startswith("stagnation_detected:")
        or error == "step_timeout"
        or error.startswith("step_timeout:")
    ):
        return False
    return any(
        error == prefix or error.startswith(prefix + ":")
        for prefix in _ABNORMAL_TERMINATION_PREFIXES
    )


def trace_is_abnormal(trace_path: Path, *, subtask_index: int = 1) -> bool:
    events = load_trajectory_events(trace_path, subtask_index=subtask_index)
    return is_abnormal_termination(_find_result(events))


def load_memory_bank(path: Path) -> list[GuiMemoryItem]:
    """Load valid entries from a GUI memory JSONL bank."""
    if not path.exists():
        return []
    items: list[GuiMemoryItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            items.append(GuiMemoryItem.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed GUI memory in %s: %s", path, exc)
    return items


def append_to_memory_bank(
    items: list[GuiMemoryItem],
    path: Path,
    *,
    similarity_threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> int:
    """Append items while collapsing near-duplicates for the same app."""
    resolved = path.expanduser().resolve(strict=False)
    lock = _BANK_LOCKS.setdefault(resolved, Lock())
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_memory_bank(path)
        seen = [
            ((item.app or "").casefold(), _dedup_token_set(item))
            for item in existing
        ]
        added = 0
        with path.open("a", encoding="utf-8") as handle:
            for item in items:
                app_key = (item.app or "").casefold()
                tokens = _dedup_token_set(item)
                if any(
                    app_key == seen_app
                    and _jaccard(tokens, seen_tokens) >= similarity_threshold
                    for seen_app, seen_tokens in seen
                ):
                    continue
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
                seen.append((app_key, tokens))
                added += 1
        return added


def find_gui_task_traces(task_dir: Path) -> list[Path]:
    if task_dir.is_file() and task_dir.name == "traj.json":
        return [task_dir]
    return sorted(task_dir.rglob("traj.json"))


def trace_step_count(trace_path: Path, *, subtask_index: int = 1) -> int:
    events = load_trajectory_events(trace_path, subtask_index=subtask_index)
    return sum(1 for event in events if _event_type(event) == "step")


def resolve_trace_app(trace_path: Path, *, subtask_index: int = 1) -> str | None:
    """Return the dominant non-system foreground app for one subtask."""
    counter: Counter[str] = Counter()
    for event in load_trajectory_events(trace_path, subtask_index=subtask_index):
        if _event_type(event) != "step":
            continue
        observation = event.get("observation") or {}
        app = str(
            observation.get("foreground_app")
            or observation.get("app")
            or event.get("app")
            or ""
        ).strip()
        if app and app.casefold() not in _LAUNCHER_OR_SYSTEM_PACKAGES:
            counter[app] += 1
    return counter.most_common(1)[0][0] if counter else None


def guess_app(task_name: str) -> str | None:
    """Fallback Android app guess used only when a trajectory has no app evidence."""
    lowered = task_name.lower()
    if "mattermost" in lowered:
        return "com.mattermost.rnbeta"
    if "mastodon" in lowered:
        return "org.joinmastodon.android.mastodon"
    if "calendar" in lowered or "schedule" in lowered or "conference" in lowered:
        return "org.fossify.calendar"
    if "alarm" in lowered or "clock" in lowered:
        return "com.google.android.deskclock"
    if "chrome" in lowered or "github" in lowered or "search" in lowered:
        return "com.android.chrome"
    if "settings" in lowered or "airplane" in lowered or "flight" in lowered:
        return "com.android.settings"
    if any(key in lowered for key in ("cart", "mall", "checkout", "item", "taodian")):
        return "com.testmall.app"
    if any(key in lowered for key in ("photo", "gallery", "wallpaper", "selfie")):
        return "gallery.photomanager.picturegalleryapp.imagegallery"
    if any(key in lowered for key in ("file", "download", "count", "sum", "bid")):
        return "com.google.android.documentsui"
    if any(
        key in lowered
        for key in ("mail", "email", "gmail", "meeting", "send", "event", "receipt", "invoice")
    ):
        return "com.gmailclone"
    return None


def _event_type(event: dict[str, Any]) -> str:
    return str(event.get("type") or event.get("event") or "")


def _find_task_goal(events: list[dict[str, Any]]) -> str:
    for event in events:
        if _event_type(event) == "metadata":
            return str(event.get("task") or "")
    return ""


def _find_result(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (event for event in reversed(events) if _event_type(event) == "result"),
        None,
    )


def _extract_thought(model_output: Any) -> str:
    if isinstance(model_output, dict):
        text = str(
            model_output.get("content")
            or model_output.get("raw_content")
            or model_output.get("action_text")
            or model_output.get("action_summary")
            or model_output.get("state_summary")
            or ""
        )
    else:
        text = str(model_output or "")
    for delimiter in ("\nAction:", "Action:"):
        index = text.find(delimiter)
        if index >= 0:
            text = text[:index]
            break
    return text.replace("Thought:", "").strip()


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _step_ui_hint(extra: dict[str, Any], *, max_chars: int) -> str:
    if not isinstance(extra, dict):
        return ""
    values: list[str] = []
    for key in ("clickable_text", "visible_text", "content_desc"):
        raw_values = extra.get(key)
        if isinstance(raw_values, list):
            values.extend(str(value).strip() for value in raw_values if str(value).strip())
    deduped = list(dict.fromkeys(values))[:12]
    return _truncate(", ".join(deduped), max_chars) if deduped else ""


def _dedup_token_set(item: GuiMemoryItem) -> frozenset[str]:
    text = f"{item.title}\n{item.content}".lower()
    tokens = {
        token
        for token in _DEDUP_TOKEN_RE.findall(text)
        if len(token) == 1 or (len(token) >= 2 and token not in _DEDUP_STOPWORDS)
    }
    return frozenset(tokens)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


__all__ = [
    "DEFAULT_DEDUP_THRESHOLD",
    "DEFAULT_MEMORY_BANK_PATH",
    "FAILURE_SYSTEM_PROMPT",
    "SUCCESS_SYSTEM_PROMPT",
    "append_to_memory_bank",
    "find_gui_task_traces",
    "format_trajectory_compact",
    "get_trace_outcome",
    "guess_app",
    "induce_memory_items",
    "is_abnormal_termination",
    "load_memory_bank",
    "parse_memory_items",
    "resolve_trace_app",
    "trace_is_abnormal",
    "trace_step_count",
]
