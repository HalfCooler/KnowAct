"""
guiclaw.trajectory.summarizer
=============================
LLM-driven trajectory summarization and shared GUI state-note formatting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from guiclaw.interfaces import LLMProvider
from guiclaw.trajectory.recorder import load_trajectory_events

logger = logging.getLogger(__name__)

_STATE_NOTE_LABELS: tuple[str, ...] = ("Status", "Done", "Remaining", "Current", "Resume")

_SUMMARY_PROMPT = """\
Summarize the following GUI automation trajectory as a compact state note.
Return exactly 5 short lines using these labels, in this order:
Status: completed|partial|blocked
Done: ...
Remaining: ...
Current: ...
Resume: ...

Rules:
- Keep each value short and readable.
- Do not add bullets, markdown, or extra lines.
- Use "none" for Remaining when the task is completed.
- Use a clear resume hint when continuation is still possible.

# Trajectory
{trajectory}

Respond with the 5-line state note only.
"""


def build_state_note(*, status: str, done: str, remaining: str, current: str, resume: str) -> str:
    """Format the compact GUI state note used across live and background summaries."""
    values = {
        "Status": status,
        "Done": done,
        "Remaining": remaining,
        "Current": current,
        "Resume": resume,
    }
    lines = []
    for label in _STATE_NOTE_LABELS:
        value = _normalize_note_value(values[label], default="none" if label != "Status" else "blocked")
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def is_state_note(text: str) -> bool:
    """Return True when ``text`` matches the strict 5-line state-note format."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != len(_STATE_NOTE_LABELS):
        return False
    for line, label in zip(lines, _STATE_NOTE_LABELS):
        prefix, separator, _ = line.partition(":")
        if separator != ":" or prefix.strip() != label:
            return False
    return True


def _extract_state_note(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    note_size = len(_STATE_NOTE_LABELS)
    for start in range(len(lines) - note_size + 1):
        candidate = "\n".join(lines[start : start + note_size])
        if is_state_note(candidate):
            return candidate
    return None


def _normalize_note_value(value: Any, *, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return " ".join(text.split())


def _model_output_text(value: Any) -> str:
    if isinstance(value, dict):
        value = (
            value.get("content")
            or value.get("raw_content")
            or value.get("action_text")
            or value.get("action_summary")
            or value.get("state_summary")
            or ""
        )
    return str(value or "").strip()


def _derive_state_note_from_events(events: list[dict[str, Any]]) -> str:
    metadata = next((event for event in events if event.get("type") == "metadata"), {})
    result = next((event for event in reversed(events) if event.get("type") == "result"), {})
    steps = [event for event in events if event.get("type") == "step"]
    status = _derive_status(result)
    done = _summarize_step_progress(steps)
    current = _describe_latest_screen(steps, metadata, status)
    remaining = _remaining_hint(status=status, error=str(result.get("error") or ""))
    resume = _resume_hint(status=status, error=str(result.get("error") or ""))
    return build_state_note(
        status=status,
        done=done,
        remaining=remaining,
        current=current,
        resume=resume,
    )


def _derive_status(result_event: dict[str, Any]) -> str:
    if result_event.get("success"):
        return "completed"
    error = str(result_event.get("error") or "")
    total_steps = int(result_event.get("total_steps") or 0)
    if total_steps == 0 and error:
        return "blocked"
    if error.startswith("stagnation_detected") or error.startswith("intervention_cancelled"):
        return "blocked"
    return "partial"


def _summarize_step_progress(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return "No GUI actions were completed."
    summaries: list[str] = []
    for event in steps[-3:]:
        summary = _model_output_text(event.get("model_output"))
        if not summary:
            action = event.get("action") if isinstance(event.get("action"), dict) else {}
            action_type = action.get("action_type", "step")
            text = action.get("text")
            summary = f"{action_type} {text}".strip() if text else action_type
        text = _normalize_note_value(summary, default="")
        if text:
            summaries.append(text.rstrip("."))
    if not summaries:
        return "No GUI actions were completed."
    return "; ".join(summaries)


def _describe_latest_screen(
    steps: list[dict[str, Any]],
    metadata: dict[str, Any],
    status: str,
) -> str:
    observation: dict[str, Any] = {}
    for event in reversed(steps):
        candidate = event.get("observation")
        if isinstance(candidate, dict):
            observation = candidate
            break
    app = observation.get("foreground_app") or observation.get("app")
    width = observation.get("screen_width")
    height = observation.get("screen_height")
    parts: list[str] = []
    if isinstance(app, str) and app.strip():
        parts.append(app.strip())
    platform = metadata.get("platform")
    if not parts and isinstance(platform, str) and platform.strip():
        parts.append(platform.strip())
    if isinstance(width, int) and isinstance(height, int):
        parts.append(f"{width}x{height}")
    if parts:
        return " ".join(parts)
    if not steps:
        return "Current screen state unavailable."
    if status == "completed":
        return "Final recorded screen state."
    return "Last recorded screen state."


def _remaining_hint(*, status: str, error: str) -> str:
    if status == "completed":
        return "none"
    if error.startswith("stagnation_detected"):
        return "Change the action sequence from the current screen."
    if error.startswith("intervention_cancelled"):
        return "Wait for the intervention blocker to be resolved."
    if error.startswith("step_timeout"):
        return "Retry the timed-out step from the current screen."
    if error.startswith("max_steps_exceeded"):
        return "Continue the remaining task from the current screen."
    if status == "blocked":
        return "Resolve the blocker before retrying."
    return "Continue from the current screen."


def _resume_hint(*, status: str, error: str) -> str:
    if status == "completed":
        return "No further action needed."
    if error.startswith("stagnation_detected"):
        return "Resume by trying a different action on the same screen."
    if error.startswith("intervention_cancelled"):
        return "Resolve the intervention blocker, then continue from the current screen."
    if error.startswith("step_timeout"):
        return "Resume from the current screen after the timeout clears."
    if error.startswith("max_steps_exceeded"):
        return "Resume from the current screen and finish the remaining steps."
    if status == "blocked":
        return "Resolve the blocker, then continue from the current screen."
    return "Resume from the current screen."


class TrajectorySummarizer:
    """Summarize trajectory JSONL files into natural language via LLM."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def summarize_file(self, trajectory_path: Path, *, subtask_index: int = 1) -> str:
        """Read one compact trajectory subtask and return a natural-language summary."""
        if not trajectory_path.exists():
            logger.warning("Trajectory file not found: %s", trajectory_path)
            return ""

        events = load_trajectory_events(trajectory_path, subtask_index=subtask_index)
        return await self.summarize_events(events)

    async def summarize_events(self, events: list[dict]) -> str:
        """Summarize pre-parsed trajectory events."""
        if not events:
            return ""

        # Build a compact representation for the LLM
        compact = self._compact_events(events)
        prompt = _SUMMARY_PROMPT.format(trajectory=compact)

        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self._llm.chat(messages)
            state_note = _extract_state_note(response.content)
            if state_note is not None:
                return state_note
            logger.debug("Trajectory summary did not match the state-note contract; using fallback.")
        except Exception:
            logger.warning("Trajectory summarization failed; using fallback state note.", exc_info=True)

        return _derive_state_note_from_events(events)

    @staticmethod
    def _compact_events(events: list[dict]) -> str:
        """Build a compact text representation of trajectory events."""
        lines: list[str] = []

        for event in events:
            etype = event.get("type", "")
            if etype == "metadata":
                lines.append(f"Task: {event.get('task', 'unknown')}")
                lines.append(f"Platform: {event.get('platform', 'unknown')}")
            elif etype == "step":
                idx = event.get("step_index", "?")
                action = event.get("action", {})
                action_type = action.get("action_type", "unknown")
                text = action.get("text", "")
                model_out = _model_output_text(event.get("model_output"))
                line = f"Step {idx}: {action_type}"
                if text:
                    line += f' text="{text}"'
                if model_out:
                    line += f" | {model_out[:100]}"
                lines.append(line)
            elif etype == "result":
                success = event.get("success", False)
                duration = event.get("duration_s", 0)
                error = event.get("error")
                status = "SUCCESS" if success else "FAILED"
                line = f"Result: {status} ({duration}s)"
                if error:
                    line += f" error={error}"
                lines.append(line)

        return "\n".join(lines)
