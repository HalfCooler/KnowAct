"""Repeat-plan detection for handing one GUI step to the planner model.

The step loop first compares canonical action types in code, then the compact
UI trees recorded with each observation. Only a matching type on a nearly
unchanged tree (difference at most 5%) triggers a yes/no model verdict. A
confirmed repeat replans that step with the large planner model once.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from guiclaw.action import Action, describe_action

RepeatJudgeModel = Literal["small", "large"]
REPEAT_JUDGE_MODELS: tuple[str, ...] = ("small", "large")
REPEAT_UI_TREE_DIFFERENCE_LIMIT = 0.05

_SKIP_REPEAT_CHECK_TYPES = frozenset({"done", "request_intervention"})
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_TRUE_VALUES = frozenset({"true", "yes", "y", "1", "repeat", "repeated"})
_FALSE_VALUES = frozenset({"false", "no", "n", "0"})
_NODE_TEXT_FIELDS = ("resource_id", "class", "text", "content_desc")
_NODE_FLAG_FIELDS = ("clickable", "scrollable", "enabled")


@dataclass(frozen=True)
class RepeatVerdict:
    """Model judgment for whether a same-type plan is an actual repeat."""

    repeated: bool
    reason: str = ""
    raw_content: str = ""


def canonicalize_repeat_judge_model(value: str | None) -> RepeatJudgeModel:
    key = "small" if value in (None, "") else str(value).strip().lower()
    if key not in REPEAT_JUDGE_MODELS:
        raise ValueError(
            f"Unsupported repeat_judge_model {value!r}. "
            f"Expected one of: {', '.join(REPEAT_JUDGE_MODELS)}."
        )
    return key  # type: ignore[return-value]


def observation_ui_tree(observation: Any) -> list[dict[str, Any]] | None:
    """Return compact UI-tree nodes from an observation, or None if unusable."""
    extra = getattr(observation, "extra", None) if observation is not None else None
    if not isinstance(extra, dict):
        return None
    tree = extra.get("ui_tree")
    if not isinstance(tree, list):
        return None
    nodes = [node for node in tree if isinstance(node, dict)]
    return nodes or None


def ui_tree_difference_ratio(
    previous_tree: list[dict[str, Any]] | None,
    current_tree: list[dict[str, Any]] | None,
) -> float | None:
    """Return how much two compact UI trees differ, in ``[0, 1]``.

    The score is the share of node signatures that do not match (multiset L1
    distance over the combined node counts). Bounds and focus are ignored so
    animation jitter does not look like a new screen. ``None`` means at least
    one tree is missing, so similarity cannot be measured.
    """
    previous_counts = _ui_tree_signature_counts(previous_tree)
    current_counts = _ui_tree_signature_counts(current_tree)
    if previous_counts is None or current_counts is None:
        return None
    added = sum((current_counts - previous_counts).values())
    removed = sum((previous_counts - current_counts).values())
    total = sum(previous_counts.values()) + sum(current_counts.values())
    return (added + removed) / total


def should_judge_repeat(
    previous: Action | None,
    proposed: Action,
    *,
    previous_tree: list[dict[str, Any]] | None = None,
    current_tree: list[dict[str, Any]] | None = None,
    max_difference: float = REPEAT_UI_TREE_DIFFERENCE_LIMIT,
) -> bool:
    """Return True when code should ask a model whether this plan is a repeat."""
    if previous is None:
        return False
    proposed_type = str(proposed.action_type or "").strip().lower()
    previous_type = str(previous.action_type or "").strip().lower()
    if not proposed_type or proposed_type in _SKIP_REPEAT_CHECK_TYPES:
        return False
    if proposed_type != previous_type:
        return False
    difference = ui_tree_difference_ratio(previous_tree, current_tree)
    return difference is not None and difference <= max_difference


def format_action_for_judge(action: Action) -> str:
    return f"type={action.action_type}; {describe_action(action)}"


def build_repeat_judge_text(
    *,
    task: str,
    previous: Action,
    proposed: Action,
) -> str:
    return (
        "Decide whether the GUI agent's proposed action is a useless repeat of "
        "the previous action.\n\n"
        f"Task:\n{task.strip() or '(none)'}\n\n"
        f"Previous action:\n{format_action_for_judge(previous)}\n\n"
        f"Proposed action:\n{format_action_for_judge(proposed)}\n\n"
        "The two actions already share the same action type, and the UI tree "
        "changed very little after the previous action. That alone is not "
        "enough. Answer repeat=true only when the new action is effectively the "
        "same operation again (same control, same text, same scroll/swipe with "
        "no new target). Answer repeat=false when it is a distinct next step of "
        "the same kind, such as tapping a different button or typing into a new "
        "field.\n\n"
        'Return JSON only: {"repeat": true, "reason": "short reason"} or '
        '{"repeat": false, "reason": "short reason"}.'
    )


def build_repeat_escalation_hint(*, previous: Action, proposed: Action, reason: str) -> str:
    detail = str(reason or "").strip() or "the proposed action was judged a repeat"
    return (
        "The previous GUI step already used this action type, and the latest "
        f"plan was rejected as a repeat ({detail}). "
        f"Rejected plan: {format_action_for_judge(proposed)}. "
        f"Previous action: {format_action_for_judge(previous)}. "
        "Take one different action that makes progress. Do not repeat the "
        "rejected operation."
    )


def parse_repeat_verdict(content: str) -> RepeatVerdict:
    """Parse a judge response. Unknown or empty output is treated as not-repeat."""
    raw = str(content or "").strip()
    if not raw:
        return RepeatVerdict(repeated=False, reason="empty_judge_response", raw_content=raw)

    payload = _extract_json_object(raw)
    if isinstance(payload, dict) and "repeat" in payload:
        repeated = _coerce_bool(payload.get("repeat"))
        if repeated is not None:
            reason = str(payload.get("reason") or "").strip()
            return RepeatVerdict(repeated=repeated, reason=reason, raw_content=raw)

    lowered = raw.lower()
    if re.search(r'"repeat"\s*:\s*true', lowered):
        return RepeatVerdict(repeated=True, reason=_reason_from_text(raw), raw_content=raw)
    if re.search(r'"repeat"\s*:\s*false', lowered):
        return RepeatVerdict(repeated=False, reason=_reason_from_text(raw), raw_content=raw)

    token = re.sub(r"[^a-z]+", "", lowered)
    if token in _TRUE_VALUES:
        return RepeatVerdict(repeated=True, reason=raw[:200], raw_content=raw)
    if token in _FALSE_VALUES:
        return RepeatVerdict(repeated=False, reason=raw[:200], raw_content=raw)
    return RepeatVerdict(repeated=False, reason="unparsed_judge_response", raw_content=raw)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))
    match = _JSON_OBJECT_RE.search(text)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value or "").strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return None


def _reason_from_text(text: str) -> str:
    payload = _extract_json_object(text)
    if isinstance(payload, dict):
        reason = str(payload.get("reason") or "").strip()
        if reason:
            return reason
    return text[:200]


def _ui_tree_signature_counts(
    tree: list[dict[str, Any]] | None,
) -> Counter[tuple[str, ...]] | None:
    if not tree:
        return None
    counts: Counter[tuple[str, ...]] = Counter()
    for node in tree:
        if isinstance(node, dict):
            counts[_node_signature(node)] += 1
    return counts or None


def _node_signature(node: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(node.get(field) or "") for field in _NODE_TEXT_FIELDS) + tuple(
        "1" if node.get(field) else "0" for field in _NODE_FLAG_FIELDS
    )
