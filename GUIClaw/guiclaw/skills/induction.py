"""Shared compact-skill induction for online and offline trajectory mining."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from guiclaw.action import normalize_action_type
from guiclaw.skills.data import Skill
from guiclaw.skills.extractor import SkillExtractor
from guiclaw.skills.trajectory_codegen import codegen_trajectory

COMPACT_TAGS: tuple[str, ...] = ("compact", "compact_extracted")

_SCROLL_ACTIONS = frozenset({"scroll", "swipe", "drag"})
_TERMINAL_TARGET_WORDS = frozenset({
    "send",
    "publish",
    "post",
    "purchase",
    "checkout",
    "pay",
    "submit",
    "delete",
    "unfollow",
    "buy",
    "order",
    "confirm",
})
_TERMINAL_TARGET_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in sorted(_TERMINAL_TARGET_WORDS)) + r")\b"
)
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _normalize_target_text(text: str) -> str:
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", str(text or "")).replace("_", " ")
    return spaced.lower()


def has_terminal_action(skill: Skill) -> bool:
    """Return whether a skill targets an irreversible or externally visible action."""
    return any(
        _TERMINAL_TARGET_RE.search(_normalize_target_text(getattr(step, "target", "")))
        for step in skill.steps
    )


def compactify_skill(
    skill: Skill,
    *,
    max_steps: int = 7,
    max_scroll_steps: int = 1,
    is_success: bool = True,
) -> Skill | None:
    """Apply the compact envelope and provenance metadata to one extracted skill."""
    steps = tuple(skill.steps)
    if not 2 <= len(steps) <= max_steps:
        return None

    scroll_count = 0
    for index, step in enumerate(steps):
        action_type = normalize_action_type(step.action_type)
        if action_type == "open_app" and index != 0:
            return None
        if action_type in _SCROLL_ACTIONS:
            scroll_count += 1
            if scroll_count > max_scroll_steps:
                return None

    if not is_success and has_terminal_action(skill):
        return None

    return replace(
        skill,
        skill_id=f"compact:{skill.app}:{skill.name}",
        tags=COMPACT_TAGS,
        success_count=1 if is_success else 0,
    )


async def induce_compact_skills_from_trace(
    extractor: SkillExtractor,
    trace_path: Path,
    *,
    is_success: bool,
    subtask_index: int = 1,
    max_steps: int = 7,
    max_scroll_steps: int = 1,
) -> list[Skill]:
    """Extract compact skills from one subtask, including single-app trajectories."""
    result = codegen_trajectory(trace_path, subtask_index=subtask_index)
    if result is None or not result.steps:
        return []

    extracted = await extractor.extract_from_codegen_result_multi(
        result,
        is_success=is_success,
    )
    compact: list[Skill] = []
    for skill in extracted:
        candidate = compactify_skill(
            skill,
            max_steps=max_steps,
            max_scroll_steps=max_scroll_steps,
            is_success=is_success,
        )
        if candidate is not None:
            compact.append(candidate)
    return compact


__all__ = [
    "COMPACT_TAGS",
    "compactify_skill",
    "has_terminal_action",
    "induce_compact_skills_from_trace",
]
