"""
guiclaw.skills._merger
======================
Pure-function skill conflict detection and merge logic.

Extracted from ``flat.py`` — no I/O, no side effects, no locks.  All functions
are deterministic given their inputs, making them independently testable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from guiclaw.skills.data import Skill, SkillStep, compute_confidence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBEDDING_CONFLICT_THRESHOLD = 0.72
_STRUCTURAL_CONFLICT_THRESHOLD = 0.70
_CLEANUP_EMBEDDING_THRESHOLD = 0.72

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "in",
        "on",
        "of",
        "for",
        "and",
        "or",
        "is",
        "it",
        "with",
        "from",
        "by",
        "at",
        "be",
        "this",
        "that",
        "do",
        "does",
        "did",
    }
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

_StepSignature = tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True)
class SkillConflict:
    skill: Skill
    score: float
    embedding_similarity: float | None
    sequence_similarity: float
    semantic_similarity: float


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


def find_best_conflict(
    incoming: Skill,
    skills: list[Skill],
    *,
    incoming_embedding: np.ndarray | None,
    existing_embeddings: dict[str, np.ndarray],
) -> SkillConflict | None:
    best: SkillConflict | None = None
    best_score = 0.0
    incoming_signature = action_signature(incoming)
    for existing in skills:
        if existing.platform != incoming.platform or existing.app != incoming.app:
            continue
        if existing.skill_id == incoming.skill_id:
            return SkillConflict(
                skill=existing,
                score=1.0,
                embedding_similarity=1.0,
                sequence_similarity=1.0,
                semantic_similarity=1.0,
            )

        existing_signature = action_signature(existing)
        sequence_sim = action_similarity(existing_signature, incoming_signature)
        semantic_sim = skill_semantic_similarity(existing, incoming)
        if is_strict_rich_prefix(existing_signature, incoming_signature) or is_strict_rich_prefix(
            incoming_signature, existing_signature
        ):
            continue
        embedding_sim: float | None = None
        score = 0.20 * sequence_sim + 0.05 * semantic_sim
        existing_embedding = existing_embeddings.get(existing.skill_id)
        if incoming_embedding is not None and existing_embedding is not None:
            embedding_sim = cosine_similarity(incoming_embedding, existing_embedding)
            score += 0.75 * embedding_sim

        if score > best_score:
            best = SkillConflict(
                skill=existing,
                score=score,
                embedding_similarity=embedding_sim,
                sequence_similarity=sequence_sim,
                semantic_similarity=semantic_sim,
            )
            best_score = score

    if best is None:
        return None
    if best.embedding_similarity is not None:
        if (
            best.embedding_similarity >= _EMBEDDING_CONFLICT_THRESHOLD
            and best.sequence_similarity >= _STRUCTURAL_CONFLICT_THRESHOLD
        ):
            return best
        if best.embedding_similarity >= 0.60 and best.sequence_similarity >= 0.78:
            return best
        return None
    if best.sequence_similarity >= 0.82 and best.semantic_similarity >= 0.25:
        return best
    if best.sequence_similarity >= 0.92:
        return best
    return None


# ---------------------------------------------------------------------------
# Merge decision
# ---------------------------------------------------------------------------


def heuristic_merge_decision(conflict: SkillConflict, new: Skill) -> str:
    old = conflict.skill
    if old.success_count > 0 and new.success_count == 0 and conflict.sequence_similarity < 0.85:
        return "KEEP_OLD"
    if new.success_count > old.success_count and old.success_count == 0:
        return "KEEP_NEW"
    return "MERGE"


def merge_skills(old: Skill, new: Skill) -> Skill:
    old_signature = action_signature(old)
    new_signature = action_signature(new)
    if old.success_count > 0 and new.success_count == 0:
        steps = old.steps
    elif new.success_count > 0 and old.success_count == 0:
        steps = new.steps
    elif is_strict_rich_prefix(old_signature, new_signature):
        steps = new.steps
    elif is_strict_rich_prefix(new_signature, old_signature):
        steps = old.steps
    elif compute_confidence(new) > compute_confidence(old):
        steps = new.steps or old.steps
    else:
        steps = old.steps or new.steps
    prefer_old_text = old.success_count > 0 and new.success_count == 0
    return Skill(
        skill_id=old.skill_id,
        name=old.name if prefer_old_text and old.name else (new.name or old.name),
        description=old.description
        if prefer_old_text and old.description
        else (new.description or old.description),
        app=new.app or old.app,
        platform=new.platform or old.platform,
        steps=steps,
        parameters=tuple(sorted(set(old.parameters) | set(new.parameters))),
        tags=tuple(sorted(set(old.tags) | set(new.tags))),
        created_at=old.created_at,
        success_count=old.success_count + new.success_count,
        failure_count=old.failure_count + new.failure_count,
    )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_superseded_prefixes(
    skills: list[Skill],
    platform: str,
    app: str,
    *,
    embeddings: dict[str, np.ndarray] | None = None,
) -> list[Skill]:
    normalized_app = app
    candidates = [
        skill for skill in skills if skill.platform == platform and skill.app == normalized_app
    ]
    removed: set[str] = set()
    for skill in candidates:
        if skill.success_count > 0:
            continue
        signature = action_signature(skill)
        for other in candidates:
            if other.skill_id == skill.skill_id:
                continue
            other_signature = action_signature(other)
            if (
                is_strict_rich_prefix(signature, other_signature)
                and compute_confidence(other) > compute_confidence(skill)
                and cleanup_same_intent(skill, other, embeddings=embeddings)
            ):
                removed.add(skill.skill_id)
                break
    if not removed:
        return skills
    return [skill for skill in skills if skill.skill_id not in removed]


def cleanup_same_intent(
    left: Skill,
    right: Skill,
    *,
    embeddings: dict[str, np.ndarray] | None,
) -> bool:
    if embeddings:
        left_embedding = embeddings.get(left.skill_id)
        right_embedding = embeddings.get(right.skill_id)
        if left_embedding is not None and right_embedding is not None:
            return (
                cosine_similarity(left_embedding, right_embedding) >= _CLEANUP_EMBEDDING_THRESHOLD
            )
    return skill_semantic_similarity(left, right) >= 0.20


# ---------------------------------------------------------------------------
# Similarity / signature helpers
# ---------------------------------------------------------------------------


def action_signature(skill_obj: Skill) -> tuple[_StepSignature, ...]:
    return tuple(step_signature(step) for step in skill_obj.steps)


def step_signature(step: SkillStep) -> _StepSignature:
    return (
        step.action_type,
        tokens(step.target),
        tokens(" ".join([*map(str, step.parameters.keys()), *map(str, step.parameters.values())])),
        tokens(step.valid_state or ""),
        tokens(stable_json(step.state_contract)),
    )


def action_similarity(
    left: tuple[_StepSignature, ...],
    right: tuple[_StepSignature, ...],
) -> float:
    if not left and not right:
        return 1.0
    min_len = min(len(left), len(right))
    max_len = max(len(left), len(right))
    if max_len == 0:
        return 1.0
    if min_len == 0:
        return 0.0
    pair_scores = [
        step_similarity(left_step, right_step)
        for left_step, right_step in zip(left, right, strict=False)
    ]
    overlap_quality = sum(pair_scores) / min_len
    tail_len = max_len - min_len
    length_penalty = 1.0 - 0.3 * (tail_len / max_len)
    return overlap_quality * length_penalty


def step_similarity(left: _StepSignature, right: _StepSignature) -> float:
    if left[0] != right[0]:
        return 0.0
    return (
        0.35
        + 0.30 * weighted_tuple_jaccard(left[1], right[1], empty_score=0.15)
        + 0.10 * weighted_tuple_jaccard(left[2], right[2], empty_score=0.50)
        + 0.15 * weighted_tuple_jaccard(left[3], right[3], empty_score=0.35)
        + 0.10 * weighted_tuple_jaccard(left[4], right[4], empty_score=0.50)
    )


def is_strict_rich_prefix(
    shorter: tuple[_StepSignature, ...],
    longer: tuple[_StepSignature, ...],
) -> bool:
    if len(shorter) >= len(longer) or not shorter:
        return False
    return all(
        step_similarity(short_step, long_step) >= 0.80
        for short_step, long_step in zip(shorter, longer, strict=False)
    )


def skill_semantic_similarity(left: Skill, right: Skill) -> float:
    name_sim = name_token_similarity(left.name, right.name)
    description_sim = name_token_similarity(left.description, right.description)
    if left.description.strip() and right.description.strip():
        return max(description_sim, 0.7 * description_sim + 0.3 * name_sim)
    return name_sim


def name_token_similarity(a: str, b: str) -> float:
    left = set(re.findall(r"\w+", a.lower())) - _STOPWORDS
    right = set(re.findall(r"\w+", b.lower())) - _STOPWORDS
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


# ---------------------------------------------------------------------------
# Low-level token / hash helpers
# ---------------------------------------------------------------------------


def tuple_jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    return len(left_set & right_set) / len(left_set | right_set)


def weighted_tuple_jaccard(
    left: tuple[str, ...],
    right: tuple[str, ...],
    *,
    empty_score: float,
) -> float:
    if not left and not right:
        return empty_score
    return tuple_jaccard(left, right)


def tokens(value: Any) -> tuple[str, ...]:
    return tuple(
        token for token in re.findall(r"\w+", str(value).lower()) if token not in _STOPWORDS
    )


def stable_json(value: Any) -> str:
    if not value:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
