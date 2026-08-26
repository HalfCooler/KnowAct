"""Use deeplink skills to simplify UI skills."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from guiclaw.skills.data import Skill, SkillStep, collect_placeholder_names

SKILL_OPTIMIZATIONS_FILENAME = "skill_optimizations.json"
SKILL_OPTIMIZATIONS_VERSION = 2
OPTIMIZED_SKILL_TAG = "shortcut_optimized"
_STORE_LOCKS: dict[Path, threading.RLock] = {}
_STORE_LOCKS_GUARD = threading.Lock()


def _store_lock(store_dir: Path) -> threading.RLock:
    key = Path(store_dir).expanduser().resolve(strict=False)
    with _STORE_LOCKS_GUARD:
        return _STORE_LOCKS.setdefault(key, threading.RLock())


class SkillOptimizationStore:
    """Store reversible prefix replacements outside canonical ``skills.py``."""

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = Path(store_dir).expanduser()
        self.path = self.store_dir / SKILL_OPTIMIZATIONS_FILENAME
        self._lock = _store_lock(self.store_dir)

    def read(self) -> dict[str, Any]:
        with self._lock:
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if payload.get("version") != SKILL_OPTIMIZATIONS_VERSION:
                payload = {}
            recipes = payload.get("recipes")
            return {
                "version": SKILL_OPTIMIZATIONS_VERSION,
                "recipes": recipes if isinstance(recipes, dict) else {},
            }

    def upsert(self, recipe: dict[str, Any]) -> None:
        with self._lock:
            payload = self.read()
            payload["recipes"][recipe["base_skill_id"]] = recipe
            self.store_dir.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.store_dir, text=True
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                    handle.write("\n")
                os.replace(temporary, self.path)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass


def apply_skill_optimizations(
    raw_skills: list[Skill] | tuple[Skill, ...],
    store: SkillOptimizationStore,
) -> list[Skill]:
    skills = list(raw_skills)
    by_id = {skill.skill_id: skill for skill in skills}
    recipes = store.read()["recipes"]
    return [
        _materialize(
            skill,
            by_id.get(str(recipes.get(skill.skill_id, {}).get("shortcut_skill_id"))),
            recipes.get(skill.skill_id, {}),
        )
        or skill
        for skill in skills
    ]


async def optimize_shortcut_prefixes(
    raw_skills: list[Skill] | tuple[Skill, ...],
    *,
    embeddings: np.ndarray,
    store_dir: Path,
    llm: Any | None,
    focus_skill_ids: list[str] | tuple[str, ...] | None = None,
    persist: bool = True,
    embedding_top_k: int = 3,
    validator: Callable[[Skill, dict[str, str]], Awaitable[tuple[bool, str]]] | None = None,
    delete_skill: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Retrieve same-app deeplinks and let the LLM replace, delete, or keep each UI skill."""
    skills = list(raw_skills)
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(skills):
        raise ValueError("skill embeddings must align with raw skills")
    report: dict[str, Any] = {
        "candidate_count": 0,
        "activated": [],
        "deleted": [],
        "proposed": [],
        "accepted": [],
        "rejected": [],
        "validation_results": [],
        "batches": [],
    }
    if llm is None:
        report["status"] = "skipped_no_llm"
        return report

    focus = {str(item) for item in focus_skill_ids or ()}
    deeplinks = [(index, skill) for index, skill in enumerate(skills) if _is_deeplink(skill)]
    bases = [
        (index, skill)
        for index, skill in enumerate(skills)
        if skill.steps and not _is_shortcut(skill)
    ]
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    # Rank every (base, deeplink) pair in one BLAS call: scores[i, j] is the cosine
    # similarity between bases[i] and deeplinks[j] (both already L2-normalized).
    base_rows = np.asarray([row for row, _ in bases], dtype=np.int64)
    shortcut_rows = np.asarray([row for row, _ in deeplinks], dtype=np.int64)
    scores = normalized[base_rows] @ normalized[shortcut_rows].T
    allowed = np.equal.outer(
        [skill.app for _, skill in bases], [skill.app for _, skill in deeplinks]
    )
    if focus:
        allowed &= np.logical_or.outer(
            [skill.skill_id in focus for _, skill in bases],
            [skill.skill_id in focus for _, skill in deeplinks],
        )
    scores = np.where(allowed, scores, -np.inf)

    top_k = max(1, embedding_top_k)
    batches: list[tuple[Skill, list[dict[str, Any]]]] = []
    retrieval: list[dict[str, Any]] = []
    candidate_index = 0
    for index, (_, base) in enumerate(bases):
        row = scores[index]
        valid = np.flatnonzero(np.isfinite(row))
        if valid.size == 0:
            continue
        take = min(top_k, valid.size)
        # Partial sort: the `take` largest scores land at the tail, then order them.
        order = np.argpartition(row, -take)[-take:]
        order = order[np.argsort(row[order])[::-1]]
        options: list[dict[str, Any]] = []
        for shortcut_row in order:
            shortcut = deeplinks[int(shortcut_row)][1]
            similarity = float(row[shortcut_row])
            candidate = {
                "candidate_id": f"candidate_{candidate_index}",
                "shortcut": shortcut,
                "similarity": similarity,
            }
            candidate_index += 1
            options.append(candidate)
            retrieval.append(
                {
                    "base_skill_id": base.skill_id,
                    "shortcut_skill_id": shortcut.skill_id,
                    "similarity": round(similarity, 6),
                }
            )
        if options:
            batches.append((base, options))

    report["candidate_count"] = candidate_index
    report["retrieval"] = retrieval
    if not batches:
        report["status"] = "no_candidates"
        return report

    store = SkillOptimizationStore(store_dir)
    usage: dict[str, int] = {}
    for batch_index, (base, options) in enumerate(batches, 1):
        try:
            response = await llm.chat(
                messages=[{"role": "user", "content": _prompt(base, options)}],
                max_tokens=512,
                reasoning_effort="none",
            )
            proposal = _parse_json(str(getattr(response, "content", "") or ""))
        except Exception as exc:
            report["batches"].append(
                {"index": batch_index, "status": "llm_error", "error": str(exc)[:300]}
            )
            continue
        for key, value in dict(getattr(response, "usage", None) or {}).items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
        action = proposal.get("action")
        report["batches"].append({"index": batch_index, "status": action or "invalid"})
        if action == "keep":
            continue
        candidate = next(
            (item for item in options if item["candidate_id"] == proposal.get("candidate_id")),
            None,
        )
        if candidate is None or action not in {"replace_prefix", "delete"}:
            report["rejected"].append(f"{base.skill_id}:invalid_decision")
            continue
        shortcut = candidate["shortcut"]
        argument_map = proposal.get("argument_map")
        if not isinstance(argument_map, dict):
            report["rejected"].append(f"{base.skill_id}:invalid_argument_map")
            continue
        argument_map = {
            str(key).removeprefix("{{").removesuffix("}}"): str(value)
            .removeprefix("{{")
            .removesuffix("}}")
            for key, value in argument_map.items()
        }
        if set(argument_map) != set(shortcut.parameters) or not set(argument_map.values()).issubset(
            base.parameters
        ):
            report["rejected"].append(f"{base.skill_id}:invalid_argument_map")
            continue
        replace_count = (
            len(base.steps) if action == "delete" else proposal.get("replace_prefix_steps")
        )
        if not isinstance(replace_count, int) or not 0 < replace_count < len(base.steps):
            if action != "delete":
                report["rejected"].append(f"{base.skill_id}:invalid_prefix")
                continue
        recipe = {
            "base_skill_id": base.skill_id,
            "shortcut_skill_id": shortcut.skill_id,
            "replace_prefix_steps": replace_count,
            "argument_map": argument_map,
            "reason": str(proposal.get("reason") or "")[:300],
        }
        optimized = _materialize(base, shortcut, recipe)
        if optimized is None:
            report["rejected"].append(f"{base.skill_id}:invalid_recipe")
            continue
        if validator is not None:
            arguments = proposal.get("validation_arguments")
            if not isinstance(arguments, dict) or set(arguments) != set(base.parameters):
                report["rejected"].append(f"{base.skill_id}:invalid_validation_arguments")
                continue
            passed, detail = await validator(
                optimized, {str(k): str(v) for k, v in arguments.items()}
            )
            report["validation_results"].append(
                {"skill_id": base.skill_id, "success": bool(passed), "detail": str(detail)[:500]}
            )
            if not passed:
                report["rejected"].append(f"{base.skill_id}:device_validation_failed")
                continue
        report["accepted"].append(
            {
                "skill_id": base.skill_id,
                "shortcut_skill_id": shortcut.skill_id,
                "action": action,
                "replace_prefix_steps": replace_count,
            }
        )
        if not persist:
            report["proposed"].append(base.skill_id)
        elif action == "delete":
            if delete_skill is None or not delete_skill(base.skill_id):
                report["rejected"].append(f"{base.skill_id}:delete_failed")
                continue
            report["deleted"].append(base.skill_id)
        else:
            store.upsert(recipe)
            report["activated"].append(base.skill_id)

    report["status"] = "processed"
    report["batch_count"] = len(batches)
    report["usage"] = usage
    return report


def _prompt(base: Skill, options: list[dict[str, Any]]) -> str:
    candidates = [
        {
            "candidate_id": item["candidate_id"],
            "similarity": round(item["similarity"], 4),
            "skill": _skill_record(item["shortcut"]),
        }
        for item in options
    ]
    return (
        "Optimize one UI skill with exact deeplink skills. Assume every deeplink executes "
        "precisely. Choose action=replace_prefix when it replaces leading UI steps, action=delete "
        "when it fully covers the UI skill, or action=keep. Return one JSON object only. "
        "For keep return only action. Otherwise return action, candidate_id, argument_map, reason, "
        "and validation_arguments. argument_map uses bare parameter names: deeplink parameter -> UI "
        "parameter. validation_arguments maps every UI parameter to a concrete test value. For "
        "replace_prefix, replace_prefix_steps must be an integer from 1 to ui_skill.step_count-1.\n"
        "It is the number of leading UI steps to remove and must include every input or submit "
        "step already performed by the deeplink.\n"
        + json.dumps(
            {"ui_skill": _skill_record(base), "deeplink_candidates": candidates},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _materialize(base: Skill, shortcut: Skill | None, recipe: dict[str, Any]) -> Skill | None:
    if shortcut is None:
        return None
    count = recipe.get("replace_prefix_steps")
    mapping = recipe.get("argument_map")
    if (
        not isinstance(count, int)
        or not 0 < count <= len(base.steps)
        or not isinstance(mapping, dict)
    ):
        return None
    mapped = tuple(_map_step(step, mapping) for step in shortcut.steps)
    if not collect_placeholder_names([step.to_dict() for step in mapped]).issubset(base.parameters):
        return None
    return replace(
        base,
        steps=(*mapped, *base.steps[count:]),
        tags=tuple(dict.fromkeys((*base.tags, OPTIMIZED_SKILL_TAG))),
    )


def _map_step(step: SkillStep, mapping: dict[str, Any]) -> SkillStep:
    def convert(value: Any) -> Any:
        if isinstance(value, str):
            return re.sub(
                r"\{\{([^{}]+)\}\}",
                lambda match: "{{" + str(mapping.get(match.group(1), match.group(1))) + "}}",
                value,
            )
        if isinstance(value, dict):
            return {convert(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(convert(item) for item in value)
        return value

    return replace(
        step,
        target=convert(step.target),
        parameters=convert(step.parameters),
        valid_state=convert(step.valid_state),
        state_contract=convert(step.state_contract),
        fixed_values=convert(step.fixed_values),
    )


def _is_shortcut(skill: Skill) -> bool:
    return "shortcut" in {tag.casefold() for tag in skill.tags} or skill.skill_id.startswith(
        "shortcut:"
    )


def _is_deeplink(skill: Skill) -> bool:
    return _is_shortcut(skill) and any(step.action_type == "open_deeplink" for step in skill.steps)


def _skill_record(skill: Skill) -> dict[str, Any]:
    return {
        "skill_id": skill.skill_id,
        "description": skill.description,
        "parameters": list(skill.parameters),
        "step_count": len(skill.steps),
        "steps": [step.to_dict() for step in skill.steps],
    }


def _parse_json(text: str) -> dict[str, Any]:
    try:
        from json_repair import loads

        payload = loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        try:
            payload = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            payload = {}
    return payload if isinstance(payload, dict) else {}


__all__ = [
    "OPTIMIZED_SKILL_TAG",
    "SKILL_OPTIMIZATIONS_FILENAME",
    "SkillOptimizationStore",
    "apply_skill_optimizations",
    "optimize_shortcut_prefixes",
]
