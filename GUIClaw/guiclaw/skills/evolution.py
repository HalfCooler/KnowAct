"""
guiclaw.skills.evolution
========================
Failure-grounded evolution for flat ``skills.py`` skills.

This module updates the reused skill that failed in a trajectory.  It does not
create a new skill candidate and does not restore the old graph/code-first stack.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from guiclaw.skills._merger import cosine_similarity
from guiclaw.skills.data import Skill, SkillStep
from guiclaw.skills.flat import FlatSkillLibrary, _skill_search_text
from guiclaw.skills.normalization import normalize_app_identifier, normalize_skill_app
from guiclaw.skills.state_contract import normalize_state_contract
from guiclaw.skills.trajectory_codegen import apply_focused_input_contracts
from guiclaw.trajectory.recorder import load_trajectory_events

logger = logging.getLogger(__name__)

EVOLUTION_DIRNAME = "evolution"
FAILURE_CASES_FILENAME = "failure_cases.jsonl"

_EVOLUTION_PROMPT = """\
You are improving one existing GUI automation skill after it failed during reuse.

Task:
{task}

Original skill JSON:
{skill_json}

Failure case:
{failure_case_json}

Prior feedback for this skill:
{feedback_json}

Full trajectory:
{trajectory_json}

Return ONLY a JSON object for the improved skill, using the same schema as the original skill.
Rules:
- Improve the original skill in place. Do not create a different skill or unrelated workflow.
- Keep the same high-level intent unless the failure shows the description caused a wrong match; then narrow the description.
- If a popup or optional obstacle appeared, add a guarded optional step before the blocked step:
  {{"action_type": "tap", "target": "Close", "parameters": {{"optional": true}}, "valid_state": "popup close button is visible"}}
  Optional steps are skipped when their valid_state is not present.
- If an action target, parameter, valid_state, or state_contract is stale for the new UI, update that step.
- Preserve reusable parameters as {{param_name}} placeholders.
- Do not include final destructive actions such as pay, delete, send, submit, or irreversible confirmation unless the original skill already required them.
- Omit skill_id; the caller will preserve the original skill_id.
"""


@dataclass(frozen=True)
class SkillFailureCase:
    trace: str
    task: str
    platform: str
    skill_id: str
    skill_name: str
    failed_step_index: int | None
    failed_target: str
    failed_valid_state: str | None
    failed_state_contract: dict[str, Any] | None
    failure_observation: dict[str, Any] | None
    failure_screenshot_path: str | None
    failure_error: str | None
    execution_error: str | None


async def evolve_failed_skill_from_trace(
    *,
    llm: Any,
    trace_path: Path,
    store_root: Path,
    task: str | None,
    platform: str,
    embedding_provider: Any | None = None,
    embedding_signature: str | None = None,
    subtask_index: int = 1,
) -> dict[str, Any]:
    events = load_trajectory_events(trace_path, subtask_index=subtask_index)
    failure_case = _build_failure_case(
        events,
        trace_path=trace_path,
        task=task,
        platform=platform,
    )
    if failure_case is None:
        return {
            "status": "no_failure_case",
            "trace": str(trace_path),
            "updated_skill_id": None,
            "updated_functions": [],
            "compiled_skill_ids": [],
        }

    store_root = Path(store_root).expanduser()
    _append_failure_case(store_root, failure_case)
    library = FlatSkillLibrary(
        store_dir=store_root,
        embedding_provider=embedding_provider,
        embedding_signature=embedding_signature,
    )
    library.record_feedback(
        failure_case.skill_id,
        task=task or failure_case.task,
        failure_case=asdict(failure_case),
        status="failure_detected",
    )
    original = library.get(failure_case.skill_id)
    if original is None:
        return {
            "status": "missing_skill",
            "trace": str(trace_path),
            "failure_case": asdict(failure_case),
            "updated_skill_id": None,
            "updated_functions": [],
            "compiled_skill_ids": [],
        }
    feedback = library.feedback_for_skill(original.skill_id)

    response = await llm.chat(
        _build_messages(
            task=task or failure_case.task,
            original=original,
            failure_case=failure_case,
            feedback=feedback,
            events=events,
        )
    )
    candidate = _parse_skill_response(response.content, original=original)
    if candidate is None:
        return {
            "status": "no_candidate",
            "trace": str(trace_path),
            "failure_case": asdict(failure_case),
            "updated_skill_id": None,
            "updated_functions": [],
            "compiled_skill_ids": [],
            "token_usage": dict(getattr(response, "usage", {}) or {}),
        }
    candidate = apply_focused_input_contracts(
        candidate,
        _focused_input_contracts_from_events(events),
    )

    rejection_reason = await _evolution_rejection_reason(
        original,
        candidate,
        embedding_provider=embedding_provider,
    )
    if rejection_reason is not None:
        library.record_feedback(
            original.skill_id,
            task=task or failure_case.task,
            status=f"rejected:{rejection_reason}",
        )
        return {
            "status": "evolution_rejected",
            "trace": str(trace_path),
            "reason": rejection_reason,
            "failure_case": asdict(failure_case),
            "updated_skill_id": None,
            "updated_functions": [],
            "compiled_skill_ids": [],
            "token_usage": dict(getattr(response, "usage", {}) or {}),
        }

    if not library.update(original.skill_id, candidate):
        return {
            "status": "missing_skill",
            "trace": str(trace_path),
            "failure_case": asdict(failure_case),
            "updated_skill_id": None,
            "updated_functions": [],
            "compiled_skill_ids": [],
            "token_usage": dict(getattr(response, "usage", {}) or {}),
        }

    library.record_feedback(
        original.skill_id,
        task=task or failure_case.task,
        status="processed_evolution",
        evolved=True,
    )
    return {
        "status": "processed_evolution",
        "trace": str(trace_path),
        "failure_case": asdict(failure_case),
        "updated_skill_id": original.skill_id,
        "updated_functions": [candidate.name],
        "compiled_skill_ids": [original.skill_id],
        "token_usage": dict(getattr(response, "usage", {}) or {}),
    }


def _build_messages(
    *,
    task: str,
    original: Skill,
    failure_case: SkillFailureCase,
    feedback: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompt = _EVOLUTION_PROMPT.format(
        task=task,
        skill_json=json.dumps(original.to_dict(), ensure_ascii=False, indent=2),
        failure_case_json=json.dumps(asdict(failure_case), ensure_ascii=False, indent=2),
        feedback_json=json.dumps(feedback, ensure_ascii=False, indent=2),
        trajectory_json=json.dumps(_compact_events(events), ensure_ascii=False, indent=2),
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, path in enumerate(_screenshot_paths(events), start=1):
        encoded = _encode_image_b64(path)
        if encoded is None:
            continue
        content.append({"type": "text", "text": f"\nScreenshot {index}: {path.name}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }
        )
    return [{"role": "user", "content": content}]


def _parse_skill_response(text: str, *, original: Skill) -> Skill | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        last_fence = cleaned.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            cleaned = cleaned[first_newline + 1 : last_fence].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse skill evolution response as JSON")
        return None
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("steps"), list)
        or not data.get("steps")
    ):
        return None

    platform = str(data.get("platform") or original.platform)
    app = normalize_app_identifier(platform, data.get("app") or original.app)
    steps: list[SkillStep] = []
    for raw_step in data.get("steps") or []:
        if not isinstance(raw_step, dict) or not raw_step.get("action_type"):
            continue
        steps.append(
            SkillStep(
                action_type=str(raw_step["action_type"]),
                target=str(raw_step.get("target") or ""),
                parameters=dict(raw_step.get("parameters") or {}),
                valid_state=raw_step.get("valid_state"),
                state_contract=normalize_state_contract(raw_step.get("state_contract")),
                fixed=bool(raw_step.get("fixed", False)),
                fixed_values=dict(raw_step.get("fixed_values") or {}),
            )
        )
    if not steps:
        return None
    return normalize_skill_app(
        Skill(
            skill_id=original.skill_id,
            name=str(data.get("name") or original.name),
            description=str(data.get("description") or original.description),
            app=app,
            platform=platform,
            steps=tuple(steps),
            parameters=tuple(str(item) for item in (data.get("parameters") or original.parameters)),
            tags=tuple(str(item) for item in (data.get("tags") or original.tags)),
            created_at=original.created_at,
            success_count=original.success_count,
            failure_count=original.failure_count,
        )
    )


def _focused_input_contracts_from_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "step":
            continue
        action = event.get("action") if isinstance(event.get("action"), dict) else {}
        if action.get("action_type") == "input_text":
            interaction_target = event.get("interaction_target")
            contract = (
                interaction_target.get("state_contract")
                if isinstance(interaction_target, dict)
                and isinstance(interaction_target.get("state_contract"), dict)
                else None
            )
            if contract:
                contracts.append(contract)
    return contracts


async def _evolution_rejection_reason(
    original: Skill,
    candidate: Skill,
    *,
    embedding_provider: Any | None,
) -> str | None:
    if candidate.platform != original.platform:
        return "platform_changed"
    if candidate.app != original.app:
        return "app_changed"
    if not candidate.steps:
        return "empty_steps"
    if embedding_provider is None:
        return None
    vectors = await embedding_provider.embed(
        [
            _skill_search_text(original),
            _skill_search_text(candidate),
        ]
    )
    if vectors is None or len(vectors) < 2:
        return None
    similarity = cosine_similarity(vectors[0], vectors[1])
    if similarity < 0.45:
        return "low_embedding_similarity"
    return None


def _build_failure_case(
    events: list[dict[str, Any]],
    *,
    trace_path: Path,
    task: str | None,
    platform: str,
) -> SkillFailureCase | None:
    failed_result_index: int | None = None
    failed_result: dict[str, Any] | None = None
    for index, event in enumerate(events):
        if event.get("type") == "skill_execution_result" and event.get("state") == "failed":
            failed_result_index = index
            failed_result = event
            break
    if failed_result_index is None or failed_result is None:
        return None

    skill_id = str(failed_result.get("skill_id") or "")
    failed_step = _failed_skill_step(events[: failed_result_index + 1], skill_id=skill_id)
    if not skill_id:
        skill_id = str(failed_step.get("skill_id") or "")
    if not skill_id:
        return None
    observation = (
        failed_step.get("observation") if isinstance(failed_step.get("observation"), dict) else None
    )
    return SkillFailureCase(
        trace=str(trace_path),
        task=str(task or ""),
        platform=platform,
        skill_id=skill_id,
        skill_name=str(failed_result.get("skill_name") or failed_step.get("skill_name") or ""),
        failed_step_index=_optional_int(failed_step.get("step_index")),
        failed_target=str(failed_step.get("target") or ""),
        failed_valid_state=failed_step.get("valid_state"),
        failed_state_contract=failed_step.get("state_contract")
        if isinstance(failed_step.get("state_contract"), dict)
        else None,
        failure_observation=observation,
        failure_screenshot_path=failed_step.get("screenshot_path"),
        failure_error=failed_step.get("error"),
        execution_error=failed_result.get("error"),
    )


def _failed_skill_step(events: list[dict[str, Any]], *, skill_id: str) -> dict[str, Any]:
    matching = [
        event
        for event in events
        if event.get("type") == "skill_step"
        and (not skill_id or str(event.get("skill_id") or "") == skill_id)
    ]
    for event in reversed(matching):
        if event.get("error") or event.get("valid_state_check") is False:
            return event
    return matching[-1] if matching else {}


def _append_failure_case(store_root: Path, failure_case: SkillFailureCase) -> None:
    path = Path(store_root).expanduser() / EVOLUTION_DIRNAME / FAILURE_CASES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(failure_case), ensure_ascii=False) + "\n")


def _compact_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        compact = dict(event)
        observation = compact.get("observation")
        if isinstance(observation, dict):
            compact["observation"] = {
                key: observation.get(key)
                for key in ("foreground_app", "app", "platform", "activity_class", "extra")
                if key in observation
            }
        out.append(compact)
    return out


def _screenshot_paths(events: list[dict[str, Any]], limit: int = 8) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for event in events:
        raw = event.get("screenshot_path")
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        paths.append(path)
        if len(paths) >= limit:
            break
    return paths


def _encode_image_b64(path: Path) -> str | None:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
