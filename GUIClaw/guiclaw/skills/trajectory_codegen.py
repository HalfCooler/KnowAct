"""Rule-based trajectory preprocessing for flat skill code extraction."""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

from guiclaw.skills.data import Skill, SkillStep
from guiclaw.skills.normalization import find_android_app_in_text, normalize_app_identifier
from guiclaw.skills.state_contract import infer_focused_input_contract
from guiclaw.skills.static_selector_filter import static_selector_from_node
from guiclaw.trajectory.recorder import load_trajectory_events

_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")
_ACTION_PARAM_KEYS = ("x", "y", "x2", "y2", "text", "key", "pixels", "direction")
_COORDLESS_ACTIONS = frozenset(
    {
        "wait",
        "back",
        "home",
        "enter",
        "app_switch",
        "done",
        "request_intervention",
        "hotkey",
        "screenshot",
    }
)
_STATE_FLAGS = ("visible", "clickable", "enabled", "focused", "scrollable")
_AMBIGUOUS_APP_IDS = frozenset({"", "unknown", "app", "app-package-or-name"})
_TRANSIENT_APP_IDS = frozenset(
    {
        "com.google.android.apps.nexuslauncher",
        "com.android.intentresolver",
        "com.android.systemui",
        "nexuslauncher",
        "intentresolver",
        "launcher",
        "systemui",
    }
)


@dataclass
class CodeStep:
    step_index: int
    app: str
    intent: str
    action_type: str
    action_params: dict
    control_info: str
    contract_json: str
    screenshot_b64: str
    succeeded: bool | None = None  # None=unknown, True=succeeded, False=failed
    suppress_extracted_contract: bool = False


@dataclass
class CodegenResult:
    task: str
    platform: str
    app: str
    app_candidates: tuple[str, ...]
    steps: list[CodeStep]
    screenshots_b64: list[str]


def codegen_trajectory(trace_path: Path, *, subtask_index: int = 1) -> CodegenResult | None:
    events = load_trajectory_events(trace_path, subtask_index=subtask_index)
    if not events:
        return None

    meta = _find_metadata(events)
    task = str(meta.get("task") or trace_path.stem)
    platform = str(meta.get("platform") or _first_platform(events) or "unknown")
    metadata_app = str(meta.get("foreground_app") or meta.get("app") or "")
    app_candidates = _trace_app_candidates(events, platform, task)
    app = _choose_trace_app(metadata_app, platform, app_candidates)

    steps: list[CodeStep] = []
    screenshots_b64: list[str] = []
    seen: set[str] = set()
    previous_observation: dict[str, Any] | None = None
    skill_failed = False  # boundary signal from skill_execution_result

    for event in events:
        event_type = event.get("type")
        if event_type == "skill_execution_result" and event.get("state") == "failed":
            skill_failed = True
            continue
        if event_type not in ("step", "skill_step"):
            continue

        # Normalize action extraction for both event types
        if event_type == "skill_step":
            action = _skill_step_action(event)
            error = event.get("error")
            succeeded = None if error is None else False
            if succeeded is False:
                skill_failed = True
        else:
            action = event.get("action") or {}
            succeeded = None  # agent steps don't encode success

        observation = event.get("observation") or {}
        action_type = str(action.get("action_type") or "")
        action_app = _open_app_action_candidate(action, platform)
        observation_app = _observation_app_candidate(observation, platform)
        if not app:
            app = action_app or observation_app or ""
        step_app = action_app or observation_app or app

        b64 = _screenshot_b64(event, trace_path)
        if b64 and b64 not in seen:
            seen.add(b64)
            screenshots_b64.append(b64)

        control_info = ""
        contract_json = ""
        # For skill_step, prefer serialized state_contract directly
        if event_type == "skill_step" and event.get("state_contract"):
            contract_json = json.dumps(
                event["state_contract"], ensure_ascii=False, separators=(",", ":")
            )
            control_info = _describe_contract_selector(event["state_contract"])
        elif isinstance(event.get("interaction_target"), dict) and isinstance(
            event["interaction_target"].get("state_contract"), dict
        ):
            contract = event["interaction_target"]["state_contract"]
            contract_json = json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
            control_info = _describe_contract_selector(contract)
        elif action_type == "input_text" and previous_observation is not None:
            contract_app = _contract_app_for_observation(
                previous_observation,
                platform,
                fallback=app,
            )
            contract = infer_focused_input_contract(
                (previous_observation.get("extra") or {}),
                app=contract_app,
            )
            if contract:
                contract_json = json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
                control_info = _describe_contract_selector(contract)
        if not contract_json and action_type not in _COORDLESS_ACTIONS:
            target_observation = _target_observation_for_action(
                action_type,
                observation,
                previous_observation,
            )
            using_pre_action_observation = target_observation is previous_observation
            pt = _action_point(action, target_observation)
            suppress_extracted_contract = False
            if pt is not None:
                target_nodes = _ui_tree(target_observation)
                suppress_extracted_contract = (
                    action_type == "tap"
                    and not using_pre_action_observation
                    and _point_hits_focused_text_input(target_nodes, *pt)
                )
                node = _target_node_at(target_nodes, *pt, action_type=action_type)
                if node is not None:
                    contract_app = _contract_app_for_observation(
                        target_observation,
                        platform,
                        fallback=app,
                    )
                    control_info = _describe_control(node)
                    contract_json = _build_contract_json(
                        node,
                        contract_app,
                        allow_class_fallback=action_type != "tap",
                    )
                elif suppress_extracted_contract:
                    control_info = "post-action focused input; omit state_contract"
        else:
            suppress_extracted_contract = False

        steps.append(
            CodeStep(
                step_index=int(event.get("step_index", len(steps))),
                app=step_app,
                intent=_step_intent(event, action_type),
                action_type=action_type,
                action_params=_extract_params(action, action_type),
                control_info=control_info,
                contract_json=contract_json,
                screenshot_b64=b64,
                succeeded=succeeded if succeeded is False else (None if skill_failed else None),
                suppress_extracted_contract=suppress_extracted_contract,
            )
        )
        previous_observation = observation if isinstance(observation, dict) else None

    return CodegenResult(
        task=task,
        platform=platform,
        app=app,
        app_candidates=app_candidates,
        steps=steps,
        screenshots_b64=screenshots_b64,
    )


def codegen_to_extraction_text(result: CodegenResult) -> str:
    lines = [
        f"Task: {result.task}",
        f"App: {result.app}",
        f"Platform: {result.platform}",
    ]
    if result.app_candidates:
        lines.append(f"Observed apps: {', '.join(result.app_candidates)}")
    lines.extend(
        [
            "",
            "Action sequence:",
        ]
    )
    for step in result.steps:
        status = ""
        if step.succeeded is False:
            status = "[FAILED] "
        elif step.succeeded is True:
            status = "[OK] "
        parts = [
            f"  [{step.step_index}] {status}{step.intent}",
            f"type={step.action_type}",
        ]
        params = _format_params(step.action_params)
        if params:
            parts.append(params)
        parts.append(f"control: {step.control_info}")
        parts.append(f"contract: {step.contract_json}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _skill_step_action(event: dict[str, Any]) -> dict[str, Any]:
    """Extract action-like dict from a skill_step event."""
    action = event.get("action") or {}
    if isinstance(action, dict):
        return action
    # Fallback: reconstruct from top-level event fields
    return {
        "action_type": event.get("action_type") or event.get("action_summary", ""),
    }


# ---- internal helpers ----


def _find_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    for e in events:
        if e.get("type") == "metadata":
            return e
    return {}


def _first_platform(events: list[dict[str, Any]]) -> str:
    for e in events:
        obs = e.get("observation") or {}
        p = obs.get("platform")
        if p:
            return str(p)
    return ""


def _trace_app_candidates(
    events: list[dict[str, Any]], platform: str, task: str
) -> tuple[str, ...]:
    candidates: list[str] = []
    task_app = _task_app_candidate(task, platform)
    if task_app:
        candidates.append(task_app)

    for event in events:
        if event.get("type") not in ("step", "skill_step"):
            continue
        action = (
            _skill_step_action(event)
            if event.get("type") == "skill_step"
            else (event.get("action") or {})
        )
        action_app = _open_app_action_candidate(action, platform)
        if action_app:
            candidates.append(action_app)
        observation = event.get("observation") or {}
        observation_app = _observation_app_candidate(observation, platform)
        if observation_app:
            candidates.append(observation_app)

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def _choose_trace_app(metadata_app: str, platform: str, app_candidates: tuple[str, ...]) -> str:
    metadata_candidate = _normalize_trace_app(metadata_app, platform)
    if _is_meaningful_trace_app(metadata_candidate):
        return metadata_candidate
    return app_candidates[0] if app_candidates else ""


def _task_app_candidate(task: str, platform: str) -> str:
    if (platform or "").strip().lower() != "android":
        return ""
    task_text = f"{task} {_split_camel_case(task)}"
    app = find_android_app_in_text(task_text)
    return app if app and _is_meaningful_trace_app(app) else ""


def _open_app_action_candidate(action: dict[str, Any], platform: str) -> str:
    if not isinstance(action, dict):
        return ""
    action_type = str(action.get("action_type") or action.get("action") or "").strip().lower()
    if action_type != "open_app":
        return ""
    raw = action.get("text") or action.get("app_name") or action.get("package")
    candidate = _normalize_trace_app(str(raw or ""), platform)
    return candidate if _is_meaningful_trace_app(candidate) else ""


def _observation_app_candidate(observation: dict[str, Any] | None, platform: str) -> str:
    if not isinstance(observation, dict):
        return ""
    raw = str(observation.get("foreground_app") or observation.get("app") or "")
    candidate = _normalize_trace_app(raw, platform)
    return candidate if _is_meaningful_trace_app(candidate) else ""


def _contract_app_for_observation(
    observation: dict[str, Any] | None,
    platform: str,
    *,
    fallback: str,
) -> str:
    candidate = _observation_app_candidate(observation, platform)
    if candidate:
        return candidate
    fallback_candidate = _normalize_trace_app(fallback, platform)
    return fallback_candidate if fallback_candidate else fallback


def _normalize_trace_app(
    app: str,
    platform: str,
) -> str:
    return normalize_app_identifier(platform, app)


def _is_meaningful_trace_app(app: str) -> bool:
    key = (app or "").strip().lower()
    return bool(key) and key not in _AMBIGUOUS_APP_IDS and key not in _TRANSIENT_APP_IDS


def _split_camel_case(text: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text or "")


def _step_intent(event: dict[str, Any], action_type: str) -> str:
    if event.get("action_intent"):
        return str(event["action_intent"])
    model_output = event.get("model_output")
    if isinstance(model_output, dict):
        model_output = model_output.get("content") or model_output.get("reasoning_content")
    if model_output:
        return str(model_output)[:200]
    if event.get("action_summary"):
        return str(event["action_summary"])
    return action_type


def _extract_params(action: dict[str, Any], action_type: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in _ACTION_PARAM_KEYS:
        if key == "direction":
            value = action.get("direction") or (
                action.get("text") if action_type == "scroll" else None
            )
        else:
            value = action.get(key)
        if value is not None:
            params[key] = value
    return params


def _format_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={_format_value(value)}" for key, value in params.items())


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _action_point(action: dict[str, Any], observation: dict[str, Any]) -> tuple[int, int] | None:
    if action.get("x") is None or action.get("y") is None:
        return None
    x = float(action["x"])
    y = float(action["y"])
    if action.get("relative"):
        w = int(observation.get("screen_width") or 0)
        h = int(observation.get("screen_height") or 0)
        if w and h:
            x = x / 999 * (w - 1)
            y = y / 999 * (h - 1)
    return _scale_point_to_ui_tree(round(x), round(y), observation)


def _target_observation_for_action(
    action_type: str,
    observation: dict[str, Any],
    previous_observation: dict[str, Any] | None,
) -> dict[str, Any]:
    if (
        action_type != "input_text"
        and previous_observation is not None
        and _ui_tree(previous_observation)
        and _same_foreground_app(previous_observation, observation)
    ):
        return previous_observation
    return observation


def _same_foreground_app(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_app = str(left.get("foreground_app") or left.get("app") or "")
    right_app = str(right.get("foreground_app") or right.get("app") or "")
    return bool(left_app and right_app and left_app == right_app)


def _scale_point_to_ui_tree(x: int, y: int, observation: dict[str, Any]) -> tuple[int, int]:
    screen_width = int(observation.get("screen_width") or 0)
    screen_height = int(observation.get("screen_height") or 0)
    bounds_width, bounds_height = _ui_tree_extent(_ui_tree(observation))
    if not screen_width or not screen_height or not bounds_width or not bounds_height:
        return x, y
    if x > screen_width or y > screen_height:
        return x, y
    if _similar_extent(screen_width, bounds_width) and _similar_extent(
        screen_height, bounds_height
    ):
        return x, y
    return round(x * bounds_width / screen_width), round(y * bounds_height / screen_height)


def _similar_extent(left: int, right: int) -> bool:
    return abs(left - right) <= max(left, right) * 0.1


def _ui_tree_extent(nodes: list[dict[str, Any]]) -> tuple[int, int]:
    max_x = 0
    max_y = 0
    for node in nodes:
        bounds = _parse_bounds(node)
        if bounds is None:
            continue
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])
    return max_x, max_y


def _ui_tree(observation: dict[str, Any]) -> list[dict[str, Any]]:
    extra = observation.get("extra") or {}
    return list(extra.get("ui_tree") or [])


def _target_node_at(
    nodes: list[dict[str, Any]],
    x: int,
    y: int,
    *,
    action_type: str,
) -> dict[str, Any] | None:
    matches = _nodes_at(nodes, x, y)
    if not matches:
        return None
    if action_type == "tap" and any(_is_focused_text_input(node) for _, node in matches):
        return None
    if action_type == "tap":
        for _, node in matches:
            if static_selector_from_node(node):
                return node
        return None
    return matches[0][1]


def _point_hits_focused_text_input(nodes: list[dict[str, Any]], x: int, y: int) -> bool:
    return any(_is_focused_text_input(node) for _, node in _nodes_at(nodes, x, y))


def _nodes_at(nodes: list[dict[str, Any]], x: int, y: int) -> list[tuple[int, dict[str, Any]]]:
    matches: list[tuple[int, dict[str, Any]]] = []
    for node in nodes:
        bounds = _parse_bounds(node)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        if x1 <= x <= x2 and y1 <= y <= y2:
            matches.append(((x2 - x1) * (y2 - y1), node))
    return sorted(matches, key=lambda item: item[0])


def _is_focused_text_input(node: dict[str, Any]) -> bool:
    class_name = str(node.get("class") or "")
    focused = node.get("focused") is True or str(node.get("focused")).lower() == "true"
    return focused and class_name == "android.widget.EditText"


def _parse_bounds(node: dict[str, Any]) -> tuple[int, int, int, int] | None:
    m = _BOUNDS_RE.search(str(node.get("bounds") or ""))
    if not m:
        return None
    return tuple(int(v) for v in m.groups())


def _describe_control(node: dict[str, Any]) -> str:
    parts = []
    for key in ("resource_id", "content_desc", "class"):
        val = node.get(key)
        if val:
            parts.append(f"{key}={val}")
    return " ".join(parts)


def _describe_contract_selector(contract: dict[str, Any]) -> str:
    required = (contract.get("signature") or {}).get("required") or []
    if not required:
        return ""
    selector = required[0].get("selector") or {}
    parts = []
    for key in ("resource_id", "content_desc", "class"):
        val = selector.get(key)
        if val:
            parts.append(f"{key}={val}")
    return " ".join(parts)


def _build_contract_json(
    node: dict[str, Any], app: str, *, allow_class_fallback: bool = True
) -> str:
    selector = static_selector_from_node(node) or _fallback_selector(
        node,
        allow_class=allow_class_fallback,
    )
    if not selector:
        return ""
    selector = dict(selector)
    state = _extract_state(node)
    for flag in _STATE_FLAGS:
        if selector.pop(flag, None) and flag not in state:
            state.append(flag)
    contract = {
        "anchor": {"app_package": app},
        "signature": {
            "required": [{"selector": selector, "state": state}],
            "forbidden": [],
        },
    }
    return json.dumps(contract, ensure_ascii=False, separators=(",", ":"))


def _fallback_selector(node: dict[str, Any], *, allow_class: bool = True) -> dict[str, str] | None:
    if node.get("resource_id"):
        return {"resource_id": str(node["resource_id"])}
    if allow_class and node.get("class"):
        return {"class": str(node["class"])}
    return None


def _extract_state(node: dict[str, Any]) -> list[str]:
    state = ["visible"]
    for flag in _STATE_FLAGS[1:]:
        val = node.get(flag)
        if val is True or str(val).lower() == "true":
            state.append(flag)
    return state


def _screenshot_b64(event: dict[str, Any], trace_path: Path) -> str:
    obs = event.get("observation") or {}
    raw = event.get("screenshot_path") or obs.get("screenshot_path")
    if not raw:
        return ""
    path = Path(str(raw))
    path = path if path.is_absolute() else trace_path.parent / path
    if not path.is_file():
        return ""
    return _scale_png(path)


def _scale_png(path: Path) -> str:
    with Image.open(path) as img:
        w, h = img.size
        max_edge = max(w, h)
        if max_edge > 1000:
            scale = 1000 / max_edge
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def apply_focused_input_contracts(skill: Skill, contracts: list[dict[str, Any]]) -> Skill:
    if not contracts:
        return skill
    changed = False
    next_contract = 0
    steps: list[SkillStep] = []
    for step in skill.steps:
        if step.action_type != "input_text" or next_contract >= len(contracts):
            steps.append(step)
            continue
        contract = contracts[next_contract]
        next_contract += 1
        if contract_requires_focused(step.state_contract):
            steps.append(step)
            continue
        changed = True
        steps.append(replace(step, state_contract=contract))
    return replace(skill, steps=tuple(steps)) if changed else skill


def contract_requires_focused(contract: dict[str, Any] | None) -> bool:
    if not isinstance(contract, dict):
        return False
    for item in (contract.get("signature") or {}).get("required") or []:
        if "focused" in (item.get("state") or []):
            return True
    return False


def _load_contract(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def apply_focused_contracts_from_codegen(skill: Skill, result: CodegenResult) -> Skill:
    contracts: list[dict[str, Any]] = []
    for step in result.steps:
        if step.action_type != "input_text" or not step.contract_json:
            continue
        contract = _load_contract(step.contract_json)
        if contract and _contract_allowed_for_skill(skill, result, step, contract):
            contracts.append(contract)
    return apply_focused_input_contracts(skill, contracts) if contracts else skill


def apply_state_contracts_from_codegen(skill: Skill, result: CodegenResult) -> Skill:
    """Make codegen-derived contracts authoritative for extracted skill steps."""
    aligned = _align_skill_steps_to_codegen(skill, result)
    if not aligned:
        return skill
    changed = False
    steps: list[SkillStep] = []
    for step, code_step in aligned:
        contract = None
        if code_step is not None and not code_step.suppress_extracted_contract:
            contract = _load_contract(code_step.contract_json) if code_step.contract_json else None
            if contract and not _contract_allowed_for_skill(skill, result, code_step, contract):
                contract = None
        if step.state_contract != contract:
            changed = True
            steps.append(replace(step, state_contract=contract))
        else:
            steps.append(step)
    return replace(skill, steps=tuple(steps)) if changed else skill


def apply_contract_constraints_from_codegen(skill: Skill, result: CodegenResult) -> Skill:
    aligned = _align_skill_steps_to_codegen(skill, result)
    if not aligned:
        return skill
    changed = False
    steps: list[SkillStep] = []
    for step, code_step in aligned:
        if (
            code_step is not None
            and code_step.suppress_extracted_contract
            and step.state_contract is not None
        ):
            changed = True
            steps.append(replace(step, state_contract=None))
        else:
            steps.append(step)
    return replace(skill, steps=tuple(steps)) if changed else skill


def _align_skill_steps_to_codegen(
    skill: Skill,
    result: CodegenResult,
) -> list[tuple[SkillStep, CodeStep | None]]:
    code_steps = list(result.steps)

    aligned: list[tuple[SkillStep, CodeStep | None]] = []
    cursor = 0
    for step in skill.steps:
        if step.action_type == "open_app":
            while cursor < len(code_steps) and code_steps[cursor].action_type == "open_app":
                cursor += 1
            aligned.append((step, None))
            continue
        match: CodeStep | None = None
        if step.state_contract is not None:
            match, cursor = _next_matching_code_step(
                step,
                code_steps,
                cursor,
                result,
                skill,
                require_contract=True,
            )
        if match is None:
            match, cursor = _next_matching_code_step(
                step,
                code_steps,
                cursor,
                result,
                skill,
                require_contract=False,
            )
        aligned.append((step, match))
    return aligned


def _next_matching_code_step(
    step: SkillStep,
    code_steps: list[CodeStep],
    cursor: int,
    result: CodegenResult,
    skill: Skill,
    *,
    require_contract: bool,
) -> tuple[CodeStep | None, int]:
    for index in range(cursor, len(code_steps)):
        candidate = code_steps[index]
        if require_contract and not candidate.contract_json:
            continue
        if _code_step_matches_skill_step(step, candidate, result, skill):
            return candidate, index + 1
    return None, cursor


def _code_step_matches_skill_step(
    step: SkillStep,
    candidate: CodeStep,
    result: CodegenResult,
    skill: Skill,
) -> bool:
    if candidate.action_type != step.action_type:
        return False

    skill_app = normalize_app_identifier(skill.platform, skill.app)
    candidate_app = _normalize_trace_app(candidate.app or result.app, result.platform)
    if (
        _is_meaningful_trace_app(skill_app)
        and _is_meaningful_trace_app(candidate_app)
        and skill_app != candidate_app
    ):
        return False

    if not step.fixed:
        return True

    coord_keys = ("x", "y", "x2", "y2")
    fixed_coords = [key for key in coord_keys if key in step.fixed_values]
    if fixed_coords:
        return all(
            key in candidate.action_params
            and _numeric_values_close(step.fixed_values[key], candidate.action_params[key])
            for key in fixed_coords
        )

    fixed_text = step.fixed_values.get("text")
    candidate_text = candidate.action_params.get("text")
    if isinstance(fixed_text, str) and isinstance(candidate_text, str):
        return fixed_text.strip() == candidate_text.strip()

    return True


def _numeric_values_close(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 2.0
    except (TypeError, ValueError):
        return False


def _contract_allowed_for_skill(
    skill: Skill,
    result: CodegenResult,
    code_step: CodeStep,
    contract: dict[str, Any],
) -> bool:
    anchor = contract.get("anchor") if isinstance(contract.get("anchor"), dict) else {}
    anchor_app_raw = str(anchor.get("app_package") or "")
    if not anchor_app_raw:
        return True

    anchor_app = _normalize_trace_app(anchor_app_raw, result.platform)
    code_app = _normalize_trace_app(code_step.app or result.app, result.platform)
    skill_app = normalize_app_identifier(skill.platform, skill.app)

    if _is_meaningful_trace_app(code_app) and anchor_app != code_app:
        return False
    if _is_meaningful_trace_app(skill_app) and anchor_app != skill_app:
        return False
    return True
