"""LLM extraction of flat Python GUI skills from code-formatted trajectories."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from guiclaw.action import VALID_ACTION_TYPES, ActionError, normalize_action_type, parse_action
from guiclaw.interfaces import LLMProvider
from guiclaw.skills.data import Skill, collect_placeholder_names
from guiclaw.skills.flat import CODE_HEADER, compile_flat_skills
from guiclaw.skills.normalization import is_unknown_app_identifier, normalize_app_identifier
from guiclaw.skills.trajectory_codegen import (
    CodegenResult,
    CodeStep,
    apply_contract_constraints_from_codegen,
    apply_focused_contracts_from_codegen,
    apply_state_contracts_from_codegen,
    codegen_to_extraction_text,
    codegen_trajectory,
)

logger = logging.getLogger(__name__)

_TRANSIENT_APP_IDS = frozenset({
    "com.google.android.apps.nexuslauncher",
    "com.android.intentresolver",
    "com.android.systemui",
    "nexuslauncher",
    "intentresolver",
    "launcher",
    "systemui",
})
_NO_VERIFY_VALID_STATE = "No need to verify"

_EXTRACT_PROMPT = """\
Return executable Python immediately. Do not analyze, explain, or use markdown fences.
The first line must be the import header shown below.

Extract one reusable GUI skill from this trajectory.

Target format:
{code_header}

@skill(app="{app}", platform="{platform}", name="short_name", description="One sentence summary")
async def skill_name(device, param1, param2):
    await action("open_app", target="{app}", fixed=True, fixed_values={{"text": "{app}"}},
                 valid_state="No need to verify")
    await action("tap", target="search button", fixed=True,
                 fixed_values={{"x": 540, "y": 960}},
                 valid_state="target is visible and clickable",
                 state_contract=C(app="...[app_package]...", required=[R(resource_id="target_id", visible=True)]))
    await action("input_text", target="{{{{param1}}}}",
                 valid_state="input field is focused")

Rules:
- Extract ONE cohesive skill that covers the core action sequence. Do NOT split into multiple tiny @skill functions.
- fixed=true + fixed_values: static UI (nav bars, toolbar, system actions, open_app). Copy exact x/y/text from the trajectory step params shown after "|".
- Use only these action types: tap, long_press, double_tap, drag, swipe, scroll, input_text, hotkey, screenshot, wait, open_app, open_deeplink, open_intent, close_app, back, home, enter, app_switch, done, request_intervention. Do not invent actions such as read_text, press_key, navigate_back, or navigate_to_folder.
- The skill function body must contain only await action(...) statements. Do not add if/for/while/try blocks, assignments, calculations, helper calls, return values, or non-action awaits.
- Do not use f-strings, string concatenation, arithmetic expressions, comparisons, comprehensions, dict/list expressions with computed values, or nested function calls in action arguments. Use literal strings or {{param}} placeholders only.
- fixed_values may contain only executable action fields such as x, y, x2, y2, text, key, pixels, direction, duration_ms, component, package, intent_action, mime_type, categories, extras, relative, status, auto_enter. Never put selectors such as resource_id, content_desc, class, or class_name in fixed_values.
- fixed=false: dynamic content (search results, variable input). Use {{{{param}}}} placeholders, omit fixed_values.
- target: Every required interactive step must have a natural-language target and valid_state. Use a concise natural-language grounding hint, e.g. "search button", "search input field", "matching video result", "skip ad button". Do not use raw class/resource_id as target unless it is also visible user-facing text.
- Collapse all app-launch steps into ONE open_app as the first step. For open_app, prefer the trajectory app package for both target and fixed_values.text when available, and always use valid_state="No need to verify".
- valid_state: Every required interactive step must have a specific present-tense valid_state, e.g. "search field is visible and enabled". For input_text, use "input field is focused" if no better state is available. If a required step has no verifiable state, remove or regenerate that step instead of leaving valid_state empty.
- state_contract: do not invent selectors. Copy only the exact contract provided by trajectory/codegen for the matched step; omit if no contract is provided. If an exact contract contains task-specific dynamic text such as a person, query, or result name, omit state_contract from the generated code instead of debating or generalizing it. The extractor postprocess will align contracts from codegen.
- R(...) supports resource_id, text, content_desc, class_, xpath, visible, clickable, enabled, focused, and scrollable only. Do not use class_name.
- Drop duplicate/redundant clicks, exploratory taps, and pointless scrolls.
- Transient popups (ads, permissions, consent): keep as optional=True step. Executor skips them when absent.
- Keep transient blockers and benign app confirmations such as skip/close/save/done as guarded optional=True steps when they appear. Omit destructive or externally visible confirmations such as pay, delete, send, submit order, publish, or irreversible consent unless the original user task explicitly requires them.
- description: MUST be generic and reusable. Mention app name, capability, and broad feature-level route only. Use parameter roles like query, media item, contact, or item. NEVER include literal values/entities, exact titles/names, or narrow qualifiers such as specific, official, first result, or top result. Avoid tap-by-tap UI actions.
{failure_note}
## Trajectory
{code_text}

Output ONLY the Python code. No markdown fences, no JSON object, no explanation.
"""

_FAILURE_NOTE = (
    "FAILURE trajectory: keep the reusable succeeded prefix. If the failure screen clearly "
    "shows one safe corrective next action, append at most one non-fixed corrective step with "
    "natural-language target and valid_state. Do not invent coordinates or state_contract for "
    "that corrective step. Use optional=True only for transient blockers/popups, and never add "
    "pay/delete/send/submit/publish/irreversible confirmation actions."
)

_RETRY_QUALITY_NOTE = """\
The previous extracted skill has quality issues:
{issues}

Regenerate the entire skill. Every required interactive step must have a natural-language target and valid_state. Use only supported action types, declare every placeholder as a function parameter, keep fixed_values executable, and do not invent state_contract selectors.
"""

_RETRY_COMPILE_NOTE = """\
The previous output did not compile as a flat GUI skill:
{issues}

Regenerate the entire skill as valid Python using the flat skill DSL only. Requirements:
- Output only import/header plus @skill functions.
- Each skill function body must contain only await action(...) statements.
- Do not use f-strings, arithmetic, assignments, conditionals, loops, helper functions, or non-action awaits.
- Use literal strings or {{param}} placeholders in action arguments.
- Use only supported action types and supported R(...) fields.
"""

_VALID_STATE_REQUIRED_ACTIONS = frozenset({
    "tap", "long_press", "double_tap", "input_text", "scroll", "swipe", "drag",
})
_TARGET_REQUIRED_ACTIONS = frozenset({"tap", "long_press", "double_tap", "input_text"})
_FIXED_VALUE_KEYS = frozenset({
    "x", "y", "x2", "y2", "text", "key", "pixels", "direction", "duration_ms",
    "component", "package", "intent_action", "mime_type", "categories", "extras",
    "relative", "status", "auto_enter",
})
_FIXED_SELECTOR_KEYS = frozenset({"resource_id", "content_desc", "class", "class_name", "xpath"})
_SEGMENT_NOISE_ACTIONS = frozenset({
    "wait", "screenshot", "home", "back", "enter", "done", "app_switch",
    "request_intervention",
})
_SEGMENT_SCROLL_ACTIONS = frozenset({"scroll", "swipe", "drag"})


class SkillExtractor:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._total_usage: dict[str, int] = {}
        self._last_diagnostics: list[dict[str, Any]] = []

    @property
    def total_usage(self) -> dict[str, int]:
        return dict(self._total_usage)

    @property
    def last_diagnostics(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._last_diagnostics]

    # -- public API --

    async def extract_from_file(
        self,
        trajectory_path: Path,
        *,
        is_success: bool = True,
        subtask_index: int = 1,
    ) -> Skill | None:
        skills = await self.extract_from_file_multi(
            trajectory_path,
            is_success=is_success,
            subtask_index=subtask_index,
        )
        return skills[0] if skills else None

    async def extract_from_file_multi(
        self,
        trajectory_path: Path,
        *,
        is_success: bool = True,
        subtask_index: int = 1,
    ) -> list[Skill]:
        self._last_diagnostics = []
        result = codegen_trajectory(trajectory_path, subtask_index=subtask_index)
        if result is None or not result.steps:
            return []
        single_app = _single_foreground_app_package(result)
        if single_app:
            return []
        raw_segments = _split_codegen_result_by_app(result)
        segments, _ = _filter_codegen_segments_for_extraction(raw_segments)
        if not segments:
            return []

        skills: list[Skill] = []
        for segment in segments:
            extracted = await self._extract_all(segment, is_success)
            skills.extend(extracted)
        if not skills:
            return []
        return skills

    async def extract_from_codegen_result_multi(
        self,
        result: CodegenResult,
        *,
        is_success: bool = True,
    ) -> list[Skill]:
        """Extract skills from an already-computed :class:`CodegenResult`.

        This accepts precomputed codegen data and does not apply the single-foreground-app guard
        (single-app success traces are exactly what compact-skill induction feeds
        in). App selection, codegen evidence, the prompt/quality loop, and contract
        alignment are all reused unchanged.

        The per-app segment split and its quality filter (launch-only / single-core
        / scroll-only rejection) are kept identical to the online extractor: a
        result whose segments are all rejected yields no skill, by design.
        """
        self._last_diagnostics = []
        if result is None or not result.steps:
            return []
        raw_segments = _split_codegen_result_by_app(result)
        segments, _skipped = _filter_codegen_segments_for_extraction(raw_segments)
        skills: list[Skill] = []
        for segment in segments:
            skills.extend(await self._extract_all(segment, is_success))
        return skills

    # -- internal --

    async def _extract_all(self, result: CodegenResult, is_success: bool) -> list[Skill]:
        code_text = codegen_to_extraction_text(result)
        prompt = _EXTRACT_PROMPT.format(
            code_header=CODE_HEADER,
            app=result.app,
            platform=result.platform,
            failure_note=_FAILURE_NOTE if not is_success else "",
            code_text=code_text,
        )
        response = await self._llm.chat(
            _build_messages(prompt, result.screenshots_b64),
            max_tokens=8192,
        )
        self._accumulate_usage(response.usage)
        compile_start = len(self._last_diagnostics)
        skills = _postprocess_skills(self._compile_all(response.content, result), result)
        compile_errors = _compile_errors_since(self._last_diagnostics, compile_start)
        if compile_errors:
            self._last_diagnostics.append({
                "phase": "compile_retry",
                "errors": list(compile_errors),
            })
            retry_prompt = f"{prompt}\n\n{_RETRY_COMPILE_NOTE.format(issues=_format_quality_issues(compile_errors))}"
            response = await self._llm.chat(
                _build_messages(retry_prompt, result.screenshots_b64),
                max_tokens=8192,
            )
            self._accumulate_usage(response.usage)
            compile_start = len(self._last_diagnostics)
            skills = _postprocess_skills(self._compile_all(response.content, result), result)
            compile_errors = _compile_errors_since(self._last_diagnostics, compile_start)
            if compile_errors:
                self._last_diagnostics.append({
                    "phase": "compile_rejected",
                    "errors": list(compile_errors),
                })
                return []
        issues = _skill_quality_issues(skills)
        if issues:
            self._last_diagnostics.append({"phase": "quality_retry", "issues": list(issues)})
            retry_prompt = f"{prompt}\n\n{_RETRY_QUALITY_NOTE.format(issues=_format_quality_issues(issues))}"
            response = await self._llm.chat(
                _build_messages(retry_prompt, result.screenshots_b64),
                max_tokens=8192,
            )
            self._accumulate_usage(response.usage)
            skills = _postprocess_skills(self._compile_all(response.content, result), result)
            issues = _skill_quality_issues(skills)
            if issues:
                self._last_diagnostics.append({"phase": "quality_rejected", "issues": list(issues)})
                logger.warning(
                    "Reject extracted skills after retry due to quality issues: %s",
                    "; ".join(issues),
                )
                return [
                    skill
                    for skill in skills
                    if not _skill_quality_issues([skill])
                ]
        return skills

    def _compile_all(self, text: str, result: CodegenResult) -> list[Skill]:
        source = _clean_code_block(text)
        compiled = compile_flat_skills(source)
        if compiled.errors:
            logger.warning("Flat skill compile failed: %s", compiled.errors)
            self._last_diagnostics.append({
                "phase": "compile",
                "errors": list(compiled.errors),
                "source": source,
            })
            return []
        skills: list[Skill] = []
        for skill in compiled.skills:
            normalized = replace(
                skill,
                app=normalize_app_identifier(skill.platform, skill.app),
                description=_generalize_skill_description(skill.description),
                steps=_normalize_extracted_steps(skill.steps),
            )
            resolved = _resolve_skill_app(normalized, result)
            if resolved is None:
                logger.warning("Reject extracted skill %s: app is unknown", skill.name)
                self._last_diagnostics.append({
                    "phase": "app_resolution",
                    "skill": skill.name,
                    "trace_app": result.app,
                    "app_candidates": list(getattr(result, "app_candidates", ())),
                    "skill_app": skill.app,
                    "open_app": _first_open_app_text(skill),
                })
                continue
            resolved = replace(
                resolved,
                steps=_normalize_extracted_steps(resolved.steps, resolved_app=resolved.app),
            )
            skills.append(resolved)
        return skills

    def _accumulate_usage(self, usage: dict[str, int]) -> None:
        for key, value in usage.items():
            self._total_usage[key] = self._total_usage.get(key, 0) + int(value)


# -- helpers --

def _build_messages(prompt: str, screenshots_b64: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64 in screenshots_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return [{"role": "user", "content": content}]


def _clean_code_block(text: str) -> str:
    t = text.strip()
    fenced_blocks: list[str] = []
    block: list[str] | None = None
    for line in t.splitlines():
        if line.strip().startswith("```"):
            if block is None:
                block = []
            else:
                fenced_blocks.append("\n".join(block).strip())
                block = None
            continue
        if block is not None:
            block.append(line)
    for candidate in fenced_blocks:
        if "@skill(" in candidate or "from guiclaw.skills.flat import" in candidate:
            t = candidate
            break
    if not t.startswith("from guiclaw"):
        t = f"{CODE_HEADER}\n\n{t}"
    return t.rstrip() + "\n"


def _generalize_skill_description(description: str) -> str:
    text = description.strip()
    replacements = (
        (r"\bspecific\s+", ""),
        (r"\bofficial\s+", ""),
        (r"\b(first|top)\s+search\s+result\b", "matching search result"),
        (r"\b(first|top)\s+result\b", "matching result"),
        (r"\ba\s+(music\s+)?video\s+query\b", "a video by query"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


def _postprocess_skills(skills: list[Skill], result: CodegenResult) -> list[Skill]:
    return [
        _ensure_valid_states(
            apply_contract_constraints_from_codegen(
                apply_state_contracts_from_codegen(
                    apply_focused_contracts_from_codegen(skill, result),
                    result,
                ),
                result,
            )
        )
        for skill in skills
    ]


def _ensure_valid_states(skill: Skill) -> Skill:
    steps = []
    changed = False
    for step in skill.steps:
        if step.valid_state and step.valid_state.strip():
            steps.append(step)
            continue
        valid_state = _default_valid_state(step)
        if valid_state is None:
            steps.append(step)
            continue
        changed = True
        steps.append(replace(step, valid_state=valid_state))
    return replace(skill, steps=tuple(steps)) if changed else skill


def _default_valid_state(step: Any) -> str | None:
    action_type = step.action_type
    target = (step.target or "").strip()
    if action_type == "open_app":
        return _NO_VERIFY_VALID_STATE
    if action_type == "input_text":
        return "input field is focused"
    if action_type in {"tap", "long_press", "double_tap"}:
        return f"{target} is visible and enabled" if target else None
    if action_type in {"scroll", "swipe", "drag"}:
        return f"{target} is ready for scrolling" if target else "screen is ready for scrolling"
    return None


def _skill_quality_issues(skills: list[Skill]) -> list[str]:
    issues: list[str] = []
    for skill in skills:
        if not _is_reusable_app(skill.app):
            issues.append(f"{skill.name} app {skill.app!r} is unknown, ambiguous, or transient")
        declared_params = set(skill.parameters)
        for index, step in enumerate(skill.steps):
            action_type = normalize_action_type(step.action_type)
            if action_type not in VALID_ACTION_TYPES:
                issues.append(f"{skill.name} step {index} has unsupported action type {step.action_type!r}")
                continue
            if action_type in _TARGET_REQUIRED_ACTIONS and not (step.target or "").strip():
                issues.append(f"{skill.name} step {index} {action_type} is missing target")
            if action_type in _VALID_STATE_REQUIRED_ACTIONS and not (step.valid_state or "").strip():
                issues.append(f"{skill.name} step {index} {action_type} is missing valid_state")
            placeholders = collect_placeholder_names({
                "target": step.target,
                "parameters": step.parameters,
                "valid_state": step.valid_state,
                "state_contract": step.state_contract,
                "fixed_values": step.fixed_values,
            })
            missing_params = sorted(placeholders - declared_params)
            if missing_params:
                issues.append(
                    f"{skill.name} step {index} uses undeclared placeholders: "
                    f"{', '.join(missing_params)}"
                )
            if step.fixed:
                fixed_keys = set(step.fixed_values)
                selector_keys = sorted(fixed_keys & _FIXED_SELECTOR_KEYS)
                if selector_keys:
                    issues.append(
                        f"{skill.name} step {index} fixed_values contains selector fields: "
                        f"{', '.join(selector_keys)}"
                    )
                unsupported_keys = sorted(fixed_keys - _FIXED_VALUE_KEYS - _FIXED_SELECTOR_KEYS)
                if unsupported_keys:
                    issues.append(
                        f"{skill.name} step {index} fixed_values has unsupported fields: "
                        f"{', '.join(unsupported_keys)}"
                    )
                if not selector_keys and not unsupported_keys:
                    payload = {"action_type": action_type, **step.fixed_values}
                    try:
                        parse_action(payload)
                    except (ActionError, TypeError, ValueError) as exc:
                        issues.append(
                            f"{skill.name} step {index} fixed {action_type} is invalid: {exc}"
                        )
    return issues


def _format_quality_issues(issues: list[str]) -> str:
    return "\n".join(f"- {issue}" for issue in issues)


def _compile_errors_since(diagnostics: list[dict[str, Any]], start: int) -> list[str]:
    errors: list[str] = []
    for item in diagnostics[start:]:
        if item.get("phase") != "compile":
            continue
        for error in item.get("errors") or ():
            errors.append(str(error))
    return errors


def _resolve_skill_app(skill: Skill, result: CodegenResult) -> Skill | None:
    trace_app = normalize_app_identifier(skill.platform, result.app)
    observed_apps = {
        normalize_app_identifier(skill.platform, app)
        for app in getattr(result, "app_candidates", ())
    }
    observed_apps = {app for app in observed_apps if _is_reusable_app(app)}
    if _is_reusable_app(skill.app) and (not observed_apps or skill.app in observed_apps):
        return skill

    open_app = _first_open_app_text(skill)
    normalized_open_app = normalize_app_identifier(skill.platform, open_app) if open_app else "unknown"
    if _is_reusable_app(normalized_open_app) and (
        not observed_apps or normalized_open_app in observed_apps
    ):
        return replace(skill, app=normalized_open_app)

    if _is_reusable_app(trace_app):
        return replace(skill, app=trace_app)

    if _is_reusable_app(skill.app):
        return skill

    if _is_reusable_app(normalized_open_app):
        return replace(skill, app=normalized_open_app)

    return None


def _normalize_extracted_steps(
    steps: tuple[Any, ...],
    *,
    resolved_app: str = "",
) -> tuple[Any, ...]:
    normalized = []
    for index, step in enumerate(steps):
        updated = step
        if index == 0 and updated.action_type == "open_app":
            updated = replace(updated, valid_state=_NO_VERIFY_VALID_STATE)
            if _is_reusable_app(resolved_app):
                fixed_values = dict(updated.fixed_values)
                fixed_values["text"] = resolved_app
                parameters = dict(updated.parameters)
                if "text" in parameters:
                    parameters["text"] = resolved_app
                updated = replace(
                    updated,
                    target=resolved_app,
                    parameters=parameters,
                    fixed=True,
                    fixed_values=fixed_values,
                )
        normalized.append(updated)
    return tuple(normalized)


def _first_open_app_text(skill: Skill) -> str:
    for step in skill.steps:
        if step.action_type != "open_app":
            continue
        for source in (step.fixed_values, step.parameters):
            value = source.get("text") if isinstance(source, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        if step.target.strip():
            return step.target.strip()
    return ""


def _is_unknown_app(app: str) -> bool:
    return is_unknown_app_identifier(app)


def _is_transient_app(app: str) -> bool:
    return (app or "").strip().lower() in _TRANSIENT_APP_IDS


def _is_reusable_app(app: str) -> bool:
    return not _is_unknown_app(app) and not _is_transient_app(app)


def _split_codegen_result_by_app(result: CodegenResult) -> list[CodegenResult]:
    """Split a trajectory into contiguous reusable foreground-app segments."""
    segments: list[CodegenResult] = []
    current_steps: list[CodeStep] = []
    current_app = ""
    pending_prefix: list[CodeStep] = []

    def flush_current() -> None:
        nonlocal current_steps, current_app
        if not current_steps or not _is_reusable_app(current_app):
            current_steps = []
            current_app = ""
            return
        step_apps = [
            _normalized_step_app(step, result)
            for step in current_steps
            if _is_reusable_app(_normalized_step_app(step, result))
        ]
        app_candidates = _dedupe([current_app, *step_apps])
        segments.append(replace(
            result,
            app=current_app,
            app_candidates=tuple(app_candidates),
            steps=list(current_steps),
            screenshots_b64=_segment_screenshots(current_steps),
        ))
        current_steps = []
        current_app = ""

    for step in result.steps:
        step_app = _normalized_step_app(step, result)
        if not _is_reusable_app(step_app):
            if current_steps:
                current_steps.append(step)
            else:
                pending_prefix.append(step)
            continue

        if not current_steps:
            current_app = step_app
            current_steps = [*pending_prefix, step]
            pending_prefix = []
            continue

        if step_app == current_app:
            current_steps.append(step)
            continue

        flush_current()
        current_app = step_app
        current_steps = [*pending_prefix, step]
        pending_prefix = []

    flush_current()
    if segments:
        return segments
    if _is_reusable_app(normalize_app_identifier(result.platform, result.app)):
        return [result]
    return []


def _filter_codegen_segments_for_extraction(
    segments: list[CodegenResult],
) -> tuple[list[CodegenResult], list[dict[str, Any]]]:
    kept: list[CodegenResult] = []
    skipped: list[dict[str, Any]] = []
    for segment in segments:
        reason = _segment_skip_reason(segment)
        if reason:
            skipped.append({
                "app": segment.app,
                "reason": reason,
                "step_indices": [step.step_index for step in segment.steps],
            })
            continue
        kept.append(segment)
    return kept, skipped


def _segment_skip_reason(segment: CodegenResult) -> str:
    core_steps = [
        step for step in segment.steps
        if step.action_type not in _SEGMENT_NOISE_ACTIONS
    ]
    if not core_steps:
        return "no_core_steps"

    interaction_steps = [
        step for step in core_steps
        if step.action_type not in {"open_app", "open_deeplink", "open_intent", "close_app"}
    ]
    if not interaction_steps:
        return "launch_only_segment"

    if len(core_steps) == 1:
        return "single_core_step_segment"

    if interaction_steps and all(
        step.action_type in _SEGMENT_SCROLL_ACTIONS for step in interaction_steps
    ):
        return "scroll_only_segment"

    return ""


def _normalized_step_app(step: CodeStep, result: CodegenResult) -> str:
    raw = getattr(step, "app", "") or result.app
    return normalize_app_identifier(result.platform, raw)


def _segment_screenshots(steps: list[CodeStep]) -> list[str]:
    screenshots: list[str] = []
    seen: set[str] = set()
    for step in steps:
        b64 = step.screenshot_b64
        if b64 and b64 not in seen:
            screenshots.append(b64)
            seen.add(b64)
    return screenshots


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _single_foreground_app_package(result: CodegenResult) -> str:
    apps: list[str] = []
    for step in result.steps:
        app = str(step.app or "").strip()
        if not app:
            return ""
        normalized = normalize_app_identifier(result.platform, app)
        if not _is_reusable_app(normalized):
            return ""
        apps.append(normalized)
    if not apps:
        return ""
    unique_apps = set(apps)
    if len(unique_apps) != 1:
        return ""
    return apps[0]
