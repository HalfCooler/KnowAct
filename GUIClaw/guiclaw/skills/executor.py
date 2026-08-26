"""
guiclaw.skills.executor
=======================
Step-by-step skill execution with agent-integrated parameter grounding,
valid-state verification, and subgoal recovery.

Execution pipeline per step:
1. Optionally verify ``valid_state`` against current screenshot (LLM-based); skip if disabled or special.
2a. If state is allowed and step is fixed → execute with ``fixed_values`` directly.
2b. If state is allowed and step is not fixed → ground parameters via ``ActionGrounder``
    (vision-LLM call) or fall back to template substitution if no grounder is available.
3. If enabled valid_state fails → run a mini recovery loop (``SubgoalRunner``) with valid_state
   as the goal; a subgoal ``done(status="success")`` hands control back to the
   original skill step without a second validator call.
4. After all steps complete (or recovery exhausted), ``execution_summary`` carries a
   narrative for the outer agent loop to use as context when it takes over.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import logging
import re
import time
import typing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from guiclaw.action import Action, ActionError, parse_action
from guiclaw.image_utils import normalize_image_scale_ratio, scale_image
from guiclaw.interfaces import DeviceBackend
from guiclaw.observation import Observation
from guiclaw.skills.data import Skill, SkillStep
from guiclaw.skills.normalization import normalize_adb_app_identifier, normalize_app_identifier
from guiclaw.skills.state_contract import evaluate_state_contract_detail

if typing.TYPE_CHECKING:
    from guiclaw.interfaces import LLMProvider
    from guiclaw.trajectory.recorder import TrajectoryRecorder

logger = logging.getLogger(__name__)

_POST_ACTION_SETTLE_SECONDS: float = 0.50
_OPEN_APP_SETTLE_SECONDS: float = 5.00
_OPEN_DEEPLINK_POST_VALIDATE_ATTEMPTS: int = 3
_OPEN_DEEPLINK_POST_VALIDATE_RETRY_SECONDS: float = 1.00
_NO_SETTLE_ACTIONS: frozenset[str] = frozenset(
    {"wait", "done", "request_intervention", "adb_command"}
)
_VALID_STATE_OPTIONAL_ACTIONS: frozenset[str] = frozenset(
    {
        "open_app",
        "wait",
        "done",
        "request_intervention",
        "screenshot",
        "hotkey",
        "adb_command",
    }
)
_TEMPLATE_COORDINATE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "tap": ("x", "y"),
    "double_tap": ("x", "y"),
    "long_press": ("x", "y"),
    "scroll": ("x", "y"),
    "swipe": ("x", "y", "x2", "y2"),
    "drag": ("x", "y", "x2", "y2"),
}
_UI_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


# ---------------------------------------------------------------------------
# Execution state
# ---------------------------------------------------------------------------


class ExecutionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ValidStateMode(str, Enum):
    """Per-step validation policy.

    ``strict``
        Deterministic ``state_contract`` (when present) plus small-model NL
        ``valid_state`` validation, with subgoal recovery on failure.
    ``contract_only``
        Deterministic ``state_contract`` only.  When no contract yields a verdict
        the step is allowed (trust grounder); the NL validator and recovery are
        never invoked.
    ``off``
        Skip all validation, contracts included.
    """

    STRICT = "strict"
    CONTRACT_ONLY = "contract_only"
    OFF = "off"


#: Source tag marking compact-induced skills (see scripts/induce_compact_skills.py).
_COMPACT_VALID_STATE_TAG = "compact_extracted"


def _coerce_valid_state_mode(value: "ValidStateMode | str | bool | None") -> ValidStateMode:
    """Normalize a mode value, mapping the legacy bool (True/False).

    The legacy ``enable_valid_state=False`` still evaluated deterministic
    contracts, so it maps to ``contract_only`` rather than ``off``.
    """
    if isinstance(value, ValidStateMode):
        return value
    if isinstance(value, bool):
        return ValidStateMode.STRICT if value else ValidStateMode.CONTRACT_ONLY
    if value is None:
        return ValidStateMode.STRICT
    return ValidStateMode(str(value))


def _skill_is_compact(skill: Skill) -> bool:
    return _COMPACT_VALID_STATE_TAG in {str(tag) for tag in (getattr(skill, "tags", ()) or ())}


# ---------------------------------------------------------------------------
# Subgoal result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgoalResult:
    """Outcome of a mini recovery loop."""

    success: bool
    steps_taken: int
    action_summaries: list[str]
    final_screenshot: Path | bytes | None = None
    final_observation: Observation | None = None
    error: str | None = None
    token_usage: dict[str, int] = dataclasses.field(default_factory=dict)
    #: True when the subgoal succeeded because the model issued a ``done``
    #: action (highest-confidence self-declaration), not via LLM state validation.
    done_judgment: bool = False


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Outcome of a single skill step."""

    step_index: int
    action: Action
    backend_result: str
    state: ExecutionState
    valid_state_check: bool = True
    # "fixed" | "llm" | "template"
    grounding_mode: str = "template"
    recovery_attempted: bool = False
    recovery_result: SubgoalResult | None = None
    action_summary: str = ""
    error: str | None = None
    token_usage: dict[str, int] = dataclasses.field(default_factory=dict)
    duration_s: float = 0.0
    # None  → validation was intentionally skipped (no valid_state / no validator)
    # float → actual LLM validation call duration in seconds
    validate_duration_s: float | None = None
    grounding_duration_s: float | None = None
    observation: dict[str, Any] | None = None
    screenshot_path: str | None = None
    contract_eval_detail: dict[str, Any] | None = None


@dataclass
class SkillExecutionResult:
    """Outcome of a full skill execution."""

    skill: Skill
    step_results: list[StepResult]
    state: ExecutionState
    # Narrative summary of all steps; injected into the agent loop as context.
    execution_summary: str = ""
    error: str | None = None
    token_usage: dict[str, int] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@typing.runtime_checkable
class StateValidator(typing.Protocol):
    """Validates current screen state against a ``valid_state`` description."""

    async def validate(
        self,
        valid_state: str,
        screenshot: Path | bytes | None = None,
    ) -> bool:
        """Return True if the current screen matches *valid_state*."""
        ...


@typing.runtime_checkable
class ActionGrounder(typing.Protocol):
    """Grounds a non-fixed SkillStep into a concrete Action via vision LLM.

    Called when ``step.fixed`` is False. The implementation sends the current
    screenshot and step description to the LLM with the ``computer_use`` tool
    and parses the response into an Action.
    """

    async def ground(
        self,
        step: SkillStep,
        screenshot: Path | bytes,
        params: dict[str, str],
    ) -> Action:
        """Return a concrete Action for *step* given the current *screenshot*."""
        ...


@typing.runtime_checkable
class SubgoalRunner(typing.Protocol):
    """Runs a mini vision-action loop to recover to a desired screen state.

    Called when ``valid_state`` does not match. The implementation executes
    up to ``max_steps`` vision-action cycles. It only recovers to a valid-state
    goal and reports success when the profile emits ``done(status="success")``.
    """

    async def run_subgoal(
        self,
        goal: str,
        screenshot: Path | bytes,
        *,
        max_steps: int = 3,
        current_observation: Observation | None = None,
    ) -> SubgoalResult:
        """Navigate towards *goal* starting from *screenshot*."""
        ...


@typing.runtime_checkable
class ScreenshotProvider(typing.Protocol):
    """Provides the current screen screenshot as a Path or bytes object."""

    async def get_screenshot(self) -> Path | bytes | None:
        """Capture and return the current screenshot."""
        ...


# ---------------------------------------------------------------------------
# LLM-based state validator
# ---------------------------------------------------------------------------

_VALIDATION_PROMPT = """\
Look at this screenshot and answer: does the current screen state match \
the following description?

Expected state: {valid_state}

Respond with ONLY a JSON object: {{"valid": true/false}}
"""


class LLMStateValidator:
    """Validate screen state using a vision LLM."""

    def __init__(self, llm: LLMProvider, image_scale_ratio: float = 0.5) -> None:
        self._llm = llm
        self._image_scale_ratio = normalize_image_scale_ratio(image_scale_ratio)
        self._usage_accum: dict[str, int] = {}
        self._ttft_samples: list[float] = []
        self._latency_samples: list[float] = []

    def drain_usage(self) -> dict[str, int]:
        """Return accumulated token usage since last drain and reset the counter."""
        usage = dict(self._usage_accum)
        self._usage_accum.clear()
        return usage

    def drain_timings(self) -> dict[str, float]:
        """Return mean ttft_s / chat_latency_s across samples and reset."""
        out: dict[str, float] = {}
        if self._ttft_samples:
            out["ttft_s"] = sum(self._ttft_samples) / len(self._ttft_samples)
        if self._latency_samples:
            out["chat_latency_s"] = sum(self._latency_samples) / len(self._latency_samples)
        self._ttft_samples.clear()
        self._latency_samples.clear()
        return out

    async def validate(
        self,
        valid_state: str,
        screenshot: Path | bytes | None = None,
    ) -> bool:
        if _should_skip_validation(valid_state):
            return True

        if screenshot is None:
            logger.warning("No screenshot for state validation; allowing execution")
            return True

        prompt = _VALIDATION_PROMPT.format(valid_state=valid_state)

        raw = screenshot.read_bytes() if isinstance(screenshot, Path) else screenshot
        image_data = base64.b64encode(
            scale_image(raw, scale_ratio=self._image_scale_ratio)
        ).decode()
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ]

        messages = [{"role": "user", "content": content}]
        try:
            response = await self._llm.chat(messages)
        except Exception as exc:
            logger.error("State validation LLM call failed: %s", exc)
            return True  # Fail-open on LLM error

        for k, v in (response.usage or {}).items():
            self._usage_accum[k] = self._usage_accum.get(k, 0) + v
        if getattr(response, "ttft_s", None) is not None:
            self._ttft_samples.append(response.ttft_s)
        if getattr(response, "latency_s", None) is not None:
            self._latency_samples.append(response.latency_s)
        return self._parse_response(response.content)

    @staticmethod
    def _parse_response(text: str) -> bool:
        import json as _json

        text = text.strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                obj = _json.loads(match.group(0))
                valid = obj.get("valid", obj.get("is_valid", obj.get("match")))
                if isinstance(valid, bool):
                    return valid
                if isinstance(valid, str):
                    return valid.strip().lower() in ("true", "yes", "matched")
            except _json.JSONDecodeError:
                pass
        lowered = text.lower()
        if "true" in lowered or "yes" in lowered or "match" in lowered:
            return True
        return False


def _should_skip_validation(valid_state: str | None) -> bool:
    """Return True when valid_state indicates no verification is needed."""
    if valid_state is None or not valid_state.strip():
        return False
    lowered = valid_state.strip().lower()
    skip_hints = ("no need to verify", "return true", "skip", "none", "n/a")
    return any(hint in lowered for hint in skip_hints)


def _post_open_app_overlay_action(observation: Observation) -> Action | None:
    ui_tree = observation.extra.get("ui_tree") if isinstance(observation.extra, dict) else None
    if not isinstance(ui_tree, list):
        return None

    parsed_nodes: list[tuple[dict[str, Any], tuple[int, int, int, int]]] = []
    for node in ui_tree:
        if isinstance(node, dict):
            bounds = _parse_ui_bounds(node.get("bounds"))
            if bounds is not None:
                parsed_nodes.append((node, bounds))
    if not parsed_nodes:
        return None

    root = (
        min(bounds[0] for _node, bounds in parsed_nodes),
        min(bounds[1] for _node, bounds in parsed_nodes),
        max(bounds[2] for _node, bounds in parsed_nodes),
        max(bounds[3] for _node, bounds in parsed_nodes),
    )
    root_width = max(1, root[2] - root[0])
    root_height = max(1, root[3] - root[1])

    skip_action: Action | None = None
    close_action: Action | None = None
    for node, bounds in parsed_nodes:
        if node.get("clickable") is not True or node.get("enabled") is not True:
            continue
        label = _ui_node_label(node)
        if not label:
            continue
        label_lower = label.lower()
        is_skip = "skip" in label_lower or "跳过" in label
        is_close = (
            "关闭" in label
            or label_lower in {"close", "x", "×"}
            or label_lower.startswith("close ")
        )
        if not is_skip and not is_close:
            continue

        x1, y1, x2, y2 = bounds
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if is_close:
            x_ratio = (cx - root[0]) / root_width
            y_ratio = (cy - root[1]) / root_height
            if x_ratio < 0.75 or 0.25 < y_ratio < 0.70:
                continue

        action = _ui_bounds_to_tap_action(bounds, root, observation)
        if is_skip and skip_action is None:
            skip_action = action
        elif is_close and close_action is None:
            close_action = action

    return skip_action or close_action


def _parse_ui_bounds(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, str):
        return None
    match = _UI_BOUNDS_RE.fullmatch(value.strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _ui_node_label(node: dict[str, Any]) -> str:
    return " ".join(
        str(value) for value in (node.get("text"), node.get("content_desc")) if value
    ).strip()


def _ui_bounds_to_tap_action(
    bounds: tuple[int, int, int, int],
    root: tuple[int, int, int, int],
    observation: Observation,
) -> Action:
    x1, y1, x2, y2 = bounds
    root_width = max(1, root[2] - root[0])
    root_height = max(1, root[3] - root[1])
    screen_width = max(1, int(observation.screen_width or 1))
    screen_height = max(1, int(observation.screen_height or 1))
    x = ((x1 + x2) / 2.0 - root[0]) / root_width * screen_width
    y = ((y1 + y2) / 2.0 - root[1]) / root_height * screen_height
    return Action(
        action_type="tap",
        x=max(0.0, min(float(screen_width - 1), x)),
        y=max(0.0, min(float(screen_height - 1), y)),
    )


# ---------------------------------------------------------------------------
# Parameter grounding helpers
# ---------------------------------------------------------------------------


def _ground_text(text: str, params: dict[str, str]) -> str:
    """Replace ``{{param}}`` placeholders with actual values."""
    for key, value in params.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    remaining = re.findall(r"\{\{(\w+)\}\}", text)
    if remaining:
        logger.warning("Unresolved placeholders: %s", remaining)
    return text


def _ground_value(value: Any, params: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _ground_text(value, params)
    if isinstance(value, list):
        return [_ground_value(item, params) for item in value]
    if isinstance(value, tuple):
        return tuple(_ground_value(item, params) for item in value)
    if isinstance(value, dict):
        return {
            _ground_value(key, params): _ground_value(item, params) for key, item in value.items()
        }
    return value


def _build_fixed_action(step: SkillStep, params: dict[str, str]) -> Action:
    """Construct an Action from ``step.fixed_values``.

    ``fixed_values`` holds concrete parameter values (coordinates, text, etc.)
    that bypass LLM grounding entirely. Template substitution is still applied
    to string values so that ``{{param}}`` placeholders in text fields resolve
    correctly.
    """
    payload: dict[str, Any] = {"action_type": step.action_type}
    for key, value in step.fixed_values.items():
        payload[key] = _ground_value(value, params)
    return parse_action(payload)


def _build_template_action(step: SkillStep, params: dict[str, str]) -> Action:
    """Construct an Action via template substitution (legacy / fallback path).

    Used when no ``ActionGrounder`` is provided or when callers do not supply
    one. Mirrors the original ``_ground_step`` behaviour.
    """
    payload = _build_template_payload(step, params)
    return _lenient_action_from_payload(payload)


def _build_template_payload(step: SkillStep, params: dict[str, str]) -> dict[str, Any]:
    """Build a raw action payload from step templates and runtime parameters."""
    target = _ground_text(step.target, params)
    grounded: dict[str, Any] = {}
    for k, v in step.parameters.items():
        grounded[k] = _ground_value(v, params)

    payload: dict[str, Any] = {"action_type": step.action_type}
    for key in (
        "x",
        "y",
        "x2",
        "y2",
        "points",
        "text",
        "key",
        "pixels",
        "direction",
        "duration_ms",
        "component",
        "package",
        "intent_action",
        "mime_type",
        "categories",
        "extras",
        "relative",
        "status",
        "auto_enter",
        "command_id",
        "command",
        "params",
        "arguments",
    ):
        if key in grounded:
            payload[key] = grounded[key]
    if "text" not in payload and step.action_type in {
        "input_text",
        "open_app",
        "open_deeplink",
        "close_app",
        "adb_command",
    }:
        payload["text"] = target
    return payload


def _lenient_action_from_payload(payload: dict[str, Any]) -> Action:
    """Create an Action while preserving legacy permissive template fallback."""
    kwargs: dict[str, Any] = {"action_type": str(payload.get("action_type") or "")}
    for key in ("x", "y", "x2", "y2"):
        if key in payload:
            kwargs[key] = float(payload[key])
    if "points" in payload:
        kwargs["points"] = parse_action(
            {
                "action_type": "click_multi",
                "points": payload["points"],
            }
        ).points
    if "text" in payload:
        kwargs["text"] = str(payload["text"])
    elif "direction" in payload:
        kwargs["text"] = str(payload["direction"]).strip().lower()
    if "key" in payload:
        raw_key = payload["key"]
        kwargs["key"] = (
            [str(item) for item in raw_key] if isinstance(raw_key, list) else [str(raw_key)]
        )
    if "pixels" in payload:
        kwargs["pixels"] = int(payload["pixels"])
    if "duration_ms" in payload:
        kwargs["duration_ms"] = int(payload["duration_ms"])
    if "component" in payload:
        kwargs["component"] = str(payload["component"])
    if "package" in payload:
        kwargs["package"] = str(payload["package"])
    if "intent_action" in payload:
        kwargs["intent_action"] = str(payload["intent_action"])
    if "mime_type" in payload:
        kwargs["mime_type"] = str(payload["mime_type"])
    if "categories" in payload:
        raw_categories = payload["categories"]
        kwargs["categories"] = (
            tuple(str(item) for item in raw_categories)
            if isinstance(raw_categories, (list, tuple))
            else (str(raw_categories),)
        )
    if "extras" in payload:
        raw_extras = payload["extras"]
        if isinstance(raw_extras, dict):
            kwargs["extras"] = tuple(raw_extras.items())
        elif isinstance(raw_extras, (list, tuple)):
            kwargs["extras"] = tuple(tuple(item) for item in raw_extras)
    if "command_id" in payload:
        kwargs["command_id"] = str(payload["command_id"])
    elif "command" in payload:
        kwargs["command_id"] = str(payload["command"])
    if "params" in payload and isinstance(payload["params"], dict):
        kwargs["params"] = dict(payload["params"])
    elif "arguments" in payload and isinstance(payload["arguments"], dict):
        kwargs["params"] = dict(payload["arguments"])
    if "relative" in payload:
        kwargs["relative"] = bool(payload["relative"])
    if "status" in payload:
        kwargs["status"] = str(payload["status"])
    if "auto_enter" in payload:
        kwargs["auto_enter"] = bool(payload["auto_enter"])
    return Action(**kwargs)


def _try_build_complete_template_action(
    step: SkillStep,
    params: dict[str, str],
) -> Action | None:
    """Return a fully-resolved template action when no visual grounding is needed."""
    try:
        action = parse_action(_build_template_payload(step, params))
    except (ActionError, TypeError, ValueError):
        return None
    required_coords = _TEMPLATE_COORDINATE_REQUIREMENTS.get(action.action_type)
    if required_coords and any(getattr(action, name, None) is None for name in required_coords):
        return None
    for value in (action.text, *(action.key or ())):
        if isinstance(value, str) and re.search(r"\{\{\w+\}\}", value):
            return None
    return action


# ---------------------------------------------------------------------------
# Execution summary formatter
# ---------------------------------------------------------------------------


def _build_execution_summary(skill: Skill, step_results: list[StepResult]) -> str:
    """Build a human-readable narrative of the skill execution for the agent loop."""
    succeeded = sum(1 for r in step_results if r.state == ExecutionState.SUCCEEDED)
    total = len(skill.steps)

    lines = [f'Skill "{skill.name}" executed ({succeeded}/{total} steps succeeded):']
    for r in step_results:
        status_tag = "succeeded" if r.state == ExecutionState.SUCCEEDED else "failed"
        mode_tag = f"[{r.grounding_mode}]"
        recovery_tag = ""
        if r.recovery_attempted:
            if r.recovery_result and r.recovery_result.success:
                recovery_tag = f" [recovered in {r.recovery_result.steps_taken} sub-steps]"
            else:
                recovery_tag = " [recovery failed]"
        summary_text = f" — {r.action_summary}" if r.action_summary else ""
        lines.append(
            f"  Step {r.step_index + 1}: {r.action.action_type} {mode_tag}"
            f"{recovery_tag}{summary_text} — {status_tag}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SkillExecutor
# ---------------------------------------------------------------------------


@dataclass
class SkillExecutor:
    """Execute a :class:`Skill` step-by-step with agent-integrated validation.

    Parameters
    ----------
    backend:
        Device backend for action execution.
    state_validator:
        Optional LLM-based validator for per-step ``valid_state`` checks.
        Falls back to pass-through (allow) when ``None``.
    valid_state_mode:
        Global per-step validation policy (:class:`ValidStateMode`, or a legacy
        bool). ``compact_extracted`` skills auto-downgrade *contracted* steps to
        ``contract_only`` under ``strict``; ``optional`` popup steps keep a guard
        unless the mode is ``off``; ``visual_guarded`` steps stay ``strict`` unless
        the mode is ``off``. Skill definitions keep their ``valid_state`` text
        regardless — the mode only controls which runtime checks execute.
    action_grounder:
        Optional vision-LLM grounder for non-fixed steps. When ``None``,
        falls back to template substitution (legacy behaviour).
    subgoal_runner:
        Optional mini agent-loop runner for recovering from a failed
        ``valid_state`` check. When ``None``, validation failures fail the
        step immediately (legacy behaviour).
    screenshot_provider:
        Optional async provider for the current screenshot. Falls back to
        ``None`` screenshots (validation is skipped) when not supplied.
    stop_on_failure:
        Whether to halt execution on the first non-optional step failure.
    max_recovery_steps:
        Maximum vision-action steps allowed inside a single recovery subgoal.
    """

    backend: DeviceBackend
    state_validator: StateValidator | None = None
    action_grounder: ActionGrounder | None = None
    subgoal_runner: SubgoalRunner | None = None
    screenshot_provider: ScreenshotProvider | None = None
    trajectory_recorder: TrajectoryRecorder | None = None
    stop_on_failure: bool = True
    max_recovery_steps: int = 3
    valid_state_mode: ValidStateMode | str | bool = ValidStateMode.STRICT
    _last_contract_eval_detail: dict[str, Any] | None = dataclasses.field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.valid_state_mode = _coerce_valid_state_mode(self.valid_state_mode)

    def set_artifacts_root(self, artifacts_root: Path) -> None:
        for component in (self.screenshot_provider, self.subgoal_runner):
            setter = getattr(component, "set_artifacts_root", None)
            if callable(setter):
                setter(artifacts_root)

    def _effective_mode(
        self,
        skill: Skill,
        step: SkillStep,
        *,
        is_optional: bool,
        visual_guarded_step: bool,
    ) -> ValidStateMode:
        """Resolve the validation mode for a single step.

        ``off`` is absolute. Otherwise ``visual_guarded`` steps stay ``strict``;
        ``optional`` steps keep a guard (``contract_only`` when contracted, else
        ``strict``) so absent popups are skipped rather than blindly executed; and
        compact skills downgrade only their *contracted* steps to ``contract_only``
        (an uncontracted required step keeps the global mode to avoid running with
        no guard at all).
        """
        mode = _coerce_valid_state_mode(self.valid_state_mode)
        if mode is ValidStateMode.OFF:
            return mode
        if visual_guarded_step:
            return ValidStateMode.STRICT
        if is_optional:
            return (
                ValidStateMode.CONTRACT_ONLY
                if step.state_contract is not None
                else ValidStateMode.STRICT
            )
        if (
            mode is ValidStateMode.STRICT
            and step.state_contract is not None
            and _skill_is_compact(skill)
        ):
            return ValidStateMode.CONTRACT_ONLY
        return mode

    async def execute(
        self,
        skill: Skill,
        params: dict[str, str] | None = None,
        *,
        timeout: float = 30.0,
    ) -> SkillExecutionResult:
        """Run all steps of *skill* sequentially with state verification."""
        params = params or {}
        step_results: list[StepResult] = []
        overall_state = ExecutionState.RUNNING
        error_msg: str | None = None
        total_token_usage: dict[str, int] = {}
        # valid_state strings confirmed by a high-confidence done judgment;
        # subsequent steps sharing the same valid_state skip LLM re-validation.
        confirmed_valid_states: set[str] = set()
        visual_guarded_skill = "visual_guarded" in {
            str(tag) for tag in (getattr(skill, "tags", ()) or ())
        }

        if self.trajectory_recorder is not None:
            self.trajectory_recorder.record_event(
                "skill_execution_start",
                skill_id=skill.skill_id,
                skill_name=skill.name,
                step_count=len(skill.steps),
                valid_state_mode=_coerce_valid_state_mode(self.valid_state_mode).value,
            )

        for i, step in enumerate(skill.steps):
            self._last_contract_eval_detail = None
            is_optional = bool(step.parameters.get("optional", False))
            post_action_contract = (
                step.action_type in {"open_deeplink", "open_intent"}
                and step.state_contract is not None
            )
            step_start = time.monotonic()
            validate_usage: dict[str, int] = {}
            grounding_usage: dict[str, int] = {}
            validate_dur = 0.0
            grounding_dur = 0.0
            visual_guard_block_reason: str | None = None

            # ------------------------------------------------------------------
            # 1. Capture current screenshot
            # ------------------------------------------------------------------
            observation, screenshot = await self._get_observation_and_screenshot()

            # ------------------------------------------------------------------
            # 2. Valid-state check
            #    Skip when the state was already confirmed via a done judgment
            #    in an earlier recovery within this execution.
            # ------------------------------------------------------------------
            visual_guarded_step = (
                visual_guarded_skill
                and not post_action_contract
                and step.action_type
                not in {
                    "open_app",
                    "open_deeplink",
                    "open_intent",
                    "adb_command",
                    "wait",
                    "done",
                    "request_intervention",
                }
            )
            mode = self._effective_mode(
                skill,
                step,
                is_optional=is_optional,
                visual_guarded_step=visual_guarded_step,
            )
            if mode is ValidStateMode.OFF:
                valid, validate_usage, validate_dur = True, {}, None
            elif visual_guarded_step and step.state_contract is None and not step.valid_state:
                visual_guard_block_reason = "missing visual valid_state"
                valid, validate_usage, validate_dur = False, {}, None
            elif visual_guarded_step and self.state_validator is None:
                visual_guard_block_reason = "missing state validator"
                valid, validate_usage, validate_dur = False, {}, None
            elif visual_guarded_step and screenshot is None:
                visual_guard_block_reason = "missing screenshot"
                valid, validate_usage, validate_dur = False, {}, None
            elif post_action_contract:
                valid, validate_usage, validate_dur = True, {}, None
            elif step.valid_state and step.valid_state in confirmed_valid_states:
                logger.debug(
                    "Step %d: valid_state %r confirmed by prior done judgment, skipping check",
                    i,
                    step.valid_state,
                )
                valid, validate_usage, validate_dur = True, {}, None
            else:
                valid, validate_usage, validate_dur = await self._validate_state(
                    step,
                    screenshot,
                    observation=observation,
                    mode=mode,
                )
            _merge_usage(total_token_usage, validate_usage)

            # ------------------------------------------------------------------
            # 3. Recovery subgoal when valid_state fails
            # ------------------------------------------------------------------
            recovery_result: SubgoalResult | None = None
            has_valid_state = bool(step.valid_state and step.valid_state.strip())
            requires_state = step.state_contract is not None or (
                step.action_type not in _VALID_STATE_OPTIONAL_ACTIONS
                and not _should_skip_validation(step.valid_state)
            )
            should_enforce_state = visual_guarded_step or (
                not post_action_contract and requires_state
            )
            if not valid and should_enforce_state and visual_guard_block_reason is None:
                can_recover_with_subgoal = (
                    mode is ValidStateMode.STRICT
                    and has_valid_state
                    and not _should_skip_validation(step.valid_state)
                    and self.subgoal_runner is not None
                    and screenshot is not None
                )
                if can_recover_with_subgoal:
                    logger.info(
                        "Step %d: valid_state failed, attempting recovery for: %r",
                        i,
                        step.valid_state,
                    )
                    recovery_result = await self.subgoal_runner.run_subgoal(
                        goal=step.valid_state or "",
                        screenshot=screenshot,
                        max_steps=self.max_recovery_steps,
                        current_observation=observation,
                    )
                    _merge_usage(total_token_usage, recovery_result.token_usage)
                    if recovery_result.success:
                        valid = True
                        observation = recovery_result.final_observation or observation
                        screenshot = recovery_result.final_screenshot or screenshot
                        logger.info(
                            "Step %d: recovery succeeded, skipping re-validation",
                            i,
                        )
                        if step.valid_state:
                            confirmed_valid_states.add(step.valid_state)
                            logger.debug(
                                "Step %d: added %r to confirmed_valid_states",
                                i,
                                step.valid_state,
                            )
                    else:
                        logger.warning(
                            "Step %d: recovery exhausted (%d sub-steps), "
                            "valid_state still not reached",
                            i,
                            recovery_result.steps_taken,
                        )
                else:
                    logger.info("Step %d: valid_state failed, no subgoal_runner available", i)

            # ------------------------------------------------------------------
            # 4. If still invalid, record failure and decide whether to continue
            # ------------------------------------------------------------------
            if not valid and should_enforce_state:
                step_dur = time.monotonic() - step_start
                state_description = (
                    step.valid_state or visual_guard_block_reason or "state_contract"
                )
                step_result = StepResult(
                    step_index=i,
                    action=Action(action_type=step.action_type),
                    backend_result="",
                    state=ExecutionState.FAILED,
                    valid_state_check=False,
                    recovery_attempted=recovery_result is not None,
                    recovery_result=recovery_result,
                    error=f"valid_state not reached: {state_description}",
                    token_usage=dict(validate_usage),
                    duration_s=step_dur,
                    validate_duration_s=validate_dur,
                    observation=_serialize_observation(observation),
                    screenshot_path=_screenshot_path_string(screenshot),
                    contract_eval_detail=self._last_contract_eval_detail,
                )
                step_results.append(step_result)
                self._record_skill_step(skill, step, step_result)
                if is_optional:
                    logger.info("Optional step %d skipped (valid_state not reached)", i)
                    continue
                overall_state = ExecutionState.FAILED
                error_msg = f"Step {i} valid_state not reached: {state_description}"
                if self.stop_on_failure:
                    break
                continue

            # ------------------------------------------------------------------
            # 5. Execute action
            # ------------------------------------------------------------------
            try:
                action, grounding_mode, grounding_usage, grounding_dur = await self._resolve_action(
                    step, screenshot, params
                )
                _merge_usage(total_token_usage, grounding_usage)
                execution_error: Exception | None = None
                try:
                    result_text = await self.backend.execute(action, timeout=timeout)
                except Exception as exc:
                    if not post_action_contract:
                        raise
                    execution_error = exc
                    result_text = str(exc)
                    logger.warning(
                        "Step %d %s raised %s; validating post-state before failing",
                        i,
                        action.action_type,
                        type(exc).__name__,
                    )
                # Allow the UI to settle before the next step's
                # screenshot / validate / grounding cycle.
                if action.action_type in {"open_app", "open_deeplink", "open_intent"}:
                    await asyncio.sleep(_OPEN_APP_SETTLE_SECONDS)
                    if action.action_type == "open_app":
                        await self._dismiss_post_open_app_overlay(timeout=timeout)
                elif action.action_type not in _NO_SETTLE_ACTIONS:
                    await asyncio.sleep(_POST_ACTION_SETTLE_SECONDS)
                if post_action_contract:
                    post_valid = False
                    attempts = (
                        _OPEN_DEEPLINK_POST_VALIDATE_ATTEMPTS
                        if action.action_type in {"open_deeplink", "open_intent"}
                        else 1
                    )
                    for attempt_index in range(attempts):
                        (
                            post_observation,
                            post_screenshot,
                        ) = await self._get_observation_and_screenshot()
                        post_valid, post_usage, post_dur = await self._validate_state(
                            step,
                            post_screenshot,
                            observation=post_observation,
                            mode=mode,
                        )
                        _merge_usage(total_token_usage, post_usage)
                        _merge_usage(validate_usage, post_usage)
                        validate_dur = _merge_optional_duration(validate_dur, post_dur)
                        if post_valid:
                            break
                        if attempt_index < attempts - 1:
                            await asyncio.sleep(_OPEN_DEEPLINK_POST_VALIDATE_RETRY_SECONDS)
                    if execution_error is not None and mode is ValidStateMode.OFF:
                        # off skips validation, but a caught backend execution error
                        # is a hard failure that must not be masked as a successful
                        # post-state (there was no positive verification under off).
                        post_valid = False
                    if (
                        post_valid
                        and execution_error is not None
                        and self.trajectory_recorder is not None
                    ):
                        self.trajectory_recorder.record_event(
                            "skill_action_error_post_state_ok",
                            skill_id=skill.skill_id,
                            skill_name=skill.name,
                            step_index=i,
                            action_type=action.action_type,
                            error=str(execution_error),
                            exception_type=type(execution_error).__name__,
                        )
                    if not post_valid:
                        step_dur = time.monotonic() - step_start
                        state_description = step.valid_state or "state_contract"
                        error_text = f"post-state not reached: {state_description}"
                        if execution_error is not None:
                            error_text = (
                                f"execution failed and post-state not reached: "
                                f"{state_description}: {execution_error}"
                            )
                        failed_token_usage = dict(validate_usage)
                        _merge_usage(failed_token_usage, grounding_usage)
                        step_result = StepResult(
                            step_index=i,
                            action=action,
                            backend_result=result_text,
                            state=ExecutionState.FAILED,
                            valid_state_check=False,
                            grounding_mode=grounding_mode,
                            recovery_attempted=recovery_result is not None,
                            recovery_result=recovery_result,
                            action_summary=f"{action.action_type} on {step.target or 'target'}",
                            error=error_text,
                            token_usage=failed_token_usage,
                            duration_s=step_dur,
                            validate_duration_s=validate_dur,
                            grounding_duration_s=grounding_dur,
                            observation=_serialize_observation(post_observation),
                            screenshot_path=_screenshot_path_string(post_screenshot),
                            contract_eval_detail=self._last_contract_eval_detail,
                        )
                        step_results.append(step_result)
                        self._record_skill_step(skill, step, step_result)
                        if is_optional:
                            logger.info("Optional deeplink step %d failed post-state validation", i)
                            continue
                        overall_state = ExecutionState.FAILED
                        error_msg = f"Step {i} post-state not reached: {state_description}"
                        if self.stop_on_failure:
                            break
                        continue
                step_dur = time.monotonic() - step_start
                step_token_usage = dict(validate_usage)
                _merge_usage(step_token_usage, grounding_usage)
                step_result = StepResult(
                    step_index=i,
                    action=action,
                    backend_result=result_text,
                    state=ExecutionState.SUCCEEDED,
                    grounding_mode=grounding_mode,
                    recovery_attempted=recovery_result is not None,
                    recovery_result=recovery_result,
                    action_summary=f"{action.action_type} on {step.target or 'target'}",
                    token_usage=step_token_usage,
                    duration_s=step_dur,
                    validate_duration_s=validate_dur,
                    grounding_duration_s=grounding_dur,
                )
                step_results.append(step_result)
                self._record_skill_step(skill, step, step_result)
            except Exception as exc:
                logger.error("Step %d execution error: %s", i, exc)
                step_dur = time.monotonic() - step_start
                step_result = StepResult(
                    step_index=i,
                    action=Action(action_type=step.action_type),
                    backend_result=str(exc),
                    state=ExecutionState.FAILED,
                    recovery_attempted=recovery_result is not None,
                    recovery_result=recovery_result,
                    error=str(exc),
                    token_usage=dict(validate_usage),
                    duration_s=step_dur,
                    validate_duration_s=validate_dur,
                    observation=_serialize_observation(observation),
                    screenshot_path=_screenshot_path_string(screenshot),
                    contract_eval_detail=self._last_contract_eval_detail,
                )
                step_results.append(step_result)
                self._record_skill_step(skill, step, step_result)
                if is_optional:
                    logger.info("Optional step %d failed with exception, continuing", i)
                    continue
                overall_state = ExecutionState.FAILED
                error_msg = f"Step {i} execution failed: {exc}"
                if self.stop_on_failure:
                    break
                continue

        if overall_state != ExecutionState.FAILED:
            overall_state = ExecutionState.SUCCEEDED

        result = SkillExecutionResult(
            skill=skill,
            step_results=step_results,
            state=overall_state,
            execution_summary=_build_execution_summary(skill, step_results),
            error=error_msg,
            token_usage=total_token_usage,
        )
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.record_event(
                "skill_execution_result",
                skill_id=skill.skill_id,
                skill_name=skill.name,
                state=result.state.value,
                execution_summary=result.execution_summary,
                error=result.error,
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_action(
        self,
        step: SkillStep,
        screenshot: Path | bytes | None,
        params: dict[str, str],
    ) -> tuple[Action, str, dict[str, int], float]:
        """Return ``(action, grounding_mode, token_usage, duration_s)`` for the given step."""
        if step.fixed:
            return self._normalize_app_action(_build_fixed_action(step, params)), "fixed", {}, 0.0

        template_action = _try_build_complete_template_action(step, params)
        if template_action is not None:
            return self._normalize_app_action(template_action), "template", {}, 0.0

        if self.action_grounder is not None and screenshot is not None:
            try:
                t0 = time.monotonic()
                action = await self.action_grounder.ground(step, screenshot, params)
                duration = time.monotonic() - t0
                usage: dict[str, int] = {}
                if hasattr(self.action_grounder, "drain_usage"):
                    usage = self.action_grounder.drain_usage()
                return self._normalize_app_action(action), "llm", usage, duration
            except Exception as exc:
                logger.warning(
                    "ActionGrounder failed for step %r, falling back to template: %s",
                    step.action_type,
                    exc,
                )

        return self._normalize_app_action(_build_template_action(step, params)), "template", {}, 0.0

    def _normalize_app_action(self, action: Action) -> Action:
        """Resolve human/cross-platform app identifiers before backend execution."""
        if action.action_type not in ("open_app", "close_app") or not action.text:
            return action
        platform = getattr(self.backend, "platform", None)
        if platform not in ("android", "ios"):
            return action

        resolved = (
            normalize_adb_app_identifier(action.text)
            if platform == "android" and hasattr(self.backend, "_run")
            else normalize_app_identifier(platform, action.text)
        )
        if resolved == action.text:
            return action
        logger.debug(
            "Resolved %s app %r -> %r for skill execution",
            platform,
            action.text,
            resolved,
        )
        return dataclasses.replace(action, text=resolved)

    async def _dismiss_post_open_app_overlay(self, *, timeout: float) -> None:
        observation, _screenshot = await self._get_observation_and_screenshot()
        if observation is None:
            return
        dismiss_action = _post_open_app_overlay_action(observation)
        if dismiss_action is None:
            return
        try:
            await self.backend.execute(dismiss_action, timeout=timeout)
        except Exception as exc:
            logger.debug("Post-open-app overlay dismiss failed: %s", exc)
            return
        if self.trajectory_recorder is not None:
            self.trajectory_recorder.record_event(
                "post_open_app_overlay_dismissed",
                action=_serialize_action(dismiss_action),
            )
        await asyncio.sleep(_POST_ACTION_SETTLE_SECONDS)

    def _record_skill_step(self, skill: Skill, step: SkillStep, step_result: StepResult) -> None:
        if self.trajectory_recorder is None:
            return
        validate_inference_time_s = step_result.validate_duration_s
        grounding_inference_time_s = (
            step_result.grounding_duration_s
            if step_result.grounding_mode == "llm"
            else None
        )
        inference_parts = [
            value
            for value in (validate_inference_time_s, grounding_inference_time_s)
            if value is not None
        ]
        timings: dict[str, float] = {}
        for src in (self.action_grounder, self.state_validator):
            drain = getattr(src, "drain_timings", None)
            if callable(drain):
                for k, v in drain().items():
                    timings.setdefault(k, v)
        self.trajectory_recorder.record_event(
            "skill_step",
            skill_id=skill.skill_id,
            skill_name=skill.name,
            step_index=step_result.step_index,
            target=step.target,
            action=_serialize_action(step_result.action),
            action_summary=step_result.action_summary,
            grounding_mode=step_result.grounding_mode,
            backend_result=step_result.backend_result,
            valid_state=step.valid_state,
            state_contract=step.state_contract,
            valid_state_mode=_coerce_valid_state_mode(self.valid_state_mode).value,
            valid_state_check=step_result.valid_state_check,
            observation=step_result.observation,
            screenshot_path=step_result.screenshot_path,
            contract_eval_detail=step_result.contract_eval_detail,
            recovery_attempted=step_result.recovery_attempted,
            recovery_success=bool(
                step_result.recovery_result and step_result.recovery_result.success
            ),
            error=step_result.error,
            token_usage=step_result.token_usage or None,
            duration_s=round(step_result.duration_s, 3) if step_result.duration_s else None,
            validate_duration_s=round(step_result.validate_duration_s, 3)
            if step_result.validate_duration_s is not None
            else None,
            grounding_duration_s=round(step_result.grounding_duration_s, 3)
            if step_result.grounding_duration_s is not None
            else None,
            inference_time_s=round(sum(inference_parts), 3) if inference_parts else None,
            validate_inference_time_s=round(validate_inference_time_s, 3)
            if validate_inference_time_s is not None
            else None,
            grounding_inference_time_s=round(grounding_inference_time_s, 3)
            if grounding_inference_time_s is not None
            else None,
            chat_latency_s=round(timings["chat_latency_s"], 3)
            if "chat_latency_s" in timings
            else None,
            ttft_s=round(timings["ttft_s"], 3) if "ttft_s" in timings else None,
        )

    async def _validate_state(
        self,
        step: SkillStep,
        screenshot: Path | bytes | None,
        *,
        observation: Observation | None = None,
        mode: ValidStateMode | str | bool | None = None,
    ) -> tuple[bool, dict[str, int], float | None]:
        """Validate per-step valid_state. Returns ``(valid, token_usage, duration_s)``.

        ``mode`` selects the policy (defaults to the executor's global mode):
        ``off`` allows immediately, ``contract_only`` returns the deterministic
        ``state_contract`` verdict and otherwise allows (no NL call), ``strict``
        falls through to small-model NL validation when the contract is silent.

        ``duration_s`` is ``None`` when validation is intentionally skipped
        (no validator configured or state confirmed by a prior done-judgment)
        or rejected before an LLM call. A positive float indicates an actual
        LLM validation call was made.

        Note: ``step.fixed`` controls *action grounding* (use pre-recorded
        coordinates instead of LLM grounding), not state validation. A fixed
        step with a meaningful ``valid_state`` still requires verification.
        """
        resolved_mode = _coerce_valid_state_mode(
            mode if mode is not None else self.valid_state_mode
        )
        if resolved_mode is ValidStateMode.OFF:
            self._last_contract_eval_detail = None
            return True, {}, None

        contract_detail = evaluate_state_contract_detail(
            step.state_contract,
            observation=observation,
        )
        self._last_contract_eval_detail = _contract_eval_detail_to_dict(contract_detail)
        if contract_detail.passed is not None:
            logger.debug(
                "State contract %s for step %s",
                "passed" if contract_detail.passed else "failed",
                step.action_type,
            )
            return bool(contract_detail.passed), {}, None

        if resolved_mode is ValidStateMode.CONTRACT_ONLY:
            logger.debug(
                "contract_only: no contract verdict for step %s; allowing", step.action_type
            )
            self._last_contract_eval_detail = None
            return True, {}, None

        if step.valid_state is None or not step.valid_state.strip():
            logger.debug("Step %s is missing valid_state", step.action_type)
            self._last_contract_eval_detail = None
            return False, {}, None

        if _should_skip_validation(step.valid_state):
            if step.state_contract is not None:
                logger.debug(
                    "State contract could not be evaluated for step %s and no valid_state fallback exists",
                    step.action_type,
                )
                return False, {}, None
            self._last_contract_eval_detail = None
            return True, {}, None

        if self.state_validator is None:
            logger.debug("No state validator; allowing step %s", step.action_type)
            self._last_contract_eval_detail = None
            return True, {}, None
        t0 = time.monotonic()
        try:
            result = await self.state_validator.validate(
                step.valid_state or "",
                screenshot=screenshot,
            )
        except Exception as exc:
            logger.error("State validator error: %s", exc)
            return True, {}, None  # Fail-open
        duration = time.monotonic() - t0
        usage: dict[str, int] = {}
        if hasattr(self.state_validator, "drain_usage"):
            usage = self.state_validator.drain_usage()
        return result, usage, duration

    async def _get_observation_and_screenshot(
        self,
    ) -> tuple[Observation | None, Path | bytes | None]:
        """Capture current observation when the provider supports it."""
        if self.screenshot_provider is not None:
            get_observation = getattr(self.screenshot_provider, "get_observation", None)
            if callable(get_observation):
                try:
                    observation = await get_observation()
                    if observation is not None and observation.screenshot_path:
                        return observation, Path(observation.screenshot_path)
                    return observation, None
                except Exception as exc:
                    logger.warning("Observation retrieval failed: %s", exc)
        return None, await self._get_screenshot()

    async def _get_screenshot(self) -> Path | bytes | None:
        """Capture the current screenshot via the ScreenshotProvider."""
        if self.screenshot_provider is not None:
            try:
                return await self.screenshot_provider.get_screenshot()
            except Exception as exc:
                logger.warning("ScreenshotProvider failed: %s", exc)
        return None


def _serialize_action(action: Action) -> dict[str, Any]:
    payload = dataclasses.asdict(action)
    return {
        key: value
        for key, value in payload.items()
        if value is not None and not (key == "relative" and value is False)
    }


def _serialize_observation(observation: Observation | None) -> dict[str, Any] | None:
    if observation is None:
        return None
    return {
        "screenshot_path": observation.screenshot_path,
        "screen_width": observation.screen_width,
        "screen_height": observation.screen_height,
        "foreground_app": observation.foreground_app,
        "platform": observation.platform,
        "extra": _scrub_trace_value(observation.extra),
    }


def _screenshot_path_string(screenshot: Path | bytes | None) -> str | None:
    return str(screenshot) if isinstance(screenshot, Path) else None


def _contract_eval_detail_to_dict(detail: Any) -> dict[str, Any]:
    return {
        "passed": getattr(detail, "passed", None),
        "score": round(float(getattr(detail, "score", 0.0) or 0.0), 3),
        "evidence_coverage": round(
            float(getattr(detail, "evidence_coverage", 0.0) or 0.0),
            3,
        ),
        "reason": getattr(detail, "reason", None),
        "matched_required": _scrub_trace_value(list(getattr(detail, "matched_required", []) or [])),
        "failed_required": _scrub_trace_value(list(getattr(detail, "failed_required", []) or [])),
        "unknown_required": _scrub_trace_value(list(getattr(detail, "unknown_required", []) or [])),
        "failed_forbidden": _scrub_trace_value(list(getattr(detail, "failed_forbidden", []) or [])),
        "unknown_forbidden": _scrub_trace_value(
            list(getattr(detail, "unknown_forbidden", []) or [])
        ),
    }


def _scrub_trace_value(value: Any) -> Any:
    if isinstance(value, dict):
        scrubbed: dict[str, Any] = {}
        action_type = (
            value.get("action_type") if isinstance(value.get("action_type"), str) else None
        )
        for key, item in value.items():
            if key == "url" and isinstance(item, str) and item.startswith("data:image/"):
                scrubbed[key] = "<omitted:image-data-url>"
            elif action_type == "input_text" and key == "text":
                scrubbed[key] = "<redacted:input_text>"
            else:
                scrubbed[key] = _scrub_trace_value(item)
        return scrubbed
    if isinstance(value, list):
        return [_scrub_trace_value(item) for item in value]
    if isinstance(value, tuple):
        return [_scrub_trace_value(item) for item in value]
    return value


def _merge_usage(target: dict[str, int], source: dict[str, int]) -> None:
    """Merge *source* token counts into *target* in-place, summing each key."""
    for k, v in source.items():
        target[k] = target.get(k, 0) + v


def _merge_optional_duration(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right
