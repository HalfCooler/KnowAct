"""
guiclaw.agent
=============
Core GUI automation agent with a vision-action loop.

``GuiAgent`` orchestrates a multi-step loop: observe the screen, call an LLM
with the screenshot, parse the tool-call response into an ``Action``, execute
it on the backend, and repeat until the task is done or max steps is reached.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from guiclaw.action import Action, ActionError, describe_action, parse_action
from guiclaw.agent_profiles import (
    build_profile_messages,
    canonicalize_agent_profile,
    coordinate_mode_for_profile,
    general_e2e_scale_factor,
    normalize_profile_response_for_observation,
    profile_llm_defaults,
    profile_uses_native_tools,
)
from guiclaw.image_utils import scale_image
from guiclaw.interfaces import (
    DeviceBackend,
    InterventionHandler,
    InterventionRequest,
    LLMProvider,
    LLMResponse,
    ProgressCallback,
    ToolCall,
)
from guiclaw.observation import Observation
from guiclaw.paths import DEFAULT_GUI_RUNS_DIR, DEFAULT_SHORTCUT_CACHE_DIR
from guiclaw.skills.compact_prompt import (
    ALWAYS_ON_SKILL_TAG,
    COMPOSITE_ACTION_DEFINITIONS,
    CompactPromptParts,
    build_compact_prompt_parts,
    composite_action_infos_from_skills,
    is_always_on_skill,
    is_shortcut_skill,
    skill_info_from_flat_skill,
)
from guiclaw.skills.deeplink import AppShortcutProfile
from guiclaw.skills.normalization import (
    find_android_app_in_text,
    normalize_adb_app_identifier,
    normalize_app_identifier,
)
from guiclaw.skills.state_contract import evaluate_state_contract, infer_interaction_target
from guiclaw.tool_schemas import build_computer_use_tool, build_shortcut_tool_defs
from guiclaw.trajectory.recorder import ExecutionPhase, TrajectoryRecorder
from guiclaw.trajectory.summarizer import build_state_note, is_state_note

logger = logging.getLogger(__name__)

_DONE_FAILURE_HINTS: tuple[str, ...] = (
    "fail",
    "failed",
    "failure",
    "unable",
    "cannot",
    "can't",
    "error",
    "not completed",
    "incomplete",
    "失败",
    "无法",
    "不能",
    "错误",
    "未完成",
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepResult:
    """Result of a single vision-action step."""

    action: Action
    tool_call_id: str
    tool_result: str
    assistant_message: dict[str, Any]
    action_summary: str
    action_intent: str | None = None
    state_summary: str | None = None
    next_observation: Observation | None = None
    interaction_target: dict[str, Any] | None = None
    prompt_snapshot: dict[str, Any] | None = None
    model_snapshot: dict[str, Any] | None = None
    execution_snapshot: dict[str, Any] | None = None
    intervention_requested: bool = False
    done: bool = False
    step_usage: dict[str, int] = dataclasses.field(default_factory=dict)
    # Token usage to attribute to this step's own trajectory event. Differs from
    # ``step_usage`` only when the step nested a skill execution (whose tokens are
    # already recorded on its own skill_step/subgoal_step events): ``step_usage``
    # keeps the merged total for run-level accounting, ``event_usage`` keeps just
    # this step's own LLM tokens so the recorder doesn't double-count.
    event_usage: dict[str, int] = dataclasses.field(default_factory=dict)
    duration_s: float = 0.0
    chat_latency_s: float | None = None
    ttft_s: float | None = None


@dataclass(frozen=True)
class HistoryTurn:
    """One completed step kept in the prompt history window."""

    step_index: int
    observation: Observation
    assistant_message: dict[str, Any]
    tool_result_message: dict[str, Any]
    action_summary: str
    action_intent: str | None = None
    state_summary: str | None = None
    raw_response_content: str | None = None


@dataclass(frozen=True)
class AgentResult:
    """Final result of a complete GUI task run (possibly with retries)."""

    success: bool
    summary: str
    model_summary: str | None = None
    trace_path: str | None = None
    steps_taken: int = 0
    error: str | None = None
    token_usage: dict[str, int] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class _ScreenFingerprint:
    """Compact screen signature for loop-stagnation detection."""

    app: str | None
    method: str
    digest: str


class _StepExecutionError(RuntimeError):
    """Runtime error raised from _run_step with optional model snapshot context."""

    def __init__(
        self,
        message: str,
        *,
        model_snapshot: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.model_snapshot = model_snapshot


# ---------------------------------------------------------------------------
# GuiAgent
# ---------------------------------------------------------------------------


_SKILL_PARAM_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_ANDROID_LAUNCHER_PACKAGES = frozenset(
    {
        "com.android.launcher",
        "com.android.launcher3",
        "com.google.android.apps.nexuslauncher",
        "com.miui.home",
    }
)


def _skill_param_names(skill: Any) -> list[str]:
    return [
        str(name).strip()
        for name in (getattr(skill, "parameters", None) or ())
        if str(name).strip()
    ]


def _skill_required_param_names(skill: Any) -> list[str]:
    names = _skill_param_names(skill)
    if not names:
        return []
    placeholders: set[str] = set()
    for step in tuple(getattr(skill, "steps", ()) or ()):
        placeholders.update(_extract_template_placeholders(getattr(step, "target", None)))
        placeholders.update(_extract_template_placeholders(getattr(step, "valid_state", None)))
        placeholders.update(_extract_template_placeholders(getattr(step, "parameters", None)))
        placeholders.update(_extract_template_placeholders(getattr(step, "fixed_values", None)))
        placeholders.update(_extract_template_placeholders(getattr(step, "state_contract", None)))
    required = [name for name in names if name in placeholders]
    return required or names


def _missing_skill_params(skill: Any, params: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for name in _skill_required_param_names(skill):
        value = params.get(name)
        if value is None or not str(value).strip():
            missing.append(name)
    return missing


def _extract_template_placeholders(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {match.group(1) for match in _SKILL_PARAM_PLACEHOLDER_RE.finditer(value)}
    if isinstance(value, dict):
        found: set[str] = set()
        for key, item in value.items():
            found.update(_extract_template_placeholders(key))
            found.update(_extract_template_placeholders(item))
        return found
    if isinstance(value, (list, tuple, set)):
        found: set[str] = set()
        for item in value:
            found.update(_extract_template_placeholders(item))
        return found
    return set()


def _skill_entry_targets_android_launcher(skill: Any, first_step: Any) -> bool:
    platform = str(getattr(skill, "platform", "") or "").strip().lower()
    if platform != "android":
        return False
    candidates: list[str] = [
        str(getattr(skill, "app", "") or ""),
        str(getattr(first_step, "target", "") or ""),
    ]
    fixed_values = getattr(first_step, "fixed_values", None)
    if isinstance(fixed_values, dict):
        for key in ("text", "package", "app", "target"):
            value = fixed_values.get(key)
            if value:
                candidates.append(str(value))
    for candidate in candidates:
        normalized = normalize_app_identifier("android", candidate)
        if normalized in _ANDROID_LAUNCHER_PACKAGES:
            return True
    return False


def _render_skill_template(text: str, params: dict[str, str]) -> str:
    """Substitute ``{{param}}`` placeholders in *text* with provided values."""
    return _SKILL_PARAM_PLACEHOLDER_RE.sub(
        lambda m: str(params.get(m.group(1), m.group(0))), str(text or "")
    )


def _observation_ui_text_blob(observation: Any) -> str:
    """Lowercased concatenation of on-screen text from an observation.

    Pulls visible text / content descriptions from the ``ui_tree`` and any
    ``*_text`` lists in ``observation.extra``. Used for a cheap, deterministic
    "is this control on screen" check without an LLM call.
    """
    extra = getattr(observation, "extra", None)
    if not isinstance(extra, dict):
        return ""
    parts: list[str] = []
    ui_tree = extra.get("ui_tree")
    if isinstance(ui_tree, list):
        for node in ui_tree:
            if isinstance(node, dict):
                for key in ("text", "content_desc"):
                    value = node.get(key)
                    if value:
                        parts.append(str(value))
    for key in ("clickable_text", "visible_text", "content_desc"):
        value = extra.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
        elif value:
            parts.append(str(value))
    return " ".join(parts).lower()


def _prompt_skill_entry_allows(skill: Any, observation: Any, params: dict[str, str]) -> bool:
    """Cheap deterministic entry guard for a prompt-selected ``use_skill``.

    Fail-closed: a skill whose first step cannot be confirmed against the current
    screen is rejected so the agent keeps control instead of having the grounder
    blindly tap a wrong screen. Entry actions (``open_app`` / deeplink / intent)
    are trusted to navigate themselves; otherwise a first-step ``state_contract``
    must match, or — when no contract exists — the rendered first-step target text
    must appear on screen.
    """
    steps = tuple(getattr(skill, "steps", ()) or ())
    if not steps:
        return False
    first = steps[0]
    first_action = str(getattr(first, "action_type", "") or "").strip().lower()
    if first_action == "open_app":
        return not _skill_entry_targets_android_launcher(skill, first)
    if first_action in {"open_deeplink", "open_intent"}:
        return True

    contract = getattr(first, "state_contract", None)
    if contract is not None:
        return evaluate_state_contract(contract, observation=observation) is True

    target = _render_skill_template(str(getattr(first, "target", "") or ""), params).strip().lower()
    if not target:
        return False  # nothing deterministic to check → fail closed
    blob = _observation_ui_text_blob(observation)
    return bool(blob) and target in blob


class GuiAgent:
    """Standalone GUI automation agent with vision-action loop.

    Args:
        llm: LLM provider conforming to :class:`~guiclaw.interfaces.LLMProvider`.
        backend: Device backend conforming to :class:`~guiclaw.interfaces.DeviceBackend`.
        model: Model name string (used for prompt customisation).
        artifacts_root: Root directory for run artifacts (traces, screenshots).
        max_steps: Maximum steps per single attempt.
        step_timeout: Timeout in seconds for each step (LLM + execute + observe).
        history_image_window: Number of recent screenshot turns kept as full
            image context, including the current screen. ``None`` keeps the
            profile default (GUI-Owl: 5; other profiles: 1).
        progress_callback: Optional async callback for progress reporting.
        stagnation_limit: Consecutive unchanged-screen transitions before abort.
    """

    _MAX_TOOL_RETRIES = 3
    _COORDINATE_ACTIONS = frozenset({"tap", "double_tap", "long_press", "swipe", "drag", "scroll"})
    _POST_ACTION_SETTLE_SECONDS = 0.50
    _OPEN_APP_SETTLE_SECONDS = 5.00
    _POST_ACTION_STABILITY_WINDOW_SECONDS = 2.0
    _POST_ACTION_STABILITY_POLL_SECONDS = 0.15
    _POST_ACTION_STABILITY_MAX_ATTEMPTS = 4
    _POST_ACTION_STABILITY_FRAMES_REQUIRED = 2
    _POST_ACTION_OBSERVE_TIMEOUT_SECONDS = 8.0
    _NO_SETTLE_ACTIONS = frozenset({"wait", "done", "request_intervention"})
    _STAGNATION_SSIM_SIZE = 64
    _STAGNATION_SSIM_THRESHOLD = 0.985

    def __init__(
        self,
        llm: LLMProvider,
        backend: DeviceBackend,
        trajectory_recorder: TrajectoryRecorder,
        model: str = "",
        artifacts_root: Path | str = DEFAULT_GUI_RUNS_DIR,
        max_steps: int = 15,
        step_timeout: float = 90.0,
        history_image_window: int | None = None,
        progress_callback: ProgressCallback | None = None,
        memory_retriever: Any = None,
        skill_library: Any = None,
        skill_executor: Any = None,
        memory_top_k: int = 5,
        shortcut_backend: DeviceBackend | None = None,
        shortcut_cache_dir: Path | str | None = DEFAULT_SHORTCUT_CACHE_DIR,
        intervention_handler: InterventionHandler | None = None,
        policy_context: str | None = None,
        agent_profile: str | None = None,
        image_scale_ratio: float = 0.5,
        stagnation_limit: int = 0,
        enable_prompt_skill_selection: bool = False,
        prompt_skill_top_k: int = 5,
        prompt_shortcut_only: bool = False,
        always_on_skill_tags: list[str] | tuple[str, ...] | None = None,
        skill_app_filter_enabled: bool = True,
        reasoning_effort: str | None = None,
    ) -> None:
        self.llm = llm
        self.backend = backend
        self.model = model
        self.agent_profile = canonicalize_agent_profile(agent_profile)
        # Per-step thinking/output budget for the main decision call. Start from
        # the profile's native defaults (e.g. seed -> reasoning_effort="high",
        # max_tokens=4096, matching MobileWorld's SeedAgent), then let an explicit
        # config override win. ``reasoning_effort`` left None/""/"auto" defers to
        # the profile default, which preserves prior behaviour for profiles that
        # declare none.
        _profile_llm = profile_llm_defaults(self.agent_profile)
        self._reasoning_effort = (
            reasoning_effort
            if reasoning_effort not in (None, "", "auto")
            else _profile_llm.get("reasoning_effort")
        )
        self._step_max_tokens = _profile_llm.get("max_tokens")
        self.artifacts_root = Path(artifacts_root)
        self.max_steps = max_steps
        self.step_timeout = step_timeout
        self.history_image_window = (
            None if history_image_window is None else max(1, history_image_window)
        )
        self.progress_callback = progress_callback
        self._trajectory_recorder = trajectory_recorder
        self._memory_retriever = memory_retriever
        self._policy_context = policy_context
        self._skill_library = skill_library
        self._skill_executor = skill_executor
        self._memory_top_k = memory_top_k
        self._shortcuts: dict[str, AppShortcutProfile] = {}
        self._shortcut_backend = shortcut_backend
        self._shortcut_cache_dir = Path(shortcut_cache_dir) if shortcut_cache_dir else None
        self._shortcut_tools: list[dict[str, Any]] = []
        self._shortcut_action_map: dict[str, tuple[str, str, str, str | None]] = {}
        self._intervention_handler = intervention_handler
        self._image_scale_ratio = image_scale_ratio
        self._enable_prompt_skill_selection = bool(enable_prompt_skill_selection)
        try:
            parsed_prompt_skill_top_k = int(prompt_skill_top_k)
        except (TypeError, ValueError):
            parsed_prompt_skill_top_k = 5
        self._prompt_skill_top_k = max(0, parsed_prompt_skill_top_k)
        self._prompt_shortcut_only = bool(prompt_shortcut_only)
        # When False, skip app-based skill-catalog filtering entirely (retrieve
        # over all apps). Off-switch for the foreground/text app detector that can
        # mis-route the catalog (e.g. "phone" -> dialer).
        self._skill_app_filter_enabled = bool(skill_app_filter_enabled)
        self._always_on_skill_tags = tuple(
            str(tag)
            for tag in (
                always_on_skill_tags if always_on_skill_tags is not None else (ALWAYS_ON_SKILL_TAG,)
            )
            if str(tag)
        )
        self._prompt_skills_by_id: dict[str, Any] = {}
        self._prompt_composite_aliases: set[str] = set()
        self._available_apps: tuple[str, ...] | None = None
        try:
            parsed_stagnation_limit = int(stagnation_limit)
        except (TypeError, ValueError):
            parsed_stagnation_limit = 0
        self.stagnation_limit = max(0, parsed_stagnation_limit)

    def _build_tools_list(self) -> list[dict[str, Any]]:
        tools = [
            build_computer_use_tool(
                allow_use_skill=bool(self._prompt_skills_by_id),
                platform=self.backend.platform,
                available_apps=self._available_apps or (),
            )
        ]
        tools.extend(self._shortcut_tools)
        return tools

    async def _ensure_shortcuts_for_app(self, foreground_app: str) -> None:
        """Lazy-load shortcuts when a new foreground app is detected."""
        app = str(foreground_app or "").strip()
        if not app or not self._shortcut_cache_dir or not self._shortcut_backend:
            return
        if (
            str(getattr(self._shortcut_backend, "platform", self.backend.platform)).lower()
            != "android"
        ):
            return
        app = normalize_adb_app_identifier(app)
        if not app or app == "unknown":
            return
        if app in self._shortcuts:
            return

        cache_file = self._shortcut_cache_dir / f"{app}.json"
        try:
            if cache_file.exists():
                profile = AppShortcutProfile.from_dict(
                    json.loads(cache_file.read_text(encoding="utf-8"))
                )
            else:
                from guiclaw.skills.deeplink import extract_app_shortcuts

                profile = await extract_app_shortcuts(self._shortcut_backend, app)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(
                    json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception as exc:
            logger.warning("Shortcut discovery failed for %s: %s", app, exc)
            try:
                self._trajectory_recorder.record_event(
                    "shortcut_discovery_failed",
                    app=app,
                    cache_file=str(cache_file),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as record_exc:
                logger.debug("Could not record shortcut discovery failure: %s", record_exc)
            self._shortcuts[app] = AppShortcutProfile(
                package=app,
                manifest_meta={
                    "status": "shortcut_discovery_failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return

        self._shortcuts[app] = profile
        if profile.deep_links or profile.deep_intents:
            self._shortcut_tools, self._shortcut_action_map = build_shortcut_tool_defs(
                self._shortcuts
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        task: str,
        *,
        max_retries: int = 3,
        app_hint: str | None = None,
    ) -> AgentResult:
        """Run the task with retry logic.

        Returns an :class:`AgentResult` summarising the outcome. On failure
        after all retries, ``success`` is ``False`` and ``error`` contains the
        last error message.
        """
        self._available_apps = None
        # 1. Start trajectory recording
        self._trajectory_recorder.start(phase=ExecutionPhase.AGENT)

        # 2. Retrieve memory context (once)
        memory_context = await self._retrieve_memory(task)
        if self._policy_context:
            self._log_policy_injection(self._policy_context)

        skill_app_filter = self._skill_app_filter(task, app_hint)
        prompt_skill_parts: CompactPromptParts | None = None
        if self._enable_prompt_skill_selection:
            prompt_skill_parts = await self._build_prompt_skill_parts(
                task,
                app=skill_app_filter,
            )

        # 3. Retry loop with free exploration. Skills, when enabled, are exposed
        # in the prompt and must be selected by the GUI model via ``use_skill``.
        last_error: str | None = None
        last_model_summary: str | None = None
        last_trace_path: str | None = None
        last_steps_taken = 0
        result: AgentResult | None = None
        total_usage: dict[str, int] = {}

        for attempt in range(max_retries):
            run_dir = self._make_run_dir(task, attempt)
            last_trace_path = str(run_dir)
            self._trajectory_recorder.set_attempt(attempt + 1)
            if self._skill_executor is not None:
                setter = getattr(type(self._skill_executor), "set_artifacts_root", None)
                if callable(setter):
                    setter(self._skill_executor, run_dir)

            await self._log_attempt_event(
                run_dir,
                "attempt_start",
                attempt=attempt,
                max_retries=max_retries,
                task=task,
            )
            try:
                result = await self._run_once(
                    task,
                    run_dir=run_dir,
                    memory_context=memory_context,
                    prompt_skill_parts=prompt_skill_parts,
                )
                for k, v in result.token_usage.items():
                    total_usage[k] = total_usage.get(k, 0) + v
                await self._log_attempt_event(
                    run_dir,
                    "attempt_result",
                    attempt=attempt,
                    success=result.success,
                    summary=result.summary,
                    model_summary=result.model_summary,
                    error=result.error,
                    steps_taken=result.steps_taken,
                    trace_path=result.trace_path,
                )
                if result.success:
                    last_error = None
                    break
                last_error = result.error
                last_model_summary = result.model_summary
                last_trace_path = result.trace_path or last_trace_path
                last_steps_taken = result.steps_taken
                if result.error and (
                    result.error.startswith("intervention_cancelled")
                    or result.error == "stagnation_detected"
                ):
                    break
                if attempt < max_retries - 1:
                    await self._log_attempt_event(
                        run_dir,
                        "retry",
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        reason=result.error or result.summary,
                    )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                model_snapshot = getattr(exc, "model_snapshot", None)
                await self._log_attempt_event(
                    run_dir,
                    "attempt_exception",
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    model_response=model_snapshot,
                )
                if attempt < max_retries - 1:
                    await self._log_attempt_event(
                        run_dir,
                        "retry",
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        reason=last_error,
                    )
        if result is None:
            status = (
                "blocked"
                if last_error
                and any(
                    keyword in last_error.lower()
                    for keyword in ("stagnation", "intervention", "preflight")
                )
                else "partial"
            )
            result = AgentResult(
                success=False,
                summary=self._build_state_note(
                    status=status,
                    history=[],
                    current_observation=None,
                    error=last_error or f"Failed after {max_retries} attempt(s).",
                ),
                model_summary=last_model_summary,
                trace_path=last_trace_path,
                steps_taken=last_steps_taken,
                error=last_error,
                token_usage=total_usage,
            )
        else:
            result = dataclasses.replace(
                result,
                token_usage=total_usage,
                trace_path=last_trace_path or result.trace_path,
                error=result.error if result.success else last_error or result.error,
            )

        # 6. Finish trajectory
        self._trajectory_recorder.finish(
            success=result.success,
            error=result.error,
            summary=result.summary,
            model_summary=result.model_summary,
        )

        return result

    # ------------------------------------------------------------------
    # Single attempt
    # ------------------------------------------------------------------

    async def _run_once(
        self,
        task: str,
        *,
        run_dir: Path,
        memory_context: str | None = None,
        prompt_skill_parts: CompactPromptParts | None = None,
    ) -> AgentResult:
        """Execute one full attempt of the task."""
        # 1. Preflight
        try:
            await self.backend.preflight()
        except Exception as exc:
            return AgentResult(
                success=False,
                summary=self._build_state_note(
                    status="blocked",
                    history=[],
                    current_observation=None,
                    error=f"Preflight failed: {exc}",
                ),
                trace_path=str(run_dir),
                error=str(exc),
            )

        await self._load_available_apps()

        # 2. Initial observation
        initial_screenshot = run_dir / "screenshots" / "000_initial.png"
        obs = await self.backend.observe(initial_screenshot, timeout=self.step_timeout)
        self._trajectory_recorder.record_screenshot(initial_screenshot, kind="initial")

        history: list[HistoryTurn] = []
        previous_fingerprint: _ScreenFingerprint | None = None
        previous_action_type: str | None = None
        stagnation_streak = 0
        if self.stagnation_limit > 0:
            previous_fingerprint = self._build_screen_fingerprint(obs)

        # 4. Step loop
        steps_taken = 0
        total_usage: dict[str, int] = {}
        for step in range(self.max_steps):
            step_index = step + 1
            messages = self._build_messages(
                task=task,
                current_observation=obs,
                history=history,
                memory_context=memory_context,
                prompt_skill_parts=prompt_skill_parts,
            )
            prompt_snapshot = None

            try:
                result = await asyncio.wait_for(
                    self._run_step(
                        messages=messages,
                        prompt_snapshot=prompt_snapshot,
                        step_index=step_index,
                        total_steps=self.max_steps,
                        current_observation=obs,
                    ),
                    timeout=self.step_timeout * 3,
                )
            except asyncio.TimeoutError:
                await self._log_attempt_event(run_dir, "timeout", step_index=step_index)
                return AgentResult(
                    success=False,
                    summary=self._build_state_note(
                        status="partial",
                        history=history,
                        current_observation=obs,
                        error="step_timeout",
                    ),
                    model_summary=None,
                    trace_path=str(run_dir),
                    steps_taken=step_index,
                    error="step_timeout",
                    token_usage=total_usage,
                )
            except _StepExecutionError as exc:
                raise _StepExecutionError(
                    str(exc),
                    model_snapshot=exc.model_snapshot,
                ) from exc

            steps_taken = step_index
            for k, v in result.step_usage.items():
                total_usage[k] = total_usage.get(k, 0) + v

            intervention_cancelled = False
            if result.intervention_requested:
                request = InterventionRequest(
                    task=task,
                    reason=result.action.text or "",
                    step_index=step_index,
                    platform=self.backend.platform,
                    foreground_app=obs.foreground_app,
                    target=dict(obs.extra),
                )
                await self._log_attempt_event(
                    run_dir,
                    "intervention_requested",
                    step_index=step_index,
                    platform=request.platform,
                    foreground_app=request.foreground_app,
                    reason=request.reason,
                    target=request.target,
                )
                if self._intervention_handler is None:
                    intervention_cancelled = True
                    cancellation_note = "missing_intervention_handler"
                else:
                    resolution = await self._intervention_handler.request_intervention(request)
                    if resolution.resume_confirmed:
                        await self._log_attempt_event(
                            run_dir,
                            "intervention_resumed",
                            step_index=step_index,
                            note=resolution.note,
                        )
                        next_screenshot = self._step_screenshot_path(
                            run_dir,
                            step_index,
                            result.action.action_type,
                        )
                        next_observation = await self.backend.observe(
                            next_screenshot,
                            timeout=self.step_timeout,
                        )
                        result = replace(
                            result,
                            tool_result="intervention_resumed",
                            next_observation=next_observation,
                            execution_snapshot={
                                "tool_result": "intervention_resumed",
                                "intervention": {
                                    "requested": True,
                                    "note": resolution.note,
                                },
                                "next_observation": self._serialize_observation(next_observation),
                                "done": False,
                            },
                        )
                    else:
                        intervention_cancelled = True
                        cancellation_note = resolution.note or "resume_not_confirmed"

                if intervention_cancelled:
                    scrubbed_cancellation_note = self._scrub_sensitive_text(cancellation_note)
                    scrubbed_intervention_summary = (
                        self._scrub_text_for_artifact_action(
                            result.state_summary or result.action_summary,
                            result.action,
                        )
                        or result.state_summary
                        or result.action_summary
                    )
                    await self._log_attempt_event(
                        run_dir,
                        "intervention_cancelled",
                        step_index=step_index,
                        note=scrubbed_cancellation_note,
                    )
                    result = replace(
                        result,
                        tool_result="intervention_cancelled",
                        execution_snapshot={
                            "tool_result": "intervention_cancelled",
                            "intervention": {
                                "requested": True,
                                "note": scrubbed_cancellation_note,
                            },
                            "next_observation": None,
                            "done": False,
                        },
                    )
                    summary_history = history + [
                        self._history_turn_from_step(
                            step_index=step_index,
                            observation=obs,
                            result=result,
                        )
                    ]
                    summary_observation = result.next_observation or obs

            await self._record_completed_step(
                run_dir=run_dir,
                step_index=step_index,
                current_observation=obs,
                result=result,
            )

            if intervention_cancelled:
                termination_summary = await self._generate_termination_summary(
                    task=task,
                    termination_reason=f"Task was interrupted by policy: {scrubbed_cancellation_note}",
                    history=summary_history,
                    run_dir=run_dir,
                )
                result_summary = self._build_state_note(
                    status="blocked",
                    history=summary_history,
                    current_observation=summary_observation,
                    error="intervention_cancelled",
                )
                return AgentResult(
                    success=False,
                    summary=termination_summary or result_summary,
                    model_summary=scrubbed_intervention_summary,
                    trace_path=str(run_dir),
                    steps_taken=steps_taken,
                    error=f"intervention_cancelled: {scrubbed_cancellation_note}",
                    token_usage=total_usage,
                )

            if result.done:
                success = self._resolve_done_status(result.action) == "success"
                return AgentResult(
                    success=success,
                    summary=self._build_state_note(
                        status="completed" if success else "blocked",
                        history=history,
                        current_observation=obs,
                        current_action_summary=result.state_summary or result.action_summary,
                        error=None if success else result.tool_result,
                    ),
                    model_summary=result.state_summary or result.action_summary,
                    trace_path=str(run_dir),
                    steps_taken=steps_taken,
                    error=None if success else result.tool_result,
                    token_usage=total_usage,
                )

            if self.stagnation_limit > 0 and result.next_observation is not None:
                current_fingerprint = self._build_screen_fingerprint(result.next_observation)
                if (
                    previous_fingerprint is not None
                    and current_fingerprint is not None
                    and (
                        previous_action_type is None
                        or previous_action_type == result.action.action_type
                    )
                    and self._is_same_screen(previous_fingerprint, current_fingerprint)
                ):
                    stagnation_streak += 1
                else:
                    stagnation_streak = 0
                previous_fingerprint = current_fingerprint
                previous_action_type = result.action.action_type

                if stagnation_streak >= self.stagnation_limit:
                    app_label = (
                        result.next_observation.foreground_app or obs.foreground_app or "unknown"
                    )
                    history_with_current_step = history + [
                        self._history_turn_from_step(
                            step_index=step_index,
                            observation=obs,
                            result=result,
                        )
                    ]
                    termination_summary = await self._generate_termination_summary(
                        task=task,
                        termination_reason=(
                            "Detected unchanged screen state for "
                            f"{stagnation_streak} consecutive step(s) in app {app_label}; "
                            "task stopped to avoid repeating the same action loop."
                        ),
                        history=history_with_current_step,
                        run_dir=run_dir,
                    )
                    await self._log_attempt_event(
                        run_dir,
                        "stagnation_detected",
                        step_index=step_index,
                        stagnation_streak=stagnation_streak,
                        stagnation_limit=self.stagnation_limit,
                        foreground_app=app_label,
                    )
                    return AgentResult(
                        success=False,
                        summary=termination_summary
                        or self._build_state_note(
                            status="blocked",
                            history=history_with_current_step,
                            current_observation=result.next_observation or obs,
                            error="stagnation_detected",
                        ),
                        model_summary=result.state_summary or result.action_summary,
                        trace_path=str(run_dir),
                        steps_taken=steps_taken,
                        error="stagnation_detected",
                        token_usage=total_usage,
                    )

            history.append(
                self._history_turn_from_step(
                    step_index=step_index,
                    observation=obs,
                    result=result,
                )
            )

            if result.next_observation is not None:
                obs = result.next_observation

        termination_summary = await self._generate_termination_summary(
            task=task,
            termination_reason=f"Reached maximum step limit ({self.max_steps})",
            history=history,
            run_dir=run_dir,
        )
        return AgentResult(
            success=False,
            summary=termination_summary
            or self._build_state_note(
                status="partial",
                history=history,
                current_observation=obs,
                error="max_steps_exceeded",
            ),
            model_summary=None,
            trace_path=str(run_dir),
            steps_taken=steps_taken,
            error="max_steps_exceeded",
            token_usage=total_usage,
        )

    def _history_turn_from_step(
        self,
        *,
        step_index: int,
        observation: Observation,
        result: StepResult,
    ) -> HistoryTurn:
        return HistoryTurn(
            step_index=step_index,
            observation=observation,
            assistant_message=self._scrub_assistant_message_for_log(
                result.assistant_message,
                result.action,
            ),
            tool_result_message={
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": self._scrub_text_for_action(
                    result.tool_result,
                    result.action,
                ),
            },
            action_summary=(
                self._scrub_text_for_action(result.action_summary, result.action)
                or result.action_summary
            ),
            action_intent=(
                self._scrub_text_for_action(result.action_intent, result.action)
                or result.action_intent
            ),
            state_summary=(
                self._scrub_text_for_action(result.state_summary, result.action)
                or result.state_summary
            ),
            raw_response_content=(
                result.model_snapshot.get("raw_content")
                if isinstance(result.model_snapshot, dict)
                else None
            ),
        )

    async def _record_completed_step(
        self,
        *,
        run_dir: Path,
        step_index: int,
        current_observation: Observation,
        result: StepResult,
    ) -> None:
        recorded_observation = result.next_observation or current_observation
        screenshot_path = (
            result.next_observation.screenshot_path
            if result.next_observation and result.next_observation.screenshot_path
            else current_observation.screenshot_path
        )
        model_snapshot = result.model_snapshot if isinstance(result.model_snapshot, dict) else {}
        if model_snapshot:
            model_output: Any = {
                "content": model_snapshot.get("raw_content") or "",
            }
            for field in ("reasoning_content", "thinking_blocks"):
                if model_snapshot.get(field) not in (None, "", []):
                    model_output[field] = model_snapshot[field]
            model_output["tool_calls"] = model_snapshot.get("tool_calls") or []
            if model_snapshot.get("finish_reason") not in (None, ""):
                model_output["finish_reason"] = model_snapshot["finish_reason"]
        else:
            model_output = (
                self._scrub_text_for_artifact_action(
                    result.action_intent or result.action_summary,
                    result.action,
                )
                or ""
            )
        self._trajectory_recorder.record_step(
            action=self._scrub_for_artifact(self._serialize_action(result.action)),
            model_output=model_output,
            screenshot_path=str(screenshot_path) if screenshot_path else None,
            foreground_app=recorded_observation.foreground_app,
            interaction_target=self._scrub_for_artifact(result.interaction_target),
            token_usage=(result.event_usage or result.step_usage) or None,
            inference_time_s=result.chat_latency_s,
        )

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    @staticmethod
    def _finalize_step_result(
        result: StepResult,
        *,
        step_usage: dict[str, int],
        step_start: float,
        step_chat_latency_s: float,
        step_ttft_s: float | None,
    ) -> StepResult:
        return replace(
            result,
            step_usage=step_usage,
            duration_s=time.monotonic() - step_start,
            chat_latency_s=step_chat_latency_s or None,
            ttft_s=step_ttft_s,
        )

    def _skipped_step_result(
        self,
        *,
        reason: str,
        step_usage: dict[str, int],
        step_start: float,
        step_chat_latency_s: float,
        step_ttft_s: float | None,
        prompt_snapshot: dict[str, Any] | None,
        current_observation: Observation,
        model_snapshot: dict[str, Any] | None = None,
    ) -> StepResult:
        """Skip a step whose model output could not be parsed.

        Instead of injecting a "Format error" message and re-querying the model
        (which polluted the context and spiralled into ``request_intervention``),
        record a benign no-op and let the outer loop continue with the next
        observation. The screen is unchanged, so repeated skips are bounded by
        ``stagnation_limit``.
        """
        note = f"Step skipped: {reason}"
        return self._finalize_step_result(
            StepResult(
                action=Action(action_type="wait"),
                tool_call_id="skipped",
                tool_result=note,
                assistant_message={"role": "assistant", "content": ""},
                action_summary=note,
                next_observation=current_observation,
                prompt_snapshot=prompt_snapshot,
                model_snapshot=model_snapshot,
                execution_snapshot={
                    "tool_result": note,
                    "next_observation": None,
                    "done": False,
                },
            ),
            step_usage=step_usage,
            step_start=step_start,
            step_chat_latency_s=step_chat_latency_s,
            step_ttft_s=step_ttft_s,
        )

    async def _run_step(
        self,
        messages: list[dict[str, Any]],
        prompt_snapshot: dict[str, Any] | None,
        step_index: int,
        total_steps: int,
        current_observation: Observation,
    ) -> StepResult:
        """Execute a single vision-action step with retries on malformed calls."""
        _step_start = time.monotonic()
        fg = str(current_observation.foreground_app or "").strip() if current_observation else ""
        await self._ensure_shortcuts_for_app(fg)
        retries_left = self._MAX_TOOL_RETRIES + 1
        step_usage: dict[str, int] = {}
        step_chat_latency_s: float = 0.0
        step_ttft_s: float | None = None

        while retries_left > 0:
            retries_left -= 1

            # Call LLM
            native_tools_enabled = profile_uses_native_tools(self.agent_profile)
            chat_kwargs: dict[str, Any] = {
                "messages": messages,
                "tools": self._build_tools_list() if native_tools_enabled else None,
                "tool_choice": "required" if native_tools_enabled else None,
            }
            if self._reasoning_effort is not None:
                chat_kwargs["reasoning_effort"] = self._reasoning_effort
            if self._step_max_tokens is not None:
                chat_kwargs["max_tokens"] = self._step_max_tokens
            inference_started_at = time.time()
            try:
                response: LLMResponse = await self.llm.chat(**chat_kwargs)
            finally:
                step_chat_latency_s += time.time() - inference_started_at
            for k, v in (response.usage or {}).items():
                step_usage[k] = step_usage.get(k, 0) + v
            if step_ttft_s is None and response.ttft_s is not None:
                step_ttft_s = response.ttft_s
            raw_response_snapshot = self._snapshot_failed_model_response(response)

            try:
                response = normalize_profile_response_for_observation(
                    self.agent_profile,
                    response,
                    current_observation,
                    model_name=self.model,
                    image_scale_ratio=self._image_scale_ratio,
                )
            except ValueError as exc:
                # Unparsable response: re-roll the LLM on the same observation
                # (no feedback injection); skip the step only after exhausting
                # retries, so a transient format glitch does not waste the step.
                if retries_left > 0:
                    continue
                return self._skipped_step_result(
                    reason=f"unparsable response: {exc}",
                    step_usage=step_usage,
                    step_start=_step_start,
                    step_chat_latency_s=step_chat_latency_s,
                    step_ttft_s=step_ttft_s,
                    prompt_snapshot=prompt_snapshot,
                    current_observation=current_observation,
                    model_snapshot=raw_response_snapshot,
                )

            # Append assistant message
            assistant_msg = self._build_assistant_message(
                response,
                include_tool_calls=native_tools_enabled,
            )
            messages.append(assistant_msg)
            assistant_snapshot = self._snapshot_failed_model_response(
                response,
                assistant_message=assistant_msg,
            )

            # Validate tool call
            if not response.tool_calls or len(response.tool_calls) == 0:
                return self._skipped_step_result(
                    reason="no action payload",
                    step_usage=step_usage,
                    step_start=_step_start,
                    step_chat_latency_s=step_chat_latency_s,
                    step_ttft_s=step_ttft_s,
                    prompt_snapshot=prompt_snapshot,
                    current_observation=current_observation,
                    model_snapshot=assistant_snapshot,
                )

            tool_call = response.tool_calls[0]
            if (
                native_tools_enabled
                and tool_call.name != "computer_use"
                and tool_call.name not in self._shortcut_action_map
            ):
                if retries_left > 0:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: unexpected tool '{tool_call.name}'.",
                        }
                    )
                    continue
                raise _StepExecutionError(
                    f"LLM called unexpected tool '{tool_call.name}'.",
                    model_snapshot=assistant_snapshot,
                )
            if tool_call.name in self._shortcut_action_map:
                action_type, text, component, mime_type = self._shortcut_action_map[tool_call.name]
                arguments = {
                    "action_type": action_type,
                    "text": text,
                    "component": component,
                    "intent": f"Shortcut: {tool_call.name}",
                    "summary": f"Executing shortcut {tool_call.name}",
                }
                if action_type == "open_intent":
                    arguments["intent_action"] = text
                    if mime_type:
                        arguments["mime_type"] = mime_type
                tool_call = replace(tool_call, arguments=arguments)
            action_intent, state_summary = self._tool_call_semantics(tool_call)

            special_action_type = (
                str(
                    (tool_call.arguments or {}).get("action_type")
                    or (tool_call.arguments or {}).get("action")
                    or ""
                )
                .strip()
                .lower()
            )
            if (
                special_action_type == "use_skill"
                or special_action_type in self._prompt_composite_aliases
            ):
                try:
                    return await self._dispatch_prompt_special_action(
                        tool_call=tool_call,
                        response=response,
                        prompt_snapshot=prompt_snapshot,
                        current_observation=current_observation,
                        step_index=step_index,
                        step_usage=step_usage,
                        step_start=_step_start,
                        step_chat_latency_s=step_chat_latency_s,
                        step_ttft_s=step_ttft_s,
                        action_intent=action_intent,
                        state_summary=state_summary,
                    )
                except ActionError as exc:
                    return self._skipped_step_result(
                        reason=f"invalid skill/composite action: {exc}",
                        step_usage=step_usage,
                        step_start=_step_start,
                        step_chat_latency_s=step_chat_latency_s,
                        step_ttft_s=step_ttft_s,
                        prompt_snapshot=prompt_snapshot,
                        current_observation=current_observation,
                        model_snapshot=assistant_snapshot,
                    )

            # Parse action
            try:
                action = parse_action(tool_call.arguments)
                action = self._normalize_relative_coordinates(action)
            except ActionError as exc:
                return self._skipped_step_result(
                    reason=f"unparsable action: {exc}",
                    step_usage=step_usage,
                    step_start=_step_start,
                    step_chat_latency_s=step_chat_latency_s,
                    step_ttft_s=step_ttft_s,
                    prompt_snapshot=prompt_snapshot,
                    current_observation=current_observation,
                    model_snapshot=assistant_snapshot,
                )

            # Report progress
            if self.progress_callback is not None:
                await self.progress_callback(
                    f"GUI step {step_index}/{total_steps}: {describe_action(action)}"
                )

            action_text = self._normalize_action_text(
                response.content,
                action,
                tool_summary=action_intent or state_summary,
            )
            action_summary = action_intent or self._action_summary(action_text)
            assistant_message = self._build_assistant_message(
                response,
                content_override=action_text,
                include_tool_calls=native_tools_enabled,
            )
            model_snapshot = self._snapshot_model_response(
                response=response,
                action=action,
                assistant_message=assistant_message,
                action_text=action_text,
                action_intent=action_summary,
                state_summary=state_summary,
            )

            # Handle terminal action (done)
            if action.action_type == "done":
                done_status = self._resolve_done_status(action)
                if action.status != done_status:
                    action = replace(action, status=done_status)
                tool_result = f"Task terminated with status: {done_status}"
                return self._finalize_step_result(
                    StepResult(
                        action=action,
                        tool_call_id=tool_call.id,
                        tool_result=tool_result,
                        assistant_message=assistant_message,
                        action_summary=action_summary,
                        action_intent=action_summary,
                        state_summary=state_summary,
                        prompt_snapshot=prompt_snapshot,
                        model_snapshot=model_snapshot,
                        execution_snapshot={
                            "tool_result": tool_result,
                            "next_observation": None,
                            "done": True,
                        },
                        done=True,
                    ),
                    step_usage=step_usage,
                    step_start=_step_start,
                    step_chat_latency_s=step_chat_latency_s,
                    step_ttft_s=step_ttft_s,
                )

            if action.action_type == "request_intervention":
                return self._finalize_step_result(
                    StepResult(
                        action=action,
                        tool_call_id=tool_call.id,
                        tool_result="intervention_requested",
                        assistant_message=assistant_message,
                        action_summary=action_summary,
                        action_intent=action_summary,
                        state_summary=state_summary,
                        prompt_snapshot=prompt_snapshot,
                        model_snapshot=model_snapshot,
                        execution_snapshot={
                            "tool_result": "intervention_requested",
                            "next_observation": None,
                            "done": False,
                        },
                        intervention_requested=True,
                    ),
                    step_usage=step_usage,
                    step_start=_step_start,
                    step_chat_latency_s=step_chat_latency_s,
                    step_ttft_s=step_ttft_s,
                )

            # Normalize app identifiers for mobile open/close actions.
            if (
                action.action_type in ("open_app", "close_app")
                and action.text
                and self.backend.platform in ("android", "ios")
            ):
                resolved = self._normalize_backend_app_identifier(action.text)
                if resolved != action.text:
                    logger.debug(
                        "Resolved %s app name %r -> %r",
                        self.backend.platform,
                        action.text,
                        resolved,
                    )
                    action = replace(action, text=resolved)

            interaction_target = infer_interaction_target(action, current_observation)

            # Execute action on backend
            try:
                result_text = await self.backend.execute(action, timeout=self.step_timeout)
            except Exception as exc:
                result_text = f"Action failed: {exc}"

            settle_seconds = self._post_action_settle_seconds(action)
            if settle_seconds > 0:
                await asyncio.sleep(settle_seconds)

            # Observe next state
            run_dir = Path(current_observation.screenshot_path or ".").parent.parent
            next_screenshot = self._step_screenshot_path(
                run_dir,
                step_index,
                action.action_type,
            )
            next_observation = await self._observe_after_action(
                next_screenshot,
                previous_observation=current_observation,
                action=action,
                timeout=self.step_timeout,
            )

            return self._finalize_step_result(
                StepResult(
                    action=action,
                    tool_call_id=tool_call.id,
                    tool_result=result_text,
                    assistant_message=assistant_message,
                    action_summary=action_summary,
                    action_intent=action_summary,
                    state_summary=state_summary,
                    next_observation=next_observation,
                    interaction_target=interaction_target,
                    prompt_snapshot=prompt_snapshot,
                    model_snapshot=model_snapshot,
                    execution_snapshot={
                        "tool_result": self._scrub_text_for_action(result_text, action),
                        "next_observation": self._serialize_observation(next_observation),
                        "done": False,
                    },
                ),
                step_usage=step_usage,
                step_start=_step_start,
                step_chat_latency_s=step_chat_latency_s,
                step_ttft_s=step_ttft_s,
            )

        raise RuntimeError("GUI model did not return a valid computer_use call after retries.")

    async def _dispatch_prompt_special_action(
        self,
        *,
        tool_call: ToolCall,
        response: LLMResponse,
        prompt_snapshot: dict[str, Any] | None,
        current_observation: Observation,
        step_index: int,
        step_usage: dict[str, int],
        step_start: float,
        step_chat_latency_s: float,
        step_ttft_s: float | None,
        action_intent: str | None,
        state_summary: str | None,
    ) -> StepResult:
        arguments = tool_call.arguments or {}
        action_type = (
            str(arguments.get("action_type") or arguments.get("action") or "").strip().lower()
        )
        if action_type == "use_skill":
            return await self._execute_prompt_skill_action(
                tool_call=tool_call,
                response=response,
                prompt_snapshot=prompt_snapshot,
                current_observation=current_observation,
                step_index=step_index,
                step_usage=step_usage,
                step_start=step_start,
                step_chat_latency_s=step_chat_latency_s,
                step_ttft_s=step_ttft_s,
                action_intent=action_intent,
                state_summary=state_summary,
            )
        if action_type in self._prompt_composite_aliases:
            return await self._execute_prompt_composite_action(
                tool_call=tool_call,
                response=response,
                prompt_snapshot=prompt_snapshot,
                current_observation=current_observation,
                step_index=step_index,
                step_usage=step_usage,
                step_start=step_start,
                step_chat_latency_s=step_chat_latency_s,
                step_ttft_s=step_ttft_s,
                action_intent=action_intent,
                state_summary=state_summary,
            )
        raise ActionError(f"Unknown prompt special action {action_type!r}.")

    async def _execute_prompt_skill_action(
        self,
        *,
        tool_call: ToolCall,
        response: LLMResponse,
        prompt_snapshot: dict[str, Any] | None,
        current_observation: Observation,
        step_index: int,
        step_usage: dict[str, int],
        step_start: float,
        step_chat_latency_s: float,
        step_ttft_s: float | None,
        action_intent: str | None,
        state_summary: str | None,
    ) -> StepResult:
        arguments = tool_call.arguments or {}
        skill_id = str(arguments.get("skill_id") or "").strip()
        if not skill_id:
            raise ActionError("use_skill requires a listed 'skill_id'.")
        skill = self._prompt_skills_by_id.get(skill_id)
        if skill is None:
            raise ActionError(f"use_skill skill_id {skill_id!r} is not in the prompt skill list.")
        if self._skill_executor is None:
            raise ActionError("use_skill requires a configured SkillExecutor.")

        raw_params = arguments.get("arguments") or arguments.get("params") or {}
        if not isinstance(raw_params, dict):
            raise ActionError("use_skill 'arguments' must be an object.")
        params = {
            str(key): self._stringify_skill_argument(value) for key, value in raw_params.items()
        }
        action = Action(action_type="use_skill", text=skill_id)
        skill_name = str(getattr(skill, "name", "") or skill_id)
        summary = f"use_skill {skill_name}"
        assistant_message = self._build_assistant_message(
            response,
            content_override=self._normalize_action_text(
                response.content,
                action,
                tool_summary=action_intent or state_summary or summary,
            ),
            include_tool_calls=profile_uses_native_tools(self.agent_profile),
        )

        self._trajectory_recorder.record_event(
            "prompt_skill_selected",
            skill_id=skill_id,
            skill_name=skill_name,
            arguments=params,
        )
        self._trajectory_recorder.set_phase(
            ExecutionPhase.SKILL,
            reason=f"Prompt-selected skill: {skill_name}",
        )
        try:
            try:
                missing_params = _missing_skill_params(skill, params)
                if missing_params:
                    self._trajectory_recorder.record_event(
                        "prompt_skill_rejected",
                        skill_id=skill_id,
                        skill_name=skill_name,
                        reason="missing_required_params",
                        missing_params=missing_params,
                    )
                    raise ActionError(
                        f"use_skill {skill_name} is missing required params: "
                        f"{', '.join(missing_params)}"
                    )
                if not _prompt_skill_entry_allows(skill, current_observation, params):
                    self._trajectory_recorder.record_event(
                        "prompt_skill_rejected",
                        skill_id=skill_id,
                        skill_name=skill_name,
                        reason="entry_precondition_not_met",
                        first_action=str(
                            getattr(skill.steps[0], "action_type", "") if skill.steps else ""
                        ),
                    )
                    raise ActionError(
                        f"use_skill {skill_name} is not applicable on the current screen; "
                        "use manual GUI actions instead"
                    )
                skill_result = await self._skill_executor.execute(
                    skill,
                    params=params,
                    timeout=self.step_timeout,
                )
            except Exception as exc:
                return await self._prompt_skill_exception_result(
                    exc,
                    skill_id=skill_id,
                    skill_name=skill_name,
                    action=action,
                    tool_call=tool_call,
                    response=response,
                    assistant_message=assistant_message,
                    prompt_snapshot=prompt_snapshot,
                    current_observation=current_observation,
                    step_index=step_index,
                    step_usage=step_usage,
                    step_start=step_start,
                    step_chat_latency_s=step_chat_latency_s,
                    step_ttft_s=step_ttft_s,
                    action_intent=action_intent,
                )
        finally:
            self._trajectory_recorder.set_phase(
                ExecutionPhase.AGENT,
                reason="Prompt-selected skill finished",
            )

        merged_usage = dict(step_usage)
        raw_skill_usage = getattr(skill_result, "token_usage", None)
        if isinstance(raw_skill_usage, dict):
            for key, value in raw_skill_usage.items():
                if isinstance(value, int):
                    merged_usage[key] = merged_usage.get(key, 0) + value

        next_observation = self._observation_from_skill_result(skill_result)
        if next_observation is None:
            run_dir = Path(current_observation.screenshot_path or ".").parent.parent
            next_screenshot = self._step_screenshot_path(
                run_dir,
                step_index,
                action.action_type,
            )
            next_observation = await self.backend.observe(
                next_screenshot, timeout=self.step_timeout
            )

        succeeded = getattr(getattr(skill_result, "state", None), "value", "") == "succeeded"
        result_text = getattr(skill_result, "execution_summary", "") or ""
        if not succeeded and getattr(skill_result, "error", None):
            result_text = f"{result_text}\nError: {skill_result.error}".strip()
        action_summary = f"{summary}: {'succeeded' if succeeded else 'failed'}"
        rich_action_intent = (response.content or "").strip() or action_intent or action_summary
        model_snapshot = self._snapshot_model_response(
            response=response,
            action=action,
            assistant_message=assistant_message,
            action_text=action_summary,
            action_intent=rich_action_intent,
            state_summary=result_text,
        )
        return self._finalize_step_result(
            StepResult(
                action=action,
                tool_call_id=tool_call.id,
                tool_result=result_text,
                assistant_message=assistant_message,
                action_summary=action_summary,
                action_intent=rich_action_intent,
                state_summary=result_text,
                next_observation=next_observation,
                prompt_snapshot=prompt_snapshot,
                model_snapshot=model_snapshot,
                execution_snapshot={
                    "tool_result": self._scrub_text_for_action(result_text, action),
                    "skill": {
                        "skill_id": skill_id,
                        "skill_name": skill_name,
                        "state": getattr(getattr(skill_result, "state", None), "value", None),
                        "error": getattr(skill_result, "error", None),
                    },
                    "next_observation": self._serialize_observation(next_observation),
                    "done": False,
                },
                event_usage=dict(step_usage),
            ),
            step_usage=merged_usage,
            step_start=step_start,
            step_chat_latency_s=step_chat_latency_s,
            step_ttft_s=step_ttft_s,
        )

    async def _prompt_skill_exception_result(
        self,
        exc: Exception,
        *,
        skill_id: str,
        skill_name: str,
        action: Action,
        tool_call: ToolCall,
        response: LLMResponse,
        assistant_message: dict[str, Any],
        prompt_snapshot: dict[str, Any] | None,
        current_observation: Observation,
        step_index: int,
        step_usage: dict[str, int],
        step_start: float,
        step_chat_latency_s: float,
        step_ttft_s: float | None,
        action_intent: str | None,
    ) -> StepResult:
        run_dir = Path(current_observation.screenshot_path or ".").parent.parent
        next_screenshot = self._step_screenshot_path(
            run_dir,
            step_index,
            action.action_type,
        )
        next_observation = await self.backend.observe(next_screenshot, timeout=self.step_timeout)
        result_text = f"Prompt-selected skill failed with {type(exc).__name__}: {exc}"
        action_summary = f"use_skill {skill_name}: failed"
        rich_action_intent = (response.content or "").strip() or action_intent or action_summary
        model_snapshot = self._snapshot_model_response(
            response=response,
            action=action,
            assistant_message=assistant_message,
            action_text=action_summary,
            action_intent=rich_action_intent,
            state_summary=result_text,
        )
        return self._finalize_step_result(
            StepResult(
                action=action,
                tool_call_id=tool_call.id,
                tool_result=result_text,
                assistant_message=assistant_message,
                action_summary=action_summary,
                action_intent=rich_action_intent,
                state_summary=result_text,
                next_observation=next_observation,
                prompt_snapshot=prompt_snapshot,
                model_snapshot=model_snapshot,
                execution_snapshot={
                    "tool_result": self._scrub_text_for_action(result_text, action),
                    "skill": {
                        "skill_id": skill_id,
                        "skill_name": skill_name,
                        "state": "failed",
                        "error": str(exc),
                    },
                    "next_observation": self._serialize_observation(next_observation),
                    "done": False,
                },
            ),
            step_usage=step_usage,
            step_start=step_start,
            step_chat_latency_s=step_chat_latency_s,
            step_ttft_s=step_ttft_s,
        )

    async def _execute_prompt_composite_action(
        self,
        *,
        tool_call: ToolCall,
        response: LLMResponse,
        prompt_snapshot: dict[str, Any] | None,
        current_observation: Observation,
        step_index: int,
        step_usage: dict[str, int],
        step_start: float,
        step_chat_latency_s: float,
        step_ttft_s: float | None,
        action_intent: str | None,
        state_summary: str | None,
    ) -> StepResult:
        arguments = tool_call.arguments or {}
        alias = str(arguments.get("action_type") or arguments.get("action") or "").strip().lower()
        if alias not in self._prompt_composite_aliases:
            raise ActionError(f"Composite action {alias!r} is not listed in the prompt.")
        if alias not in COMPOSITE_ACTION_DEFINITIONS:
            raise ActionError(f"Composite action {alias!r} has no executor.")

        action = Action(action_type=alias, text=self._composite_action_text(arguments))
        assistant_message = self._build_assistant_message(
            response,
            content_override=self._normalize_action_text(
                response.content,
                action,
                tool_summary=action_intent or state_summary or describe_action(action),
            ),
            include_tool_calls=profile_uses_native_tools(self.agent_profile),
        )
        executed_actions = self._build_composite_actions(alias, arguments, current_observation)
        result_lines: list[str] = []
        for index, inner_action in enumerate(executed_actions):
            if index > 0:
                await asyncio.sleep(self._POST_ACTION_SETTLE_SECONDS)
            try:
                result_lines.append(
                    await self.backend.execute(inner_action, timeout=self.step_timeout)
                )
            except Exception as exc:
                result_lines.append(f"Action failed: {exc}")
                break

        await asyncio.sleep(self._POST_ACTION_SETTLE_SECONDS)
        run_dir = Path(current_observation.screenshot_path or ".").parent.parent
        next_screenshot = self._step_screenshot_path(
            run_dir,
            step_index,
            action.action_type,
        )
        next_observation = await self._observe_after_action(
            next_screenshot,
            previous_observation=current_observation,
            action=action,
            timeout=self.step_timeout,
        )
        result_text = "\n".join(result_lines)
        action_summary = f"{alias}: {self._composite_action_summary(alias, arguments)}"
        rich_action_intent = (response.content or "").strip() or action_intent or action_summary
        model_snapshot = self._snapshot_model_response(
            response=response,
            action=action,
            assistant_message=assistant_message,
            action_text=action_summary,
            action_intent=rich_action_intent,
            state_summary=state_summary,
        )
        return self._finalize_step_result(
            StepResult(
                action=action,
                tool_call_id=tool_call.id,
                tool_result=result_text,
                assistant_message=assistant_message,
                action_summary=action_summary,
                action_intent=rich_action_intent,
                state_summary=state_summary,
                next_observation=next_observation,
                interaction_target={"composite_action": alias},
                prompt_snapshot=prompt_snapshot,
                model_snapshot=model_snapshot,
                execution_snapshot={
                    "tool_result": self._scrub_text_for_action(result_text, action),
                    "inner_actions": [
                        self._serialize_action(inner_action) for inner_action in executed_actions
                    ],
                    "next_observation": self._serialize_observation(next_observation),
                    "done": False,
                },
            ),
            step_usage=step_usage,
            step_start=step_start,
            step_chat_latency_s=step_chat_latency_s,
            step_ttft_s=step_ttft_s,
        )

    def _build_composite_actions(
        self,
        alias: str,
        arguments: dict[str, Any],
        observation: Observation,
    ) -> list[Action]:
        if alias == "click_then_type":
            x, y = self._point_from_payload(arguments, observation)
            text = str(arguments.get("text") or "")
            if not text:
                raise ActionError(f"{alias} requires 'text'.")
            return [
                Action(action_type="tap", x=x, y=y),
                Action(
                    action_type="input_text",
                    text=text,
                    auto_enter=self._bool_argument(arguments.get("auto_enter"), False),
                ),
            ]
        if alias == "click_multi":
            points = arguments.get("coordinates") or arguments.get("points") or []
            if not isinstance(points, list) or not points:
                single = arguments.get("coordinate")
                points = [single] if single is not None else []
            actions = [
                Action(action_type="tap", x=x, y=y)
                for x, y in (self._point_to_screen(point, observation) for point in points)
            ]
            if not actions:
                raise ActionError("click_multi requires non-empty 'coordinates'.")
            return actions
        raise ActionError(f"Unsupported composite action {alias!r}.")

    def _point_from_payload(
        self,
        payload: dict[str, Any],
        observation: Observation,
    ) -> tuple[int, int]:
        if "coordinate" in payload:
            return self._point_to_screen(payload["coordinate"], observation)
        if "x" in payload and "y" in payload:
            return self._point_to_screen([payload["x"], payload["y"]], observation)
        raise ActionError("Composite action requires 'coordinate' or 'x'/'y'.")

    def _point_to_screen(self, point: Any, observation: Observation) -> tuple[int, int]:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ActionError(f"Invalid coordinate {point!r}.")
        try:
            raw_x = float(point[0])
            raw_y = float(point[1])
        except (TypeError, ValueError) as exc:
            raise ActionError(f"Invalid coordinate {point!r}.") from exc
        width = int(observation.screen_width or 1000)
        height = int(observation.screen_height or 1000)
        # Scale identically to the normal-action coordinate path, per profile/model:
        # kimi-k -> [0,1]; qwen/default -> [0,1000]; claude/opus -> image dims.
        # (Previously hard-coded /999, which mis-scaled kimi's [0,1] coords to ~(1,1).)
        scale = general_e2e_scale_factor(self.model, width, height)
        scale_x, scale_y = (scale, scale) if isinstance(scale, int) else scale
        px = int(round(raw_x * width / scale_x))
        py = int(round(raw_y * height / scale_y))
        return max(0, min(px, width - 1)), max(0, min(py, height - 1))

    @staticmethod
    def _composite_action_text(arguments: dict[str, Any]) -> str:
        text = arguments.get("text")
        if isinstance(text, str) and text:
            return text
        return str(arguments.get("coordinates") or arguments.get("coordinate") or "")

    @staticmethod
    def _composite_action_summary(alias: str, arguments: dict[str, Any]) -> str:
        if alias == "click_then_type":
            return "tap target and type text"
        if alias == "click_multi":
            points = arguments.get("coordinates") or arguments.get("points") or []
            return f"tap {len(points) if isinstance(points, list) else 1} target(s)"
        return alias

    @staticmethod
    def _bool_argument(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)

    @staticmethod
    def _stringify_skill_argument(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)) or value is None:
            return "" if value is None else str(value)
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _observation_from_skill_result(skill_result: Any) -> Observation | None:
        for step_result in reversed(getattr(skill_result, "step_results", ()) or ()):
            raw = getattr(step_result, "observation", None)
            if not isinstance(raw, dict):
                continue
            screenshot_path = raw.get("screenshot_path")
            if not screenshot_path:
                continue
            return Observation(
                screenshot_path=str(screenshot_path),
                screen_width=int(raw.get("screen_width") or 1000),
                screen_height=int(raw.get("screen_height") or 1000),
                foreground_app=raw.get("foreground_app"),
                platform=raw.get("platform") or "unknown",
                extra=raw.get("extra") if isinstance(raw.get("extra"), dict) else {},
            )
        return None

    def _coordinate_mode(self) -> str:
        return coordinate_mode_for_profile(self.agent_profile, self.model)

    def _model_uses_relative_grid(self) -> bool:
        return self._coordinate_mode() == "relative_999"

    def _normalize_relative_coordinates(self, action: Action) -> Action:
        if action.relative or action.action_type not in self._COORDINATE_ACTIONS:
            return action
        if not self._model_uses_relative_grid():
            return action
        coords = [
            value for value in (action.x, action.y, action.x2, action.y2) if value is not None
        ]
        if coords and all(0 <= value <= 999 for value in coords):
            return replace(action, relative=True)
        return action

    def _post_action_settle_seconds(self, action: Action) -> float:
        if action.action_type in self._NO_SETTLE_ACTIONS:
            return 0.0
        if action.action_type == "open_app":
            return self._OPEN_APP_SETTLE_SECONDS
        return self._POST_ACTION_SETTLE_SECONDS

    async def _observe_after_action(
        self,
        screenshot_path: Path,
        *,
        previous_observation: Observation | None = None,
        action: Action | None = None,
        timeout: float,
    ) -> Observation | None:
        # Wait for the UI to reach a short stable state before returning the
        # observation used by the next planner turn.
        max_attempts = self._POST_ACTION_STABILITY_MAX_ATTEMPTS
        if action is not None and action.action_type in self._NO_SETTLE_ACTIONS:
            max_attempts = 1
        if action is None or self._post_action_settle_seconds(action) <= 0:
            max_attempts = min(max_attempts, 1)

        window_seconds = min(timeout, self._POST_ACTION_STABILITY_WINDOW_SECONDS)
        poll_interval = self._POST_ACTION_STABILITY_POLL_SECONDS
        observe_timeout = min(timeout, self._POST_ACTION_OBSERVE_TIMEOUT_SECONDS)
        attempts_within_window = max(
            1,
            int(window_seconds / max(poll_interval, 1e-6)),
        )
        max_attempts = max(1, min(max_attempts, attempts_within_window))

        previous_fingerprint = (
            self._build_screen_fingerprint(previous_observation)
            if previous_observation is not None
            else None
        )
        stable_count = 0
        stable_required = self._POST_ACTION_STABILITY_FRAMES_REQUIRED
        last_observation: Observation | None = None
        deadline = time.monotonic() + window_seconds

        for attempt in range(max_attempts):
            try:
                observation = await self.backend.observe(
                    screenshot_path,
                    timeout=observe_timeout,
                )
                last_observation = observation
                current_fingerprint = self._build_screen_fingerprint(observation)

                if previous_fingerprint is not None and current_fingerprint is not None:
                    if self._is_same_screen(previous_fingerprint, current_fingerprint):
                        stable_count += 1
                    else:
                        stable_count = 1
                        previous_fingerprint = current_fingerprint
                else:
                    # Fingerprints are best-effort. If unavailable, just use the
                    # latest sampled observation as the fallback signal.
                    stable_count = min(stable_count + 1, stable_required)

                if stable_count >= stable_required:
                    return observation
            except Exception:
                pass

            if max_attempts > 1 and time.monotonic() < deadline:
                await asyncio.sleep(poll_interval)

        if last_observation is not None:
            return last_observation
        return None

    def _build_screen_fingerprint(self, observation: Observation) -> _ScreenFingerprint | None:
        screenshot = observation.screenshot_path
        if not screenshot:
            return None
        screenshot_path = Path(screenshot)
        if not screenshot_path.exists():
            return None

        app_name = self._normalize_stagnation_app(observation.foreground_app)
        try:
            data = screenshot_path.read_bytes()
        except OSError:
            return None

        try:
            from PIL import Image

            with Image.open(screenshot_path) as img:
                resampling = getattr(Image, "Resampling", Image)
                ssim_size = self._STAGNATION_SSIM_SIZE
                grayscale = img.convert("L").resize(
                    (ssim_size, ssim_size),
                    resampling.BILINEAR,
                )
                pixels = list(grayscale.tobytes())
                if pixels:
                    return _ScreenFingerprint(
                        app=app_name,
                        method="ssim",
                        digest=base64.b64encode(bytes(pixels)).decode("ascii"),
                    )
        except Exception:
            pass

        return _ScreenFingerprint(
            app=app_name,
            method="sha256",
            digest=hashlib.sha256(data).hexdigest(),
        )

    @classmethod
    def _is_same_screen(
        cls,
        previous: _ScreenFingerprint,
        current: _ScreenFingerprint,
    ) -> bool:
        if previous.app and current.app and previous.app != current.app:
            return False

        if previous.method == "ssim" and current.method == "ssim":
            try:
                return cls._ssim_is_similar(previous.digest, current.digest)
            except Exception:
                return previous.digest == current.digest

        if previous.method != current.method:
            return False
        return previous.digest == current.digest

    @classmethod
    def _ssim_is_similar(cls, previous_digest: str, current_digest: str) -> bool:
        previous_pixels = base64.b64decode(previous_digest)
        current_pixels = base64.b64decode(current_digest)
        if len(previous_pixels) != len(current_pixels) or len(previous_pixels) == 0:
            return False
        return cls._ssim_score(previous_pixels, current_pixels) >= cls._STAGNATION_SSIM_THRESHOLD

    @classmethod
    def _ssim_score(cls, previous_pixels: bytes, current_pixels: bytes) -> float:
        del cls

        if len(previous_pixels) != len(current_pixels) or len(previous_pixels) == 0:
            return 0.0

        n = len(previous_pixels)
        previous_values = [value for value in previous_pixels]
        current_values = [value for value in current_pixels]

        previous_mean = sum(previous_values) / n
        current_mean = sum(current_values) / n

        previous_variance = sum((value - previous_mean) ** 2 for value in previous_values) / n
        current_variance = sum((value - current_mean) ** 2 for value in current_values) / n
        covariance = (
            sum(
                (previous_value - previous_mean) * (current_value - current_mean)
                for previous_value, current_value in zip(previous_values, current_values)
            )
            / n
        )

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        denominator = (previous_mean * previous_mean + current_mean * current_mean + c1) * (
            previous_variance + current_variance + c2
        )

        if denominator == 0:
            return 1.0 if previous_mean == current_mean else 0.0

        numerator = (2 * previous_mean * current_mean + c1) * (2 * covariance + c2)
        return numerator / denominator

    def _normalize_stagnation_app(self, app: str | None) -> str | None:
        if not app:
            return None
        normalized = self._normalize_backend_app_identifier(app)
        return None if normalized == "unknown" else normalized

    def _normalize_backend_app_identifier(self, app: str) -> str:
        if self.backend.platform == "android" and hasattr(self.backend, "_run"):
            return normalize_adb_app_identifier(app)
        return normalize_app_identifier(self.backend.platform, app)

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        *,
        task: str,
        current_observation: Observation,
        history: list[HistoryTurn],
        memory_context: str | None = None,
        prompt_skill_parts: CompactPromptParts | None = None,
    ) -> list[dict[str, Any]]:
        task_context: list[str] = [task]
        if self._policy_context:
            task_context.extend(
                [
                    "",
                    "Advisory Policy Hints (guidance only; not guaranteed enforcement):",
                    self._policy_context,
                ]
            )
        if memory_context:
            task_context.extend(["", "Relevant Knowledge:", memory_context])
        return build_profile_messages(
            self.agent_profile,
            task="\n".join(task_context),
            current_observation=current_observation,
            history=history,
            model_name=self.model,
            history_image_window=self.history_image_window,
            compact_prompt_parts=prompt_skill_parts,
            available_apps=self._available_apps or (),
            image_scale_ratio=self._image_scale_ratio,
        )

    async def _load_available_apps(self) -> None:
        if self._available_apps is not None:
            return
        if str(self.backend.platform).lower() not in {"macos", "windows"}:
            self._available_apps = ()
            return
        try:
            apps = await self.backend.list_apps()
        except Exception as exc:
            logger.warning("Unable to list %s applications: %s", self.backend.platform, exc)
            apps = []
        self._available_apps = tuple(
            sorted(
                {" ".join(str(name).split()) for name in apps if str(name).strip()},
                key=str.casefold,
            )
        )

    @staticmethod
    def _build_state_note(
        *,
        status: str,
        history: list[HistoryTurn],
        current_observation: Observation | None,
        current_action_summary: str | None = None,
        error: str | None = None,
    ) -> str:
        return build_state_note(
            status=status,
            done=GuiAgent._summarize_progress(history, current_action_summary),
            remaining=GuiAgent._remaining_hint(status=status, error=error),
            current=GuiAgent._describe_observation_state(current_observation),
            resume=GuiAgent._resume_hint(status=status, error=error),
        )

    @staticmethod
    def _summarize_progress(
        history: list[HistoryTurn], current_action_summary: str | None = None
    ) -> str:
        summaries = [
            (turn.state_summary or turn.action_summary).strip()
            for turn in history
            if (turn.state_summary or turn.action_summary).strip()
        ]
        if current_action_summary and current_action_summary.strip():
            summaries.append(current_action_summary.strip())
        if not summaries:
            return "No GUI actions were completed."
        return "; ".join(summary.rstrip(".") for summary in summaries[-3:])

    @staticmethod
    def _describe_observation_state(observation: Observation | None) -> str:
        if observation is None:
            return "Current screen state unavailable."
        parts: list[str] = []
        if observation.foreground_app and observation.foreground_app.strip():
            parts.append(observation.foreground_app.strip())
        if isinstance(observation.screen_width, int) and isinstance(observation.screen_height, int):
            parts.append(f"{observation.screen_width}x{observation.screen_height}")
        if parts:
            return " ".join(parts)
        if observation.platform:
            return observation.platform
        return "Current screen state unavailable."

    @staticmethod
    def _remaining_hint(*, status: str, error: str | None) -> str:
        error_text = (error or "").lower()
        if status == "completed":
            return "none"
        if "stagnation_detected" in error_text:
            return "Change the action sequence from the current screen."
        if "intervention_cancelled" in error_text or "interruption" in error_text:
            return "Wait for the intervention blocker to be resolved."
        if "step_timeout" in error_text:
            return "Retry the timed-out step from the current screen."
        if "max_steps_exceeded" in error_text:
            return "Continue the remaining task from the current screen."
        if status == "blocked":
            return "Resolve the blocker before retrying."
        return "Continue from the current screen."

    @staticmethod
    def _resume_hint(*, status: str, error: str | None) -> str:
        error_text = (error or "").lower()
        if status == "completed":
            return "No further action needed."
        if "stagnation_detected" in error_text:
            return "Resume by trying a different action on the same screen."
        if "intervention_cancelled" in error_text or "interruption" in error_text:
            return "Resolve the intervention blocker, then continue from the current screen."
        if "step_timeout" in error_text:
            return "Resume from the current screen after the timeout clears."
        if "max_steps_exceeded" in error_text:
            return "Resume from the current screen and finish the remaining steps."
        if status == "blocked":
            return "Resolve the blocker, then continue from the current screen."
        return "Resume from the current screen."

    async def _generate_termination_summary(
        self,
        *,
        task: str,
        termination_reason: str,
        history: list[HistoryTurn],
        run_dir: Path,
    ) -> str | None:
        """Ask the LLM for a brief state note when the task terminates abnormally."""
        steps_text = (
            "\n".join(f"  {i}. {turn.action_summary}" for i, turn in enumerate(history, 1))
            or "  (no steps completed)"
        )
        fallback_status = (
            "blocked"
            if any(
                keyword in termination_reason.lower()
                for keyword in ("interrupted", "cancel", "loop")
            )
            else "partial"
        )
        observation: Observation | None = None
        try:
            # Take a fresh screenshot for the summary
            screenshot_path = run_dir / "screenshots" / "termination_summary.png"
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            observation = await self.backend.observe(
                screenshot_path,
                timeout=self.step_timeout,
            )
            prompt_text = (
                "Return a compact GUI state note for the terminated task.\n\n"
                f"Task: {task}\n"
                f"Termination reason: {termination_reason}\n"
                f"Steps executed:\n{steps_text}\n\n"
                "Use exactly these 5 labels and keep each value short:\n"
                "Status: completed|partial|blocked\n"
                "Done: ...\n"
                "Remaining: ...\n"
                "Current: ...\n"
                "Resume: ...\n\n"
                "Rules:\n"
                "- Do not add bullets, markdown, or extra lines.\n"
                "- Use a clear resume hint if continuation is still possible.\n"
                "- Use 'none' for Remaining when the task is completed."
            )
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
            if observation.screenshot_path and Path(observation.screenshot_path).exists():
                content.append(self._image_block(Path(observation.screenshot_path)))

            response = await self.llm.chat(
                messages=[{"role": "user", "content": content}],
                tools=None,
            )
            text = response.content.strip()
            if text and is_state_note(text):
                return text
        except Exception as exc:
            logger.warning("Failed to generate termination summary: %s", exc)

        return self._build_state_note(
            status=fallback_status,
            history=history,
            current_observation=observation,
            error=termination_reason,
        )

    @staticmethod
    def _build_assistant_message(
        response: LLMResponse,
        *,
        content_override: str | None = None,
        include_tool_calls: bool = True,
    ) -> dict[str, Any]:
        """Build an assistant message dict from an LLM response."""
        msg: dict[str, Any] = {"role": "assistant"}

        content = content_override if content_override is not None else response.content
        if content:
            msg["content"] = content

        if include_tool_calls and response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                        if isinstance(tc.arguments, dict)
                        else str(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]

        return msg

    @staticmethod
    def _extract_action_line_from_response(content: str) -> str | None:
        parts = content.split("Action:", 1)
        if len(parts) != 2:
            return None

        action_line = parts[1].strip().splitlines()
        if not action_line:
            return None

        first_line = action_line[0].strip()
        if not first_line:
            return None

        first_line = first_line.strip('"')
        return first_line.strip()

    def _snapshot_model_response(
        self,
        *,
        response: LLMResponse,
        action: Action,
        assistant_message: dict[str, Any],
        action_text: str,
        action_intent: str | None = None,
        state_summary: str | None = None,
    ) -> dict[str, Any]:
        snapshot = {
            "raw_content": self._scrub_text_for_artifact_action(response.content, action),
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": self._scrub_for_artifact(tool_call.arguments),
                }
                for tool_call in (response.tool_calls or [])
            ],
            "assistant_message": self._scrub_assistant_message_for_artifact(
                assistant_message, action
            ),
            "parsed_action": self._scrub_for_artifact(self._serialize_action(action)),
            "action_text": self._scrub_text_for_artifact_action(action_text, action),
            "action_summary": self._scrub_text_for_artifact_action(
                self._action_summary(action_text), action
            ),
            "action_intent": self._scrub_text_for_artifact_action(action_intent, action),
            "state_summary": self._scrub_text_for_artifact_action(state_summary, action),
        }
        self._add_provider_response_fields(snapshot, response, action=action)
        return snapshot

    def _snapshot_failed_model_response(
        self,
        response: LLMResponse,
        *,
        assistant_message: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "raw_content": self._scrub_text_for_artifact_action(response.content, None),
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": self._scrub_for_artifact(tool_call.arguments),
                }
                for tool_call in (response.tool_calls or [])
            ],
        }
        if assistant_message is not None:
            snapshot["assistant_message"] = self._scrub_for_artifact(assistant_message)
        self._add_provider_response_fields(snapshot, response, action=None)
        return snapshot

    def _add_provider_response_fields(
        self,
        snapshot: dict[str, Any],
        response: LLMResponse,
        *,
        action: Action | None,
    ) -> None:
        raw = response.raw

        def get_field(name: str) -> Any:
            if isinstance(raw, dict):
                return raw.get(name)
            return getattr(raw, name, None)

        reasoning_content = get_field("reasoning_content")
        if reasoning_content:
            snapshot["reasoning_content"] = self._scrub_text_for_artifact_action(
                str(reasoning_content), action
            )
        thinking_blocks = get_field("thinking_blocks")
        if thinking_blocks:
            snapshot["thinking_blocks"] = self._scrub_for_artifact(thinking_blocks)
        finish_reason = get_field("finish_reason")
        if finish_reason:
            snapshot["finish_reason"] = str(finish_reason)

    @staticmethod
    def _normalize_action_text(
        content: str,
        action: Action,
        *,
        tool_summary: str | None = None,
    ) -> str:
        summary = GuiAgent._clean_action_summary(tool_summary)
        if summary:
            return f"Action: {summary}"
        text = content.strip() if content else ""
        action_line = GuiAgent._extract_action_line_from_response(text)
        if action_line:
            return f"Action: {action_line}"
        if text:
            first_line = text.splitlines()[0].strip()
            if first_line.lower().startswith("action:"):
                return first_line
            return f"Action: {first_line}"
        return f"Action: {describe_action(action)}"

    @staticmethod
    def _tool_call_semantics(tool_call: ToolCall) -> tuple[str | None, str | None]:
        arguments = tool_call.arguments or {}
        intent = GuiAgent._clean_action_summary(arguments.get("intent"))
        summary = GuiAgent._clean_action_summary(arguments.get("summary"))
        return intent, summary

    @staticmethod
    def _clean_action_summary(value: Any) -> str | None:
        if value is None:
            return None
        text = " ".join(str(value).split()).strip()
        if not text:
            return None
        lowered = text.casefold()
        if lowered.startswith("action:"):
            text = text.split(":", 1)[1].strip()
        text = text.strip("`\"'")
        return text or None

    @staticmethod
    def _action_summary(action_text: str) -> str:
        if action_text.lower().startswith("action:"):
            return action_text.split(":", 1)[1].strip()
        return action_text.strip()

    @staticmethod
    def _resolve_done_status(action: Action) -> str:
        """Resolve terminal status for done actions with safe fallback rules."""
        if action.status in {"success", "failure"}:
            return action.status
        text = (action.text or "").strip().lower()
        if any(hint in text for hint in _DONE_FAILURE_HINTS):
            return "failure"
        # Missing status is common for some providers; default to success so
        # we do not retry already-completed tasks.
        return "success"

    def _image_block(self, path: Path) -> dict[str, Any]:
        """Create a base64 image content block for an LLM message."""
        b64 = base64.b64encode(
            scale_image(path.read_bytes(), scale_ratio=self._image_scale_ratio)
        ).decode()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }

    @staticmethod
    def _serialize_action(action: Action) -> dict[str, Any]:
        payload = dataclasses.asdict(action)
        return {
            key: value
            for key, value in payload.items()
            if value is not None and not (key == "relative" and value is False)
        }

    @staticmethod
    def _serialize_observation(observation: Observation | None) -> dict[str, Any] | None:
        if observation is None:
            return None
        return {
            "screenshot_path": observation.screenshot_path,
            "screen_width": observation.screen_width,
            "screen_height": observation.screen_height,
            "foreground_app": observation.foreground_app,
            "platform": observation.platform,
            "extra": GuiAgent._scrub_for_log(observation.extra),
        }

    @staticmethod
    def _scrub_for_log(value: Any) -> Any:
        return GuiAgent._scrub_value(value, redact_input_text=True)

    @staticmethod
    def _scrub_for_artifact(value: Any) -> Any:
        return GuiAgent._scrub_value(value, redact_input_text=False)

    @staticmethod
    def _scrub_value(value: Any, *, redact_input_text: bool) -> Any:
        if isinstance(value, dict):
            scrubbed: dict[str, Any] = {}
            action_type = (
                value.get("action_type") if isinstance(value.get("action_type"), str) else None
            )
            intervention_text = (
                value.get("text")
                if action_type == "request_intervention" and isinstance(value.get("text"), str)
                else None
            )
            for key, item in value.items():
                if key == "url" and isinstance(item, str) and item.startswith("data:image/"):
                    scrubbed[key] = "<omitted:image-data-url>"
                elif redact_input_text and action_type == "input_text" and key == "text":
                    scrubbed[key] = "<redacted:input_text>"
                elif (action_type == "request_intervention" and key == "text") or key == "reason":
                    scrubbed[key] = "<redacted:intervention_reason>"
                elif any(
                    token in key.lower()
                    for token in ("password", "secret", "token", "otp", "credential")
                ):
                    scrubbed[key] = "<redacted:sensitive_field>"
                elif intervention_text and isinstance(item, str):
                    scrubbed[key] = GuiAgent._scrub_sensitive_text(item).replace(
                        intervention_text,
                        "<redacted:intervention_reason>",
                    )
                else:
                    scrubbed[key] = GuiAgent._scrub_value(item, redact_input_text=redact_input_text)
            return scrubbed
        if isinstance(value, list):
            return [
                GuiAgent._scrub_value(item, redact_input_text=redact_input_text) for item in value
            ]
        if isinstance(value, str):
            return GuiAgent._scrub_sensitive_text(value)
        return value

    @staticmethod
    def _scrub_text_for_action(text: str | None, action: Action | None) -> str | None:
        return GuiAgent._scrub_text(text, action, redact_input_text=True)

    @staticmethod
    def _scrub_text_for_artifact_action(text: str | None, action: Action | None) -> str | None:
        return GuiAgent._scrub_text(text, action, redact_input_text=False)

    @staticmethod
    def _scrub_text(
        text: str | None, action: Action | None, *, redact_input_text: bool
    ) -> str | None:
        if text is None:
            return None
        scrubbed = GuiAgent._scrub_sensitive_text(text)
        if action is None:
            return scrubbed
        if redact_input_text and action.action_type == "input_text" and action.text:
            scrubbed = scrubbed.replace(action.text, "<redacted:input_text>")
        if action.action_type == "request_intervention" and action.text:
            scrubbed = scrubbed.replace(action.text, "<redacted:intervention_reason>")
        return scrubbed

    @staticmethod
    def _scrub_sensitive_text(text: str) -> str:
        return re.sub(
            r"(?i)(\b[\w-]*(?:password|secret|token|otp|credential)[\w-]*\b\s*[:=]\s*)([^\s,}\]]+)",
            r"\1<redacted:sensitive_field>",
            text,
        )

    @classmethod
    def _scrub_assistant_message_for_log(
        cls,
        assistant_message: dict[str, Any],
        action: Action,
    ) -> dict[str, Any]:
        scrubbed = cls._scrub_for_log(assistant_message)
        content = scrubbed.get("content")
        if isinstance(content, str):
            scrubbed["content"] = cls._scrub_text_for_action(content, action)
        for tool_call in scrubbed.get("tool_calls", []):
            if not isinstance(tool_call, dict):
                continue
            function_payload = tool_call.get("function")
            if not isinstance(function_payload, dict):
                continue
            arguments = function_payload.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                function_payload["arguments"] = json.dumps(
                    cls._scrub_for_log(json.loads(arguments)),
                    ensure_ascii=False,
                )
            except json.JSONDecodeError:
                function_payload["arguments"] = cls._scrub_text_for_action(arguments, action)
        return scrubbed

    @classmethod
    def _scrub_assistant_message_for_artifact(
        cls,
        assistant_message: dict[str, Any],
        action: Action,
    ) -> dict[str, Any]:
        scrubbed = cls._scrub_for_artifact(assistant_message)
        content = scrubbed.get("content")
        if isinstance(content, str):
            scrubbed["content"] = cls._scrub_text_for_artifact_action(content, action)
        for tool_call in scrubbed.get("tool_calls", []):
            if not isinstance(tool_call, dict):
                continue
            function_payload = tool_call.get("function")
            if not isinstance(function_payload, dict):
                continue
            arguments = function_payload.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                function_payload["arguments"] = json.dumps(
                    cls._scrub_for_artifact(json.loads(arguments)),
                    ensure_ascii=False,
                )
            except json.JSONDecodeError:
                function_payload["arguments"] = cls._scrub_text_for_artifact_action(
                    arguments, action
                )
        return scrubbed

    # ------------------------------------------------------------------
    # Run directory and trace
    # ------------------------------------------------------------------

    def _make_run_dir(self, task: str, attempt: int) -> Path:
        """Create the stable artifact directory for one task attempt."""
        del task
        run_dir = self.artifacts_root / f"attempt_{attempt + 1:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "screenshots").mkdir(exist_ok=True)
        return run_dir

    @staticmethod
    def _step_screenshot_path(run_dir: Path, step_index: int, action_type: str) -> Path:
        kind = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(action_type or "action")).strip("_")
        return run_dir / "screenshots" / f"{step_index:03d}_{kind or 'action'}.png"

    async def _log_attempt_event(
        self,
        run_dir: Path,
        event: str,
        **payload: Any,
    ) -> None:
        del run_dir
        scrubbed_payload = self._scrub_for_log(payload)
        self._trajectory_recorder.record_event(event, **scrubbed_payload)

    # ------------------------------------------------------------------
    # Memory / skill / trajectory helpers
    # ------------------------------------------------------------------

    async def _retrieve_memory(self, task: str) -> str | None:
        """Return retrieved knowledge without replacing the independent policy hint."""
        if self._memory_retriever is None:
            return None
        from guiclaw.memory.types import MemoryType

        # Fetch relevant entries by query
        results = await self._memory_retriever.search(task, top_k=self._memory_top_k + 10)

        # Separate POLICY entries from search results
        policies = [(e, s) for e, s in results if e.memory_type == MemoryType.POLICY]
        others = [(e, s) for e, s in results if e.memory_type != MemoryType.POLICY][
            : self._memory_top_k
        ]

        # Also fetch all POLICY entries separately (they must always be included)
        policy_results = await self._memory_retriever.search(
            task,
            memory_type=MemoryType.POLICY,
            top_k=50,
        )
        # Merge: add any POLICY entries not already in the list
        seen_ids = {e.entry_id for e, _ in policies}
        for entry, score in policy_results:
            if entry.entry_id not in seen_ids:
                policies.append((entry, score))
                seen_ids.add(entry.entry_id)

        memory_entries = policies + others
        if not memory_entries:
            logger.info("Memory retrieval: no hits for task=%r", task)
            self._trajectory_recorder.record_event(
                "memory_retrieval",
                task=task,
                hit_count=0,
                hits=[],
                context="",
            )
            return None
        context = self._memory_retriever.format_context(memory_entries)
        self._log_memory_retrieval(task, memory_entries, context)
        return context

    def _log_policy_injection(self, context: str) -> None:
        """Record a trajectory event for direct policy context injection."""
        line_count = context.count("\n") + 1
        logger.info("Policy context injected directly: %d line(s)", line_count)
        self._trajectory_recorder.record_event(
            "memory_retrieval",
            task="(policy_direct_injection)",
            hit_count=line_count,
            hits=[],
            context=context[:200],
        )

    def _log_memory_retrieval(
        self,
        task: str,
        memory_entries: list[tuple[Any, float]],
        context: str,
    ) -> None:
        hits: list[dict[str, Any]] = []
        logger.info("Memory retrieval: %d hit(s) for task=%r", len(memory_entries), task)
        for entry, score in memory_entries:
            preview = re.sub(r"\s+", " ", entry.content).strip()[:160]
            hit = {
                "entry_id": entry.entry_id,
                "memory_type": entry.memory_type.value,
                "platform": entry.platform,
                "app": entry.app,
                "score": round(float(score), 4),
                "content_preview": preview,
            }
            hits.append(hit)
            logger.info(
                "Memory hit id=%s type=%s score=%.4f platform=%s app=%s content=%s",
                entry.entry_id,
                entry.memory_type.value,
                float(score),
                entry.platform,
                entry.app or "-",
                preview,
            )

        self._trajectory_recorder.record_event(
            "memory_retrieval",
            task=task,
            hit_count=len(hits),
            hits=hits,
            context=context,
        )

    @staticmethod
    def _task_without_advisory_hints(task: str) -> str:
        """Return the user task text without injected advisory memory hints."""
        return str(task or "").split("\n\nAdvisory hints from past GUI memory:", 1)[0]

    def _skill_app_filter(self, task: str, app_hint: str | None) -> str | None:
        if not self._skill_app_filter_enabled:
            return None
        platform = self.backend.platform
        if app_hint:
            normalized = normalize_app_identifier(platform, app_hint)
            if normalized and normalized != "unknown":
                return normalized
        if platform.strip().lower() == "android":
            return find_android_app_in_text(self._task_without_advisory_hints(task))
        return None

    async def _build_prompt_skill_parts(
        self,
        task: str,
        *,
        app: str | None = None,
    ) -> CompactPromptParts:
        """Retrieve prompt-visible skills and always-on composite actions."""
        self._prompt_skills_by_id = {}
        self._prompt_composite_aliases = set()
        if self._skill_library is None:
            self._trajectory_recorder.record_event(
                "prompt_skill_context",
                task=task,
                hit_count=0,
                composite_aliases=[],
                reason="no_library",
                app_filter=app,
            )
            return CompactPromptParts()

        all_skills = [
            skill
            for skill in self._skill_library.list_all(platform=self.backend.platform)
            if self._skill_matches_app_filter(skill, app)
        ]
        composite_actions = composite_action_infos_from_skills(
            all_skills,
            tags=self._always_on_skill_tags,
        )
        self._prompt_composite_aliases = {action.alias for action in composite_actions}

        skill_query = self._task_without_advisory_hints(task)
        retrieved_infos = []
        if self._prompt_skill_top_k > 0:
            search_k = max(
                self._prompt_skill_top_k,
                self._prompt_skill_top_k * 5
                if self._prompt_shortcut_only or self._always_on_skill_tags
                else self._prompt_skill_top_k,
            )
            results = await self._skill_library.search(
                skill_query,
                platform=self.backend.platform,
                app=app,
                top_k=search_k,
            )
            for skill, _score in results:
                if is_always_on_skill(skill, self._always_on_skill_tags):
                    continue
                if self._prompt_shortcut_only and not is_shortcut_skill(skill):
                    continue
                skill_id = str(getattr(skill, "skill_id", "") or "")
                if not skill_id:
                    continue
                self._prompt_skills_by_id[skill_id] = skill
                retrieved_infos.append(skill_info_from_flat_skill(skill))
                if len(retrieved_infos) >= self._prompt_skill_top_k:
                    break

        parts = build_compact_prompt_parts(
            retrieved_skills=retrieved_infos,
            composite_actions=composite_actions,
        )
        self._trajectory_recorder.record_event(
            "prompt_skill_context",
            task=task,
            hit_count=len(retrieved_infos),
            skill_ids=list(parts.skill_ids),
            composite_aliases=list(parts.composite_aliases),
            app_filter=app,
            shortcut_only=self._prompt_shortcut_only,
        )
        return parts

    @staticmethod
    def _skill_matches_app_filter(skill: Any, app: str | None) -> bool:
        if not app:
            return True
        skill_app = str(getattr(skill, "app", "") or "").strip()
        return skill_app in {"", "*", "any", "unknown"} or skill_app == app
