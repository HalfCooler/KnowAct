"""
guiclaw.skills.subgoal_runner
=============================
Mini vision-action loop for skill subgoal recovery.

Runs an isolated observe → LLM → execute cycle, independent of the main agent
loop.  Consumed by ``SkillExecutor``.  This was previously an inner class in
``guiclaw.agent``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from guiclaw.action import Action, ActionError, parse_action
from guiclaw.agent_profiles import (
    build_profile_messages,
    canonicalize_agent_profile,
    coordinate_mode_for_profile,
    normalize_profile_response_for_observation,
    profile_uses_native_tools,
)
from guiclaw.interfaces import DeviceBackend, LLMProvider
from guiclaw.observation import Observation
from guiclaw.skills.executor import SubgoalResult
from guiclaw.tool_schemas import COMPUTER_USE_TOOL, image_dimensions

if TYPE_CHECKING:
    from guiclaw.trajectory.recorder import TrajectoryRecorder

logger = logging.getLogger(__name__)


class SubgoalRunner:
    """Mini vision-action loop to recover to a desired screen state.

    Maintains an isolated history (not shared with the main agent loop) and
    runs up to ``max_steps`` observe → LLM → execute cycles until the profile
    emits ``done(status="success")``.
    """

    _COORDINATE_ACTIONS = frozenset({"tap", "double_tap", "long_press", "swipe", "drag", "scroll"})
    _POST_ACTION_SETTLE_SECONDS = 0.50
    _NO_SETTLE_ACTIONS = frozenset({"wait", "done", "request_intervention"})
    _PROFILE_PARSE_RETRIES = 3
    _HISTORY_IMAGE_WINDOW = 3

    def __init__(
        self,
        llm: LLMProvider,
        backend: DeviceBackend,
        state_validator: Any,
        model: str,
        artifacts_root: Path,
        trajectory_recorder: TrajectoryRecorder | None = None,
        agent_profile: str | None = None,
        step_timeout: float = 30.0,
        image_scale_ratio: float = 0.5,
        history_image_window: int | None = None,
    ) -> None:
        self._llm = llm
        self._backend = backend
        self._state_validator = state_validator
        self._model = model
        self._artifacts_root = Path(artifacts_root)
        self._trajectory_recorder = trajectory_recorder
        self._step_counter = 0
        self._agent_profile = canonicalize_agent_profile(agent_profile)
        self._step_timeout = step_timeout
        self._image_scale_ratio = image_scale_ratio
        self._history_image_window = (
            self._HISTORY_IMAGE_WINDOW
            if history_image_window is None
            else max(1, history_image_window)
        )

    def set_artifacts_root(self, artifacts_root: Path) -> None:
        self._artifacts_root = Path(artifacts_root)
        self._step_counter = 0

    async def run_subgoal(
        self,
        goal: str,
        screenshot: Path | bytes,
        *,
        max_steps: int = 3,
        current_observation: Observation | None = None,
    ) -> SubgoalResult:
        summaries: list[str] = []
        history: list[Any] = []
        subgoal_usage: dict[str, int] = {}
        current_observation, current_screenshot = self._initial_observation(
            screenshot,
            current_observation,
        )
        task = self._subgoal_task(goal)

        if self._trajectory_recorder is not None:
            self._trajectory_recorder.record_event(
                "subgoal_start", goal=goal, max_steps=max_steps,
            )

        for i in range(max_steps):
            substep_start = time.monotonic()
            messages = build_profile_messages(
                self._agent_profile,
                task=task,
                current_observation=current_observation,
                history=history,
                model_name=self._model,
                history_image_window=self._history_image_window,
                image_scale_ratio=self._image_scale_ratio,
            )
            parsed = await self._next_action(
                messages=messages,
                current_observation=current_observation,
                subgoal_usage=subgoal_usage,
            )
            if parsed.get("error"):
                error = str(parsed["error"])
                response_content = parsed.get("response_content")
                logger.warning("Subgoal parse failed at sub-step %d: %s", i, error)
                substep_dur = time.monotonic() - substep_start
                self._record_subgoal_step(
                    goal=goal, substep_index=i + 1,
                    model_output=response_content if isinstance(response_content, str) else None,
                    action=None, action_summary=None, screenshot_path=None,
                    goal_reached=False, error=error,
                    duration_s=substep_dur, token_usage=None,
                )
                if self._trajectory_recorder is not None:
                    self._trajectory_recorder.record_event(
                        "subgoal_result", goal=goal, success=False,
                        steps_taken=i, action_summaries=summaries, error=error,
                    )
                return SubgoalResult(
                    success=False, steps_taken=i, action_summaries=summaries,
                    final_screenshot=current_screenshot,
                    final_observation=current_observation,
                    error=error,
                    token_usage=dict(subgoal_usage),
                )

            response = parsed["response"]
            action = parsed["action"]

            if action.action_type == "done":
                substep_dur = time.monotonic() - substep_start
                goal_reached = getattr(action, "status", None) == "success"
                summary = "goal reached by model judgment" if goal_reached else "model declared goal unreachable"
                summaries.append(f"Sub-step {i+1}: {summary}")
                self._record_subgoal_step(
                    goal=goal, substep_index=i + 1, model_output=response.content,
                    action=action, action_summary=summary, screenshot_path=None,
                    goal_reached=goal_reached,
                    error=None if goal_reached else "model declared goal unreachable",
                    duration_s=substep_dur, token_usage=None,
                )
                if goal_reached:
                    if self._trajectory_recorder is not None:
                        self._trajectory_recorder.record_event(
                            "subgoal_result", goal=goal, success=True,
                            steps_taken=i + 1, action_summaries=summaries, error=None,
                        )
                    return SubgoalResult(
                        success=True, steps_taken=i + 1,
                        action_summaries=summaries,
                        final_screenshot=current_screenshot,
                        final_observation=current_observation,
                        done_judgment=True, token_usage=dict(subgoal_usage),
                    )
                break

            if action.action_type == "request_intervention":
                summaries.append(f"Sub-step {i+1}: terminal action skipped in subgoal")
                substep_dur = time.monotonic() - substep_start
                self._record_subgoal_step(
                    goal=goal, substep_index=i + 1, model_output=response.content,
                    action=action, action_summary="terminal action skipped in subgoal",
                    screenshot_path=None, goal_reached=False,
                    error="terminal action skipped in subgoal",
                    duration_s=substep_dur, token_usage=None,
                )
                continue

            try:
                await self._backend.execute(action, timeout=self._step_timeout)
                result_text = f"executed:{action.action_type}"
            except Exception as exc:
                logger.warning("Subgoal execution failed at sub-step %d: %s", i, exc)
                summaries.append(f"Sub-step {i+1}: {action.action_type} — execution error: {exc}")
                substep_dur = time.monotonic() - substep_start
                self._record_subgoal_step(
                    goal=goal, substep_index=i + 1, model_output=response.content,
                    action=action, action_summary=f"{action.action_type}",
                    screenshot_path=None, goal_reached=False,
                    error=f"execution error: {exc}",
                    duration_s=substep_dur, token_usage=None,
                )
                continue

            settle = self._post_action_settle_seconds(action)
            if settle > 0:
                await asyncio.sleep(settle)

            self._step_counter += 1
            subgoal_dir = self._artifacts_root / "screenshots"
            subgoal_dir.mkdir(parents=True, exist_ok=True)
            next_path = subgoal_dir / f"subgoal_{self._step_counter:03d}_{action.action_type}.png"
            screenshot_path: str | None = None
            try:
                obs = await self._backend.observe(next_path, timeout=self._step_timeout)
                current_observation = obs
                current_screenshot = Path(obs.screenshot_path) if obs.screenshot_path else current_screenshot
                screenshot_path = obs.screenshot_path
            except Exception as exc:
                logger.warning("Subgoal observe failed at sub-step %d: %s", i, exc)

            action_desc = f"{action.action_type}"
            if hasattr(action, "x") and action.x is not None:
                action_desc += f" at ({action.x}, {action.y})"
            elif hasattr(action, "text") and action.text:
                action_desc += f" '{action.text}'"
            summaries.append(f"Sub-step {i+1}: {action_desc}")
            history.append(self._history_turn(
                observation=parsed["observation"],
                response=response,
                action_summary=action_desc,
                tool_result=result_text,
            ))

            substep_dur = time.monotonic() - substep_start
            substep_usage = dict(subgoal_usage)
            self._record_subgoal_step(
                goal=goal, substep_index=i + 1, model_output=response.content,
                action=action, action_summary=action_desc,
                screenshot_path=screenshot_path, goal_reached=False,
                error=None, duration_s=substep_dur,
                validate_duration_s=0.0, token_usage=substep_usage,
            )

        if self._trajectory_recorder is not None:
            self._trajectory_recorder.record_event(
                "subgoal_result", goal=goal, success=False,
                steps_taken=max_steps, action_summaries=summaries,
                error=f"Subgoal not reached after {max_steps} sub-steps",
            )
        return SubgoalResult(
            success=False, steps_taken=max_steps,
            action_summaries=summaries,
            final_screenshot=current_screenshot,
            final_observation=current_observation,
            error=f"Subgoal not reached after {max_steps} sub-steps",
            token_usage=dict(subgoal_usage),
        )

    # -- helpers -------------------------------------------------------------

    async def _next_action(
        self,
        *,
        messages: list[dict[str, Any]],
        current_observation: Observation,
        subgoal_usage: dict[str, int],
    ) -> dict[str, Any]:
        native_tools_enabled = profile_uses_native_tools(self._agent_profile)
        last_response_content: str | None = None
        for attempt in range(self._PROFILE_PARSE_RETRIES):
            try:
                response = await self._llm.chat(
                    messages=messages,
                    tools=[COMPUTER_USE_TOOL] if native_tools_enabled else None,
                    tool_choice="required" if native_tools_enabled else None,
                    model=self._model or None,
                )
            except Exception as exc:
                return {"error": str(exc), "response_content": last_response_content}
            for k, v in (response.usage or {}).items():
                subgoal_usage[k] = subgoal_usage.get(k, 0) + v
            last_response_content = response.content
            try:
                response = normalize_profile_response_for_observation(
                    self._agent_profile,
                    response,
                    current_observation,
                    model_name=self._model,
                    image_scale_ratio=self._image_scale_ratio,
                )
                if not response.tool_calls:
                    raise ValueError("no valid action returned")
                tool_call = response.tool_calls[0]
                action = parse_action(tool_call.arguments)
                action = self._normalize_relative_coordinates(action)
                return {
                    "response": response,
                    "action": action,
                    "observation": current_observation,
                }
            except (ActionError, ValueError) as exc:
                last_response_content = response.content
                if attempt >= self._PROFILE_PARSE_RETRIES - 1:
                    return {
                        "error": f"profile/action parse error after retries: {exc}",
                        "response_content": response.content,
                    }
                messages.append(self._format_error_message(exc))
        return {"error": "profile/action parse error after retries", "response_content": last_response_content}

    def _format_error_message(self, exc: Exception) -> dict[str, Any]:
        return {
            "role": "user",
            "content": (
                f"Format error: {exc}. Return exactly one action in the configured "
                f"{self._agent_profile} MobileWorld response format."
            ),
        }

    def _initial_observation(
        self,
        screenshot: Path | bytes,
        current_observation: Observation | None,
    ) -> tuple[Observation, Path | bytes]:
        screenshot_path = self._ensure_screenshot_path(screenshot)
        if current_observation is not None:
            if not current_observation.screenshot_path:
                current_observation.screenshot_path = str(screenshot_path)
            return current_observation, screenshot_path
        img_bytes = screenshot_path.read_bytes()
        width, height = image_dimensions(img_bytes)
        return (
            Observation(
                screenshot_path=str(screenshot_path),
                screen_width=width,
                screen_height=height,
                platform=getattr(self._backend, "platform", "unknown"),
            ),
            screenshot_path,
        )

    def _ensure_screenshot_path(self, screenshot: Path | bytes) -> Path:
        if isinstance(screenshot, Path):
            return screenshot
        subgoal_dir = self._artifacts_root / "screenshots"
        subgoal_dir.mkdir(parents=True, exist_ok=True)
        path = subgoal_dir / f"subgoal_input_{self._step_counter:03d}.png"
        path.write_bytes(screenshot)
        return path

    @staticmethod
    def _subgoal_task(goal: str) -> str:
        return (
            f"Sub-goal recovery: reach this screen state: {goal}. "
            "If already satisfied, return done(status=\"success\"). "
            "Otherwise perform exactly one action toward that state."
        )

    @staticmethod
    def _history_turn(
        *,
        observation: Observation,
        response: Any,
        action_summary: str,
        tool_result: str,
    ) -> Any:
        content = response.content or action_summary
        return SimpleNamespace(
            observation=observation,
            action_summary=action_summary,
            tool_result_message={"content": tool_result},
            assistant_message={"role": "assistant", "content": [{"type": "text", "text": content}]},
            raw_response_content=content,
            state_summary=action_summary,
        )

    def _record_subgoal_step(
        self, *, goal: str, substep_index: int,
        model_output: str | None, action: Action | None,
        action_summary: str | None, screenshot_path: str | None,
        goal_reached: bool, error: str | None,
        duration_s: float = 0.0, validate_duration_s: float = 0.0,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        if self._trajectory_recorder is None:
            return
        self._trajectory_recorder.record_event(
            "subgoal_step", goal=goal, substep_index=substep_index,
            model_output=model_output,
            action=self._serialize_action(action) if action is not None else None,
            action_summary=action_summary, screenshot_path=screenshot_path,
            goal_reached=goal_reached, error=error,
            duration_s=round(duration_s, 3) if duration_s else None,
            validate_duration_s=round(validate_duration_s, 3) if validate_duration_s else None,
            token_usage=token_usage or None,
        )

    @staticmethod
    def _serialize_action(action: Action) -> dict[str, Any]:
        payload = dataclasses.asdict(action)
        return {
            key: value
            for key, value in payload.items()
            if value is not None and not (key == "relative" and value is False)
        }

    def _coordinate_mode(self) -> str:
        return coordinate_mode_for_profile(self._agent_profile, self._model)

    def _model_uses_relative_grid(self) -> bool:
        return self._coordinate_mode() == "relative_999"

    def _normalize_relative_coordinates(self, action: Action) -> Action:
        if action.relative or action.action_type not in self._COORDINATE_ACTIONS:
            return action
        if not self._model_uses_relative_grid():
            return action
        coords = [v for v in (action.x, action.y, action.x2, action.y2) if v is not None]
        if coords and all(0 <= v <= 999 for v in coords):
            return replace(action, relative=True)
        return action

    def _post_action_settle_seconds(self, action: Action) -> float:
        if action.action_type in self._NO_SETTLE_ACTIONS:
            return 0.0
        return self._POST_ACTION_SETTLE_SECONDS
