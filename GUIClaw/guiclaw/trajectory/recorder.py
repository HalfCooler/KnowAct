"""Compact JSON trajectory recording for GUIClaw runs."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class ExecutionPhase(str, Enum):
    """Execution mode associated with a recorded step."""

    AGENT = "agent"
    SKILL = "skill"
    RETRY = "retry"
    RECOVERY = "recovery"


@dataclass
class TrajectoryRecorder:
    """Write one compact ``traj.json`` and ``result.json`` per GUI run."""

    output_dir: Path
    task: str
    platform: str = "unknown"
    event_callback: Callable[[dict[str, Any]], None] | None = None
    instruction: str | None = None
    subtask_index: int = 1
    app_hint: str | None = None

    _path: Path | None = field(default=None, init=False, repr=False)
    _result_path: Path | None = field(default=None, init=False, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _current_phase: ExecutionPhase = field(default=ExecutionPhase.AGENT, init=False, repr=False)
    _attempt: int = field(default=1, init=False, repr=False)
    _pending_skill: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.subtask_index = max(1, int(self.subtask_index))

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def result_path(self) -> Path | None:
        return self._result_path

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def current_phase(self) -> ExecutionPhase:
        return self._current_phase

    def start(self, *, phase: ExecutionPhase = ExecutionPhase.AGENT) -> Path:
        """Start or join a run-scoped compact trajectory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / "traj.json"
        self._result_path = self.output_dir / "result.json"
        self._start_time = time.time()
        self._closed = False
        self._current_phase = phase
        self._attempt = 1
        self._pending_skill = {}

        trajectory = self._load_trajectory()
        self._register_subtask(trajectory)
        self._step_count = len(trajectory["steps"])
        self._write_json(self._path, trajectory)
        self._emit(
            {
                "type": "metadata",
                "task": self.task,
                "platform": self.platform,
                "subtask": self.subtask_index,
                "initial_phase": phase.value,
            }
        )
        return self._path

    def set_attempt(self, attempt: int) -> None:
        if self._path is None:
            raise RuntimeError("Recorder not started; call start() first")
        if self._closed:
            raise RuntimeError("Recorder already closed")
        self._attempt = max(1, int(attempt))

    def set_phase(self, phase: ExecutionPhase, *, reason: str = "") -> None:
        if self._path is None:
            raise RuntimeError("Recorder not started; call start() first")
        if self._closed:
            raise RuntimeError("Recorder already closed")
        previous = self._current_phase
        self._current_phase = phase
        self._emit(
            {
                "type": "phase_change",
                "from_phase": previous.value,
                "to_phase": phase.value,
                "reason": reason,
                "at_step": self._step_count,
                "subtask": self.subtask_index,
            }
        )

    def record_screenshot(self, screenshot_path: str | Path, *, kind: str) -> None:
        """Add one screenshot manifest entry without duplicating an existing path."""
        trajectory = self._require_trajectory()
        relative = self._relative_path(screenshot_path)
        if any(item.get("file") == relative for item in trajectory["screenshots"]):
            return
        trajectory["screenshots"].append(
            {
                "subtask": self.subtask_index,
                "attempt": self._attempt,
                "sequence": _screenshot_sequence(relative, len(trajectory["screenshots"])),
                "kind": str(kind or "screenshot"),
                "file": relative,
            }
        )
        self._write_json(self._path, trajectory)

    def record_event(self, event_type: str, **payload: Any) -> None:
        """Publish a lifecycle event without persisting verbose event payloads."""
        if self._path is None:
            raise RuntimeError("Recorder not started; call start() first")
        if self._closed:
            raise RuntimeError("Recorder already closed")
        self._capture_skill_event(event_type, payload)
        self._emit(
            {
                "type": event_type,
                "at_step": self._step_count,
                "subtask": self.subtask_index,
                "attempt": self._attempt,
                **payload,
            }
        )

    def record_step(
        self,
        *,
        action: dict[str, Any],
        model_output: Any = "",
        screenshot_path: str | None = None,
        foreground_app: str | None = None,
        interaction_target: dict[str, Any] | None = None,
        phase: ExecutionPhase | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        """Persist one model or skill action using only reusable compact fields."""
        trajectory = self._require_trajectory()
        screenshot = self._relative_path(screenshot_path) if screenshot_path else None
        action_type = str(action.get("action_type") or "screenshot")
        if screenshot_path:
            self.record_screenshot(screenshot_path, kind=action_type)
            trajectory = self._require_trajectory()

        step: dict[str, Any] = {
            "step": len(trajectory["steps"]) + 1,
            "subtask": self.subtask_index,
            "attempt": self._attempt,
            "phase": (phase or self._current_phase).value,
            "model_output": (
                model_output if isinstance(model_output, (dict, list)) else str(model_output or "")
            ),
            "action": dict(action),
        }
        if screenshot:
            step["screenshot"] = screenshot
        if foreground_app:
            step["app"] = foreground_app
        if interaction_target:
            step["interaction_target"] = interaction_target
        cleaned_usage = _clean_token_usage(token_usage)
        if cleaned_usage:
            step["token_usage"] = cleaned_usage
        if action_type == "use_skill" and self._pending_skill:
            step["skill"] = dict(self._pending_skill)
            self._pending_skill = {}

        trajectory["steps"].append(step)
        self._step_count = len(trajectory["steps"])
        self._write_json(self._path, trajectory)
        self._emit({"type": "step", **step})

    def finish(
        self,
        *,
        success: bool,
        error: str | None = None,
        summary: str | None = None,
        model_summary: str | None = None,
    ) -> Path:
        """Finalize the current subtask and update the compact run result."""
        trajectory = self._require_trajectory()
        duration = round(time.time() - self._start_time, 2)
        subtask_steps = [
            step for step in trajectory["steps"] if step.get("subtask") == self.subtask_index
        ]
        subtask_record = next(
            item for item in trajectory["subtasks"] if item.get("subtask") == self.subtask_index
        )
        subtask_result = {
            **subtask_record,
            "success": bool(success),
            "summary": summary,
            "error": error,
            "steps_taken": len(subtask_steps),
            "duration_s": duration,
        }

        result = self._load_result()
        subtasks = [
            item for item in result.get("subtasks", []) if item.get("subtask") != self.subtask_index
        ]
        subtasks.append(subtask_result)
        subtasks.sort(key=lambda item: int(item.get("subtask") or 0))
        total_duration = round(
            sum(float(item.get("duration_s") or 0) for item in subtasks if isinstance(item, dict)),
            2,
        )
        result["run"] = {
            "success": all(bool(item.get("success")) for item in subtasks),
            "summary": summary,
            "model_summary": model_summary,
            "error": error,
            "workflow_mode": "multi_app" if len(subtasks) > 1 else "single",
            "steps_taken": len(trajectory["steps"]),
            "duration_s": total_duration,
            "token_usage": _sum_step_token_usage(trajectory["steps"]),
        }
        result["subtasks"] = subtasks
        self._write_json(self._result_path, result)
        self._closed = True
        self._emit(
            {
                "type": "result",
                "success": bool(success),
                "total_steps": len(subtask_steps),
                "duration_s": duration,
                "error": error,
                "subtask": self.subtask_index,
            }
        )
        return self._path  # type: ignore[return-value]

    def _load_trajectory(self) -> dict[str, Any]:
        if self._path is not None and self._path.exists():
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("instruction", self.instruction or self.task)
                payload.setdefault("platform", self.platform)
                payload.setdefault("subtasks", [])
                payload.setdefault("steps", [])
                payload.setdefault("screenshots", [])
                return payload
        return {
            "instruction": self.instruction or self.task,
            "platform": self.platform,
            "subtasks": [],
            "steps": [],
            "screenshots": [],
        }

    def _load_result(self) -> dict[str, Any]:
        if self._result_path is not None and self._result_path.exists():
            payload = json.loads(self._result_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        return {}

    def _require_trajectory(self) -> dict[str, Any]:
        if self._path is None:
            raise RuntimeError("Recorder not started; call start() first")
        if self._closed:
            raise RuntimeError("Recorder already closed")
        return self._load_trajectory()

    def _register_subtask(self, trajectory: dict[str, Any]) -> None:
        subtasks = trajectory["subtasks"]
        existing = next(
            (item for item in subtasks if item.get("subtask") == self.subtask_index),
            None,
        )
        record = {"subtask": self.subtask_index, "task": self.task, "app_hint": self.app_hint}
        if existing is None:
            subtasks.append(record)
            subtasks.sort(key=lambda item: int(item.get("subtask") or 0))
        else:
            existing.update(record)

    def _capture_skill_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "skill_execution_start":
            self._pending_skill = _skill_identity(payload)
            return
        if event_type == "skill_step":
            if payload.get("error") is None and payload.get("valid_state_check") is not False:
                return
            self._pending_skill.update(_skill_identity(payload))
            failed_step = {
                key: payload[key]
                for key in (
                    "step_index",
                    "target",
                    "valid_state",
                    "state_contract",
                    "observation",
                    "screenshot_path",
                    "error",
                )
                if payload.get(key) is not None
            }
            self._pending_skill["failed_step"] = failed_step
            return
        if event_type == "skill_execution_result":
            self._pending_skill.update(_skill_identity(payload))
            self._pending_skill["state"] = payload.get("state")
            if payload.get("error") is not None:
                self._pending_skill["error"] = payload["error"]

    def _relative_path(self, path: str | Path) -> str:
        candidate = Path(path)
        try:
            return candidate.resolve().relative_to(self.output_dir.resolve()).as_posix()
        except ValueError:
            return candidate.as_posix()

    @staticmethod
    def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
        if path is None:
            raise RuntimeError("Recorder not started; call start() first")
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(dict(event))


def _screenshot_sequence(path: str, fallback: int) -> int:
    match = re.match(r"^(\d+)_", Path(path).name)
    return int(match.group(1)) if match else fallback


def _clean_token_usage(usage: dict[str, int] | None) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    return {key: value for key, value in usage.items() if isinstance(value, int)}


def _sum_step_token_usage(steps: list[dict[str, Any]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for step in steps:
        for key, value in _clean_token_usage(step.get("token_usage")).items():
            total[key] = total.get(key, 0) + value
    return total


def _skill_identity(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in ("skill_id", "skill_name") if payload.get(key) is not None}


def load_trajectory_events(
    trajectory_path: str | Path,
    *,
    subtask_index: int | None = None,
) -> list[dict[str, Any]]:
    """Return the compact run as the normalized event view used by learning code."""
    trajectory_path = Path(trajectory_path)
    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    subtasks = [item for item in payload.get("subtasks", []) if isinstance(item, dict)]
    if subtask_index is None:
        task = payload.get("instruction")
    else:
        task = next(
            (item.get("task") for item in subtasks if item.get("subtask") == subtask_index),
            payload.get("instruction"),
        )
    events: list[dict[str, Any]] = [
        {
            "type": "metadata",
            "task": task,
            "platform": payload.get("platform") or "unknown",
            "subtask": subtask_index,
        }
    ]
    for step in payload.get("steps", []):
        if not isinstance(step, dict):
            continue
        if subtask_index is not None and step.get("subtask") != subtask_index:
            continue
        skill = step.get("skill") if isinstance(step.get("skill"), dict) else None
        if skill and skill.get("state") == "failed":
            failed_step = skill.get("failed_step")
            if isinstance(failed_step, dict):
                events.append({"type": "skill_step", **_skill_identity(skill), **failed_step})
            events.append(
                {
                    "type": "skill_execution_result",
                    **_skill_identity(skill),
                    "state": "failed",
                    "error": skill.get("error"),
                }
            )
        events.append(_step_event(step, platform=str(payload.get("platform") or "unknown")))
        if skill and skill.get("state") != "failed":
            events.append(
                {
                    "type": "skill_execution_result",
                    **_skill_identity(skill),
                    "state": skill.get("state"),
                    "error": skill.get("error"),
                }
            )

    result_path = trajectory_path.parent / "result.json"
    result = {}
    if result_path.exists():
        loaded_result = json.loads(result_path.read_text(encoding="utf-8"))
        if isinstance(loaded_result, dict):
            result = loaded_result
    if subtask_index is None:
        subtask_result = result.get("run") if isinstance(result.get("run"), dict) else None
    else:
        subtask_result = next(
            (
                item
                for item in result.get("subtasks", [])
                if isinstance(item, dict) and item.get("subtask") == subtask_index
            ),
            None,
        )
    if isinstance(subtask_result, dict):
        events.append(
            {
                "type": "result",
                "success": bool(subtask_result.get("success")),
                "total_steps": int(subtask_result.get("steps_taken") or 0),
                "duration_s": subtask_result.get("duration_s"),
                "error": subtask_result.get("error"),
                "summary": subtask_result.get("summary"),
            }
        )
    return events


def trajectory_subtask_indices(trajectory_path: str | Path) -> tuple[int, ...]:
    """Return the ordered subtask ids stored in one compact trajectory."""
    payload = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ()
    indexes = {
        int(item["subtask"])
        for item in payload.get("subtasks", [])
        if isinstance(item, dict) and isinstance(item.get("subtask"), int)
    }
    return tuple(sorted(indexes)) or (1,)


def _step_event(step: dict[str, Any], *, platform: str) -> dict[str, Any]:
    screenshot_path = step.get("screenshot")
    observation = {
        "platform": platform,
        "foreground_app": step.get("app"),
        "screenshot_path": screenshot_path,
    }
    return {
        "type": "step",
        "step_index": max(0, int(step.get("step") or 1) - 1),
        "phase": step.get("phase"),
        "action": step.get("action") if isinstance(step.get("action"), dict) else {},
        "model_output": step.get("model_output") or "",
        "screenshot_path": screenshot_path,
        "observation": observation,
        "interaction_target": step.get("interaction_target"),
        "token_usage": step.get("token_usage"),
    }


def finalize_run_result(output_dir: str | Path, payload: dict[str, Any]) -> Path:
    """Merge the router-level outcome into the run's single ``result.json``."""
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    existing: dict[str, Any] = {}
    if result_path.exists():
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded

    stored_subtasks = [
        dict(item) for item in existing.get("subtasks", []) if isinstance(item, dict)
    ]
    routed_subtasks = payload.get("subtasks")
    if isinstance(routed_subtasks, list):
        merged_subtasks: list[dict[str, Any]] = []
        for index, routed in enumerate(routed_subtasks, start=1):
            stored = next(
                (item for item in stored_subtasks if item.get("subtask") == index),
                {},
            )
            merged = {**stored, "subtask": index}
            if isinstance(routed, dict):
                for key in ("task", "app_hint", "success", "summary", "error"):
                    if key in routed:
                        merged[key] = routed[key]
            merged_subtasks.append(merged)
        stored_subtasks = merged_subtasks

    run = {
        "success": bool(payload.get("success")),
        "summary": payload.get("summary"),
        "model_summary": payload.get("model_summary"),
        "error": payload.get("error"),
        "workflow_mode": payload.get("workflow_mode") or "single",
        "steps_taken": int(payload.get("steps_taken") or 0),
        "duration_s": payload.get("duration_s"),
        "token_usage": _clean_token_usage(payload.get("token_usage")),
    }
    if payload.get("missing_outputs"):
        run["missing_outputs"] = list(payload["missing_outputs"])
    existing["run"] = run
    existing["subtasks"] = stored_subtasks
    if isinstance(payload.get("blackboard"), dict):
        existing["blackboard"] = dict(payload["blackboard"])

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_name(f".{result_path.name}.tmp")
    temporary.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    return result_path


def update_result_section(
    output_dir: str | Path,
    section: str,
    subtask_index: int,
    payload: dict[str, Any],
) -> Path:
    """Atomically update one per-subtask post-processing section."""
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    result: dict[str, Any] = {}
    if result_path.exists():
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            result = loaded
    section_payload = result.setdefault(section, {})
    if not isinstance(section_payload, dict):
        section_payload = {}
        result[section] = section_payload
    section_payload[str(max(1, int(subtask_index)))] = _compact_section_payload(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_name(f".{result_path.name}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    return result_path


def _compact_section_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redundant = {
        "trace",
        "trace_path",
        "timestamp",
        "token_usage",
        "output_path",
        "instruction",
        "task_id",
        "source_path",
        "is_success",
        "agent_success",
        "evaluation_success",
        "platform",
        "task",
    }

    def compact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: compact(item)
                for key, item in value.items()
                if key not in redundant and item not in (None, [], {}, "")
            }
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    return compact(payload)
