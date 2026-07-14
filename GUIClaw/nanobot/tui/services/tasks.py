"""Task launch services for the TUI backend."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from guiclaw.action import Action
from guiclaw.backends.desktop import LocalDesktopBackend
from guiclaw.backends.dry_run import DryRunBackend
from nanobot.agent.tools.gui import GuiSubagentTool
from nanobot.tui.contracts import TaskLaunchContract
from nanobot.tui.schemas import LaunchRunResponse, TaskContractResponse, TaskLaunchRequest
from nanobot.tui.schemas.tasks import (
    GUIClawLaunchAppRequest,
    GUIClawOpenSettingsRequest,
    NanobotOpenSettingsLaunchRequest,
    NanobotOpenUrlLaunchRequest,
)
from nanobot.tui.services.operations_registry import OperationsRegistry


@dataclass(slots=True)
class TaskRunResult:
    """Internal normalized result for launched operations."""

    summary: str
    steps_taken: int = 0
    trace_ref: str | None = None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class TaskLaunchService:
    """Browser-safe launcher for the supported task allowlist."""

    def __init__(
        self,
        contract: TaskLaunchContract | None = None,
        registry: OperationsRegistry | None = None,
        *,
        nanobot_runner: Any = None,
        guiclaw_runner: Any = None,
        run_id_factory: Any = None,
        now_factory: Any = None,
    ) -> None:
        self._contract = contract or TaskLaunchContract(
            describe_capability=lambda: {
                "name": "task-launch",
                "mutable": True,
                "phase": 19,
                "status": "typed-allowlist",
            },
            launch_task=None,
        )
        self._registry = registry
        self._nanobot_runner = nanobot_runner
        self._guiclaw_runner = guiclaw_runner
        self._run_id_factory = run_id_factory or (lambda: f"run-{uuid4().hex}")
        self._now_factory = now_factory or _utc_now
        self._inflight_tasks: set[asyncio.Task[Any]] = set()

    def describe_capability(self) -> TaskContractResponse:
        return TaskContractResponse.model_validate(self._contract.describe_capability())

    def launch_task(self, payload: TaskLaunchRequest | dict[str, Any]) -> LaunchRunResponse:
        request = self._coerce_request(payload)
        if self._contract.launch_task is not None:
            return self._contract.launch_task(request)
        if self._registry is None:
            raise RuntimeError("Task launches are unavailable")

        run_id = self._run_id_factory()
        accepted_at = self._now_factory()
        self._registry.start_run(
            run_id=run_id,
            task_kind=request.kind,
            status="queued",
            summary=self._queued_summary(request),
            started_at=accepted_at,
        )
        task = asyncio.create_task(self._execute_run(run_id, request))
        self._inflight_tasks.add(task)
        task.add_done_callback(self._inflight_tasks.discard)
        return LaunchRunResponse(
            run_id=run_id,
            status="queued",
            accepted_at=accepted_at,
        )

    async def _execute_run(self, run_id: str, payload: TaskLaunchRequest) -> None:
        assert self._registry is not None
        self._registry.update_run(
            run_id,
            status="running",
            summary=self._running_summary(payload),
        )
        try:
            result = await self._dispatch(payload)
        except Exception as exc:
            self._registry.finish_run(
                run_id,
                status="failed",
                summary=self._clean_summary(str(exc)) or "Task launch failed",
            )
            return

        self._registry.finish_run(
            run_id,
            status="succeeded",
            summary=self._clean_summary(result.summary) or self._running_summary(payload),
            steps_taken=result.steps_taken,
            trace_ref=result.trace_ref,
        )

    async def _dispatch(self, payload: TaskLaunchRequest) -> TaskRunResult:
        if isinstance(payload, (NanobotOpenUrlLaunchRequest, NanobotOpenSettingsLaunchRequest)):
            if self._nanobot_runner is None:
                raise RuntimeError("Nanobot launch adapter is unavailable")
            return await self._run_adapter(self._nanobot_runner, payload)

        if self._guiclaw_runner is None:
            raise RuntimeError("GUIClaw launch adapter is unavailable")
        return await self._run_adapter(self._guiclaw_runner, payload)

    @staticmethod
    async def _run_adapter(adapter: Any, payload: TaskLaunchRequest) -> TaskRunResult:
        result = adapter(payload)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, TaskRunResult):
            return result
        if isinstance(result, dict):
            return TaskRunResult(
                summary=str(result.get("summary") or ""),
                steps_taken=int(result.get("steps_taken") or 0),
                trace_ref=result.get("trace_ref"),
            )
        raise RuntimeError("Launch adapter returned an unsupported result")

    @staticmethod
    def _coerce_request(payload: TaskLaunchRequest | dict[str, Any]) -> TaskLaunchRequest:
        if isinstance(
            payload,
            (
                NanobotOpenUrlLaunchRequest,
                NanobotOpenSettingsLaunchRequest,
                GUIClawLaunchAppRequest,
                GUIClawOpenSettingsRequest,
            ),
        ):
            return payload
        from pydantic import TypeAdapter

        return TypeAdapter(TaskLaunchRequest).validate_python(payload)

    @staticmethod
    def _queued_summary(payload: TaskLaunchRequest) -> str:
        return f"Queued {payload.kind}"

    @staticmethod
    def _running_summary(payload: TaskLaunchRequest) -> str:
        return f"Running {payload.kind}"

    @staticmethod
    def _clean_summary(value: str) -> str | None:
        cleaned = " ".join(value.split())
        return cleaned[:240] if cleaned else None


async def run_nanobot_launch(
    payload: NanobotOpenUrlLaunchRequest | NanobotOpenSettingsLaunchRequest,
    *,
    gui_tool: GuiSubagentTool,
) -> TaskRunResult:
    task = _nanobot_task_text(payload)
    raw_result = await gui_tool.execute(
        task=task,
        require_background_isolation=payload.require_background_isolation,
        acknowledge_background_fallback=payload.acknowledge_background_fallback,
    )
    try:
        parsed = json.loads(raw_result)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GuiSubagentTool returned non-JSON launch output") from exc

    summary = str(parsed.get("summary") or task)
    success = bool(parsed.get("success"))
    if not success:
        raise RuntimeError(summary)
    return TaskRunResult(
        summary=summary,
        steps_taken=int(parsed.get("steps_taken") or 0),
        trace_ref=parsed.get("trace_path"),
    )


async def run_guiclaw_launch(
    payload: GUIClawLaunchAppRequest | GUIClawOpenSettingsRequest,
) -> TaskRunResult:
    backend_name = payload.backend or "local"
    backend = _build_guiclaw_backend(backend_name)
    if hasattr(backend, "preflight"):
        await backend.preflight()

    target = _resolve_guiclaw_target(payload)
    action = Action(action_type="open_app", text=target)
    summary = await backend.execute(action)
    if isinstance(payload, GUIClawOpenSettingsRequest):
        summary = f"{summary} ({payload.panel})"
    return TaskRunResult(summary=summary, steps_taken=1)


def _nanobot_task_text(
    payload: NanobotOpenUrlLaunchRequest | NanobotOpenSettingsLaunchRequest,
) -> str:
    if isinstance(payload, NanobotOpenUrlLaunchRequest):
        return f"Open URL {payload.url}"
    return f"Open the {payload.panel} settings panel"


def _build_guiclaw_backend(backend_name: str) -> LocalDesktopBackend | DryRunBackend:
    if backend_name == "dry-run":
        return DryRunBackend()
    if backend_name == "local":
        return LocalDesktopBackend()
    raise RuntimeError(f"Unsupported GUIClaw backend: {backend_name}")


def _resolve_guiclaw_target(
    payload: GUIClawLaunchAppRequest | GUIClawOpenSettingsRequest,
) -> str:
    if isinstance(payload, GUIClawOpenSettingsRequest):
        return _platform_settings_target()
    return _platform_app_target(payload.app_id)


def _platform_settings_target() -> str:
    if sys.platform == "darwin":
        return "System Settings"
    if sys.platform == "win32":
        return "ms-settings:"
    return "Settings"


def _platform_app_target(app_id: str) -> str:
    if sys.platform == "darwin":
        return {
            "calculator": "Calculator",
            "notepad": "TextEdit",
            "settings": "System Settings",
            "terminal": "Terminal",
        }[app_id]
    if sys.platform == "win32":
        return {
            "calculator": "calc.exe",
            "notepad": "notepad.exe",
            "settings": "ms-settings:",
            "terminal": "wt.exe",
        }[app_id]
    return {
        "calculator": "gnome-calculator",
        "notepad": "gedit",
        "settings": "gnome-control-center",
        "terminal": "x-terminal-emulator",
    }[app_id]
