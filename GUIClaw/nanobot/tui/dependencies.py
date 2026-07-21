"""Dependency providers for the isolated TUI backend."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Request

from guiclaw.paths import resolve_guiclaw_data_dir
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.gui import GuiSubagentTool
from nanobot.bus.queue import MessageBus
from nanobot.cli.commands import _load_runtime_config
from nanobot.config.loader import load_config
from nanobot.config.paths import get_cron_dir
from nanobot.config.schema import Config, ModelPresetConfig
from nanobot.cron.service import CronService
from nanobot.providers.factory import make_provider
from nanobot.session.manager import SessionManager
from nanobot.tui.contracts import (
    RuntimeInspectionContract,
    SessionContract,
    TaskLaunchContract,
)
from nanobot.tui.services import (
    ChatWorkspaceService,
    EventStreamBroker,
    OperationsRegistry,
    RuntimeService,
    SessionService,
    TaskLaunchService,
    TraceInspectionService,
)
from nanobot.tui.services.tasks import run_guiclaw_launch, run_nanobot_launch


def _resolve_workspace_path(
    workspace: Path | None = None,
) -> Path:
    if workspace is not None:
        return workspace
    return workspace or load_config().workspace_path


def _resolve_runtime_config(request: Request | None = None) -> Config:
    if request is not None:
        config = getattr(request.app.state, "nanobot_config", None)
        if isinstance(config, Config):
            return config
    return _load_runtime_config()


def _resolve_gui_runtime(
    config: Config,
    *,
    host_provider: Any | None = None,
) -> tuple[Any | None, str | None]:
    gui_config = config.gui
    if gui_config is None:
        return None, None

    resolved = config.resolve_preset()
    gui_model = gui_config.model or resolved.model
    gui_provider_name = gui_config.provider
    inherited_provider = gui_provider_name in (None, "auto", resolved.provider)
    if gui_model == resolved.model and inherited_provider:
        return host_provider or make_provider(config), gui_model

    gui_preset = ModelPresetConfig(
        model=gui_model,
        provider=gui_provider_name or resolved.provider,
        max_tokens=resolved.max_tokens,
        context_window_tokens=resolved.context_window_tokens,
        temperature=resolved.temperature,
        reasoning_effort=resolved.reasoning_effort,
    )
    return make_provider(config, preset=gui_preset), gui_model


def get_session_contract(
    workspace: Path | None = None,
) -> SessionContract:
    """Return a lazy session contract backed by SessionManager."""

    resolved_workspace = _resolve_workspace_path(workspace=workspace)

    def _list_sessions() -> list[dict[str, Any]]:
        return SessionManager(resolved_workspace).list_sessions()

    return SessionContract(
        workspace_path=resolved_workspace,
        list_sessions=_list_sessions,
    )


def get_task_launch_contract() -> TaskLaunchContract:
    """Return a future task-launch contract kept non-mutating for Phase 17."""

    def _describe_capability() -> dict[str, Any]:
        return {
            "name": "task-launch",
            "mutable": False,
            "phase": 17,
            "status": "contract-only",
        }

    return TaskLaunchContract(
        describe_capability=_describe_capability,
        launch_task=None,
    )


def get_runtime_inspection_contract() -> RuntimeInspectionContract:
    """Return a lazy runtime-inspection contract without booting the agent runtime."""

    def _inspect_runtime() -> dict[str, Any]:
        return {
            "status": "idle",
            "channel_runtime_booted": False,
            "agent_loop_booted": False,
            "task_launch_available": False,
            "session_stats": {
                "total": 0,
                "active": 0,
                "most_recent_session_id": None,
            },
            "active_runs": [],
            "recent_failures": [],
        }

    return RuntimeInspectionContract(inspect_runtime=_inspect_runtime)


def get_session_service() -> SessionService:
    """Build the read-only session service for browser-facing routes."""

    return SessionService(get_session_contract())


def get_operations_registry(request: Request) -> OperationsRegistry:
    """Return the shared process-local operations registry."""

    registry = getattr(request.app.state, "operations_registry", None)
    if isinstance(registry, OperationsRegistry):
        return registry
    registry = OperationsRegistry()
    request.app.state.operations_registry = registry
    return registry


def _build_runtime_service(
    *,
    config: Config,
    registry: OperationsRegistry,
) -> RuntimeService:
    session_contract = get_session_contract(workspace=config.workspace_path)
    artifacts_root = resolve_guiclaw_data_dir(config.gui.artifacts_dir)
    return RuntimeService(
        session_contract,
        registry,
        artifacts_root=artifacts_root,
        task_launch_available=True,
    )


def get_runtime_service(request: Request) -> RuntimeService:
    """Build the read-only runtime inspection service."""

    config = _resolve_runtime_config(request)
    registry = get_operations_registry(request)
    return _build_runtime_service(config=config, registry=registry)


def get_trace_inspection_service(request: Request) -> TraceInspectionService:
    """Build the browser-safe trace and log inspection service."""

    config = _resolve_runtime_config(request)
    registry = get_operations_registry(request)
    artifacts_root = resolve_guiclaw_data_dir(config.gui.artifacts_dir)
    return TraceInspectionService(
        registry=registry,
        artifacts_root=artifacts_root,
    )


def get_task_launch_service(request: Request) -> TaskLaunchService:
    """Build the typed task-launch service."""

    config = _resolve_runtime_config(request)
    registry = get_operations_registry(request)

    gui_config = config.gui
    host_provider = make_provider(config)
    host_model = config.resolve_preset().model
    gui_provider, gui_model = (
        _resolve_gui_runtime(config, host_provider=host_provider)
        if gui_config is not None
        else (None, None)
    )

    def _gui_tool() -> Any:
        if gui_config is None or gui_provider is None or gui_model is None:
            raise RuntimeError("nanobot gui launches require gui config")
        return GuiSubagentTool(
            gui_config=gui_config,
            provider=gui_provider,
            model=gui_model,
            workspace=config.workspace_path,
            postprocess_provider=host_provider,
            postprocess_model=host_model,
        )

    async def _nanobot_runner(payload: Any) -> dict[str, Any]:
        result = await run_nanobot_launch(payload, gui_tool=_gui_tool())
        return {
            "summary": result.summary,
            "steps_taken": result.steps_taken,
            "trace_ref": result.trace_ref,
        }

    async def _guiclaw_runner(payload: Any) -> dict[str, Any]:
        result = await run_guiclaw_launch(payload)
        return {
            "summary": result.summary,
            "steps_taken": result.steps_taken,
            "trace_ref": result.trace_ref,
        }

    return TaskLaunchService(
        TaskLaunchContract(
            describe_capability=lambda: {
                "name": "task-launch",
                "mutable": True,
                "phase": 19,
                "status": "typed-allowlist",
            },
            launch_task=None,
        ),
        registry,
        nanobot_runner=_nanobot_runner,
        guiclaw_runner=_guiclaw_runner,
    )


def get_chat_runtime_factory(
    *,
    config: Config | None = None,
    session_manager: SessionManager | None = None,
) -> Callable[[], AgentLoop]:
    """Build a browser-chat runtime factory without importing Typer routing."""

    resolved_config = config or _load_runtime_config()
    shared_sessions = session_manager or SessionManager(resolved_config.workspace_path)

    def _factory() -> AgentLoop:
        cron_store_path = get_cron_dir() / "jobs.json"
        cron = CronService(cron_store_path)
        return AgentLoop.from_config(
            resolved_config,
            bus=MessageBus(),
            cron_service=cron,
            session_manager=shared_sessions,
        )

    return _factory


def get_chat_workspace_service(request: Request) -> ChatWorkspaceService:
    """Build the browser chat workspace service."""

    config = _resolve_runtime_config(request)
    session_manager = SessionManager(config.workspace_path)
    return ChatWorkspaceService(
        session_manager=session_manager,
        event_broker=get_chat_event_broker(request),
        runtime_factory=get_chat_runtime_factory(
            config=config,
            session_manager=session_manager,
        ),
    )


def get_chat_event_broker(request: Request) -> EventStreamBroker:
    """Return the shared in-process broker for browser chat SSE events."""

    broker = getattr(request.app.state, "chat_event_broker", None)
    if isinstance(broker, EventStreamBroker):
        return broker
    broker = EventStreamBroker()
    request.app.state.chat_event_broker = broker
    return broker
