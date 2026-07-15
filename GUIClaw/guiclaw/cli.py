"""Standalone CLI for driving GUIClaw without nanobot runtime imports."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import json_repair
import numpy as np
import yaml
from openai import AsyncOpenAI

from guiclaw.agent import AgentResult, GuiAgent
from guiclaw.agent_profiles import SUPPORTED_AGENT_PROFILES
from guiclaw.backends.adb import AdbBackend
from guiclaw.backends.dry_run import DryRunBackend
from guiclaw.interfaces import (
    InterventionHandler,
    InterventionRequest,
    InterventionResolution,
    LLMResponse,
    ToolCall,
)
from guiclaw.memory.policy import load_policy_context
from guiclaw.memory.retrieval import MemoryRetriever
from guiclaw.memory.store import MemoryStore
from guiclaw.paths import DEFAULT_GUI_RUNS_DIR
from guiclaw.postprocessing import PostRunProcessor
from guiclaw.skills.action_grounder import ActionGrounder as _AgentActionGrounder
from guiclaw.skills.executor import LLMStateValidator, SkillExecutor
from guiclaw.skills.flat import DEFAULT_SKILLS_STORE_DIR, FlatSkillLibrary
from guiclaw.skills.observation_provider import AgentScreenshotProvider as _AgentScreenshotProvider
from guiclaw.skills.subgoal_runner import SubgoalRunner as _AgentSubgoalRunner
from guiclaw.trajectory.recorder import TrajectoryRecorder

LocalDesktopBackend = None
BackgroundDesktopBackend = None
WindowsIsolatedBackend = None
probe_isolated_background_support = None
resolve_run_mode = None
log_mode_resolution = None

logger = logging.getLogger(__name__)
_SAFE_INTERVENTION_TARGET_KEYS = frozenset(
    {"display_id", "monitor_index", "desktop_name", "width", "height", "platform"}
)

DEFAULT_CONFIG_PATH = Path.home() / ".guiclaw" / "config.yaml"
DEFAULT_MEMORY_DIR = Path.home() / ".guiclaw" / "memory"
DEFAULT_SKILLS_DIR = DEFAULT_SKILLS_STORE_DIR
_EMBEDDING_BATCH_SIZE = 10
WINDOWS_TARGET_APP_CLASSES = ("classic-win32", "uwp", "directx", "gpu-heavy", "electron-gpu")


@dataclass(slots=True)
class ProviderConfig:
    base_url: str
    model: str
    api_key: str | None = None


@dataclass(slots=True)
class EmbeddingConfig:
    base_url: str
    model: str
    api_key: str | None = None


@dataclass(slots=True)
class AdbConfig:
    serial: str | None = None
    adb_path: str = "adb"


@dataclass(slots=True)
class ScrcpyConfig:
    max_fps: int = 12
    jpeg_quality: int = 80
    frame_timeout_ms: int = 3000
    max_frame_age_ms: int = 1000


@dataclass(slots=True)
class IosConfig:
    wda_url: str = "http://localhost:8100"


@dataclass(slots=True)
class HdcConfig:
    serial: str | None = None
    hdc_path: str = "hdc"


@dataclass(slots=True)
class BackgroundConfig:
    """Settings for isolated background displays used with --background."""

    display_num: int = 99
    width: int = 1280
    height: int = 720


@dataclass(slots=True)
class CliConfig:
    provider: ProviderConfig
    embedding: EmbeddingConfig | None = None
    adb: AdbConfig = field(default_factory=AdbConfig)
    scrcpy: ScrcpyConfig = field(default_factory=ScrcpyConfig)
    ios: IosConfig = field(default_factory=IosConfig)
    hdc: HdcConfig = field(default_factory=HdcConfig)
    max_steps: int = 15
    stagnation_limit: int = 0
    image_scale_ratio: float = 0.5
    memory_dir: Path | None = None
    skills_dir: Path | None = None
    enable_skill_execution: bool = False
    enable_skill_extraction: bool = False
    enable_memory_extraction: bool = False
    agent_profile: str | None = None
    background: bool = False
    background_config: BackgroundConfig = field(default_factory=BackgroundConfig)


class OpenAICompatibleLLMProvider:
    """OpenAI-compatible chat bridge that satisfies guiclaw's LLM protocol."""

    def __init__(self, *, base_url: str, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key or "no-key",
            base_url=base_url,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": _sanitize_messages(messages),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = await self._client.chat.completions.create(**kwargs)
        if not response.choices:
            raise RuntimeError("OpenAI-compatible API returned no choices")

        choice = response.choices[0]
        message = choice.message
        parsed_tool_calls: list[ToolCall] = []
        for index, tool_call in enumerate(message.tool_calls or []):
            parsed_tool_calls.append(
                ToolCall(
                    id=tool_call.id or f"tool-call-{index}",
                    name=tool_call.function.name,
                    arguments=_parse_tool_arguments(tool_call.function.arguments),
                )
            )

        usage_obj = getattr(response, "usage", None)
        usage: dict[str, int] = (
            {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }
            if usage_obj is not None
            else {}
        )

        return LLMResponse(
            content=_coerce_message_content(message.content),
            tool_calls=parsed_tool_calls or None,
            raw=response,
            usage=usage,
        )


class OpenAICompatibleEmbeddingProvider:
    """OpenAI-compatible embedding bridge for optional memory and skill search."""

    def __init__(self, *, base_url: str, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key or "no-key",
            base_url=base_url,
        )

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + _EMBEDDING_BATCH_SIZE]
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            vectors.extend(item.embedding for item in response.data)
        return np.array(vectors, dtype=np.float32)


def build_embedding_provider(
    config: CliConfig,
) -> OpenAICompatibleEmbeddingProvider | None:
    if config.embedding is None:
        return None
    return OpenAICompatibleEmbeddingProvider(
        base_url=config.embedding.base_url,
        model=config.embedding.model,
        api_key=config.embedding.api_key,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m guiclaw.cli")
    parser.add_argument("task_input", nargs="?", help="Task description")
    parser.add_argument("--task", dest="task_flag", help="Task description")
    parser.add_argument(
        "--backend",
        choices=("adb", "ios", "hdc", "local", "dry-run"),
        default="local",
        help="Execution backend",
    )
    parser.add_argument("--dry-run", action="store_true", help="Shortcut for --backend dry-run")
    parser.add_argument(
        "--agent-profile",
        choices=SUPPORTED_AGENT_PROFILES,
        default=None,
        help="Prompt/action profile to emulate for the GUI agent.",
    )
    parser.add_argument("--json", dest="json_output", action="store_true", help="Emit JSON output")
    parser.add_argument("--config", type=Path, help="Config file path")
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run on virtual Xvfb display (Linux only)",
    )
    parser.add_argument(
        "--require-isolation",
        action="store_true",
        help="Block instead of falling back when isolated background execution is unavailable",
    )
    parser.add_argument(
        "--target-app-class",
        choices=WINDOWS_TARGET_APP_CLASSES,
        default=None,
        help="Windows app class hint for isolated background probing.",
    )
    parser.add_argument(
        "--display-num",
        type=int,
        default=None,
        help="Xvfb display number (default: 99)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Display width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Display height in pixels (default: 720)",
    )

    args = parser.parse_args(argv)
    if not args.task_input and not args.task_flag:
        parser.error("task is required via positional input or --task")
    if args.background and args.backend in ("adb", "ios", "hdc", "dry-run"):
        parser.error("--background requires --backend local (or omit --backend)")
    if args.background and args.dry_run:
        parser.error("--background is incompatible with --dry-run")
    if args.require_isolation and not args.background:
        parser.error("--require-isolation requires --background")
    return args


def resolve_task(args: argparse.Namespace) -> str:
    task_flag = (args.task_flag or "").strip()
    task_input = (args.task_input or "").strip()
    if task_flag and task_input and task_flag != task_input:
        raise ValueError("Positional task and --task disagree")
    task = task_flag or task_input
    if not task:
        raise ValueError("Task is required")
    return task


def resolve_backend_name(args: argparse.Namespace) -> str:
    if args.dry_run:
        return "dry-run"
    if getattr(args, "background", False):
        return "local"
    return args.backend


def resolve_target_app_class(
    args: argparse.Namespace, *, sys_platform: str | None = None
) -> str | None:
    if not getattr(args, "background", False):
        return None
    if resolve_backend_name(args) != "local":
        return None
    if (sys_platform or sys.platform) != "win32":
        return None
    return args.target_app_class or "classic-win32"


def load_config(path: Path | None = None) -> CliConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    provider_raw = _require_mapping(raw, "provider")
    provider_api_key = _optional_string(provider_raw, "api_key") or os.getenv("OPENAI_API_KEY")
    provider = ProviderConfig(
        base_url=_require_string(provider_raw, "base_url"),
        model=_require_string(provider_raw, "model"),
        api_key=provider_api_key,
    )

    embedding_raw = raw.get("embedding")
    embedding: EmbeddingConfig | None = None
    if embedding_raw is not None:
        if not isinstance(embedding_raw, dict):
            raise ValueError("embedding config must be a mapping")
        embedding = EmbeddingConfig(
            base_url=_require_string(embedding_raw, "base_url"),
            model=_require_string(embedding_raw, "model"),
            api_key=_optional_string(embedding_raw, "api_key") or provider_api_key,
        )

    adb_raw = raw.get("adb") or {}
    if not isinstance(adb_raw, dict):
        raise ValueError("adb config must be a mapping")
    adb = AdbConfig(
        serial=_optional_string(adb_raw, "serial"),
        adb_path=_optional_string(adb_raw, "adb_path") or "adb",
    )

    scrcpy_raw = raw.get("scrcpy") or {}
    if not isinstance(scrcpy_raw, dict):
        raise ValueError("scrcpy config must be a mapping")
    scrcpy = ScrcpyConfig(
        max_fps=_coerce_positive_int(scrcpy_raw.get("max_fps"), default=12),
        jpeg_quality=_coerce_positive_int(scrcpy_raw.get("jpeg_quality"), default=80),
        frame_timeout_ms=_coerce_positive_int(scrcpy_raw.get("frame_timeout_ms"), default=3000),
        max_frame_age_ms=_coerce_positive_int(scrcpy_raw.get("max_frame_age_ms"), default=1000),
    )

    ios_raw = raw.get("ios") or {}
    if not isinstance(ios_raw, dict):
        raise ValueError("ios config must be a mapping")
    ios = IosConfig(
        wda_url=_optional_string(ios_raw, "wda_url") or "http://localhost:8100",
    )

    hdc_raw = raw.get("hdc") or {}
    if not isinstance(hdc_raw, dict):
        raise ValueError("hdc config must be a mapping")
    hdc = HdcConfig(
        serial=_optional_string(hdc_raw, "serial"),
        hdc_path=_optional_string(hdc_raw, "hdc_path") or "hdc",
    )

    return CliConfig(
        provider=provider,
        embedding=embedding,
        adb=adb,
        scrcpy=scrcpy,
        ios=ios,
        hdc=hdc,
        max_steps=_coerce_positive_int(raw.get("max_steps"), default=15),
        stagnation_limit=_coerce_non_negative_int(raw.get("stagnation_limit"), default=0),
        image_scale_ratio=_coerce_image_scale_ratio(raw.get("image_scale_ratio"), default=0.5),
        memory_dir=_optional_path(raw.get("memory_dir")),
        skills_dir=_optional_path(raw.get("skills_dir")),
        enable_skill_execution=_coerce_bool(raw.get("enable_skill_execution"), default=False),
        enable_skill_extraction=_coerce_bool(raw.get("enable_skill_extraction"), default=False),
        enable_memory_extraction=_coerce_bool(raw.get("enable_memory_extraction"), default=False),
        agent_profile=_optional_string(raw, "agent_profile"),
    )


def build_backend(name: str, config: CliConfig) -> Any:
    if name == "adb":
        base_kwargs = {
            "serial": config.adb.serial,
            "adb_path": config.adb.adb_path or "adb",
        }
        extra_kwargs = {
            "scrcpy_max_fps": config.scrcpy.max_fps,
            "scrcpy_jpeg_quality": config.scrcpy.jpeg_quality,
            "scrcpy_frame_timeout_ms": config.scrcpy.frame_timeout_ms,
            "scrcpy_max_frame_age_ms": config.scrcpy.max_frame_age_ms,
            "collect_ui_tree": True,
            "collect_ui_tree_nodes": True,
        }
        kwargs = {**base_kwargs}
        try:
            signature = inspect.signature(AdbBackend.__init__)
            kwargs.update(
                {key: value for key, value in extra_kwargs.items() if key in signature.parameters}
            )
        except (TypeError, ValueError):
            pass
        return AdbBackend(**kwargs)
    if name == "ios":
        from guiclaw.backends.ios_wda import WdaBackend

        return WdaBackend(wda_url=config.ios.wda_url)
    if name == "hdc":
        from guiclaw.backends.hdc import HdcBackend

        return HdcBackend(serial=config.hdc.serial, hdc_path=config.hdc.hdc_path or "hdc")
    if name == "local":
        desktop_backend_cls = LocalDesktopBackend
        if desktop_backend_cls is None:
            from guiclaw.backends.desktop import LocalDesktopBackend as ImportedLocalDesktopBackend

            desktop_backend_cls = ImportedLocalDesktopBackend
        return desktop_backend_cls()
    if name == "dry-run":
        return DryRunBackend()
    raise ValueError(f"Unsupported backend: {name}")


async def build_optional_components(
    config: CliConfig,
    *,
    provider: OpenAICompatibleLLMProvider,
    backend: Any,
    model_name: str,
    artifacts_root: Path,
    embedding_provider: OpenAICompatibleEmbeddingProvider | None = None,
) -> tuple[Any | None, Any | None, Any | None]:
    if embedding_provider is None and config.embedding is not None:
        embedding_provider = OpenAICompatibleEmbeddingProvider(
            base_url=config.embedding.base_url,
            model=config.embedding.model,
            api_key=config.embedding.api_key,
        )

    memory_retriever = None
    if embedding_provider is not None:
        memory_store = MemoryStore(config.memory_dir or DEFAULT_MEMORY_DIR)
        memory_retriever = MemoryRetriever(embedding_provider=embedding_provider, top_k=5)
        await memory_retriever.index(memory_store.list_all())

    if not config.enable_skill_execution:
        return memory_retriever, None, None

    try:
        skill_library = FlatSkillLibrary(
            store_dir=config.skills_dir or DEFAULT_SKILLS_DIR,
            embedding_provider=embedding_provider,
            merge_llm=provider,
            embedding_signature=config.embedding.model if config.embedding else None,
        )
    except TypeError as exc:
        if "embedding_signature" not in str(exc):
            raise
        skill_library = FlatSkillLibrary(
            store_dir=config.skills_dir or DEFAULT_SKILLS_DIR,
            embedding_provider=embedding_provider,
            merge_llm=provider,
        )
    state_validator = LLMStateValidator(
        provider,
        image_scale_ratio=config.image_scale_ratio,
    )
    skill_executor = SkillExecutor(
        backend=backend,
        state_validator=state_validator,
        action_grounder=_AgentActionGrounder(
            llm=provider,
            model=model_name,
            agent_profile=config.agent_profile,
            image_scale_ratio=config.image_scale_ratio,
        ),
        subgoal_runner=_AgentSubgoalRunner(
            llm=provider,
            backend=backend,
            state_validator=state_validator,
            model=model_name,
            artifacts_root=artifacts_root,
            agent_profile=config.agent_profile,
            step_timeout=60.0,
            image_scale_ratio=config.image_scale_ratio,
        ),
        screenshot_provider=_AgentScreenshotProvider(
            backend=backend,
            artifacts_root=artifacts_root,
        ),
    )
    return memory_retriever, skill_library, skill_executor


async def _execute_agent(
    args: argparse.Namespace,
    config: CliConfig,
    backend: Any,
    provider: OpenAICompatibleLLMProvider,
    task: str,
) -> AgentResult:
    """Assemble and run the GUI agent with the given backend and provider."""
    run_root = DEFAULT_GUI_RUNS_DIR / datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S_%f")
    embedding_provider = build_embedding_provider(config)
    memory_retriever, skill_library, skill_executor = await build_optional_components(
        config,
        provider=provider,
        backend=backend,
        model_name=config.provider.model,
        artifacts_root=run_root,
        embedding_provider=embedding_provider,
    )

    recorder = TrajectoryRecorder(output_dir=run_root, task=task, platform=backend.platform)
    if skill_executor is not None:
        skill_executor.trajectory_recorder = recorder
        if getattr(skill_executor, "subgoal_runner", None) is not None:
            skill_executor.subgoal_runner._trajectory_recorder = recorder
    agent = GuiAgent(
        llm=provider,
        backend=backend,
        trajectory_recorder=recorder,
        model=config.provider.model,
        artifacts_root=run_root,
        max_steps=config.max_steps or 15,
        progress_callback=_make_progress_printer(json_output=args.json_output),
        memory_retriever=memory_retriever,
        skill_library=skill_library,
        skill_executor=skill_executor,
        intervention_handler=_build_intervention_handler(backend),
        policy_context=load_policy_context(config.memory_dir or DEFAULT_MEMORY_DIR),
        agent_profile=args.agent_profile or config.agent_profile,
        enable_prompt_skill_selection=config.enable_skill_execution,
        image_scale_ratio=config.image_scale_ratio,
        stagnation_limit=config.stagnation_limit,
    )
    result = await agent.run(task)
    if config.enable_skill_extraction or config.enable_memory_extraction:
        postprocessor = PostRunProcessor(
            llm=provider,
            merge_llm=provider,
            embedding_provider=embedding_provider,
            embedding_signature=config.embedding.model if config.embedding else None,
            skill_store_root=config.skills_dir or DEFAULT_SKILLS_DIR,
            enable_skill_extraction=config.enable_skill_extraction,
            enable_memory_extraction=config.enable_memory_extraction,
            memory_bank_path=(config.memory_dir or DEFAULT_MEMORY_DIR) / "gui_memory_bank.jsonl",
        )
        postprocessor.schedule(
            Path(result.trace_path) if result.trace_path else None,
            is_success=result.success,
            platform=backend.platform,
            task=task,
        )
        await postprocessor.drain()
    return result


async def run_cli(args: argparse.Namespace) -> AgentResult:
    task = resolve_task(args)
    config = load_config(args.config)
    backend = build_backend(resolve_backend_name(args), config)
    provider = OpenAICompatibleLLMProvider(
        base_url=config.provider.base_url,
        model=config.provider.model,
        api_key=config.provider.api_key,
    )

    if getattr(args, "background", False):
        probe_fn = probe_isolated_background_support
        resolve_fn = resolve_run_mode
        log_fn = log_mode_resolution
        if probe_fn is None:
            from guiclaw.backends.background_runtime import (
                probe_isolated_background_support as runtime_probe_isolated_background_support,
            )

            probe_fn = runtime_probe_isolated_background_support
        if resolve_fn is None:
            from guiclaw.backends.background_runtime import (
                resolve_run_mode as runtime_resolve_run_mode,
            )

            resolve_fn = runtime_resolve_run_mode
        if log_fn is None:
            from guiclaw.backends.background_runtime import (
                log_mode_resolution as runtime_log_mode_resolution,
            )

            log_fn = runtime_log_mode_resolution

        resolved_target_app_class = resolve_target_app_class(args, sys_platform=sys.platform)
        probe = probe_fn(
            sys_platform=sys.platform,
            target_app_class=resolved_target_app_class,
        )
        decision = resolve_fn(
            probe,
            require_isolation=args.require_isolation,
            require_acknowledgement_for_fallback=False,
        )
        log_fn(logger, decision, owner="cli", task=task)

        if decision.mode == "blocked":
            raise RuntimeError(decision.message)

        if decision.mode == "isolated":
            mgr = _build_isolated_display_manager(args, probe)
            if probe.backend_name == "windows_isolated_desktop":
                backend_cls = WindowsIsolatedBackend
                if backend_cls is None:
                    from guiclaw.backends.windows_isolated import (
                        WindowsIsolatedBackend as ImportedWindowsIsolatedBackend,
                    )

                    backend_cls = ImportedWindowsIsolatedBackend
                wrapped_backend = backend_cls(
                    backend,
                    mgr,
                    run_metadata={"owner": "cli", "task": task},
                )
            else:
                backend_cls = BackgroundDesktopBackend
                if backend_cls is None:
                    from guiclaw.backends.background import (
                        BackgroundDesktopBackend as ImportedBackgroundDesktopBackend,
                    )

                    backend_cls = ImportedBackgroundDesktopBackend
                wrapped_backend = backend_cls(
                    backend, mgr, run_metadata={"owner": "cli", "task": task}
                )
            try:
                return await _execute_agent(args, config, wrapped_backend, provider, task)
            finally:
                await wrapped_backend.shutdown()

    return await _execute_agent(args, config, backend, provider, task)


def main(argv: list[str] | None = None) -> int:
    command_args = list(sys.argv[1:] if argv is None else argv)
    if command_args and command_args[0] == "shortcuts":
        from guiclaw.shortcuts import main as shortcuts_main

        return shortcuts_main(command_args[1:])
    args = parse_args(command_args)
    try:
        result = asyncio.run(run_cli(args))
    except Exception as exc:
        result = AgentResult(
            success=False,
            summary="CLI execution failed.",
            model_summary=None,
            trace_path=None,
            steps_taken=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    if args.json_output:
        payload = {
            "success": result.success,
            "summary": result.summary,
            "model_summary": result.model_summary,
            "trace_path": result.trace_path,
            "steps_taken": result.steps_taken,
            "error": result.error,
        }
        print(json.dumps(payload))
    else:
        _print_human_result(result)
    return 0 if result.success else 1


def _print_human_result(result: AgentResult) -> None:
    print(f"status: {'success' if result.success else 'failure'}")
    print(f"success: {'true' if result.success else 'false'}")
    print(f"summary: {result.summary}")
    if result.model_summary is not None:
        print(f"model_summary: {result.model_summary}")
    print(f"trace_path: {result.trace_path}")
    print(f"steps_taken: {result.steps_taken}")
    if result.error is not None:
        print(f"error: {result.error}")


def _build_isolated_display_manager(args: argparse.Namespace, probe: Any) -> Any:
    width = args.width if args.width is not None else 1280
    height = args.height if args.height is not None else 720

    if probe.backend_name == "xvfb":
        from guiclaw.backends.displays.xvfb import XvfbDisplayManager

        display_num = args.display_num if args.display_num is not None else 99
        return XvfbDisplayManager(display_num=display_num, width=width, height=height)

    if probe.backend_name == "windows_isolated_desktop":
        from guiclaw.backends.displays.win32desktop import Win32DesktopManager

        return Win32DesktopManager(width=width, height=height)

    raise RuntimeError(f"Unsupported isolated backend: {probe.backend_name}")


def _make_progress_printer(*, json_output: bool) -> Any:
    async def _progress(message: str) -> None:
        print(_scrub_progress_message(message), file=sys.stderr if json_output else sys.stdout)

    return _progress


class _CliInterventionHandler:
    def __init__(self, backend: Any) -> None:
        self._backend = backend

    async def request_intervention(
        self,
        request: InterventionRequest,
    ) -> InterventionResolution:
        payload = {
            "reason": request.reason,
            "target": _resolve_intervention_target(self._backend, request),
        }
        scrubbed = _scrub_intervention_payload(payload)
        print("intervention requested")
        print(f"reason: {scrubbed['reason']}")
        if scrubbed["target"]:
            print(f"target: {json.dumps(scrubbed['target'], ensure_ascii=False, sort_keys=True)}")
        response = await asyncio.to_thread(input, "type 'resume' to continue: ")
        if response == "resume":
            return InterventionResolution(resume_confirmed=True)
        return InterventionResolution(resume_confirmed=False, note="resume_not_confirmed")


def _build_intervention_handler(backend: Any) -> InterventionHandler:
    return _CliInterventionHandler(backend)


def _resolve_intervention_target(backend: Any, request: InterventionRequest) -> dict[str, Any]:
    target = dict(request.target)
    get_target = getattr(backend, "get_intervention_target", None)
    if callable(get_target):
        backend_target = get_target() or {}
        if isinstance(backend_target, dict):
            target.update(backend_target)
    return {key: value for key, value in target.items() if key in _SAFE_INTERVENTION_TARGET_KEYS}


def _scrub_intervention_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return GuiAgent._scrub_for_log(payload)


def _scrub_progress_message(message: str) -> str:
    if "request intervention:" in message:
        prefix, _separator, _rest = message.partition("request intervention:")
        return f"{prefix}request intervention: <redacted:intervention_reason>"
    if "input text:" in message:
        prefix, _separator, _rest = message.partition("input text:")
        return f"{prefix}input text: <redacted:input_text>"
    return message


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        normalized = dict(message)
        if normalized.get("content") is None:
            normalized["content"] = ""
        sanitized.append(normalized)
    return sanitized


def _coerce_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "".join(parts)
    return str(content)


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        parsed = json_repair.loads(arguments)
    else:
        parsed = arguments
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} config must be a mapping")
    return value


def _require_string(raw: dict[str, Any], key: str) -> str:
    value = _optional_string(raw, key)
    if not value:
        raise ValueError(f"Missing required config key: {key}")
    return value


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _coerce_positive_int(value: Any, *, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected positive integer, got {value!r}") from exc
    return parsed if parsed > 0 else default


def _coerce_non_negative_int(value: Any, *, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected non-negative integer, got {value!r}") from exc
    return parsed if parsed >= 0 else default


def _coerce_image_scale_ratio(value: Any, *, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected image_scale_ratio in (0, 1], got {value!r}") from exc
    if not (0 < parsed <= 1):
        raise ValueError(f"Expected image_scale_ratio in (0, 1], got {value!r}")
    return parsed


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean, got {value!r}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
