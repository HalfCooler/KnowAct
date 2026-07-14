"""Phase 3 nanobot integration tests.

Wave 0 starts with xfail stubs for the full phase boundary, then later tasks in
this plan promote the adapter/config coverage to real passing tests.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

try:
    from guiclaw.interfaces import LLMProvider as GUIClawLLMProvider
    from guiclaw.interfaces import LLMResponse as GUIClawLLMResponse
    from guiclaw.interfaces import ToolCall as GUIClawToolCall
    from guiclaw.memory.retrieval import EmbeddingProvider as GUIClawEmbeddingProvider
except Exception as exc:  # pragma: no cover - only used if imports break
    GUIClawLLMProvider = None
    GUIClawLLMResponse = None
    GUIClawToolCall = None
    GUIClawEmbeddingProvider = None
    _GUICLAW_IMPORT_ERROR: Exception | None = exc
else:
    _GUICLAW_IMPORT_ERROR = None

try:
    from nanobot.config.schema import Config
    from nanobot.providers.base import LLMProvider as NanobotLLMProvider
    from nanobot.providers.base import LLMResponse as NanobotLLMResponse
    from nanobot.providers.base import ToolCallRequest
except Exception as exc:  # pragma: no cover - only used if imports break
    Config = None
    NanobotLLMProvider = object
    NanobotLLMResponse = None
    ToolCallRequest = None
    _NANOBOT_IMPORT_ERROR: Exception | None = exc
else:
    _NANOBOT_IMPORT_ERROR = None


@dataclass
class _QueuedResponse:
    response: Any


def _nanobot_tool_response(
    *,
    content: str,
    arguments: dict[str, Any],
    call_id: str,
) -> Any:
    return NanobotLLMResponse(
        content=_profile_content(content, arguments),
        tool_calls=[
            ToolCallRequest(
                id=call_id,
                name="computer_use",
                arguments=arguments,
            )
        ],
    )


def _profile_content(thought: str, arguments: dict[str, Any]) -> str:
    action_type = arguments.get("action_type")
    if action_type == "done":
        goal_status = "complete" if arguments.get("status", "success") == "success" else "failed"
        action = {"action_type": "status", "goal_status": goal_status}
    elif action_type == "wait":
        action = {"action_type": "wait"}
    elif action_type == "input_text":
        action = {"action_type": "input_text", "text": arguments.get("text", "")}
    elif action_type == "request_intervention":
        action = {"action_type": "request_intervention", "text": arguments.get("text", "")}
    else:
        action = arguments
    return f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}"


def _nanobot_text_response(content: str) -> Any:
    return NanobotLLMResponse(content=content)


def _skill_candidate(*, app: str = "Settings", name: str = "open_settings") -> Any:
    from guiclaw.skills.data import Skill, SkillStep

    return Skill(
        skill_id=f"flat:{name}",
        name=name,
        description="Open Settings from the dry-run menu",
        app=app,
        platform="dry-run",
        tags=("settings",),
        steps=(
            SkillStep(
                action_type="tap",
                target="Menu",
                parameters={"x": 100, "y": 100, "text": "Menu"},
                state_contract={
                    "anchor": {"app_package": app},
                    "signature": {
                        "required": [
                            {
                                "selector": {"resource_id": "dryrun:id/menu"},
                                "state": ["clickable"],
                            }
                        ],
                        "forbidden": [],
                    },
                },
            ),
        ),
    )


def _install_code_skill_trace_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from guiclaw.agent import AgentResult

    async def fake_run(self, task: str, *, max_retries: int = 3, app_hint: str | None = None):
        del task, max_retries, app_hint
        self._trajectory_recorder.start()
        self._trajectory_recorder.record_step(
            action={"action_type": "tap", "x": 100, "y": 100, "text": "Menu"},
            model_output="tap menu",
            foreground_app="Settings",
            interaction_target={
                "selector": {"resource_id": "dryrun:id/menu"},
                "state_contract": {
                    "anchor": {"app_package": "Settings"},
                    "signature": {
                        "required": [
                            {
                                "selector": {"resource_id": "dryrun:id/menu"},
                                "state": ["visible", "clickable"],
                            }
                        ],
                        "forbidden": [],
                    },
                    "mask_rules": [],
                },
            },
            token_usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        )
        trace_path = self._trajectory_recorder.finish(success=True)
        return AgentResult(
            success=True,
            summary="Status: completed",
            trace_path=str(trace_path),
            steps_taken=1,
            error=None,
        )

    monkeypatch.setattr("guiclaw.agent.GuiAgent.run", fake_run)


if _NANOBOT_IMPORT_ERROR is None:

    class _MockNanobotProvider(NanobotLLMProvider):
        """Minimal nanobot LLM provider for adapter tests."""

        def __init__(self, responses: list[Any]) -> None:
            super().__init__(api_key="test-key")
            self._responses = [_QueuedResponse(response=r) for r in responses]
            self.calls: list[dict[str, Any]] = []

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            model: str | None = None,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            reasoning_effort: str | None = None,
            tool_choice: str | dict[str, Any] | None = None,
        ) -> Any:
            return await self.chat_with_retry(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )

        async def chat_with_retry(self, messages, tools=None, model=None, **kwargs) -> Any:
            self.calls.append(
                {
                    "messages": messages,
                    "tools": tools,
                    "model": model,
                    **kwargs,
                }
            )
            if not self._responses:
                raise AssertionError("No scripted nanobot responses left")
            return self._responses.pop(0).response

        def get_default_model(self) -> str:
            return "test-model"

else:

    class _MockNanobotProvider:
        def __init__(self, responses: list[Any]) -> None:
            raise RuntimeError("nanobot provider imports unavailable") from _NANOBOT_IMPORT_ERROR


@pytest.fixture
def tmp_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    guiclaw_home = tmp_path / ".guiclaw"
    (guiclaw_home / "gui_runs").mkdir(parents=True)
    (tmp_path / "gui_skills").mkdir()
    monkeypatch.setattr("guiclaw.paths.DEFAULT_GUICLAW_HOME", guiclaw_home)
    return tmp_path


def test_gui_tool_registered(tmp_workspace: Path) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool
    from nanobot.agent.tools.registry import ToolRegistry

    provider = _MockNanobotProvider([])
    registry = ToolRegistry()
    tool = GuiSubagentTool(
        gui_config=Config(gui={"backend": "dry-run"}).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )
    registry.register(tool)

    assert tool.name == "gui_task"
    assert tool.parameters["required"] == ["task"]
    assert tool.parameters["properties"]["backend"]["enum"] == [
        "adb",
        "ios",
        "hdc",
        "local",
        "dry-run",
    ]
    definitions = registry.get_definitions()
    assert any(defn["function"]["name"] == tool.name for defn in definitions)
    assert tool.description
    assert "do not invent low-level UI paths" in tool.description
    assert (
        "avoid speculative step-by-step UI navigation"
        in tool.parameters["properties"]["task"]["description"]
    )


def test_agent_loop_registers_gui_tool_with_gui_runtime_override(tmp_workspace: Path) -> None:
    from nanobot.agent.loop import AgentLoop

    bus = MagicMock()
    bus.publish_outbound = AsyncMock()
    main_provider = MagicMock()
    main_provider.get_default_model.return_value = "main-model"
    main_provider.generation = SimpleNamespace(max_tokens=4096)
    gui_provider = MagicMock()

    with patch(
        "nanobot.agent.tools.gui.GuiSubagentTool", return_value=MagicMock()
    ) as mock_gui_tool:
        AgentLoop(
            bus=bus,
            provider=main_provider,
            workspace=tmp_workspace,
            model="main-model",
            gui_config=Config(gui={"backend": "dry-run"}).gui,
            gui_provider=gui_provider,
            gui_model="gui-model",
        )

    kwargs = mock_gui_tool.call_args.kwargs
    assert kwargs["provider"] is gui_provider
    assert kwargs["model"] == "gui-model"


@pytest.mark.asyncio
async def test_llm_adapter_maps_response() -> None:
    from nanobot.agent.gui_adapter import NanobotLLMAdapter

    provider = _MockNanobotProvider(
        [
            NanobotLLMResponse(
                content="tap settings",
                tool_calls=[
                    ToolCallRequest(
                        id="tc1",
                        name="computer_use",
                        arguments={"action_type": "tap", "x": 100, "y": 200},
                    )
                ],
            )
        ]
    )
    adapter = NanobotLLMAdapter(provider=provider, model=provider.get_default_model())

    result = await adapter.chat(messages=[{"role": "user", "content": "Open Settings"}])

    assert isinstance(adapter, GUIClawLLMProvider)
    assert isinstance(result, GUIClawLLMResponse)
    assert result.content == "tap settings"
    assert result.tool_calls is not None
    assert isinstance(result.tool_calls[0], GUIClawToolCall)
    assert result.tool_calls[0] == GUIClawToolCall(
        id="tc1",
        name="computer_use",
        arguments={"action_type": "tap", "x": 100, "y": 200},
    )
    assert provider.calls[0]["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_llm_adapter_empty_tool_calls() -> None:
    from nanobot.agent.gui_adapter import NanobotLLMAdapter

    provider = _MockNanobotProvider([NanobotLLMResponse(content="done", tool_calls=[])])
    adapter = NanobotLLMAdapter(provider=provider, model=provider.get_default_model())

    result = await adapter.chat(messages=[{"role": "user", "content": "Finish"}])

    assert result.tool_calls is None


@pytest.mark.asyncio
async def test_llm_adapter_content_none() -> None:
    from nanobot.agent.gui_adapter import NanobotLLMAdapter

    provider = _MockNanobotProvider([NanobotLLMResponse(content=None, tool_calls=[])])
    adapter = NanobotLLMAdapter(provider=provider, model=provider.get_default_model())

    result = await adapter.chat(messages=[{"role": "user", "content": "Continue"}])

    assert result.content == ""


@pytest.mark.asyncio
async def test_llm_adapter_raw_preserved() -> None:
    from nanobot.agent.gui_adapter import NanobotLLMAdapter

    response = NanobotLLMResponse(content="done", tool_calls=[])
    provider = _MockNanobotProvider([response])
    adapter = NanobotLLMAdapter(provider=provider, model=provider.get_default_model())

    result = await adapter.chat(messages=[{"role": "user", "content": "Finish"}])

    assert result.raw is response


@pytest.mark.asyncio
async def test_llm_adapter_tool_choice_passthrough() -> None:
    from nanobot.agent.gui_adapter import NanobotLLMAdapter

    provider = _MockNanobotProvider([NanobotLLMResponse(content="done", tool_calls=[])])
    adapter = NanobotLLMAdapter(provider=provider, model=provider.get_default_model())
    tools = [{"type": "function", "function": {"name": "computer_use"}}]

    await adapter.chat(
        messages=[{"role": "user", "content": "Use the computer"}],
        tools=tools,
        tool_choice="required",
    )

    assert provider.calls[0]["tool_choice"] == "required"
    assert provider.calls[0]["model"] == "test-model"
    assert provider.calls[0]["tools"] == tools


@pytest.mark.asyncio
async def test_embedding_adapter() -> None:
    from nanobot.agent.gui_adapter import NanobotEmbeddingAdapter

    calls: list[list[str]] = []

    async def embed_fn(texts: list[str]) -> np.ndarray:
        calls.append(texts)
        return np.array([[1.0, 2.0]], dtype=np.float32)

    adapter = NanobotEmbeddingAdapter(embed_fn)
    result = await adapter.embed(["hello"])

    assert isinstance(adapter, GUIClawEmbeddingProvider)
    assert calls == [["hello"]]
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (1, 2)


def test_backend_selection(tmp_workspace: Path) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool

    provider = _MockNanobotProvider([])
    gui_config = Config(gui={"backend": "dry-run"}).gui
    assert gui_config is not None
    tool = GuiSubagentTool(
        gui_config=gui_config,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )

    assert tool._backend.platform == "dry-run"
    assert tool._skill_libraries == {}


def test_gui_config_defaults() -> None:
    from nanobot.config.schema import GuiConfig

    config = GuiConfig()

    assert config.backend == "adb"
    assert config.adb.serial is None
    assert config.ios.wda_url == "http://localhost:8100"
    assert config.artifacts_dir == "gui_runs"
    assert config.shortcut_cache_dir == "shortcut_cache"
    assert config.max_steps == 15
    assert config.stagnation_limit == 0
    assert config.image_scale_ratio == pytest.approx(0.5)
    assert config.agent_profile is None


def test_gui_config_validation() -> None:
    from nanobot.config.schema import GuiConfig

    assert GuiConfig(backend="dry-run").backend == "dry-run"
    assert GuiConfig(backend="ios").backend == "ios"
    assert GuiConfig(agent_profile="default").agent_profile == "default"
    assert GuiConfig(agent_profile="gui_owl").agent_profile == "gui_owl"
    assert GuiConfig(agent_profile="qwen3vl").agent_profile == "qwen3vl"
    with pytest.raises(ValidationError, match="Unsupported agent profile"):
        GuiConfig(agent_profile="mobileworld_general_e2e_compact_skill")
    assert GuiConfig.model_validate({"agentProfile": "gelab"}).agent_profile == "gelab"
    assert GuiConfig.model_validate({"imageScaleRatio": 0.25}).image_scale_ratio == pytest.approx(
        0.25
    )
    assert GuiConfig.model_validate({"stagnationLimit": 3}).stagnation_limit == 3
    with pytest.raises(ValidationError):
        GuiConfig(backend="invalid")
    with pytest.raises(ValidationError):
        GuiConfig(agent_profile="invalid-profile")
    with pytest.raises(ValidationError):
        GuiConfig(stagnation_limit=-1)
    with pytest.raises(ValidationError):
        GuiConfig(image_scale_ratio=0)
    with pytest.raises(ValidationError):
        GuiConfig(image_scale_ratio=1.2)


def test_config_gui_none_by_default() -> None:
    config = Config()

    assert config.gui is None


def test_gui_config_nested_aliases() -> None:
    config = Config(
        gui={
            "backend": "ios",
            "ios": {"wdaUrl": "http://127.0.0.1:18100"},
            "artifactsDir": "custom_runs",
            "shortcutCacheDir": "custom_shortcuts",
        }
    )

    assert config.gui is not None
    assert config.gui.backend == "ios"
    assert config.gui.ios.wda_url == "http://127.0.0.1:18100"
    assert config.gui.artifacts_dir == "custom_runs"
    assert config.gui.shortcut_cache_dir == "custom_shortcuts"


@pytest.mark.asyncio
async def test_trajectory_saved_to_guiclaw_home(tmp_workspace: Path) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool

    provider = _MockNanobotProvider(
        [
            _nanobot_text_response('{"mode": "single", "subtasks": []}'),
            _nanobot_tool_response(
                content='Action: {"action_type": "wait"}',
                arguments={"action_type": "wait", "duration_ms": 1},
                call_id="tc_wait",
            ),
            _nanobot_tool_response(
                content='Action: {"action_type": "status", "goal_status": "complete"}',
                arguments={"action_type": "done", "status": "success"},
                call_id="tc_done",
            ),
        ]
    )
    tool = GuiSubagentTool(
        gui_config=Config(gui={"backend": "dry-run"}).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )
    result = json.loads(await tool.execute(task="Open Settings"))

    run_dirs = list((tmp_workspace / ".guiclaw" / "gui_runs").iterdir())
    assert set(result) == {
        "success",
        "summary",
        "model_summary",
        "trace_path",
        "steps_taken",
        "error",
        "post_run_state",
        "duration_s",
        "token_usage",
        "workflow_mode",
    }
    assert result["success"] is True
    assert result["summary"].startswith("Status: completed")
    assert result["post_run_state"]["current_state"] == result["summary"]
    assert result["steps_taken"] == 2
    assert result["error"] is None
    assert Path(result["trace_path"]).is_file()
    assert result["duration_s"] is not None
    assert isinstance(result["token_usage"], dict)
    assert result["workflow_mode"] == "single"
    assert len(run_dirs) == 1
    assert (tmp_workspace / ".guiclaw" / "shortcut_cache").is_dir()
    assert not (tmp_workspace / "shortcut_cache").exists()
    run_dir = run_dirs[0]
    trajectory = json.loads((run_dir / "traj.json").read_text(encoding="utf-8"))
    compact_result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert trajectory["instruction"] == "Open Settings"
    assert [step["action"]["action_type"] for step in trajectory["steps"]] == [
        "wait",
        "done",
    ]
    assert compact_result["run"]["success"] is True
    assert set(compact_result) >= {"run", "subtasks"}
    screenshots = run_dir / "subtasks" / "01_Open_Settings" / "attempt_01" / "screenshots"
    assert (screenshots / "000_initial.png").is_file()
    assert (screenshots / "001_wait.png").is_file()
    assert not list(run_dir.rglob("*.jsonl"))


@pytest.mark.asyncio
async def test_gui_task_workflow_planner_single_falls_back_to_one_agent_run(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent.tools.gui import GuiRouterContext, GuiSubagentTool, GuiWorkflowPlan

    provider = _MockNanobotProvider([])
    tool = GuiSubagentTool(
        gui_config=Config(gui={"backend": "dry-run"}).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )

    plan_workflow = AsyncMock(return_value=GuiWorkflowPlan(mode="single", subtasks=[]))
    monkeypatch.setattr(
        "nanobot.agent.tools.gui.GuiWorkflowRunner._plan_workflow",
        plan_workflow,
    )
    monkeypatch.setattr(
        "nanobot.agent.tools.gui.GuiRouterMemoryRetriever.retrieve",
        lambda *_args, **_kwargs: GuiRouterContext(),
    )
    run_task = AsyncMock(
        return_value=json.dumps(
            {
                "success": True,
                "summary": "done",
                "model_summary": None,
                "trace_path": None,
                "steps_taken": 1,
                "error": None,
            }
        )
    )
    monkeypatch.setattr(GuiSubagentTool, "_run_task", run_task)

    result = json.loads(await tool.execute(task="Open Settings"))

    plan_workflow.assert_awaited_once()
    assert plan_workflow.await_args.args == ("Open Settings",)
    assert plan_workflow.await_args.kwargs["router_context"] is None
    run_task.assert_awaited_once()
    assert run_task.await_args.args == (tool._backend, "Open Settings")
    assert run_task.await_args.kwargs["original_task"] == "Open Settings"
    assert run_task.await_args.kwargs["subtask_index"] == 1
    assert run_task.await_args.kwargs["run_root"].parent == (
        tmp_workspace / ".guiclaw" / "gui_runs"
    )
    assert result["success"] is True
    assert result["summary"] == "done"
    assert result["workflow_mode"] == "single"


@pytest.mark.asyncio
async def test_gui_task_workflow_planner_prompt_keeps_subtasks_high_level() -> None:
    from nanobot.agent.gui_adapter import NanobotLLMAdapter
    from nanobot.agent.tools.gui import GuiWorkflowRunner

    provider = _MockNanobotProvider([_nanobot_text_response('{"mode":"single","subtasks":[]}')])
    runner = GuiWorkflowRunner(
        llm=NanobotLLMAdapter(provider=provider, model=provider.get_default_model()),
        run_task=AsyncMock(),
        load_latest_step_event=lambda _path: {},
    )

    plan = await runner._plan_workflow("Open WeChat and send hello")

    assert plan.mode == "single"
    prompt = provider.calls[0]["messages"][0]["content"]
    assert "high-level app-scoped goal" in prompt
    assert "not a UI action script" in prompt
    assert "leave UI action planning to GuiAgent" in prompt


def test_gui_router_memory_retriever_reads_workspace_evidence(tmp_workspace: Path) -> None:
    from nanobot.agent.tools.gui import GuiRouterMemoryRetriever

    memory_dir = tmp_workspace / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "MEMORY.md").write_text(
        "\n".join(
            [
                "- GUI automation: Meituan triggers 身份核实/验证码 pages that block GUI automation.",
                "- GUI deeplink evidence: tv.danmaku.bili supports `bilibili://search?keyword=关键词` for video search.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (memory_dir / "history.jsonl").write_text(
        json.dumps(
            {
                "cursor": 187,
                "timestamp": "2026-04-21 14:25",
                "content": "Discussed claw-gui skill task decomposition strategy - compound tasks decomposed, single GUI tasks not split.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_workspace / "android_deeplinks.md").write_text(
        "| 哔哩哔哩 | tv.danmaku.bili | bilibili:// | `bilibili://search?keyword=关键词` | 搜索视频 |\n",
        encoding="utf-8",
    )

    context = GuiRouterMemoryRetriever(tmp_workspace).retrieve(
        "在 B 站搜索播放华强买瓜",
        platform="android",
    )

    assert context.app_candidates == ("tv.danmaku.bili",)
    assert any("bilibili://search" in item.text for item in context.evidence)
    assert all(item.source for item in context.evidence)

    generic_context = GuiRouterMemoryRetriever(tmp_workspace).retrieve(
        "搜索并播放一个视频",
        platform="android",
    )
    assert generic_context.app_candidates == ()
    assert generic_context.evidence == ()


@pytest.mark.asyncio
async def test_gui_task_workflow_planner_receives_router_context() -> None:
    from nanobot.agent.gui_adapter import NanobotLLMAdapter
    from nanobot.agent.tools.gui import GuiRouterContext, GuiRouterMemoryEvidence, GuiWorkflowRunner

    provider = _MockNanobotProvider([_nanobot_text_response('{"mode":"single","subtasks":[]}')])
    runner = GuiWorkflowRunner(
        llm=NanobotLLMAdapter(provider=provider, model=provider.get_default_model()),
        run_task=AsyncMock(),
        load_latest_step_event=lambda _path: {},
    )
    context = GuiRouterContext(
        app_candidates=("tv.danmaku.bili",),
        evidence=(
            GuiRouterMemoryEvidence(
                source="memory/history.jsonl:187",
                text="compound tasks decomposed, single GUI tasks not split.",
            ),
        ),
    )

    await runner._plan_workflow("在 B 站搜索播放华强买瓜", router_context=context)

    system_prompt = provider.calls[0]["messages"][0]["content"]
    user_prompt = provider.calls[0]["messages"][1]["content"]
    assert "Memory evidence is advisory" in system_prompt
    assert "lacks an in-app control or follows a system-level setting" in system_prompt
    assert 'app_hint="com.android.settings"' in system_prompt
    assert "Deterministic app candidates" in user_prompt
    assert "tv.danmaku.bili" in user_prompt
    assert "compound tasks decomposed" in user_prompt


@pytest.mark.asyncio
async def test_gui_task_workflow_planner_price_comparison_requires_outputs() -> None:
    from nanobot.agent.gui_adapter import NanobotLLMAdapter
    from nanobot.agent.tools.gui import GuiWorkflowRunner

    task = "帮我在京东、淘宝、拼多多的旗舰店/自营店，对比一加15 16g+512G的价格"
    response = json.dumps(
        {
            "mode": "multi_app",
            "subtasks": [
                {
                    "app_hint": "京东",
                    "task": "在京东查询一加15 16G+512G，仅记录旗舰店或自营店的当前价格。",
                    "inputs": [],
                    "outputs": ["jd_price"],
                },
                {
                    "app_hint": "淘宝",
                    "task": "在淘宝查询一加15 16G+512G，仅记录旗舰店或自营店的当前价格。",
                    "inputs": [],
                    "outputs": ["taobao_price"],
                },
                {
                    "app_hint": "拼多多",
                    "task": "在拼多多查询一加15 16G+512G，仅记录旗舰店或自营店的当前价格。",
                    "inputs": [],
                    "outputs": ["pinduoduo_price"],
                },
            ],
        },
        ensure_ascii=False,
    )
    provider = _MockNanobotProvider([_nanobot_text_response(response)])
    runner = GuiWorkflowRunner(
        llm=NanobotLLMAdapter(provider=provider, model=provider.get_default_model()),
        run_task=AsyncMock(),
        load_latest_step_event=lambda _path: {},
    )

    plan = await runner._plan_workflow(task)
    normalized = runner._normalize_plan_app_hints(plan, platform="android")

    assert plan.mode == "multi_app"
    assert len(plan.subtasks) == 3
    assert [subtask.app_hint for subtask in plan.subtasks] == ["京东", "淘宝", "拼多多"]
    assert [subtask.inputs for subtask in plan.subtasks] == [(), (), ()]
    assert [subtask.outputs for subtask in plan.subtasks] == [
        ("jd_price",),
        ("taobao_price",),
        ("pinduoduo_price",),
    ]
    assert [subtask.app_hint for subtask in normalized.subtasks] == [
        "com.jingdong.app.mall",
        "com.taobao.taobao",
        "com.xunmeng.pinduoduo",
    ]
    prompt = provider.calls[0]["messages"][0]["content"]
    assert "final answer" in prompt
    assert "comparison or research tasks across apps" in prompt
    for subtask in plan.subtasks:
        assert "click" not in subtask.task.lower()
        assert "tap" not in subtask.task.lower()
        assert "swipe" not in subtask.task.lower()


@pytest.mark.asyncio
async def test_gui_task_multi_app_workflow_injects_blackboard_values(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool, GuiWorkflowPlan, GuiWorkflowSubtask

    provider = _MockNanobotProvider([])
    tool = GuiSubagentTool(
        gui_config=Config(gui={"backend": "dry-run"}).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )

    plan = GuiWorkflowPlan(
        mode="multi_app",
        subtasks=[
            GuiWorkflowSubtask(
                task="Open Messages and read the verification code.",
                app_hint="Messages",
                outputs=("code",),
            ),
            GuiWorkflowSubtask(
                task="Open Browser and enter the verification code.",
                app_hint="Browser",
                inputs=("code",),
            ),
        ],
    )
    monkeypatch.setattr(
        "nanobot.agent.tools.gui.GuiWorkflowRunner._plan_workflow",
        AsyncMock(return_value=plan),
    )
    monkeypatch.setattr(
        "nanobot.agent.tools.gui.GuiWorkflowRunner._extract_outputs",
        AsyncMock(return_value={"code": "1234"}),
    )

    run_roots: list[Path] = []
    subtask_indexes: list[int] = []

    async def fake_run_task(active_backend: Any, task: str, **kwargs: Any) -> str:
        del active_backend
        run_root = kwargs["run_root"]
        run_roots.append(run_root)
        subtask_indexes.append(kwargs["subtask_index"])
        return json.dumps(
            {
                "success": True,
                "summary": f"completed: {task}",
                "model_summary": None,
                "trace_path": str(run_root / "traj.json"),
                "steps_taken": 1,
                "error": None,
            }
        )

    run_task = AsyncMock(side_effect=fake_run_task)
    monkeypatch.setattr(GuiSubagentTool, "_run_task", run_task)

    result = json.loads(await tool.execute(task="Copy a code from Messages into Browser"))

    assert run_task.await_count == 2
    second_task = run_task.await_args_list[1].args[1]
    assert "Open Browser and enter the verification code." in second_task
    assert "You must use these known values if relevant: code=1234" in second_task
    assert run_task.await_args_list[0].kwargs["app_hint"] == "messages"
    assert run_task.await_args_list[1].kwargs["app_hint"] == "browser"
    assert run_roots[0] == run_roots[1]
    assert run_roots[0].parent == tmp_workspace / ".guiclaw" / "gui_runs"
    assert subtask_indexes == [1, 2]
    assert all(
        call.kwargs["original_task"] == "Copy a code from Messages into Browser"
        for call in run_task.await_args_list
    )
    assert len(list((tmp_workspace / ".guiclaw" / "gui_runs").iterdir())) == 1
    assert result["success"] is True
    assert result["trace_path"] == str(run_roots[0] / "traj.json")
    assert result["workflow_mode"] == "multi_app"
    assert result["blackboard"] == {"code": "1234"}
    assert result["subtasks"][0]["app_hint"] == "messages"
    assert result["subtasks"][1]["app_hint"] == "browser"
    saved_result = json.loads((run_roots[0] / "result.json").read_text(encoding="utf-8"))
    assert saved_result["run"]["success"] is True
    assert saved_result["run"]["workflow_mode"] == "multi_app"
    assert saved_result["blackboard"] == {"code": "1234"}
    assert len(saved_result["subtasks"]) == 2


def test_gui_task_schedules_postprocessing_once_per_routed_subtask(
    tmp_workspace: Path,
) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool

    provider = _MockNanobotProvider([])
    tool = GuiSubagentTool(
        gui_config=Config(
            gui={
                "backend": "dry-run",
                "enableMemoryExtraction": True,
                "enableSkillExtraction": True,
            }
        ).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )
    schedule = MagicMock()
    tool._postprocessor.schedule = schedule
    run_root = tmp_workspace / "gui_runs" / "one-run"

    tool._schedule_postprocessing(
        run_root,
        {
            "success": True,
            "subtasks": [
                {"task": "Read the message", "success": True},
                {"task": "Enter the code", "success": True},
            ],
        },
        platform="android",
        original_task="Copy a code between apps",
    )

    assert schedule.call_count == 2
    assert [call.args[0] for call in schedule.call_args_list] == [
        run_root / "traj.json",
        run_root / "traj.json",
    ]
    assert [call.kwargs["subtask_index"] for call in schedule.call_args_list] == [1, 2]
    assert [call.kwargs["task"] for call in schedule.call_args_list] == [
        "Read the message",
        "Enter the code",
    ]


@pytest.mark.asyncio
async def test_gui_task_multi_app_workflow_stops_on_missing_declared_output(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool, GuiWorkflowPlan, GuiWorkflowSubtask

    provider = _MockNanobotProvider([])
    tool = GuiSubagentTool(
        gui_config=Config(gui={"backend": "dry-run"}).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )

    plan = GuiWorkflowPlan(
        mode="multi_app",
        subtasks=[
            GuiWorkflowSubtask(
                task="Open Shop and read the tracking number.",
                app_hint="Shop",
                outputs=("tracking_number",),
            ),
            GuiWorkflowSubtask(
                task="Open Notes and save the tracking number.",
                app_hint="Notes",
                inputs=("tracking_number",),
            ),
        ],
    )
    monkeypatch.setattr(
        "nanobot.agent.tools.gui.GuiWorkflowRunner._plan_workflow",
        AsyncMock(return_value=plan),
    )
    monkeypatch.setattr(
        "nanobot.agent.tools.gui.GuiWorkflowRunner._extract_outputs",
        AsyncMock(return_value={}),
    )
    run_task = AsyncMock(
        return_value=json.dumps(
            {
                "success": True,
                "summary": "Opened Shop but no tracking number was visible.",
                "model_summary": None,
                "trace_path": None,
                "steps_taken": 1,
                "error": None,
            }
        )
    )
    monkeypatch.setattr(GuiSubagentTool, "_run_task", run_task)

    result = json.loads(await tool.execute(task="Copy a tracking number from Shop into Notes"))

    run_task.assert_awaited_once()
    assert result["success"] is False
    assert result["workflow_mode"] == "multi_app"
    assert result["error"] == "missing_workflow_output"
    assert result["missing_outputs"] == ["tracking_number"]
    assert "tracking_number" in result["summary"]
    assert result["subtasks"][0]["success"] is True
    assert result["blackboard"] == {}


@pytest.mark.asyncio
async def test_gui_task_workflow_normalizes_android_app_hints(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool, GuiWorkflowPlan, GuiWorkflowSubtask

    provider = _MockNanobotProvider([])
    tool = GuiSubagentTool(
        gui_config=Config(gui={"backend": "dry-run"}).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )
    tool._backend = SimpleNamespace(platform="android")

    monkeypatch.setattr(
        "nanobot.agent.tools.gui.GuiWorkflowRunner._plan_workflow",
        AsyncMock(
            return_value=GuiWorkflowPlan(
                mode="multi_app",
                subtasks=[
                    GuiWorkflowSubtask(
                        task="Open Settings and copy the device name.",
                        app_hint="Settings",
                        outputs=("device_name",),
                    ),
                    GuiWorkflowSubtask(
                        task="Open Chrome and search for the device name.",
                        app_hint="Chrome",
                        inputs=("device_name",),
                    ),
                ],
            )
        ),
    )
    monkeypatch.setattr(
        "nanobot.agent.tools.gui.GuiWorkflowRunner._extract_outputs",
        AsyncMock(return_value={"device_name": "Pixel"}),
    )
    run_task = AsyncMock(
        return_value=json.dumps(
            {
                "success": True,
                "summary": "done",
                "model_summary": None,
                "trace_path": None,
                "steps_taken": 1,
                "error": None,
            }
        )
    )
    monkeypatch.setattr(GuiSubagentTool, "_run_task", run_task)

    result = json.loads(await tool.execute(task="Use Settings value in Chrome"))

    assert result["success"] is True
    assert run_task.await_args_list[0].kwargs["app_hint"] == "com.android.settings"
    assert run_task.await_args_list[1].kwargs["app_hint"] == "com.android.chrome"
    assert result["subtasks"][0]["app_hint"] == "com.android.settings"
    assert result["subtasks"][1]["app_hint"] == "com.android.chrome"


@pytest.mark.asyncio
async def test_gui_task_returns_state_note_for_partial_run(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from guiclaw.agent import GuiAgent as GUIClawAgent
    from nanobot.agent.tools.gui import GuiSubagentTool

    original_run = GUIClawAgent.run

    async def _single_attempt_run(
        self, task: str, *, max_retries: int = 3, app_hint: str | None = None
    ):
        del max_retries
        return await original_run(self, task, max_retries=1, app_hint=app_hint)

    monkeypatch.setattr(
        "guiclaw.postprocessing.PostRunProcessor._summarize_trajectory",
        AsyncMock(return_value=""),
    )
    monkeypatch.setattr("guiclaw.agent.GuiAgent.run", _single_attempt_run)

    provider = _MockNanobotProvider(
        [
            _nanobot_text_response('{"mode": "single", "subtasks": []}'),
            _nanobot_tool_response(
                content='Action: {"action_type": "wait"}',
                arguments={"action_type": "wait", "duration_ms": 1},
                call_id="tc_wait",
            ),
        ]
    )
    tool = GuiSubagentTool(
        gui_config=Config(gui={"backend": "dry-run", "maxSteps": 1}).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )
    result = json.loads(await tool.execute(task="Open Settings"))

    assert set(result) == {
        "success",
        "summary",
        "model_summary",
        "trace_path",
        "steps_taken",
        "error",
        "post_run_state",
        "duration_s",
        "token_usage",
        "workflow_mode",
    }
    assert result["success"] is False
    assert result["summary"].startswith("Status: partial")
    assert "Done:" in result["summary"]
    assert "Remaining:" in result["summary"]
    assert "Current:" in result["summary"]
    assert "Resume:" in result["summary"]
    assert result["post_run_state"]["current_state"] == result["summary"]
    assert result["error"] == "max_steps_exceeded"
    trace_path = Path(result["trace_path"])
    assert trace_path.name == "traj.json"
    assert (trace_path.parent / "result.json").is_file()
    assert result["duration_s"] is not None
    assert isinstance(result["token_usage"], dict)
    assert result["workflow_mode"] == "single"


@pytest.mark.asyncio
async def test_auto_skill_extraction(tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool

    extract_calls: list[dict[str, Any]] = []

    async def fake_induce_compact_skills_from_trace(
        extractor: object,
        trace_path: Path,
        *,
        is_success: bool,
        subtask_index: int,
    ):
        del extractor
        extract_calls.append(
            {
                "trace_path": trace_path,
                "is_success": is_success,
                "subtask_index": subtask_index,
            }
        )
        return [_skill_candidate()]

    _install_code_skill_trace_run(monkeypatch)
    monkeypatch.setattr(
        "guiclaw.postprocessing.induce_compact_skills_from_trace",
        fake_induce_compact_skills_from_trace,
    )
    monkeypatch.setattr(
        "guiclaw.postprocessing.PostRunProcessor._summarize_trajectory", AsyncMock(return_value="")
    )

    provider = _MockNanobotProvider([])
    tool = GuiSubagentTool(
        gui_config=Config(
            gui={
                "backend": "dry-run",
                "enableSkillExtraction": True,
                "enableSkillExecution": False,
            }
        ).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )
    result = json.loads(await tool.execute(task="Open calculator"))
    await tool._wait_for_pending_postprocessing()

    assert result["success"] is True
    assert len(extract_calls) == 1
    assert extract_calls[0]["is_success"] is True
    assert extract_calls[0]["trace_path"] == Path(result["trace_path"])
    run_result = json.loads(
        (Path(result["trace_path"]).parent / "result.json").read_text(encoding="utf-8")
    )
    extraction_result = run_result["extraction"]["1"]
    assert extraction_result["status"] == "processed_code"
    assert run_result["subtasks"][0]["task"] == "Open calculator"
    assert "open_settings" in extraction_result["updated_functions"]
    assert (tmp_workspace / "gui_skills" / "skills.py").is_file()


@pytest.mark.asyncio
async def test_auto_skill_extraction_none_is_graceful(
    tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool

    extract = AsyncMock(return_value=[])
    _install_code_skill_trace_run(monkeypatch)
    monkeypatch.setattr("guiclaw.postprocessing.induce_compact_skills_from_trace", extract)
    monkeypatch.setattr(
        "guiclaw.postprocessing.PostRunProcessor._summarize_trajectory", AsyncMock(return_value="")
    )

    provider = _MockNanobotProvider([])
    tool = GuiSubagentTool(
        gui_config=Config(
            gui={
                "backend": "dry-run",
                "enableSkillExtraction": True,
                "enableSkillExecution": False,
            }
        ).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )

    result = json.loads(await tool.execute(task="Open calculator"))
    await tool._wait_for_pending_postprocessing()

    assert result["success"] is True
    extract.assert_awaited_once()
    run_result = json.loads(
        (Path(result["trace_path"]).parent / "result.json").read_text(encoding="utf-8")
    )
    extraction_result = run_result["extraction"]["1"]
    assert extraction_result["status"] == "no_candidate"


@pytest.mark.asyncio
async def test_auto_skill_extraction_persists_to_normalized_bucket(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool

    async def fake_extract_multi(extractor: object, *args: Any, **kwargs: Any):
        del extractor, args, kwargs
        return [_skill_candidate(app=" Settings ")]

    _install_code_skill_trace_run(monkeypatch)
    monkeypatch.setattr(
        "guiclaw.postprocessing.induce_compact_skills_from_trace",
        fake_extract_multi,
    )
    monkeypatch.setattr(
        "guiclaw.postprocessing.PostRunProcessor._summarize_trajectory", AsyncMock(return_value="")
    )

    provider = _MockNanobotProvider([])
    tool = GuiSubagentTool(
        gui_config=Config(
            gui={
                "backend": "dry-run",
                "enableSkillExtraction": True,
                "enableSkillExecution": False,
            }
        ).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )

    result = json.loads(await tool.execute(task="Open calculator"))
    await tool._wait_for_pending_postprocessing()
    code_store = tmp_workspace / "gui_skills" / "skills.py"
    reloaded = tool._get_skill_library("dry-run")
    reloaded.load_all()

    assert result["success"] is True
    assert code_store.is_file()
    assert len(reloaded.list_all(platform="dry-run", app="Settings")) == 1
    assert reloaded.list_all(platform="dry-run", app="settings")[0].app == "settings"


@pytest.mark.asyncio
async def test_execute_creates_fresh_trajectory_recorder(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from guiclaw.agent import AgentResult
    from nanobot.agent.tools.gui import GuiSubagentTool

    recorders: list[Any] = []

    async def fake_run(self, task: str, *, max_retries: int = 3, app_hint: str | None = None):
        recorders.append(self._trajectory_recorder)
        self._trajectory_recorder.start()
        self._trajectory_recorder.record_step(action={"action_type": "wait"}, model_output="wait")
        self._trajectory_recorder.record_step(action={"action_type": "done"}, model_output="done")
        trace_path = self._trajectory_recorder.finish(success=True)
        return AgentResult(
            success=True,
            summary=f"completed {task}",
            trace_path=str(trace_path),
            steps_taken=2,
            error=None,
        )

    monkeypatch.setattr("guiclaw.agent.GuiAgent.run", fake_run)
    monkeypatch.setattr(
        "guiclaw.skills.extractor.SkillExtractor.extract_from_file",
        AsyncMock(return_value=None),
    )

    provider = _MockNanobotProvider([])
    tool = GuiSubagentTool(
        gui_config=Config(gui={"backend": "dry-run"}).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )

    first = json.loads(await tool.execute(task="Open app"))
    second = json.loads(await tool.execute(task="Open app"))

    assert len(recorders) == 2
    assert recorders[0] is not recorders[1]
    assert first["trace_path"] != second["trace_path"]


def test_agent_loop_registers_gui_tool(tmp_workspace: Path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_workspace,
        model="test-model",
        gui_config=Config(gui={"backend": "dry-run"}).gui,
    )

    assert loop.tools.has("gui_task")


def test_agent_loop_no_gui_config(tmp_workspace: Path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_workspace,
        model="test-model",
    )

    assert not loop.tools.has("gui_task")


def test_scaffolding_uses_phase3_patterns() -> None:
    """Sanity check the helper imports used by later tasks."""
    assert asyncio.iscoroutinefunction(_MockNanobotProvider.chat_with_retry)
    assert json.dumps({"phase": 3}) == '{"phase": 3}'
