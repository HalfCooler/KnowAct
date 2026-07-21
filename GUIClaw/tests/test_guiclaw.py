from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import logging
import time
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from PIL import Image

import guiclaw.backends.adb as adb_backend_module
import guiclaw.backends.hdc as hdc_backend_module
import guiclaw.backends.ios_wda as ios_wda_module
import guiclaw.skills.deeplink as deeplink_module
import guiclaw.skills.executor as skill_executor_module
from guiclaw.action import Action, ActionError, parse_action, resolve_coordinate
from guiclaw.agent import GuiAgent, StepResult
from guiclaw.agent_profiles import (
    canonicalize_agent_profile,
    normalize_profile_response_for_screen,
    profile_tool_definition,
)
from guiclaw.backends.adb import AdbBackend, AdbError
from guiclaw.backends.dry_run import DryRunBackend
from guiclaw.backends.hdc import HdcBackend
from guiclaw.image_utils import scale_image
from guiclaw.interfaces import LLMResponse, ToolCall
from guiclaw.observation import Observation
from guiclaw.skills.action_grounder import ActionGrounder as _AgentActionGrounder
from guiclaw.skills.data import Skill, SkillStep
from guiclaw.skills.deeplink import AppShortcutProfile, DeepIntent, DeepLink
from guiclaw.skills.flat import FlatSkillLibrary, compile_flat_skills, export_skills_to_source
from guiclaw.skills.subgoal_runner import SubgoalRunner as _AgentSubgoalRunner
from guiclaw.tool_schemas import build_shortcut_tool_defs
from guiclaw.trajectory.recorder import TrajectoryRecorder


def _make_recorder(
    tmp_path: Path,
    task: str = "test task",
    *,
    events: list[dict[str, Any]] | None = None,
) -> TrajectoryRecorder:
    return TrajectoryRecorder(
        output_dir=tmp_path / "traj",
        task=task,
        event_callback=events.append if events is not None else None,
    )


def _write_test_png(path: Path, *, size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path, format="PNG")


def _normalize_test_profile_response(profile_name: str, response: LLMResponse) -> LLMResponse:
    return normalize_profile_response_for_screen(
        profile_name,
        response,
        screen_width=999,
        screen_height=999,
        fallback_relative=True,
    )


def _mobileworld_response(
    action: dict,
    *,
    thought: str = "Proceed with the next MobileWorld action.",
    usage: dict[str, int] | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}",
        tool_calls=None,
        usage=usage or {},
    )


def _messages_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
    return "\n".join(part for part in parts if part)


def _last_user_message_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _messages_text([message])
    return ""


def _qwen_response(
    arguments: dict,
    *,
    thought: str = "Proceed with the next MobileWorld action.",
    action_text: str = "Execute the next action",
    usage: dict[str, int] | None = None,
) -> LLMResponse:
    tool_call = {"name": "mobile_use", "arguments": arguments}
    return LLMResponse(
        content=(
            f"Thought: {thought}\n"
            f'Action: "{action_text}"\n'
            f"<tool_call>{json.dumps(tool_call, ensure_ascii=False)}</tool_call>"
        ),
        tool_calls=None,
        usage=usage or {},
    )


def _mobileworld_action_from_tool_args(args: dict) -> dict:
    action_type = str(args.get("action_type") or args.get("action") or "").strip().lower()
    if action_type in {"tap", "click", "long_press", "double_tap", "double_click"}:
        mapped = {"tap": "click", "double_click": "double_tap"}.get(action_type, action_type)
        return {
            "action_type": mapped,
            "coordinate": [
                args.get("x", 500),
                args.get("y", 500),
            ],
        }
    if action_type == "drag":
        return {
            "action_type": "drag",
            "start_coordinate": [args.get("x", 500), args.get("y", 500)],
            "end_coordinate": [args.get("x2", 500), args.get("y2", 500)],
        }
    if action_type == "scroll":
        return {
            "action_type": "scroll",
            "direction": args.get("direction") or args.get("text", "down"),
        }
    if action_type == "input_text":
        return {"action_type": "input_text", "text": args.get("text", "")}
    if action_type == "open_app":
        return {"action_type": "open_app", "app_name": args.get("text", "")}
    if action_type in {"back", "navigate_back"}:
        return {"action_type": "navigate_back"}
    if action_type in {"home", "navigate_home"}:
        return {"action_type": "navigate_home"}
    if action_type in {"enter", "keyboard_enter"}:
        return {"action_type": "keyboard_enter"}
    if action_type == "wait":
        return {"action_type": "wait"}
    if action_type == "done":
        status = str(args.get("status") or "").lower()
        text = str(args.get("text") or "").lower()
        failed = status == "failure" or "fail" in text or "unable" in text
        return {"action_type": "status", "goal_status": "infeasible" if failed else "complete"}
    if action_type == "request_intervention":
        return {"action_type": "ask_user", "text": args.get("text", "")}
    return {"action_type": action_type or "wait"}


def _coerce_mobileworld_response(response: LLMResponse) -> LLMResponse:
    if not response.tool_calls:
        return response
    tool_call = response.tool_calls[0]
    action = _mobileworld_action_from_tool_args(tool_call.arguments)
    thought = response.content or "Use the scripted action."
    return _mobileworld_response(action, thought=thought, usage=response.usage)


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    return ""


class _ScriptedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        if not self._responses:
            raise AssertionError("No scripted responses left.")
        response = self._responses.pop(0)
        return response if tools else _coerce_mobileworld_response(response)


class _RecordingLLM(_ScriptedLLM):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(responses)
        self.calls: list[list[dict]] = []

    async def chat(
        self, messages, tools=None, tool_choice=None, model=None, max_tokens=None
    ) -> LLMResponse:
        self.calls.append(copy.deepcopy(messages))
        return await super().chat(messages, tools=tools, tool_choice=tool_choice)


class _RecordingValidator:
    def __init__(self, returns: list[bool]) -> None:
        self._returns = list(returns)
        self.calls: list[dict[str, object]] = []

    async def validate(self, valid_state: str, screenshot=None) -> bool:
        self.calls.append({"valid_state": valid_state, "screenshot": screenshot})
        if not self._returns:
            raise AssertionError("No validator responses left.")
        return self._returns.pop(0)


class _SkillTestBackend:
    def __init__(self) -> None:
        self.platform = "dry-run"
        self.executed_actions: list[object] = []
        self.observe_calls: list[Path] = []

    async def execute(self, action, timeout: float = 5.0) -> str:
        del timeout
        self.executed_actions.append(action)
        return f"executed:{action.action_type}"

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        self.observe_calls.append(screenshot_path)
        _write_test_png(screenshot_path)
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=1000,
            screen_height=1000,
            foreground_app="DryRun",
            platform=self.platform,
        )

    async def preflight(self) -> None:
        return None

    async def list_apps(self) -> list[str]:
        return []


class _FakePromptSkillExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[Skill, dict[str, str]]] = []

    async def execute(self, skill: Skill, params: dict[str, str], *, timeout: float = 30.0):
        del timeout
        self.calls.append((skill, params))
        return skill_executor_module.SkillExecutionResult(
            skill=skill,
            step_results=[],
            state=skill_executor_module.ExecutionState.SUCCEEDED,
            execution_summary=f"Skill {skill.name} executed",
            token_usage={"prompt_tokens": 2},
        )


@pytest.mark.asyncio
async def test_prompt_skill_selection_injects_and_dispatches_use_skill(tmp_path: Path) -> None:
    library = FlatSkillLibrary(store_dir=tmp_path / "skills")
    shortcut_skill = Skill(
        skill_id="shortcut:dl:dry:search",
        name="dry_search",
        description="Search videos by query",
        app="dry.app",
        platform="dry-run",
        tags=("shortcut", "deeplink", "validated"),
        parameters=("query",),
        steps=(
            SkillStep(
                action_type="open_deeplink",
                target="dry://search?q={{query}}",
                parameters={"text": "dry://search?q={{query}}", "package": "dry.app"},
            ),
        ),
    )
    composite_skill = Skill(
        skill_id="manual:click-and-type",
        name="click_then_type",
        description="Tap an input and type text",
        app="dry.app",
        platform="dry-run",
        tags=("compact_action", "action_alias:click_then_type"),
        steps=(),
    )
    library.add(shortcut_skill)
    library.add(composite_skill)
    executor = _FakePromptSkillExecutor()
    llm = _RecordingLLM(
        [
            _mobileworld_response(
                {
                    "action_type": "use_skill",
                    "skill_id": "shortcut:dl:dry:search",
                    "skill_name": "dry_search",
                    "arguments": {"query": "cats"},
                }
            ),
            _mobileworld_response({"action_type": "status", "goal_status": "complete"}),
        ]
    )
    agent = GuiAgent(
        llm,
        _SkillTestBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "prompt skill"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        skill_library=library,
        skill_executor=executor,
        enable_prompt_skill_selection=True,
        prompt_skill_top_k=3,
        prompt_shortcut_only=True,
        agent_profile="general_e2e",
    )

    result = await agent.run("Search videos by query cats", max_retries=1)

    assert result.success
    assert len(executor.calls) == 1
    executed_skill, executed_params = executor.calls[0]
    assert executed_skill.skill_id == "shortcut:dl:dry:search"
    assert executed_params == {"query": "cats"}
    first_prompt = _messages_text(llm.calls[0])
    assert "`use_skill`" in first_prompt
    assert "skill_id=shortcut:dl:dry:search" in first_prompt
    assert "`click_then_type`" in first_prompt


@pytest.mark.asyncio
async def test_default_prompt_skill_selection_dispatches_native_use_skill(
    tmp_path: Path,
) -> None:
    skill_id = "compact:dry.app:search"
    library = FlatSkillLibrary(store_dir=tmp_path / "skills")
    skill = Skill(
        skill_id=skill_id,
        name="dry_search",
        description="Search videos by query",
        app="dry.app",
        platform="dry-run",
        tags=("compact", "compact_extracted"),
        parameters=("query",),
        steps=(
            SkillStep(
                action_type="open_app",
                target="dry.app",
                fixed=True,
                fixed_values={"text": "dry.app"},
            ),
        ),
    )
    library.add(skill)
    executor = _FakePromptSkillExecutor()

    class RecordingNativeLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.responses = [
                LLMResponse(
                    content="Action: Run the matching search skill",
                    tool_calls=[
                        ToolCall(
                            id="skill-1",
                            name="computer_use",
                            arguments={
                                "action_type": "use_skill",
                                "skill_id": skill_id,
                                "arguments": {"query": "cats"},
                                "intent": "Use the matching search shortcut",
                                "summary": "A matching retrieved skill is available",
                            },
                        )
                    ],
                ),
                LLMResponse(
                    content="Action: Finish the completed task",
                    tool_calls=[
                        ToolCall(
                            id="done-1",
                            name="computer_use",
                            arguments={
                                "action_type": "done",
                                "status": "success",
                                "text": "Search completed",
                                "intent": "Finish",
                                "summary": "The skill completed the task",
                            },
                        )
                    ],
                ),
            ]

        async def chat(self, **kwargs: Any) -> LLMResponse:
            self.calls.append(copy.deepcopy(kwargs))
            return self.responses.pop(0)

    llm = RecordingNativeLLM()
    agent = GuiAgent(
        llm,
        _SkillTestBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "native prompt skill"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        skill_library=library,
        skill_executor=executor,
        enable_prompt_skill_selection=True,
        prompt_skill_top_k=3,
        prompt_shortcut_only=False,
        agent_profile="default",
    )

    result = await agent.run("Search videos by query cats", max_retries=1)

    assert result.success
    assert len(executor.calls) == 1
    executed_skill, executed_params = executor.calls[0]
    assert executed_skill.skill_id == skill_id
    assert executed_params == {"query": "cats"}
    first_call = llm.calls[0]
    first_prompt = _messages_text(first_call["messages"])
    assert skill_id in first_prompt
    assert "does not need to match the entire task" in first_prompt
    assert "a useful prefix, or an intermediate subgoal" in first_prompt
    assert "continue the remaining GUI steps" in first_prompt
    assert "MUST call `use_skill` before `open_app`" in first_prompt
    parameters = first_call["tools"][0]["function"]["parameters"]
    assert "use_skill" in parameters["properties"]["action_type"]["enum"]
    assert "skill_id" in parameters["properties"]
    assert "arguments" in parameters["properties"]


@pytest.mark.asyncio
async def test_prompt_composite_action_executes_without_skill_executor(tmp_path: Path) -> None:
    library = FlatSkillLibrary(store_dir=tmp_path / "skills")
    library.add(
        Skill(
            skill_id="manual:click-and-type",
            name="click_then_type",
            description="Tap an input and type text",
            app="dry.app",
            platform="dry-run",
            tags=("compact_action", "action_alias:click_then_type"),
            steps=(),
        )
    )
    backend = _SkillTestBackend()
    llm = _RecordingLLM(
        [
            _mobileworld_response(
                {
                    "action_type": "click_then_type",
                    "coordinate": [500, 400],
                    "text": "hello",
                }
            ),
            _mobileworld_response({"action_type": "status", "goal_status": "complete"}),
        ]
    )
    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "prompt composite"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        skill_library=library,
        enable_prompt_skill_selection=True,
        prompt_skill_top_k=0,
        agent_profile="general_e2e",
    )

    result = await agent.run("Tap the field and type hello", max_retries=1)

    assert result.success
    assert [action.action_type for action in backend.executed_actions] == ["tap", "input_text"]
    assert backend.executed_actions[0].x == 500
    assert backend.executed_actions[0].y == 400
    assert backend.executed_actions[1].text == "hello"


def test_parse_scroll_allows_center_default() -> None:
    action = parse_action(
        {
            "action_type": "scroll",
            "direction": "left",
            "pixels": 180,
        }
    )

    assert action.x is None
    assert action.y is None
    assert action.text == "left"


def test_parse_scroll_rejects_partial_coordinates() -> None:
    with pytest.raises(ActionError, match="requires both 'x' and 'y'"):
        parse_action(
            {
                "action_type": "scroll",
                "x": 100,
                "pixels": 180,
            }
        )


def test_parse_action_unwraps_singleton_coordinate_lists() -> None:
    action = parse_action(
        {
            "action": "tap",
            "x": [417],
            "y": [129],
            "relative": True,
        }
    )

    assert action.action_type == "tap"
    assert action.x == 417.0
    assert action.y == 129.0
    assert action.relative is True


def test_parse_action_splits_paired_coordinates_from_x_list() -> None:
    action = parse_action(
        {
            "action": "tap",
            "x": [498, 441],
            "relative": True,
        }
    )

    assert action.action_type == "tap"
    assert action.x == 498.0
    assert action.y == 441.0
    assert action.relative is True


def test_parse_action_splits_paired_coordinates_from_stringified_x_list() -> None:
    action = parse_action(
        {
            "action": "tap",
            "x": "[903, 130]",
            "relative": "true",
        }
    )

    assert action.action_type == "tap"
    assert action.x == 903.0
    assert action.y == 130.0
    assert action.relative is True


def test_parse_action_maps_coordinate_alias_pair() -> None:
    action = parse_action(
        {
            "action": "tap",
            "coordinate": [500, 261],
            "relative": True,
        }
    )

    assert action.action_type == "tap"
    assert action.x == 500.0
    assert action.y == 261.0
    assert action.relative is True


def test_parse_action_maps_stringified_coordinate_alias_pair() -> None:
    action = parse_action(
        {
            "action": "tap",
            "coordinate": "[780, 503]",
        }
    )

    assert action.action_type == "tap"
    assert action.x == 780.0
    assert action.y == 503.0


def test_relative_observation_prompt_normalizes_ui_tree_bounds_without_screen_resolution() -> None:
    observation = Observation(
        screenshot_path=None,
        screen_width=488,
        screen_height=1080,
        foreground_app="com.example",
        platform="android",
        extra={
            "capture_source": "scrcpy",
            "ui_tree": [
                {"class": "android.widget.FrameLayout", "bounds": "[0,0][1080,2376]"},
                {
                    "text": "我的订单",
                    "class": "android.widget.TextView",
                    "bounds": "[258,723][390,768]",
                },
            ],
        },
    )

    text = observation.to_user_text(
        "open orders",
        step_index=4,
        coordinate_instruction="Use relative coordinates in [0, 999] for both x and y, and set relative=true.",
    )

    assert "Screen:" not in text
    assert "488 x 1080" not in text
    assert "Platform: android" in text
    assert "'bounds':" not in text
    assert '"bounds":' not in text
    assert "relative_bounds" in text
    assert "[0,0][999,999]" in text
    assert "[239,304][361,323]" in text
    assert "[258,723][390,768]" not in text


@pytest.mark.asyncio
async def test_adb_ui_tree_capture_retries_without_compressed() -> None:
    backend = AdbBackend(use_scrcpy=False, collect_ui_tree=True)
    calls: list[tuple[str, ...]] = []
    xml = (
        '<hierarchy><node text="Open" class="android.widget.Button" '
        'clickable="true" enabled="true" bounds="[0,0][10,10]" /></hierarchy>'
    )

    async def fake_run(*args: str, timeout: float = 5.0) -> str:
        del timeout
        calls.append(args)
        if args == (
            "shell",
            "uiautomator",
            "dump",
            "--compressed",
            adb_backend_module._DEVICE_UI_XML_PATH,
        ):
            raise RuntimeError("compressed dump failed")
        if args == (
            "shell",
            "uiautomator",
            "dump",
            adb_backend_module._DEVICE_UI_XML_PATH,
        ):
            return "UI hierchary dumped"
        if args == ("shell", "cat", adb_backend_module._DEVICE_UI_XML_PATH):
            return xml
        raise AssertionError(f"Unexpected adb call: {args}")

    backend._run = fake_run  # type: ignore[method-assign]

    extra = await backend._collect_ui_tree_extra(1.0)

    assert extra["visible_text"] == ["Open"]
    assert extra["clickable_text"] == ["Open"]
    assert calls == [
        (
            "shell",
            "uiautomator",
            "dump",
            "--compressed",
            adb_backend_module._DEVICE_UI_XML_PATH,
        ),
        (
            "shell",
            "uiautomator",
            "dump",
            adb_backend_module._DEVICE_UI_XML_PATH,
        ),
        ("shell", "cat", adb_backend_module._DEVICE_UI_XML_PATH),
    ]


@pytest.mark.asyncio
async def test_adb_ui_tree_capture_logs_warning_after_retry_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = AdbBackend(use_scrcpy=False, collect_ui_tree=True)
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout: float = 5.0) -> str:
        del timeout
        calls.append(args)
        raise RuntimeError("dump failed")

    backend._run = fake_run  # type: ignore[method-assign]
    caplog.set_level(logging.WARNING, logger=adb_backend_module.__name__)

    extra = await backend._collect_ui_tree_extra(1.0)

    assert extra == {
        "ui_tree_error": "dump failed",
        "ui_tree_timeout_s": 15.0,
    }

    assert calls == [
        (
            "shell",
            "uiautomator",
            "dump",
            "--compressed",
            adb_backend_module._DEVICE_UI_XML_PATH,
        ),
        (
            "shell",
            "uiautomator",
            "dump",
            adb_backend_module._DEVICE_UI_XML_PATH,
        ),
    ]
    assert "ADB UI-tree capture unavailable after retries" in caplog.text


@pytest.mark.asyncio
async def test_adb_open_intent_uses_uri_extra_for_stream_payload() -> None:
    backend = AdbBackend()
    calls: list[list[str]] = []

    async def fake_run_am_start(remote_args: list[str], *, timeout: float, label: str) -> str:
        del timeout, label
        calls.append(remote_args)
        return "ok"

    backend._run_am_start_with_fallbacks = fake_run_am_start  # type: ignore[method-assign]

    result = await backend._open_intent(
        Action(
            action_type="open_intent",
            intent_action="android.intent.action.SEND",
            package="com.google.android.youtube",
            mime_type="image/*",
            extras=(("android.intent.extra.STREAM", "file:///sdcard/Download/probe.png"),),
        ),
        timeout=1.0,
    )

    assert result == "ok"
    remote_args = calls[0]
    assert "--grant-read-uri-permission" in remote_args
    assert "--eu" in remote_args
    assert remote_args[remote_args.index("--eu") + 2] == "file:///sdcard/Download/probe.png"


@pytest.mark.asyncio
async def test_agent_lazy_loads_shortcuts_for_foreground_app_and_updates_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _SkillTestBackend()
    backend.platform = "android"
    cache_dir = tmp_path / "shortcut-cache"
    profile = AppShortcutProfile(
        package="com.example.app",
        deep_links=(
            DeepLink(
                uri_template="example://home",
                scheme="example",
                host="home",
                path=None,
                component="com.example.app/.MainActivity",
                description="Open example home",
            ),
        ),
    )
    calls: list[tuple[object, str]] = []

    async def fake_extract(shortcut_backend: object, package: str) -> AppShortcutProfile:
        calls.append((shortcut_backend, package))
        return profile

    monkeypatch.setattr(deeplink_module, "extract_app_shortcuts", fake_extract)
    agent = GuiAgent(
        _ScriptedLLM([]),
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "lazy shortcuts"),
        shortcut_backend=backend,
        shortcut_cache_dir=str(cache_dir),
    )

    await agent._ensure_shortcuts_for_app("com.example.app")
    await agent._ensure_shortcuts_for_app("com.example.app")

    assert calls == [(backend, "com.example.app")]
    assert (cache_dir / "com.example.app.json").exists()
    shortcut_name = next(iter(agent._shortcut_action_map))
    assert shortcut_name.startswith("com_example_app__example_home__")
    assert agent._shortcut_action_map[shortcut_name] == (
        "open_deeplink",
        "example://home",
        "com.example.app/.MainActivity",
        None,
    )


@pytest.mark.asyncio
async def test_agent_shortcut_discovery_uses_adb_alias_and_fails_soft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _SkillTestBackend()
    backend.platform = "android"
    cache_dir = tmp_path / "shortcut-cache"
    calls: list[tuple[object, str]] = []

    async def fake_extract(shortcut_backend: object, package: str) -> AppShortcutProfile:
        calls.append((shortcut_backend, package))
        raise AdbError(f"adb failed (exit 1): adb shell pm path {package}", returncode=1)

    monkeypatch.setattr(deeplink_module, "extract_app_shortcuts", fake_extract)
    agent = GuiAgent(
        _ScriptedLLM([]),
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "shortcut fail soft"),
        shortcut_backend=backend,
        shortcut_cache_dir=str(cache_dir),
    )

    await agent._ensure_shortcuts_for_app("org.joinmastodon.android")

    assert calls == [(backend, "org.joinmastodon.android")]
    assert "org.joinmastodon.android" in agent._shortcuts
    profile = agent._shortcuts["org.joinmastodon.android"]
    assert profile.manifest_meta["status"] == "shortcut_discovery_failed"
    assert not (cache_dir / "org.joinmastodon.android.json").exists()
    assert "org.joinmastodon.android.mastodon" not in agent._shortcuts


def test_static_shortcut_extraction_ignores_path_without_host_and_pattern_paths() -> None:
    manifest = ET.fromstring(
        """
        <manifest xmlns:android="http://schemas.android.com/apk/res/android" package="com.example.app">
          <application>
            <activity android:name=".NoHost">
              <intent-filter>
                <action android:name="android.intent.action.VIEW" />
                <category android:name="android.intent.category.BROWSABLE" />
                <data android:scheme="foo" android:path="/ignored" />
              </intent-filter>
            </activity>
            <activity android:name=".Pattern">
              <intent-filter>
                <action android:name="android.intent.action.VIEW" />
                <category android:name="android.intent.category.BROWSABLE" />
                <data android:scheme="bar" android:host="user" android:pathPattern="/profile/.*" />
              </intent-filter>
            </activity>
            <activity android:name=".Prefix">
              <intent-filter>
                <action android:name="android.intent.action.VIEW" />
                <category android:name="android.intent.category.BROWSABLE" />
                <data android:scheme="baz" android:host="user" android:pathPrefix="/profile" />
              </intent-filter>
            </activity>
          </application>
        </manifest>
        """
    )

    filters = deeplink_module._extract_all_filters(manifest, "com.example.app")
    links = deeplink_module._classify_deep_links(filters)

    assert [link.uri_template for link in links] == ["foo:", "baz://user/profile"]
    assert links[0].path is None
    assert links[1].path_kind == "pathPrefix"
    assert "Static Android deep link candidate" in links[1].description
    assert "not page-validated" in links[1].description


def test_static_shortcut_extraction_parses_deep_intent_mime_type() -> None:
    manifest = ET.fromstring(
        """
        <manifest xmlns:android="http://schemas.android.com/apk/res/android" package="com.example.app">
          <application>
            <activity android:name=".Share">
              <intent-filter>
                <action android:name="android.intent.action.SEND" />
                <category android:name="android.intent.category.DEFAULT" />
                <data android:mimeType="text/plain" />
              </intent-filter>
            </activity>
          </application>
        </manifest>
        """
    )

    filters = deeplink_module._extract_all_filters(manifest, "com.example.app")
    intents = deeplink_module._classify_deep_intents(filters)

    assert len(intents) == 1
    assert intents[0].action == "android.intent.action.SEND"
    assert intents[0].mime_type == "text/plain"
    assert "Static Android intent candidate" in intents[0].description
    assert "not page-validated" in intents[0].description


def test_shortcut_agent_tool_names_do_not_collide() -> None:
    profile = AppShortcutProfile(
        package="com.example.app",
        deep_links=(
            DeepLink(
                uri_template="example://user/profile",
                scheme="example",
                host="user",
                path="/profile",
                component="com.example.app/.Router",
                description="Static Android deep link candidate",
                path_kind="path",
            ),
            DeepLink(
                uri_template="example://user/settings",
                scheme="example",
                host="user",
                path="/settings",
                component="com.example.app/.Router",
                description="Static Android deep link candidate",
                path_kind="path",
            ),
        ),
        deep_intents=(
            DeepIntent(
                action="android.intent.action.SEND",
                component="com.example.app/.Share",
                mime_type="text/plain",
                description="Static Android intent candidate",
            ),
            DeepIntent(
                action="android.intent.action.SEND",
                component="com.example.app/.Share",
                mime_type="image/png",
                description="Static Android intent candidate",
            ),
        ),
    )

    tools, action_map = build_shortcut_tool_defs({"com.example.app": profile})
    names = [tool["function"]["name"] for tool in tools]
    assert len(set(names)) == len(names)
    assert len(action_map) == 4
    intent_entries = [value for value in action_map.values() if value[0] == "open_intent"]
    assert {entry[3] for entry in intent_entries} == {"text/plain", "image/png"}


def test_validated_deeplink_promotes_to_concise_flat_skill() -> None:
    skill = deeplink_module.validated_shortcut_to_skill(
        {
            "package": "com.taobao.taobao",
            "kind": "deeplink",
            "status": "page_validated",
            "description": "淘宝搜索商品",
            "name": "taobao_search",
            "uri_template": "taobao://search?q={{query}}",
            "component": "com.taobao.taobao/.RouterActivity",
            "valid_state": "淘宝搜索结果页已打开",
        }
    )

    assert skill is not None
    assert skill.app == "com.taobao.taobao"
    assert skill.description == "淘宝搜索商品"
    assert skill.tags == ("shortcut", "deeplink", "validated")
    assert skill.parameters == ("query",)
    assert skill.success_count == 1
    assert skill.steps[0].action_type == "open_deeplink"
    assert skill.steps[0].parameters["text"] == "taobao://search?q={{query}}"
    assert skill.steps[0].parameters["package"] == "com.taobao.taobao"

    source = export_skills_to_source([skill])
    assert "description='淘宝搜索商品'" in source
    assert "am start" not in source
    compiled = compile_flat_skills(source)
    assert compiled.errors == ()
    assert compiled.skills[0].parameters == ("query",)


@pytest.mark.asyncio
async def test_validated_intent_promotes_through_flat_library_add_or_merge(tmp_path: Path) -> None:
    library = FlatSkillLibrary(store_dir=tmp_path)
    record = {
        "package": "com.example.app",
        "kind": "intent",
        "status": "page_validated",
        "description": "在应用内搜索内容",
        "name": "example_search",
        "intent_action": "android.intent.action.SEARCH",
        "extras": [["query", "{{query}}"]],
        "valid_state": "搜索结果页已打开",
    }

    decision, skill_id = await deeplink_module.add_validated_shortcut_skill(library, record)
    decision_again, skill_id_again = await deeplink_module.add_validated_shortcut_skill(
        library, record
    )

    assert decision == "ADD"
    assert decision_again == "KEEP_NEW"
    assert skill_id == skill_id_again
    skills = library.list_all()
    assert len(skills) == 1
    skill = skills[0]
    assert skill.description == "在应用内搜索内容"
    assert skill.steps[0].action_type == "open_intent"
    assert skill.steps[0].parameters["intent_action"] == "android.intent.action.SEARCH"
    assert skill.steps[0].parameters["extras"] == (("query", "{{query}}"),)
    assert skill.parameters == ("query",)


def test_validated_shortcut_skips_unvalidated_record() -> None:
    skill = deeplink_module.validated_shortcut_to_skill(
        {
            "package": "com.example.app",
            "kind": "deeplink",
            "status": "launchable",
            "description": "打开示例页面",
            "uri_template": "example://home",
        }
    )

    assert skill is None


def test_intent_extra_placeholders_are_grounded_recursively() -> None:
    step = SkillStep(
        action_type="open_intent",
        target="android.intent.action.SEARCH",
        parameters={
            "intent_action": "android.intent.action.SEARCH",
            "extras": (("query", "{{query}}"),),
        },
    )

    action = skill_executor_module._build_template_action(step, {"query": "手机 壳"})

    assert action.action_type == "open_intent"
    assert action.intent_action == "android.intent.action.SEARCH"
    assert action.extras == (("query", "手机 壳"),)


def test_parse_swipe_maps_start_and_end_coordinate_aliases() -> None:
    action = parse_action(
        {
            "action_type": "swipe",
            "start_coordinate": [120, 340],
            "end_coordinate": [760, 355],
            "relative": True,
        }
    )

    assert action.action_type == "swipe"
    assert action.x == 120.0
    assert action.y == 340.0
    assert action.x2 == 760.0
    assert action.y2 == 355.0
    assert action.relative is True


def test_parse_swipe_maps_coordinate_and_coordinate2_aliases() -> None:
    action = parse_action(
        {
            "action": "swipe",
            "coordinate": [120, 340],
            "coordinate2": [760, 355],
        }
    )

    assert action.action_type == "swipe"
    assert action.x == 120.0
    assert action.y == 340.0
    assert action.x2 == 760.0
    assert action.y2 == 355.0


def test_parse_action_unwraps_duplicated_y_list() -> None:
    action = parse_action(
        {
            "action_type": "tap",
            "x": 321,
            "y": [957, 957],
            "relative": True,
        }
    )

    assert action.action_type == "tap"
    assert action.x == 321.0
    assert action.y == 957.0
    assert action.relative is True


def test_parse_action_unwraps_duplicated_stringified_y_list() -> None:
    action = parse_action(
        {
            "action_type": "tap",
            "x": 321,
            "y": "[957, 957]",
            "relative": True,
        }
    )

    assert action.action_type == "tap"
    assert action.x == 321.0
    assert action.y == 957.0
    assert action.relative is True


def test_scale_image_accepts_custom_ratio() -> None:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (120, 80), color=(255, 0, 0)).save(buf, format="PNG")

    scaled = scale_image(buf.getvalue(), scale_ratio=0.25)
    with Image.open(io.BytesIO(scaled)) as img:
        assert img.size == (30, 20)


def test_scale_image_ratio_one_keeps_original_bytes() -> None:
    raw = b"not-an-image"
    assert scale_image(raw, scale_ratio=1.0) == raw


def test_parse_swipe_splits_all_coordinates_from_x_list() -> None:
    action = parse_action(
        {
            "action_type": "swipe",
            "x": [120, 340, 760, 355],
            "relative": True,
        }
    )

    assert action.action_type == "swipe"
    assert action.x == 120.0
    assert action.y == 340.0
    assert action.x2 == 760.0
    assert action.y2 == 355.0
    assert action.relative is True


def test_parse_swipe_splits_all_coordinates_from_stringified_x_list() -> None:
    action = parse_action(
        {
            "action_type": "swipe",
            "x": "[120, 340, 760, 355]",
            "relative": True,
        }
    )

    assert action.action_type == "swipe"
    assert action.x == 120.0
    assert action.y == 340.0
    assert action.x2 == 760.0
    assert action.y2 == 355.0
    assert action.relative is True


def test_parse_swipe_rejects_noop_path() -> None:
    with pytest.raises(ActionError, match="start and end coordinates must differ"):
        parse_action(
            {
                "action_type": "swipe",
                "x": 500,
                "y": 700,
                "x2": 500,
                "y2": 700,
                "relative": True,
            }
        )


def test_parse_swipe_accepts_endpoint_only_coordinates() -> None:
    action = parse_action(
        {
            "action_type": "swipe",
            "x2": 500,
            "y2": 749,
            "relative": True,
        }
    )

    assert action.action_type == "swipe"
    assert action.x == 500.0
    assert action.y2 == 749.0
    assert action.y is not None
    assert action.y != action.y2


def test_parse_action_accepts_mobileworld_navigation_aliases() -> None:
    enter = parse_action({"action_type": "keyboard_enter"})
    recents = parse_action({"action_type": "recents"})

    assert enter.action_type == "enter"
    assert recents.action_type == "app_switch"


def test_default_profile_tool_definition_uses_native_computer_schema() -> None:
    params = profile_tool_definition("default")["function"]["parameters"]

    assert profile_tool_definition("default")["function"]["name"] == "computer_use"
    assert params["required"] == ["action_type", "intent", "summary"]
    assert "action_type" in params["properties"]


def test_general_e2e_compact_skill_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported agent profile"):
        canonicalize_agent_profile("general_e2e_compact_skill")


def test_annotate_android_apps_filters_unmapped_packages() -> None:
    from guiclaw.skills.normalization import annotate_android_apps

    result = annotate_android_apps(["com.sankuai.meituan", "com.unknown.xyz"])

    assert len(result) == 1, f"Expected 1 entry, got {len(result)}: {result}"
    assert "美团" in result[0] or "Meituan" in result[0]
    assert not any("com.unknown.xyz" in entry for entry in result)


def test_android_normalization_supports_common_chinese_and_english_aliases() -> None:
    from guiclaw.skills.normalization import (
        find_android_app_in_text,
        find_android_apps_in_text,
        normalize_adb_app_identifier,
        normalize_app_identifier,
    )

    cases = {
        "喜马拉雅": "com.ximalaya.ting.android",
        "喜马拉雅App": "com.ximalaya.ting.android",
        "打开喜马拉雅App": "com.ximalaya.ting.android",
        "ximalaya app": "com.ximalaya.ting.android",
        "open ximalaya app": "com.ximalaya.ting.android",
        "himalaya": "com.ximalaya.ting.android",
        "B站": "tv.danmaku.bili",
        "小红书客户端": "com.xingin.xhs",
        "QQ音乐": "com.tencent.qqmusic",
        "百度地图": "com.baidu.BaiduMap",
        "Google Maps app": "com.google.android.apps.maps",
        "Chrome browser": "com.android.chrome",
        "钉钉": "com.alibaba.android.rimet",
        "Messages": "com.google.android.apps.messaging",
        "SMS app": "com.google.android.apps.messaging",
        "com.android.mms": "com.google.android.apps.messaging",
        "Calendar": "org.fossify.calendar",
        "com.google.android.calendar": "org.fossify.calendar",
        "Documents": "com.google.android.documentsui",
        "com.android.documentsui": "com.google.android.documentsui",
        "Mattermost": "com.mattermost.rnbeta",
        "com.mattermost.rn": "com.mattermost.rnbeta",
        "Mastodon": "org.joinmastodon.android.mastodon",
        "deskclock": "com.google.android.deskclock",
        "Taobao": "com.testmall.app",
        "com.taobao.taobao": "com.testmall.app",
        "Gallery": "gallery.photomanager.picturegalleryapp.imagegallery",
    }

    for alias, package in cases.items():
        assert normalize_app_identifier("android", alias) == package

    adb_cases = {
        "Mastodon": "org.joinmastodon.android",
        "Mastodon App": "org.joinmastodon.android",
        "org.joinmastodon.android": "org.joinmastodon.android",
        "org.joinmastodon.android.mastodon": "org.joinmastodon.android",
        "com.taobao.taobao": "com.taobao.taobao",
    }
    for alias, package in adb_cases.items():
        assert normalize_adb_app_identifier(alias) == package

    text_cases = {
        "In YouTube, search for Never Gonna Give You Up.": "com.google.android.youtube",
        "帮我在油管搜索 Rick Astley": "com.google.android.youtube",
        "在 B 站搜索播放华强买瓜": "tv.danmaku.bili",
        "打开小破站看看推荐视频": "tv.danmaku.bili",
        "帮我用喜马拉雅FM播放三国演义": "com.ximalaya.ting.android",
        "In the messaging app, reply OK to the invitation.": "com.google.android.apps.messaging",
        "In the calendar app, schedule a lunch event.": "org.fossify.calendar",
        "Open Mattermost and check the support channel.": "com.mattermost.rnbeta",
    }
    for text, package in text_cases.items():
        assert find_android_app_in_text(text) == package

    assert set(find_android_apps_in_text("帮我在京东、淘宝、拼多多对比价格")) == {
        "com.xunmeng.pinduoduo",
        "com.taobao.taobao",
        "com.jingdong.app.mall",
    }


def test_gui_agent_skill_app_filter_prefers_hint_then_task_text(tmp_path: Path) -> None:
    class AndroidDryRunBackend(DryRunBackend):
        @property
        def platform(self) -> str:
            return "android"

    agent = GuiAgent(
        _ScriptedLLM([]),
        AndroidDryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path),
        artifacts_root=tmp_path / "runs",
    )

    assert (
        agent._skill_app_filter("In YouTube, search for a video.", None)
        == "com.google.android.youtube"
    )
    assert agent._skill_app_filter("Search for a video.", "B站") == "tv.danmaku.bili"
    assert agent._skill_app_filter("Search for a video.", None) is None
    task_with_hints = (
        "Send the file 'waiver.jpg' as an email attachment to bob@gmail.com. "
        "Title the email 'Updated waiver'.\n\n"
        "Advisory hints from past GUI memory:\n"
        "- [guiclaw/gui_memory:org.joinmastodon.android.mastodon:failure] "
        "Locate server features within Settings > Server"
    )
    assert agent._skill_app_filter(task_with_hints, None) == "com.gmailclone"


@pytest.mark.asyncio
async def test_adb_backend_ensure_yadb_pushes_packaged_asset_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = AdbBackend()
    yadb_asset = tmp_path / "yadb"
    yadb_asset.write_bytes(b"jar-data")

    monkeypatch.setattr(backend, "_get_packaged_yadb_path", lambda: yadb_asset)
    run_mock = AsyncMock(
        side_effect=[
            AdbError("missing"),
            "",
            "",
        ]
    )
    monkeypatch.setattr(backend, "_run", run_mock)

    assert await backend._ensure_yadb_available(timeout=5.0) is True
    assert run_mock.await_args_list[0].args == ("shell", "ls", "/data/local/tmp/yadb")
    assert run_mock.await_args_list[1].args == ("push", str(yadb_asset), "/data/local/tmp/yadb")
    assert run_mock.await_args_list[2].args == ("shell", "chmod", "755", "/data/local/tmp/yadb")


@pytest.mark.asyncio
async def test_adb_backend_ensure_yadb_skips_push_when_device_already_has_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = AdbBackend()
    yadb_asset = tmp_path / "yadb"
    yadb_asset.write_bytes(b"jar-data")

    monkeypatch.setattr(backend, "_get_packaged_yadb_path", lambda: yadb_asset)
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    assert await backend._ensure_yadb_available(timeout=5.0) is True
    run_mock.assert_awaited_once_with("shell", "ls", "/data/local/tmp/yadb", timeout=5.0)


@pytest.mark.asyncio
async def test_adb_backend_resolves_relative_tap(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = AdbBackend()
    backend._screen_width = 200
    backend._screen_height = 400
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = parse_action(
        {
            "action_type": "tap",
            "x": 500,
            "y": 250,
            "relative": True,
        }
    )
    await backend.execute(action)

    expected_x = resolve_coordinate(500, 200, relative=True)
    expected_y = resolve_coordinate(250, 400, relative=True)
    run_mock.assert_awaited_once_with(
        "shell",
        "input",
        "tap",
        str(expected_x),
        str(expected_y),
        timeout=5.0,
    )


@pytest.mark.asyncio
async def test_adb_backend_observe_prefers_screenshot_size(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = AdbBackend(use_scrcpy=False)
    run_mock = AsyncMock(return_value="")
    screen_size_mock = AsyncMock(return_value=(1080, 2376))
    foreground_mock = AsyncMock(return_value="com.example.app")

    monkeypatch.setattr(backend, "_run", run_mock)
    monkeypatch.setattr(backend, "_query_screen_size", screen_size_mock)
    monkeypatch.setattr(backend, "_query_foreground_app", foreground_mock)
    monkeypatch.setattr(adb_backend_module, "read_png_size", lambda _path: (2376, 1080))

    obs = await backend.observe(Path("/tmp/adb-orientation.png"))

    assert obs.screen_width == 2376
    assert obs.screen_height == 1080
    assert backend._screen_width == 2376
    assert backend._screen_height == 1080
    screen_size_mock.assert_not_awaited()
    foreground_mock.assert_awaited_once_with(30.0)


@pytest.mark.asyncio
async def test_adb_backend_observe_falls_back_when_screenshot_size_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend(use_scrcpy=False)
    run_mock = AsyncMock(return_value="")
    screen_size_mock = AsyncMock(return_value=(1080, 2376))
    foreground_mock = AsyncMock(return_value="com.example.app")

    monkeypatch.setattr(backend, "_run", run_mock)
    monkeypatch.setattr(backend, "_query_screen_size", screen_size_mock)
    monkeypatch.setattr(backend, "_query_foreground_app", foreground_mock)
    monkeypatch.setattr(adb_backend_module, "read_png_size", lambda _path: None)

    obs = await backend.observe(Path("/tmp/adb-orientation-fallback.png"))

    assert obs.screen_width == 1080
    assert obs.screen_height == 2376
    screen_size_mock.assert_awaited_once_with(30.0)
    foreground_mock.assert_awaited_once_with(30.0)


@pytest.mark.asyncio
async def test_adb_backend_foreground_app_uses_activity_dump_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(
        return_value="""
      mResumedActivity: ActivityRecord{123 u0 com.coloros.calendar/.MainActivity t12}
    """
    )
    monkeypatch.setattr(backend, "_run", run_mock)

    assert await backend._query_foreground_app(timeout=5.0) == "com.coloros.calendar"
    run_mock.assert_awaited_once_with(
        "shell",
        "dumpsys",
        "activity",
        "activities",
        timeout=10.0,
    )


@pytest.mark.asyncio
async def test_adb_backend_foreground_app_falls_back_to_window_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(
        side_effect=[
            "header without resumed activity",
            "mCurrentFocus=Window{42 u0 com.android.settings/com.android.settings.Settings}",
        ]
    )
    monkeypatch.setattr(backend, "_run", run_mock)

    assert await backend._query_foreground_app(timeout=5.0) == "com.android.settings"
    assert run_mock.await_args_list[0].args == ("shell", "dumpsys", "activity", "activities")
    assert run_mock.await_args_list[1].args == ("shell", "dumpsys", "window", "windows")


def test_adb_backend_extract_foreground_app_supports_multiple_android_signals() -> None:
    assert (
        AdbBackend._extract_foreground_app(
            "topResumedActivity=ActivityRecord{7 u0 com.heytap.browser/.Main t11}"
        )
        == "com.heytap.browser"
    )
    assert (
        AdbBackend._extract_foreground_app(
            "mFocusedApp=AppWindowToken{ token=Token{ ActivityRecord{7 u0 com.android.settings/.Settings t11}}}"
        )
        == "com.android.settings"
    )
    assert AdbBackend._extract_foreground_app("no foreground info here") == "unknown"


@pytest.mark.asyncio
async def test_hdc_backend_observe_prefers_screenshot_size(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = HdcBackend()
    run_mock = AsyncMock(return_value="")
    screen_size_mock = AsyncMock(return_value=(1080, 2376))
    foreground_mock = AsyncMock(return_value="com.example.harmony")

    class _FakeImageContext:
        def __enter__(self) -> "_FakeImageContext":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

        def save(self, path: str, format: str = "PNG") -> None:
            del format
            Path(path).write_bytes(b"png")

    class _FakeImage:
        @staticmethod
        def open(_path: str) -> _FakeImageContext:
            return _FakeImageContext()

    monkeypatch.setattr(backend, "_run", run_mock)
    monkeypatch.setattr(backend, "_query_screen_size", screen_size_mock)
    monkeypatch.setattr(backend, "_query_foreground_app", foreground_mock)
    monkeypatch.setattr(hdc_backend_module, "_import_pil_image", lambda: _FakeImage)
    monkeypatch.setattr(hdc_backend_module, "read_png_size", lambda _path: (2376, 1080))

    obs = await backend.observe(Path("/tmp/hdc-orientation.png"))

    assert obs.screen_width == 2376
    assert obs.screen_height == 1080
    assert backend._screen_width == 2376
    assert backend._screen_height == 1080
    screen_size_mock.assert_not_awaited()
    foreground_mock.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_hdc_backend_observe_falls_back_when_screenshot_size_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = HdcBackend()
    run_mock = AsyncMock(return_value="")
    screen_size_mock = AsyncMock(return_value=(1080, 2376))
    foreground_mock = AsyncMock(return_value="com.example.harmony")

    class _FakeImageContext:
        def __enter__(self) -> "_FakeImageContext":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

        def save(self, path: str, format: str = "PNG") -> None:
            del format
            Path(path).write_bytes(b"png")

    class _FakeImage:
        @staticmethod
        def open(_path: str) -> _FakeImageContext:
            return _FakeImageContext()

    monkeypatch.setattr(backend, "_run", run_mock)
    monkeypatch.setattr(backend, "_query_screen_size", screen_size_mock)
    monkeypatch.setattr(backend, "_query_foreground_app", foreground_mock)
    monkeypatch.setattr(hdc_backend_module, "_import_pil_image", lambda: _FakeImage)
    monkeypatch.setattr(hdc_backend_module, "read_png_size", lambda _path: None)

    obs = await backend.observe(Path("/tmp/hdc-orientation-fallback.png"))

    assert obs.screen_width == 1080
    assert obs.screen_height == 2376
    screen_size_mock.assert_awaited_once_with(5.0)
    foreground_mock.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_hdc_backend_query_screen_size_preserves_landscape_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = HdcBackend()
    run_mock = AsyncMock(return_value="Display Resolution: 2376x1080")
    monkeypatch.setattr(backend, "_run", run_mock)

    width, height = await backend._query_screen_size(timeout=5.0)

    assert width == 2376
    assert height == 1080


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direction", "expected_y2"),
    [("down", 280), ("up", 520)],
)
async def test_hdc_backend_scroll_vertical_direction_is_inverted(
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
    expected_y2: int,
) -> None:
    backend = HdcBackend()
    backend._screen_width = 400
    backend._screen_height = 800
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = parse_action(
        {
            "action_type": "scroll",
            "direction": direction,
            "pixels": 120,
        }
    )
    await backend.execute(action)

    run_mock.assert_awaited_once_with(
        "shell",
        "uitest",
        "uiInput",
        "swipe",
        "200",
        "400",
        "200",
        str(expected_y2),
        "2000",
        timeout=5.0,
    )


@pytest.mark.asyncio
async def test_ios_backend_observe_swaps_window_size_when_orientation_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    screenshot_path = tmp_path / "ios-mismatch.png"

    class _FakeImage:
        size = (2376, 1080)

        def save(self, path: str, format: str = "PNG") -> None:
            del format
            Path(path).write_bytes(b"png")

    class _FakeClient:
        def __init__(self, _url: str) -> None:
            pass

        def screenshot(self) -> _FakeImage:
            return _FakeImage()

        def window_size(self) -> dict[str, int]:
            return {"width": 1080, "height": 2376}

        def app_current(self) -> dict[str, str]:
            return {"bundleId": "com.example.ios"}

    class _FakeWdaModule:
        Client = _FakeClient

    monkeypatch.setattr(ios_wda_module, "_import_wda", lambda: _FakeWdaModule)
    backend = ios_wda_module.WdaBackend()

    obs = await backend.observe(screenshot_path)

    assert obs.screen_width == 2376
    assert obs.screen_height == 1080
    assert backend._screen_width == 2376
    assert backend._screen_height == 1080


@pytest.mark.asyncio
async def test_ios_backend_observe_keeps_window_size_when_orientation_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    screenshot_path = tmp_path / "ios-match.png"

    class _FakeImage:
        size = (1080, 2376)

        def save(self, path: str, format: str = "PNG") -> None:
            del format
            Path(path).write_bytes(b"png")

    class _FakeClient:
        def __init__(self, _url: str) -> None:
            pass

        def screenshot(self) -> _FakeImage:
            return _FakeImage()

        def window_size(self) -> dict[str, int]:
            return {"width": 1080, "height": 2376}

        def app_current(self) -> dict[str, str]:
            return {"bundleId": "com.example.ios"}

    class _FakeWdaModule:
        Client = _FakeClient

    monkeypatch.setattr(ios_wda_module, "_import_wda", lambda: _FakeWdaModule)
    backend = ios_wda_module.WdaBackend()

    obs = await backend.observe(screenshot_path)

    assert obs.screen_width == 1080
    assert obs.screen_height == 2376
    assert backend._screen_width == 1080
    assert backend._screen_height == 2376


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direction", "expected_y2"),
    [("down", 280), ("up", 520)],
)
async def test_ios_backend_scroll_vertical_direction_is_inverted(
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
    expected_y2: int,
) -> None:
    class _FakeSession:
        def __init__(self) -> None:
            self.swipe_calls: list[tuple[int, int, int, int, float]] = []

        def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_s: float) -> None:
            self.swipe_calls.append((x1, y1, x2, y2, duration_s))

    class _FakeClient:
        def __init__(self, _url: str) -> None:
            self._session = _FakeSession()

        def session(self) -> _FakeSession:
            return self._session

        def window_size(self) -> dict[str, int]:
            return {"width": 1080, "height": 2376}

        def app_current(self) -> dict[str, str]:
            return {"bundleId": "com.example.ios"}

    class _FakeWdaModule:
        Client = _FakeClient

    monkeypatch.setattr(ios_wda_module, "_import_wda", lambda: _FakeWdaModule)
    backend = ios_wda_module.WdaBackend()
    monkeypatch.setattr(backend, "_wda_call", AsyncMock(side_effect=lambda fn, *args: fn(*args)))
    backend._screen_width = 400
    backend._screen_height = 800

    action = parse_action(
        {
            "action_type": "scroll",
            "direction": direction,
            "pixels": 120,
        }
    )
    await backend.execute(action)

    assert backend._client._session.swipe_calls == [(200, 400, 200, expected_y2, 0.3)]


def test_agent_uses_absolute_coordinates_for_mobileworld_profiles(tmp_path: Path) -> None:
    qwen_agent = GuiAgent(
        _ScriptedLLM([]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "qwen"),
        model="qwen-vl-max",
        agent_profile="general_e2e",
    )
    gemini_agent = GuiAgent(
        _ScriptedLLM([]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "gemini"),
        model="gemini-2.5-pro",
        agent_profile="general_e2e",
    )

    action = parse_action({"action_type": "tap", "x": 500, "y": 250})

    assert qwen_agent._normalize_relative_coordinates(action).relative is False
    assert gemini_agent._normalize_relative_coordinates(action).relative is False
    assert qwen_agent._coordinate_mode() == "absolute"


def test_agent_keeps_absolute_coordinates_for_other_models(tmp_path: Path) -> None:
    agent = GuiAgent(
        _ScriptedLLM([]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "other"),
        model="gpt-5.2",
    )

    action = parse_action({"action_type": "tap", "x": 500, "y": 250})

    assert agent._normalize_relative_coordinates(action).relative is False
    assert agent._coordinate_mode() == "absolute"


def test_qwen3vl_profile_normalizes_content_only_response() -> None:
    response = LLMResponse(
        content=(
            "Thought: I found the target\n"
            'Action: "Tap target"\n'
            '<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,250]}}</tool_call>'
        ),
        tool_calls=None,
    )

    normalized = _normalize_test_profile_response("qwen3vl", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].name == "computer_use"
    assert normalized.tool_calls[0].arguments == {
        "action_type": "tap",
        "x": 500,
        "y": 250,
        "relative": True,
        "summary": "Tap target",
        "intent": "Tap target",
    }


def test_qwen3vl_profile_rejects_action_json_without_tool_call() -> None:
    response = LLMResponse(
        content=('Thought: I found a target\nAction: {"action":"click","coordinate":[500,250]}'),
        tool_calls=None,
    )

    with pytest.raises(ValueError):
        _normalize_test_profile_response("qwen3vl", response)


def test_qwen3vl_profile_uses_mobileworld_conclusion_as_summary() -> None:
    response = LLMResponse(
        content=(
            "Thought: I found the target\n"
            'Action: "Tap login"\n'
            '<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,250],"summary":"tap login button","intent":"tap login button"}}</tool_call>'
        ),
        tool_calls=None,
    )

    normalized = _normalize_test_profile_response("qwen3vl", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].arguments == {
        "action_type": "tap",
        "x": 500,
        "y": 250,
        "relative": True,
        "summary": "Tap login",
        "intent": "Tap login",
    }


def test_qwen3vl_profile_parses_action_type_key() -> None:
    response = LLMResponse(
        content=(
            "Thought: Wait here\n"
            'Action: "Wait"\n'
            '<tool_call>{"name":"mobile_use","arguments":{"action":"wait"}}</tool_call>'
        ),
        tool_calls=None,
    )

    normalized = _normalize_test_profile_response("qwen3vl", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].arguments == {
        "action_type": "wait",
        "summary": "Wait",
        "intent": "Wait",
    }


def test_qwen3vl_profile_prefers_content_contract_over_provider_tool_calls() -> None:
    response = LLMResponse(
        content=(
            "Thought: Open Chrome\n"
            'Action: "Open Chrome"\n'
            '<tool_call>{"name":"mobile_use","arguments":{"action":"open","text":"chrome"}}</tool_call>'
        ),
        tool_calls=[
            ToolCall(
                id="provider-tool-call-0",
                name="computer_use",
                arguments={"action_type": "open_app", "text": "chrome"},
            )
        ],
    )

    normalized = _normalize_test_profile_response("qwen3vl", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].id == "content-tool-call-0"
    assert normalized.tool_calls[0].name == "computer_use"
    assert normalized.tool_calls[0].arguments == {
        "action_type": "open_app",
        "text": "chrome",
        "summary": "Open Chrome",
        "intent": "Open Chrome",
    }


def test_qwen3vl_profile_rejects_provider_tool_calls_when_content_contract_is_missing() -> None:
    response = LLMResponse(
        content="Thought: Continue\nAction: Tap the next result",
        tool_calls=[
            ToolCall(
                id="provider-tool-call-0",
                name="computer_use",
                arguments={"action_type": "tap", "x": 321, "y": 654, "relative": True},
            )
        ],
    )

    with pytest.raises(ValueError):
        _normalize_test_profile_response("qwen3vl", response)


def test_qwen3vl_profile_rejects_provider_mobile_use_tool_calls() -> None:
    response = LLMResponse(
        content="Action: Tap the next result",
        tool_calls=[
            ToolCall(
                id="provider-tool-call-0",
                name="mobile_use",
                arguments={"action": "click", "coordinate": [903, 130]},
            )
        ],
    )

    with pytest.raises(ValueError):
        _normalize_test_profile_response("qwen3vl", response)


def test_qwen3vl_profile_requires_action_text_before_tool_call() -> None:
    response = LLMResponse(
        content='Thought: Tap target\n<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,250]}}</tool_call>',
        tool_calls=None,
    )

    with pytest.raises(ValueError):
        _normalize_test_profile_response("qwen3vl", response)


def test_qwen3vl_profile_normalizes_wait_with_time() -> None:
    response = LLMResponse(
        content=(
            "Thought: Need to wait\n"
            'Action: "Pause for a moment"\n'
            '<tool_call>{"name":"mobile_use","arguments":{"action":"wait","time":1.5}}</tool_call>'
        ),
        tool_calls=None,
    )

    normalized = _normalize_test_profile_response("qwen3vl", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].arguments == {
        "action_type": "wait",
        "summary": "Pause for a moment",
        "intent": "Pause for a moment",
    }


def test_mai_ui_profile_parses_swipe_direction_scroll() -> None:
    response = LLMResponse(
        content=(
            "<thinking>Need to scroll down.</thinking>\n"
            '<tool_call>{"name":"mobile_use","arguments":{"action":"swipe","direction":"down","coordinate":[300,700]}}</tool_call>'
        ),
        tool_calls=None,
    )

    normalized = _normalize_test_profile_response("mai_ui", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].arguments == {
        "action_type": "scroll",
        "direction": "up",
        "pixels": 400,
        "relative": True,
        "x": 300,
        "y": 700,
        "summary": "Need to scroll down.",
        "intent": "Need to scroll down.",
    }


def test_mai_ui_profile_rejects_action_json_without_tool_call() -> None:
    response = LLMResponse(
        content=('<thinking>open app</thinking>\nAction: {"action":"open","text":"Settings"}\n'),
        tool_calls=None,
    )

    with pytest.raises(ValueError):
        _normalize_test_profile_response("mai_ui", response)


def test_qwen3vl_profile_parses_tool_call_block_with_text_action() -> None:
    response = LLMResponse(
        content=(
            "Thought: Continue\n"
            "Action: Tap the next result\n"
            '<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[700,240]}}</tool_call>'
        ),
        tool_calls=None,
    )

    normalized = _normalize_test_profile_response("qwen3vl", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].name == "computer_use"
    assert normalized.tool_calls[0].arguments == {
        "action_type": "tap",
        "x": 700,
        "y": 240,
        "relative": True,
        "summary": "Tap the next result",
        "intent": "Tap the next result",
    }


def test_mai_ui_profile_parses_tool_call_block() -> None:
    response = LLMResponse(
        content=(
            "<thinking>\n"
            "The next step is to open details.\n"
            "</thinking>\n"
            '<tool_call>{"name":"mobile_use","arguments":{"action":"open","text":"Settings"}}</tool_call>'
        ),
        tool_calls=None,
    )

    normalized = _normalize_test_profile_response("mai_ui", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].name == "computer_use"
    assert normalized.tool_calls[0].arguments == {
        "action_type": "open_app",
        "text": "Settings",
        "summary": "The next step is to open details.",
        "intent": "The next step is to open details.",
    }


@pytest.mark.asyncio
async def test_agent_action_grounder_uses_profile_seam_for_qwen3vl(tmp_path: Path) -> None:
    screenshot = tmp_path / "grounder.png"
    _write_test_png(screenshot, size=(1000, 1000))
    llm = _RecordingLLM(
        [
            LLMResponse(
                content=(
                    "Thought: Tap the button\n"
                    'Action: "Tap login"\n'
                    '<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,250]}}</tool_call>'
                ),
                tool_calls=None,
            )
        ]
    )
    grounder = _AgentActionGrounder(llm, model="qwen-vl-max", agent_profile="qwen3vl")
    step = SkillStep(action_type="tap", target="Login button", parameters={"x_hint": "unused"})

    action = await grounder.ground(step, screenshot, {})

    assert action.action_type == "tap"
    assert action.x == 501
    assert action.y == 250
    assert action.relative is False
    assert len(llm.calls) == 1
    assert llm.calls[0][0]["role"] == "user"


@pytest.mark.asyncio
async def test_agent_subgoal_runner_uses_profile_seam_for_qwen3vl(tmp_path: Path) -> None:
    screenshot = tmp_path / "subgoal.png"
    _write_test_png(screenshot, size=(1000, 1000))
    llm = _RecordingLLM(
        [
            _qwen_response(
                {"action": "click", "coordinate": [400, 300]},
                action_text="Tap settings",
            ),
            _qwen_response(
                {"action": "terminate", "status": "success"},
                thought="The target state is reached.",
                action_text="Finish successfully",
            ),
        ]
    )
    backend = _SkillTestBackend()
    validator = _RecordingValidator([])
    runner = _AgentSubgoalRunner(
        llm=llm,
        backend=backend,
        state_validator=validator,
        model="qwen-vl-max",
        artifacts_root=tmp_path / "artifacts",
        agent_profile="qwen3vl",
    )

    result = await runner.run_subgoal("Settings screen visible", screenshot, max_steps=2)

    assert result.success is True
    assert result.done_judgment is True
    assert backend.executed_actions
    action = backend.executed_actions[0]
    assert action.action_type == "tap"
    assert action.relative is False
    assert validator.calls == []


@pytest.mark.asyncio
async def test_agent_subgoal_runner_records_events(tmp_path: Path) -> None:
    screenshot = tmp_path / "subgoal-record.png"
    _write_test_png(screenshot, size=(1000, 1000))
    llm = _RecordingLLM(
        [
            _mobileworld_response({"action_type": "click", "coordinate": [0.4, 0.3]}),
            _mobileworld_response(
                {"action_type": "status", "goal_status": "complete"},
                thought="The target state is visible.",
            ),
        ]
    )
    backend = _SkillTestBackend()
    validator = _RecordingValidator([])
    events: list[dict[str, Any]] = []
    recorder = _make_recorder(tmp_path, "subgoal trace", events=events)
    recorder.start()
    runner = _AgentSubgoalRunner(
        llm=llm,
        backend=backend,
        state_validator=validator,
        model="test-model",
        artifacts_root=tmp_path / "artifacts",
        trajectory_recorder=recorder,
        agent_profile="general_e2e",
    )

    result = await runner.run_subgoal("Settings screen visible", screenshot, max_steps=2)
    recorder.finish(success=True)

    assert result.success is True
    names = [event["type"] for event in events]
    assert "subgoal_start" in names
    assert "subgoal_step" in names
    assert "subgoal_result" in names
    subgoal_steps = [event for event in events if event["type"] == "subgoal_step"]
    assert len(subgoal_steps) == 2
    assert subgoal_steps[0]["model_output"]
    assert subgoal_steps[0]["goal_reached"] is False
    assert subgoal_steps[0]["action"]["action_type"] == "tap"
    assert subgoal_steps[1]["goal_reached"] is True
    assert subgoal_steps[1]["action"]["action_type"] == "done"
    assert validator.calls == []


@pytest.mark.asyncio
async def test_agent_subgoal_runner_records_parse_failure(tmp_path: Path) -> None:
    screenshot = tmp_path / "subgoal-failure.png"
    _write_test_png(screenshot, size=(1000, 1000))
    llm = _RecordingLLM(
        [
            LLMResponse(content="No valid tool call", tool_calls=None),
            LLMResponse(content="Still invalid", tool_calls=None),
            LLMResponse(content="Invalid again", tool_calls=None),
        ]
    )
    events: list[dict[str, Any]] = []
    recorder = _make_recorder(tmp_path, "subgoal trace failure", events=events)
    recorder.start()
    runner = _AgentSubgoalRunner(
        llm=llm,
        backend=_SkillTestBackend(),
        state_validator=_RecordingValidator([False]),
        model="test-model",
        artifacts_root=tmp_path / "artifacts-failure",
        trajectory_recorder=recorder,
    )

    result = await runner.run_subgoal("Settings screen visible", screenshot, max_steps=1)
    recorder.finish(success=True)

    assert result.success is False
    assert result.steps_taken == 0
    subgoal_step = next(event for event in events if event["type"] == "subgoal_step")
    assert subgoal_step["error"].startswith("profile/action parse error after retries:")
    subgoal_result = next(event for event in events if event["type"] == "subgoal_result")
    assert subgoal_result["success"] is False
    assert subgoal_result["steps_taken"] == 0


@pytest.mark.asyncio
async def test_agent_subgoal_runner_retries_parse_without_consuming_step(tmp_path: Path) -> None:
    screenshot = tmp_path / "subgoal-retry.png"
    _write_test_png(screenshot, size=(1000, 1000))
    llm = _RecordingLLM(
        [
            LLMResponse(content="No valid action", tool_calls=None),
            _mobileworld_response(
                {"action_type": "status", "goal_status": "complete"},
                thought="The target state is already visible.",
            ),
        ]
    )
    runner = _AgentSubgoalRunner(
        llm=llm,
        backend=_SkillTestBackend(),
        state_validator=_RecordingValidator([]),
        model="test-model",
        artifacts_root=tmp_path / "artifacts-retry",
        agent_profile="general_e2e",
    )

    result = await runner.run_subgoal("Settings screen visible", screenshot, max_steps=1)

    assert result.success is True
    assert result.steps_taken == 1
    assert len(llm.calls) == 2


def test_recorder_keeps_step_usage_once_in_trajectory_and_result(tmp_path: Path) -> None:
    recorder = _make_recorder(tmp_path, "compact usage")
    recorder.start()
    recorder.record_event(
        "skill_step",
        skill_id="code:open_menu",
        skill_name="open_menu",
        step_index=0,
        action={"action_type": "tap"},
        token_usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        duration_s=1.25,
    )
    recorder.record_step(
        action={"action_type": "done"},
        model_output="done",
        token_usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    )

    recorder.finish(success=True)

    assert recorder.path is not None
    trajectory = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert len(trajectory["steps"]) == 1
    assert trajectory["steps"][0]["token_usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }
    result = json.loads((recorder.path.parent / "result.json").read_text(encoding="utf-8"))
    assert result["run"]["token_usage"] == trajectory["steps"][0]["token_usage"]
    assert not (recorder.path.parent / "gui_metrics.json").exists()


def test_recorder_result_prefers_step_usage_over_explicit_finish_usage(tmp_path: Path) -> None:
    recorder = _make_recorder(tmp_path, "step usage wins over explicit")
    recorder.start()
    recorder.record_step(
        action={"action_type": "wait", "duration_ms": 1000},
        model_output="wait",
        token_usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    )
    recorder.finish(success=True)

    assert recorder.path is not None
    result = json.loads((recorder.path.parent / "result.json").read_text(encoding="utf-8"))
    assert result["run"]["token_usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }


@pytest.mark.asyncio
async def test_subgoal_runner_normalizes_relative_coordinates_for_gemini(tmp_path: Path) -> None:
    """MobileWorld profiles parse textual actions into absolute screenshot pixels."""
    screenshot = tmp_path / "subgoal.png"
    _write_test_png(screenshot, size=(1000, 1000))
    llm = _RecordingLLM(
        [
            _mobileworld_response({"action_type": "click", "coordinate": [0.9, 0.941]}),
            _mobileworld_response(
                {"action_type": "status", "goal_status": "complete"},
                thought="The target state is reached.",
            ),
        ]
    )
    backend = _SkillTestBackend()
    validator = _RecordingValidator([])
    runner = _AgentSubgoalRunner(
        llm=llm,
        backend=backend,
        state_validator=validator,
        model="gemini-2.0-flash",
        artifacts_root=tmp_path / "artifacts",
        agent_profile="general_e2e",
    )

    result = await runner.run_subgoal("Tap the target button", screenshot, max_steps=2)

    assert result.success is True
    assert backend.executed_actions
    action = backend.executed_actions[0]
    assert action.action_type == "tap"
    assert action.relative is False


@pytest.mark.asyncio
async def test_subgoal_runner_uses_configured_step_timeout(tmp_path: Path) -> None:
    """execute() and observe() must receive the step_timeout passed at construction."""
    screenshot = tmp_path / "subgoal.png"
    _write_test_png(screenshot, size=(1000, 1000))

    execute_timeouts: list[float] = []
    observe_timeouts: list[float] = []

    class _TimeoutCapturingBackend(_SkillTestBackend):
        async def execute(self, action, timeout: float = 5.0) -> str:
            execute_timeouts.append(timeout)
            return await super().execute(action, timeout=timeout)

        async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
            observe_timeouts.append(timeout)
            return await super().observe(screenshot_path, timeout=timeout)

    llm = _RecordingLLM(
        [
            _mobileworld_response({"action_type": "click", "coordinate": [0.01, 0.02]}),
            _mobileworld_response(
                {"action_type": "status", "goal_status": "complete"},
                thought="Timeout propagation checked.",
            ),
        ]
    )
    runner = _AgentSubgoalRunner(
        llm=llm,
        backend=_TimeoutCapturingBackend(),
        state_validator=_RecordingValidator([True]),
        model="test-model",
        artifacts_root=tmp_path / "artifacts",
        step_timeout=42.0,
        agent_profile="general_e2e",
    )

    await runner.run_subgoal("Confirm timeout propagation", screenshot, max_steps=2)

    assert execute_timeouts == [42.0]
    assert observe_timeouts == [42.0]


@pytest.mark.asyncio
async def test_agent_post_action_observe_allows_ui_tree_timeout_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observe_timeouts: list[float] = []

    class _TimeoutCapturingBackend(_SkillTestBackend):
        async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
            observe_timeouts.append(timeout)
            return await super().observe(screenshot_path, timeout=timeout)

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("guiclaw.agent.asyncio.sleep", fake_sleep)
    agent = GuiAgent(
        _ScriptedLLM([]),
        _TimeoutCapturingBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "post-action observe timeout"),
        artifacts_root=tmp_path / "runs",
        step_timeout=30.0,
    )

    observation = await agent._observe_after_action(
        tmp_path / "runs" / "screenshots" / "step_001.png",
        action=Action(action_type="tap", x=10, y=20),
        timeout=30.0,
    )

    assert isinstance(observation, Observation)
    assert observe_timeouts
    assert set(observe_timeouts) == {8.0}


def test_agent_builds_history_turn_from_step_result(tmp_path: Path) -> None:
    observation = Observation(
        screenshot_path=str(tmp_path / "screen.png"),
        screen_width=100,
        screen_height=200,
        foreground_app="Settings",
        platform="android",
    )
    action = Action(action_type="tap", x=10, y=20)
    result = StepResult(
        action=action,
        tool_call_id="call-1",
        tool_result="tap complete",
        assistant_message={"role": "assistant", "content": "Action: tap Settings"},
        action_summary="tap Settings",
        action_intent="Open Settings",
        state_summary="Settings visible",
        model_snapshot={"raw_content": "raw model response"},
    )
    agent = GuiAgent(
        _ScriptedLLM([]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "history helper"),
    )

    turn = agent._history_turn_from_step(
        step_index=3,
        observation=observation,
        result=result,
    )

    assert turn.step_index == 3
    assert turn.observation is observation
    assert turn.assistant_message["content"] == "Action: tap Settings"
    assert turn.tool_result_message == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "tap complete",
    }
    assert turn.action_summary == "tap Settings"
    assert turn.action_intent == "Open Settings"
    assert turn.state_summary == "Settings visible"
    assert turn.raw_response_content == "raw model response"


@pytest.mark.asyncio
async def test_agent_records_complete_model_output_without_prompt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    screenshot = run_dir / "screenshots" / "001_tap.png"
    _write_test_png(screenshot)
    observation = Observation(
        screenshot_path=str(screenshot),
        screen_width=32,
        screen_height=32,
        foreground_app="Settings",
        platform="android",
        extra={"activity": "MainActivity"},
    )
    recorder = _make_recorder(tmp_path, "record completed step")
    recorder.start()
    agent = GuiAgent(
        _ScriptedLLM([]),
        DryRunBackend(),
        trajectory_recorder=recorder,
    )
    result = StepResult(
        action=Action(action_type="tap", x=10, y=20),
        tool_call_id="call-1",
        tool_result="tap complete",
        assistant_message={"role": "assistant", "content": "Action: tap Settings"},
        action_summary="tap Settings",
        action_intent="Open Settings",
        state_summary="Settings visible",
        next_observation=observation,
        prompt_snapshot={"step_index": 1},
        model_snapshot={
            "raw_content": "raw model response",
            "reasoning_content": "inspect the visible Settings control",
            "thinking_blocks": [{"type": "thinking", "thinking": "locate Settings"}],
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "computer_use",
                    "arguments": {"action_type": "tap", "x": 10, "y": 20},
                }
            ],
            "finish_reason": "tool_calls",
        },
        execution_snapshot={"tool_result": "tap complete", "done": False},
        step_usage={"total_tokens": 4},
        duration_s=0.25,
    )

    await agent._record_completed_step(
        run_dir=run_dir,
        step_index=1,
        current_observation=observation,
        result=result,
    )

    assert not (run_dir / "trace.jsonl").exists()
    assert recorder.path is not None
    trajectory = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert trajectory["steps"][-1]["action"]["action_type"] == "tap"
    assert trajectory["steps"][-1]["model_output"] == {
        "content": "raw model response",
        "reasoning_content": "inspect the visible Settings control",
        "thinking_blocks": [{"type": "thinking", "thinking": "locate Settings"}],
        "tool_calls": [
            {
                "id": "call-1",
                "name": "computer_use",
                "arguments": {"action_type": "tap", "x": 10, "y": 20},
            }
        ],
        "finish_reason": "tool_calls",
    }
    assert trajectory["steps"][-1]["token_usage"] == {"total_tokens": 4}
    assert "prompt" not in trajectory["steps"][-1]
    assert "execution" not in trajectory["steps"][-1]


def test_agent_snapshot_preserves_provider_reasoning_and_finish_reason(tmp_path: Path) -> None:
    agent = GuiAgent(
        _ScriptedLLM([]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "model snapshot"),
    )
    action = Action(action_type="wait")
    response = LLMResponse(
        content="wait for the page",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 500},
            )
        ],
        raw=SimpleNamespace(
            reasoning_content="the page is still loading",
            thinking_blocks=[{"type": "thinking", "thinking": "observe loading state"}],
            finish_reason="tool_calls",
        ),
    )
    assistant_message = agent._build_assistant_message(response)

    snapshot = agent._snapshot_model_response(
        response=response,
        action=action,
        assistant_message=assistant_message,
        action_text="Action: wait",
    )

    assert snapshot["reasoning_content"] == "the page is still loading"
    assert snapshot["thinking_blocks"] == [
        {"type": "thinking", "thinking": "observe loading state"}
    ]
    assert snapshot["finish_reason"] == "tool_calls"


def test_agent_finalizes_step_result_with_shared_metrics(tmp_path: Path) -> None:
    agent = GuiAgent(
        _ScriptedLLM([]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "step result helper"),
    )
    result = StepResult(
        action=Action(action_type="wait"),
        tool_call_id="call-1",
        tool_result="waited",
        assistant_message={"role": "assistant", "content": "Action: wait"},
        action_summary="wait",
        event_usage={"total_tokens": 2},
    )

    finalized = agent._finalize_step_result(
        result,
        step_usage={"total_tokens": 5},
        step_start=0.0,
        step_chat_latency_s=0.0,
        step_ttft_s=0.2,
    )

    assert finalized.step_usage == {"total_tokens": 5}
    assert finalized.event_usage == {"total_tokens": 2}
    assert finalized.duration_s > 0
    assert finalized.chat_latency_s is None
    assert finalized.ttft_s == 0.2


@pytest.mark.asyncio
async def test_subgoal_runner_settle_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tap gets a 0.5s settle before observe; wait action does not."""
    screenshot = tmp_path / "subgoal.png"
    _write_test_png(screenshot, size=(1000, 1000))
    sleep_calls: list[float] = []
    original_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        await original_sleep(0)

    monkeypatch.setattr("guiclaw.skills.subgoal_runner.asyncio.sleep", fake_sleep)

    llm = _RecordingLLM(
        [
            _mobileworld_response({"action_type": "click", "coordinate": [0.01, 0.02]}),
            _mobileworld_response({"action_type": "wait"}),
            _mobileworld_response(
                {"action_type": "status", "goal_status": "complete"},
                thought="Settle behavior checked.",
            ),
        ]
    )
    runner = _AgentSubgoalRunner(
        llm=llm,
        backend=_SkillTestBackend(),
        state_validator=_RecordingValidator([]),
        model="test-model",
        artifacts_root=tmp_path / "artifacts",
        agent_profile="general_e2e",
    )

    result = await runner.run_subgoal("Settle behavior check", screenshot, max_steps=3)

    assert result.success is True
    # tap must produce exactly one settle sleep of 0.5s
    assert sleep_calls == [0.5]


@pytest.mark.asyncio
async def test_agent_runs_with_qwen3vl_content_only_profile(tmp_path: Path) -> None:
    llm = _RecordingLLM(
        [
            LLMResponse(
                content=(
                    "Thought: I should wait briefly\n"
                    "Action: Wait briefly\n"
                    '<tool_call>{"name":"mobile_use","arguments":{"action":"wait"}}</tool_call>'
                ),
                tool_calls=None,
            ),
            LLMResponse(
                content=(
                    "Thought: The task is complete\n"
                    "Action: Finish successfully\n"
                    '<tool_call>{"name":"mobile_use","arguments":{"action":"terminate","status":"success"}}</tool_call>'
                ),
                tool_calls=None,
            ),
        ]
    )
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "qwen profile"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        agent_profile="qwen3vl",
    )

    result = await agent.run("Wait and finish", max_retries=1)

    assert result.success
    assert len(llm.calls) == 2
    assert llm.calls[0][0]["role"] == "system"
    assert "mobile_use" in llm.calls[0][0]["content"][0]["text"]


@pytest.mark.asyncio
async def test_agent_runs_with_qwen3vl_mobileworld_coordinates(
    tmp_path: Path,
) -> None:
    llm = _RecordingLLM(
        [
            _qwen_response(
                {"action": "click", "coordinate": [410, 125]}, action_text="Tap the search bar"
            ),
            _qwen_response(
                {"action": "terminate", "status": "success"}, action_text="Finish successfully"
            ),
        ]
    )
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "qwen provider stringified coords"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        agent_profile="qwen3vl",
    )

    result = await agent.run("Tap then finish", max_retries=1)

    assert result.success is True
    assert result.summary


@pytest.mark.asyncio
async def test_qwen_profile_history_assistant_has_no_tool_calls(tmp_path: Path) -> None:
    llm = _RecordingLLM(
        [
            _qwen_response({"action": "wait"}, action_text="Wait briefly"),
            _qwen_response(
                {"action": "terminate", "status": "success"}, action_text="Finish successfully"
            ),
        ]
    )
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "qwen profile history tool calls"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        agent_profile="qwen3vl",
    )

    result = await agent.run("Wait and finish", max_retries=1)

    assert result.success
    assert len(llm.calls) == 2

    second_call = llm.calls[1]
    assert [message["role"] for message in second_call] == ["system", "user"]
    assert "Step 1: Wait briefly" in _message_text(second_call[1])


@pytest.mark.asyncio
async def test_agent_runs_with_qwen3vl_provider_mobile_use_tool_call(tmp_path: Path) -> None:
    llm = _RecordingLLM(
        [
            _qwen_response(
                {"action": "terminate", "status": "success"}, action_text="Finish successfully"
            ),
        ]
    )
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "qwen provider tool"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        agent_profile="qwen3vl",
    )

    result = await agent.run("Finish", max_retries=1)

    assert result.success is True
    assert result.summary


@pytest.mark.asyncio
async def test_agent_done_without_status_defaults_to_success(tmp_path: Path) -> None:
    llm = _RecordingLLM(
        [
            LLMResponse(
                content=(
                    "Thought: Task completed successfully and search results are visible.\n"
                    'Action: {"action_type": "answer", "text": "Task completed successfully and search results are visible."}'
                ),
                tool_calls=None,
            ),
        ]
    )
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "done without status success"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        agent_profile="general_e2e",
    )

    result = await agent.run("Finish", max_retries=1)

    assert result.success is True
    assert result.error is None


@pytest.mark.asyncio
async def test_agent_done_without_status_with_failure_text_marks_failure(tmp_path: Path) -> None:
    llm = _RecordingLLM(
        [
            LLMResponse(
                content=(
                    "Thought: Task failed because login is required.\n"
                    'Action: {"action_type": "answer", "text": "Task failed because login is required."}'
                ),
                tool_calls=None,
            ),
        ]
    )
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "done without status failure"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        agent_profile="general_e2e",
    )

    result = await agent.run("Finish", max_retries=1)

    assert result.success is False
    assert result.error == "Task terminated with status: failure"


@pytest.mark.asyncio
async def test_agent_trajectory_records_only_compact_step_details(tmp_path: Path) -> None:
    llm = _RecordingLLM(
        [
            LLMResponse(
                content='Thought: Action: wait briefly\nAction: {"action_type": "wait"}',
                tool_calls=None,
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            ),
            LLMResponse(
                content=(
                    'Thought: Action: done\nAction: {"action_type": "status", "goal_status": "complete"}'
                ),
                tool_calls=None,
                usage={"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9},
            ),
        ]
    )
    recorder = _make_recorder(tmp_path, "Open Settings")
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        agent_profile="general_e2e",
    )

    result = await agent.run("Open Settings", max_retries=1)

    assert result.success
    assert result.trace_path is not None
    assert recorder.path is not None
    trajectory = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert trajectory["instruction"] == "Open Settings"
    assert len(trajectory["steps"]) == 2
    step = trajectory["steps"][0]
    assert "prompt" not in step
    assert "messages" not in step
    assert step["model_output"]["content"] == (
        'Thought: Action: wait briefly\nAction: {"action_type": "wait"}'
    )
    assert step["model_output"]["tool_calls"][0]["name"] == "computer_use"
    assert step["action"]["action_type"] == "wait"
    assert step["app"] == "DryRun"
    assert step["token_usage"]["total_tokens"] == 12
    result_payload = json.loads((recorder.path.parent / "result.json").read_text(encoding="utf-8"))
    assert result_payload["run"]["success"] is True
    assert result_payload["run"]["steps_taken"] == 2
    assert result_payload["run"]["token_usage"] == {
        "prompt_tokens": 18,
        "completion_tokens": 3,
        "total_tokens": 21,
    }


@pytest.mark.asyncio
async def test_adb_backend_scrolls_horizontally_from_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    backend._screen_width = 400
    backend._screen_height = 800
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = parse_action(
        {
            "action_type": "scroll",
            "direction": "left",
            "pixels": 120,
        }
    )
    await backend.execute(action)

    run_mock.assert_awaited_once_with(
        "shell",
        "input",
        "swipe",
        "200",
        "400",
        "80",
        "400",
        "300",
        timeout=5.0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direction", "expected_y2"),
    [("down", "280"), ("up", "520")],
)
async def test_adb_backend_scroll_vertical_direction_is_inverted(
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
    expected_y2: str,
) -> None:
    backend = AdbBackend()
    backend._screen_width = 400
    backend._screen_height = 800
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = parse_action(
        {
            "action_type": "scroll",
            "direction": direction,
            "pixels": 120,
        }
    )
    await backend.execute(action)

    run_mock.assert_awaited_once_with(
        "shell",
        "input",
        "swipe",
        "200",
        "400",
        "200",
        expected_y2,
        "300",
        timeout=5.0,
    )


@pytest.mark.asyncio
async def test_adb_backend_input_text_prefers_yadb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)
    ensure_yadb_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(backend, "_ensure_yadb_available", ensure_yadb_mock)
    monkeypatch.setattr(
        backend, "_write_local_temp_text", lambda text: Path("/tmp/guiclaw-yadb-input.txt")
    )
    monkeypatch.setattr(
        backend, "_make_yadb_device_text_path", lambda: "/data/local/tmp/guiclaw-yadb-input.txt"
    )
    monkeypatch.setattr(
        backend, "_write_local_temp_yadb_script", lambda: Path("/tmp/guiclaw-yadb-input.sh")
    )
    monkeypatch.setattr(
        backend, "_make_yadb_device_script_path", lambda: "/data/local/tmp/guiclaw-yadb-input.sh"
    )

    text = "你好，GUIClaw"
    action = parse_action({"action_type": "input_text", "text": text})
    await backend.execute(action)

    ensure_yadb_mock.assert_awaited_once_with(5.0)
    assert run_mock.await_args_list[0].args == (
        "push",
        "/tmp/guiclaw-yadb-input.txt",
        "/data/local/tmp/guiclaw-yadb-input.txt",
    )
    assert run_mock.await_args_list[1].args == (
        "push",
        "/tmp/guiclaw-yadb-input.sh",
        "/data/local/tmp/guiclaw-yadb-input.sh",
    )
    assert run_mock.await_args_list[2].args == (
        "shell",
        "chmod",
        "755",
        "/data/local/tmp/guiclaw-yadb-input.sh",
    )
    assert run_mock.await_args_list[3].args == (
        "shell",
        "sh",
        "/data/local/tmp/guiclaw-yadb-input.sh",
        "/data/local/tmp/guiclaw-yadb-input.txt",
    )
    assert run_mock.await_args_list[4].args == (
        "shell",
        "rm",
        "-f",
        "/data/local/tmp/guiclaw-yadb-input.txt",
    )
    assert run_mock.await_args_list[5].args == (
        "shell",
        "rm",
        "-f",
        "/data/local/tmp/guiclaw-yadb-input.sh",
    )


@pytest.mark.asyncio
async def test_adb_backend_input_text_multiline_sends_each_line_and_enter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()

    async def fake_run(*args: object, timeout: float = 5.0) -> str:
        return ""

    run_mock = AsyncMock(side_effect=fake_run)
    monkeypatch.setattr(backend, "_run", run_mock)
    ensure_yadb_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(backend, "_ensure_yadb_available", ensure_yadb_mock)
    local_paths = iter(
        [
            Path("/tmp/guiclaw-yadb-input-1.txt"),
            Path("/tmp/guiclaw-yadb-input-2.txt"),
        ]
    )
    device_paths = iter(
        [
            "/data/local/tmp/guiclaw-yadb-input-1.txt",
            "/data/local/tmp/guiclaw-yadb-input-2.txt",
        ]
    )
    script_local_paths = iter(
        [
            Path("/tmp/guiclaw-yadb-input-1.sh"),
            Path("/tmp/guiclaw-yadb-input-2.sh"),
        ]
    )
    script_device_paths = iter(
        [
            "/data/local/tmp/guiclaw-yadb-input-1.sh",
            "/data/local/tmp/guiclaw-yadb-input-2.sh",
        ]
    )
    monkeypatch.setattr(backend, "_write_local_temp_text", lambda text: next(local_paths))
    monkeypatch.setattr(backend, "_make_yadb_device_text_path", lambda: next(device_paths))
    monkeypatch.setattr(backend, "_write_local_temp_yadb_script", lambda: next(script_local_paths))
    monkeypatch.setattr(backend, "_make_yadb_device_script_path", lambda: next(script_device_paths))

    action = parse_action({"action_type": "input_text", "text": "第一行\n第二行"})
    await backend.execute(action)

    assert run_mock.await_args_list[0].args == (
        "push",
        "/tmp/guiclaw-yadb-input-1.txt",
        "/data/local/tmp/guiclaw-yadb-input-1.txt",
    )
    assert run_mock.await_args_list[1].args == (
        "push",
        "/tmp/guiclaw-yadb-input-1.sh",
        "/data/local/tmp/guiclaw-yadb-input-1.sh",
    )
    assert run_mock.await_args_list[2].args == (
        "shell",
        "chmod",
        "755",
        "/data/local/tmp/guiclaw-yadb-input-1.sh",
    )
    assert run_mock.await_args_list[3].args == (
        "shell",
        "sh",
        "/data/local/tmp/guiclaw-yadb-input-1.sh",
        "/data/local/tmp/guiclaw-yadb-input-1.txt",
    )
    assert run_mock.await_args_list[4].args == (
        "shell",
        "rm",
        "-f",
        "/data/local/tmp/guiclaw-yadb-input-1.txt",
    )
    assert run_mock.await_args_list[5].args == (
        "shell",
        "rm",
        "-f",
        "/data/local/tmp/guiclaw-yadb-input-1.sh",
    )
    assert run_mock.await_args_list[6].args == (
        "shell",
        "input",
        "keyevent",
        "KEYCODE_ENTER",
    )
    assert run_mock.await_args_list[7].args == (
        "push",
        "/tmp/guiclaw-yadb-input-2.txt",
        "/data/local/tmp/guiclaw-yadb-input-2.txt",
    )
    assert run_mock.await_args_list[8].args == (
        "push",
        "/tmp/guiclaw-yadb-input-2.sh",
        "/data/local/tmp/guiclaw-yadb-input-2.sh",
    )
    assert run_mock.await_args_list[9].args == (
        "shell",
        "chmod",
        "755",
        "/data/local/tmp/guiclaw-yadb-input-2.sh",
    )
    assert run_mock.await_args_list[10].args == (
        "shell",
        "sh",
        "/data/local/tmp/guiclaw-yadb-input-2.sh",
        "/data/local/tmp/guiclaw-yadb-input-2.txt",
    )
    assert run_mock.await_args_list[11].args == (
        "shell",
        "rm",
        "-f",
        "/data/local/tmp/guiclaw-yadb-input-2.txt",
    )
    assert run_mock.await_args_list[12].args == (
        "shell",
        "rm",
        "-f",
        "/data/local/tmp/guiclaw-yadb-input-2.sh",
    )


@pytest.mark.asyncio
async def test_adb_backend_input_text_falls_back_to_adb_keyboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(
        side_effect=[
            "com.example.ime/.ExampleIme",
            "com.android.adbkeyboard/.AdbIME\ncom.example.ime/.ExampleIme",
            "",
            "",
            "",
        ]
    )
    monkeypatch.setattr(backend, "_run", run_mock)
    ensure_yadb_mock = AsyncMock(return_value=False)
    monkeypatch.setattr(backend, "_ensure_yadb_available", ensure_yadb_mock)

    text = "你好，GUIClaw"
    action = parse_action({"action_type": "input_text", "text": text})
    await backend.execute(action)

    expected_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    ensure_yadb_mock.assert_awaited_once_with(5.0)
    assert run_mock.await_args_list[0].args == (
        "shell",
        "settings",
        "get",
        "secure",
        "default_input_method",
    )
    assert run_mock.await_args_list[1].args == (
        "shell",
        "ime",
        "list",
        "-s",
    )
    assert run_mock.await_args_list[2].args == (
        "shell",
        "ime",
        "set",
        "com.android.adbkeyboard/.AdbIME",
    )
    assert run_mock.await_args_list[3].args == (
        "shell",
        "am",
        "broadcast",
        "-a",
        "ADB_INPUT_B64",
        "--es",
        "msg",
        expected_b64,
    )
    assert run_mock.await_args_list[4].args == (
        "shell",
        "input",
        "keyevent",
        "KEYCODE_ENTER",
    )


@pytest.mark.asyncio
async def test_adb_backend_input_text_enables_adb_keyboard_before_switching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(
        side_effect=[
            "com.example.ime/.ExampleIme",
            "com.android.adbkeyboard/.AdbIME\ncom.example.ime/.ExampleIme",
            "",
            "",
            "",
            "",
        ]
    )
    monkeypatch.setattr(backend, "_run", run_mock)
    enable_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(backend, "_needs_ime_enable_before_set", enable_mock)
    monkeypatch.setattr(backend, "_ensure_yadb_available", AsyncMock(return_value=False))

    action = parse_action({"action_type": "input_text", "text": "你好"})
    await backend.execute(action)

    enable_mock.assert_awaited_once_with(timeout=5.0)
    assert run_mock.await_args_list[2].args == (
        "shell",
        "ime",
        "enable",
        "com.android.adbkeyboard/.AdbIME",
    )
    assert run_mock.await_args_list[3].args == (
        "shell",
        "ime",
        "set",
        "com.android.adbkeyboard/.AdbIME",
    )


@pytest.mark.asyncio
async def test_adb_backend_input_text_falls_back_to_yadb_for_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)
    ensure_yadb_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(backend, "_ensure_yadb_available", ensure_yadb_mock)
    monkeypatch.setattr(
        backend, "_write_local_temp_text", lambda text: Path("/tmp/guiclaw-yadb-input.txt")
    )
    monkeypatch.setattr(
        backend, "_make_yadb_device_text_path", lambda: "/data/local/tmp/guiclaw-yadb-input.txt"
    )
    monkeypatch.setattr(
        backend, "_write_local_temp_yadb_script", lambda: Path("/tmp/guiclaw-yadb-input.sh")
    )
    monkeypatch.setattr(
        backend, "_make_yadb_device_script_path", lambda: "/data/local/tmp/guiclaw-yadb-input.sh"
    )

    text = "你好，GUIClaw"
    action = parse_action({"action_type": "input_text", "text": text})
    await backend.execute(action)

    ensure_yadb_mock.assert_awaited_once_with(5.0)

    assert run_mock.await_args_list[0].args == (
        "push",
        "/tmp/guiclaw-yadb-input.txt",
        "/data/local/tmp/guiclaw-yadb-input.txt",
    )
    assert run_mock.await_args_list[1].args == (
        "push",
        "/tmp/guiclaw-yadb-input.sh",
        "/data/local/tmp/guiclaw-yadb-input.sh",
    )
    assert run_mock.await_args_list[2].args == (
        "shell",
        "chmod",
        "755",
        "/data/local/tmp/guiclaw-yadb-input.sh",
    )
    assert run_mock.await_args_list[3].args == (
        "shell",
        "sh",
        "/data/local/tmp/guiclaw-yadb-input.sh",
        "/data/local/tmp/guiclaw-yadb-input.txt",
    )
    assert run_mock.await_args_list[4].args == (
        "shell",
        "rm",
        "-f",
        "/data/local/tmp/guiclaw-yadb-input.txt",
    )
    assert run_mock.await_args_list[5].args == (
        "shell",
        "rm",
        "-f",
        "/data/local/tmp/guiclaw-yadb-input.sh",
    )


@pytest.mark.asyncio
async def test_adb_backend_input_text_falls_back_to_shell_input_for_ascii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(
        side_effect=[
            "com.example.ime/.ExampleIme",
            "com.other.ime/.OtherIme",
            "",
            "",
        ]
    )
    monkeypatch.setattr(backend, "_run", run_mock)
    ensure_yadb_mock = AsyncMock(return_value=False)
    monkeypatch.setattr(backend, "_ensure_yadb_available", ensure_yadb_mock)

    text = "hello world"
    action = parse_action({"action_type": "input_text", "text": text})
    await backend.execute(action)

    ensure_yadb_mock.assert_awaited_once_with(5.0)
    assert run_mock.await_args_list[2].args == (
        "shell",
        "input",
        "text",
        "hello%sworld",
    )
    assert run_mock.await_args_list[2].kwargs == {"timeout": 5.0}
    assert run_mock.await_args_list[3].args == (
        "shell",
        "input",
        "keyevent",
        "KEYCODE_ENTER",
    )


def test_adb_backend_text_segmentation_splits_emoji_boundaries() -> None:
    from guiclaw.backends.adb import _iter_text_input_segments

    assert _iter_text_input_segments("杭州✅明天☁️有雨") == [
        "杭州",
        "✅",
        "明天",
        "☁️",
        "有雨",
    ]


@pytest.mark.asyncio
async def test_adb_backend_input_text_sends_text_after_emoji_as_later_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    segments: list[str] = []
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    async def fake_input_single_text(text: str, timeout: float) -> None:
        assert timeout == 5.0
        segments.append(text)

    monkeypatch.setattr(backend, "_input_single_text", fake_input_single_text)

    action = parse_action({"action_type": "input_text", "text": "已发送给苏✅请查收"})
    await backend.execute(action)

    assert segments == ["已发送给苏", "✅", "请查收"]
    assert run_mock.await_args_list[0].args == (
        "shell",
        "input",
        "keyevent",
        "KEYCODE_ENTER",
    )


@pytest.mark.asyncio
async def test_adb_backend_enter_and_app_switch_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    await backend.execute(parse_action({"action_type": "hotkey", "key": ["ENTER"]}))
    await backend.execute(Action(action_type="enter"))
    await backend.execute(Action(action_type="app_switch"))

    assert run_mock.await_args_list[0].args == (
        "shell",
        "input",
        "keyevent",
        "KEYCODE_ENTER",
    )
    assert run_mock.await_args_list[0].kwargs == {"timeout": 5.0}
    assert run_mock.await_args_list[1].args == (
        "shell",
        "input",
        "keyevent",
        "KEYCODE_ENTER",
    )
    assert run_mock.await_args_list[1].kwargs == {"timeout": 5.0}
    assert run_mock.await_args_list[2].args == (
        "shell",
        "input",
        "keyevent",
        "KEYCODE_APP_SWITCH",
    )
    assert run_mock.await_args_list[2].kwargs == {"timeout": 5.0}


@pytest.mark.asyncio
async def test_adb_backend_hotkey_uses_keycombination_for_multiple_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    await backend.execute(parse_action({"action_type": "hotkey", "key": ["power", "volume_down"]}))

    assert run_mock.await_count == 1
    assert run_mock.await_args_list[0].args == (
        "shell",
        "input",
        "keycombination",
        "KEYCODE_POWER",
        "KEYCODE_VOLUME_DOWN",
    )
    assert run_mock.await_args_list[0].kwargs == {"timeout": 5.0}


@pytest.mark.asyncio
async def test_adb_backend_hotkey_multi_key_unsupported_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(
        side_effect=AdbError(
            "adb failed (exit 1): adb shell input keycombination KEYCODE_POWER KEYCODE_VOLUME_DOWN\n"
            "stderr: Error: Unknown command: keycombination"
        )
    )
    monkeypatch.setattr(backend, "_run", run_mock)

    with pytest.raises(AdbError, match="does not support simultaneous multi-key hotkeys"):
        await backend.execute(
            parse_action({"action_type": "hotkey", "key": ["power", "volume_down"]})
        )

    assert run_mock.await_count == 1


# -- root_available error classification -------------------------------------


@pytest.mark.asyncio
async def test_root_available_does_not_cache_false_on_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """root_available() must not cache False — only True is cached."""
    backend = AdbBackend()
    backend._root_available_cache = None

    probe_order: list[str] = []

    async def fake_run(_self: Any, command: str, *args: Any, **kwargs: Any) -> str:
        probe_order.append(command)
        raise AdbError(
            "adb failed (exit 1): /system/bin/sh: su: inaccessible or not found",
            returncode=1,
            stderr="/system/bin/sh: su: inaccessible or not found",
        )

    monkeypatch.setattr(AdbBackend, "_run", fake_run)
    assert await backend.root_available(timeout=0.1) is False
    assert (
        not hasattr(backend, "_root_available_cache") or backend._root_available_cache is not False
    )
    # Verify both probes were attempted (no early cache return)
    assert len(probe_order) == 2


@pytest.mark.asyncio
async def test_root_available_caches_true_and_returns_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """root_available() caches True and short-circuits subsequent calls."""
    backend = AdbBackend()

    call_count = 0

    async def fake_run(_self: Any, command: str, *args: Any, **kwargs: Any) -> str:
        nonlocal call_count
        call_count += 1
        return "uid=0(root) gid=0(root)"

    monkeypatch.setattr(AdbBackend, "_run", fake_run)
    assert await backend.root_available(timeout=0.1) is True
    assert backend._root_available_cache is True
    assert call_count == 1
    # Second call must return immediately from cache
    assert await backend.root_available(timeout=0.1) is True
    assert call_count == 1


@pytest.mark.asyncio
async def test_root_available_propagates_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport errors (device offline) must propagate, not be treated as 'no root'."""
    backend = AdbBackend()
    backend._root_available_cache = None

    async def fake_run(_self: Any, command: str, *args: Any, **kwargs: Any) -> str:
        raise AdbError(
            "adb: device offline",
            returncode=1,
            stderr="error: device offline",
        )

    monkeypatch.setattr(AdbBackend, "_run", fake_run)
    with pytest.raises(AdbError):
        await backend.root_available(timeout=0.1)


@pytest.mark.asyncio
async def test_root_available_propagates_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeouts must propagate as transport failures."""
    backend = AdbBackend()
    backend._root_available_cache = None

    async def fake_run(_self: Any, command: str, *args: Any, **kwargs: Any) -> str:
        raise asyncio.TimeoutError("adb timed out")

    monkeypatch.setattr(AdbBackend, "_run", fake_run)
    with pytest.raises(asyncio.TimeoutError):
        await backend.root_available(timeout=0.1)


@pytest.mark.asyncio
async def test_agent_failure_keeps_last_trace_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_clock = iter((100.0, 101.25))
    monkeypatch.setattr(
        "guiclaw.agent.time",
        SimpleNamespace(time=lambda: next(inference_clock), monotonic=time.monotonic),
    )
    agent = GuiAgent(
        _ScriptedLLM(
            [
                LLMResponse(
                    content="wait",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="computer_use",
                            arguments={"action_type": "wait", "duration_ms": 1},
                        )
                    ],
                ),
            ]
        ),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "never finishes"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
    )

    result = await agent.run("never finishes", max_retries=1)

    assert not result.success
    assert result.error == "max_steps_exceeded"
    assert result.steps_taken == 1
    assert result.trace_path is not None
    assert Path(result.trace_path).exists()
    assert result.summary.startswith("Status: partial")
    assert "Done:" in result.summary
    assert "Remaining:" in result.summary
    assert "Current:" in result.summary
    assert "Resume:" in result.summary
    trajectory = json.loads((tmp_path / "traj" / "traj.json").read_text(encoding="utf-8"))
    assert trajectory["steps"][0]["inference_time_s"] == 1.25


@pytest.mark.asyncio
async def test_agent_stagnation_detection_terminates_before_max_steps(tmp_path: Path) -> None:
    responses = [
        LLMResponse(
            content=f"wait #{idx}",
            tool_calls=[
                ToolCall(
                    id=f"call-{idx}",
                    name="computer_use",
                    arguments={"action_type": "wait", "duration_ms": 1},
                )
            ],
        )
        for idx in range(1, 7)
    ]
    agent = GuiAgent(
        _ScriptedLLM(responses),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "stagnation"),
        artifacts_root=tmp_path / "runs",
        max_steps=6,
        stagnation_limit=3,
    )

    result = await agent.run("wait on unchanged screen", max_retries=1)

    assert not result.success
    assert result.error == "stagnation_detected"
    assert result.steps_taken == 3
    assert result.steps_taken < agent.max_steps
    assert result.summary.startswith("Status: blocked")
    assert "Done:" in result.summary
    assert "Remaining:" in result.summary
    assert "Current:" in result.summary
    assert "Resume:" in result.summary


@pytest.mark.asyncio
async def test_agent_success_uses_compact_state_note(tmp_path: Path) -> None:
    agent = GuiAgent(
        _ScriptedLLM(
            [
                LLMResponse(
                    content="wait",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="computer_use",
                            arguments={"action_type": "wait", "duration_ms": 1},
                        )
                    ],
                ),
                LLMResponse(
                    content="finish task",
                    tool_calls=[
                        ToolCall(
                            id="call-2",
                            name="computer_use",
                            arguments={"action_type": "done", "status": "success"},
                        )
                    ],
                ),
            ]
        ),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "completed note"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
    )

    result = await agent.run("complete the task", max_retries=1)

    assert result.success
    assert result.summary.startswith("Status: completed")
    assert "Done:" in result.summary
    assert "Remaining: none" in result.summary
    assert "Current:" in result.summary
    assert "Resume: No further action needed." in result.summary


@pytest.mark.asyncio
async def test_agent_intervention_cancelled_returns_blocked_note(tmp_path: Path) -> None:
    agent = GuiAgent(
        _ScriptedLLM(
            [
                LLMResponse(
                    content="request help",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="computer_use",
                            arguments={
                                "action_type": "request_intervention",
                                "text": "Need human input",
                            },
                        )
                    ],
                ),
            ]
        ),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "intervention note"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
    )

    result = await agent.run("pause for review", max_retries=1)

    assert not result.success
    assert result.error and result.error.startswith("intervention_cancelled")
    assert result.summary.startswith("Status: blocked")
    assert "Done:" in result.summary
    assert "Remaining:" in result.summary
    assert "Current:" in result.summary
    assert "Resume:" in result.summary


@pytest.mark.asyncio
async def test_agent_stagnation_counter_resets_when_screen_changes(tmp_path: Path) -> None:
    class _ChangingBackend(DryRunBackend):
        def __init__(self) -> None:
            super().__init__()
            self._observe_calls = 0

        async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
            del timeout
            from PIL import Image, ImageDraw

            self._observe_calls += 1
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.new("L", (32, 32), color=0)
            draw = ImageDraw.Draw(img)
            if self._observe_calls in {1, 2}:
                draw.rectangle((16, 0, 31, 31), fill=255)  # right half bright
            else:
                draw.rectangle((0, 16, 31, 31), fill=255)  # bottom half bright
            img.save(screenshot_path, format="PNG")
            return Observation(
                screenshot_path=str(screenshot_path),
                screen_width=32,
                screen_height=32,
                foreground_app="DryRun",
                platform=self.platform,
            )

    llm = _ScriptedLLM(
        [
            LLMResponse(
                content="wait 1",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="computer_use",
                        arguments={"action_type": "wait", "duration_ms": 1},
                    )
                ],
            ),
            LLMResponse(
                content="wait 2",
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="computer_use",
                        arguments={"action_type": "wait", "duration_ms": 1},
                    )
                ],
            ),
            LLMResponse(
                content="wait 3",
                tool_calls=[
                    ToolCall(
                        id="call-3",
                        name="computer_use",
                        arguments={"action_type": "wait", "duration_ms": 1},
                    )
                ],
            ),
            LLMResponse(
                content="wait 4",
                tool_calls=[
                    ToolCall(
                        id="call-4",
                        name="computer_use",
                        arguments={"action_type": "wait", "duration_ms": 1},
                    )
                ],
            ),
            LLMResponse(
                content="finish",
                tool_calls=[
                    ToolCall(
                        id="call-5",
                        name="computer_use",
                        arguments={"action_type": "done", "status": "success"},
                    )
                ],
            ),
        ]
    )
    agent = GuiAgent(
        llm,
        _ChangingBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "stagnation reset"),
        artifacts_root=tmp_path / "runs",
        max_steps=6,
        stagnation_limit=3,
    )

    result = await agent.run("wait with one screen change", max_retries=1)

    assert result.success
    assert result.error is None


@pytest.mark.asyncio
async def test_stagnation_detection_short_circuits_retries(tmp_path: Path) -> None:
    class _InfiniteWaitLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
            del messages, tools, tool_choice
            self.calls += 1
            return _mobileworld_response({"action_type": "wait"}, thought=f"wait {self.calls}")

    events: list[dict[str, Any]] = []
    recorder = _make_recorder(
        tmp_path,
        "stagnation retry short-circuit",
        events=events,
    )
    llm = _InfiniteWaitLLM()
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=8,
        stagnation_limit=3,
    )

    result = await agent.run("repeat wait", max_retries=3)

    assert not result.success
    assert result.error == "stagnation_detected"
    assert llm.calls == 4
    assert recorder.path is not None
    assert sum(1 for event in events if event["type"] == "attempt_start") == 1
    assert not any(event["type"] == "retry" for event in events)


@pytest.mark.asyncio
async def test_stagnation_detection_generates_termination_summary(tmp_path: Path) -> None:
    summary_text = "Task appears to loop on the same screen. Aborted to avoid repeated actions."
    llm = _RecordingLLM(
        [
            LLMResponse(
                content="wait briefly",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="computer_use",
                        arguments={"action_type": "wait", "duration_ms": 1},
                    )
                ],
            ),
            LLMResponse(
                content="wait again",
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="computer_use",
                        arguments={"action_type": "wait", "duration_ms": 1},
                    )
                ],
            ),
            LLMResponse(content=summary_text),
        ]
    )

    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "stagnation summary"),
        artifacts_root=tmp_path / "runs",
        max_steps=5,
        stagnation_limit=2,
    )

    result = await agent.run("detect loop", max_retries=1)

    assert not result.success
    assert result.error == "stagnation_detected"
    assert result.summary.startswith("Status: blocked")
    assert "Done:" in result.summary
    assert "Remaining:" in result.summary
    assert "Current:" in result.summary
    assert "Resume:" in result.summary
    assert len(llm.calls) == 3
    summary_call = llm.calls[2]
    content = summary_call[-1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    prompt_text = "\n".join(block["text"] for block in content if block.get("type") == "text")
    assert "Status: completed|partial|blocked" in prompt_text


@pytest.mark.asyncio
async def test_agent_stagnation_requires_same_action_type(tmp_path: Path) -> None:
    class _StaticScreenshotBackend(DryRunBackend):
        async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
            del timeout
            from PIL import Image

            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), color=(90, 90, 90)).save(
                screenshot_path,
                format="PNG",
            )
            return Observation(
                screenshot_path=str(screenshot_path),
                screen_width=64,
                screen_height=64,
                foreground_app="DryRun",
                platform=self.platform,
            )

    agent = GuiAgent(
        _ScriptedLLM(
            [
                LLMResponse(
                    content="wait",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="computer_use",
                            arguments={"action_type": "wait", "duration_ms": 1},
                        )
                    ],
                ),
                LLMResponse(
                    content="tap",
                    tool_calls=[
                        ToolCall(
                            id="call-2",
                            name="computer_use",
                            arguments={
                                "action_type": "tap",
                                "x": 10,
                                "y": 10,
                                "relative": True,
                            },
                        )
                    ],
                ),
                LLMResponse(
                    content="wait",
                    tool_calls=[
                        ToolCall(
                            id="call-3",
                            name="computer_use",
                            arguments={"action_type": "wait", "duration_ms": 1},
                        )
                    ],
                ),
                LLMResponse(
                    content="tap",
                    tool_calls=[
                        ToolCall(
                            id="call-4",
                            name="computer_use",
                            arguments={
                                "action_type": "tap",
                                "x": 20,
                                "y": 20,
                                "relative": True,
                            },
                        )
                    ],
                ),
                LLMResponse(
                    content="done",
                    tool_calls=[
                        ToolCall(
                            id="call-5",
                            name="computer_use",
                            arguments={"action_type": "done", "status": "success"},
                        )
                    ],
                ),
            ]
        ),
        _StaticScreenshotBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "stagnation action type"),
        artifacts_root=tmp_path / "runs",
        max_steps=5,
        stagnation_limit=2,
    )

    result = await agent.run("alternate actions", max_retries=1)

    assert result.success
    assert result.error is None


@pytest.mark.asyncio
async def test_stagnation_similarity_uses_ssim(tmp_path: Path) -> None:
    from PIL import Image

    base = tmp_path / "base.png"
    similar = tmp_path / "similar.png"
    changed = tmp_path / "changed.png"

    width = 64
    height = 64

    def _pattern(path: Path, invert: bool = False, perturb: bool = False) -> None:
        img = Image.new("L", (width, height))
        for y in range(height):
            for x in range(width):
                value = (x * 3 + y * 5) % 256
                if invert:
                    value = 255 - value
                img.putpixel((x, y), value)
        if perturb:
            img.putpixel((1, 1), (img.getpixel((1, 1)) + 1) % 256)
        img.save(path, format="PNG")

    _pattern(base)
    _pattern(similar, perturb=True)
    _pattern(changed, invert=True)

    agent = GuiAgent(
        _ScriptedLLM([]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "ssim stagnation"),
        artifacts_root=tmp_path / "runs",
    )

    base_fp = agent._build_screen_fingerprint(
        Observation(
            screenshot_path=str(base),
            screen_width=width,
            screen_height=height,
            foreground_app="DryRun",
            platform="dry-run",
        ),
    )
    similar_fp = agent._build_screen_fingerprint(
        Observation(
            screenshot_path=str(similar),
            screen_width=width,
            screen_height=height,
            foreground_app="DryRun",
            platform="dry-run",
        ),
    )
    changed_fp = agent._build_screen_fingerprint(
        Observation(
            screenshot_path=str(changed),
            screen_width=width,
            screen_height=height,
            foreground_app="DryRun",
            platform="dry-run",
        ),
    )

    assert base_fp is not None
    assert similar_fp is not None
    assert changed_fp is not None
    assert agent._is_same_screen(base_fp, similar_fp)
    assert not agent._is_same_screen(base_fp, changed_fp)


@pytest.mark.asyncio
async def test_agent_records_attempt_exception_and_retry_events(tmp_path: Path) -> None:
    class _FlakyLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("provider exploded")
            return _mobileworld_response(
                {"action_type": "status", "goal_status": "complete"},
                thought="finish task",
            )

    events: list[dict[str, Any]] = []
    recorder = _make_recorder(tmp_path, "retry task", events=events)
    agent = GuiAgent(
        _FlakyLLM(),
        DryRunBackend(),
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        agent_profile="general_e2e",
    )

    result = await agent.run("retry task", max_retries=2)

    assert result.success
    assert result.error is None
    assert recorder.path is not None
    types = [event["type"] for event in events]
    assert "attempt_start" in types
    assert "attempt_exception" in types
    assert "retry" in types
    attempt_exception = next(event for event in events if event["type"] == "attempt_exception")
    assert attempt_exception["error_type"] == "RuntimeError"
    assert attempt_exception["error_message"] == "provider exploded"


@pytest.mark.asyncio
async def test_agent_retries_profile_parse_error_three_times_within_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_clock = iter((0.0, 0.25, 1.0, 1.25, 2.0, 2.25, 3.0, 3.25))
    monkeypatch.setattr(
        "guiclaw.agent.time",
        SimpleNamespace(time=lambda: next(inference_clock), monotonic=time.monotonic),
    )

    class _MalformedThenRecoverLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
            self.calls += 1
            if self.calls <= 3:
                return _mobileworld_response(
                    {"action_type": "click"},
                    thought="malformed click missing coordinate",
                )
            return _mobileworld_response(
                {"action_type": "status", "goal_status": "complete"},
                thought="finish task",
            )

    llm = _MalformedThenRecoverLLM()
    events: list[dict[str, Any]] = []
    recorder = _make_recorder(
        tmp_path,
        "retry malformed profile response",
        events=events,
    )
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        agent_profile="general_e2e",
    )

    result = await agent.run("retry malformed profile response", max_retries=1)

    assert result.success
    assert llm.calls == 4
    assert recorder.path is not None
    assert not any(event["type"] == "attempt_exception" for event in events)
    trajectory = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert trajectory["steps"][0]["inference_time_s"] == 1.0


@pytest.mark.asyncio
async def test_agent_records_model_response_on_attempt_exception(tmp_path: Path) -> None:
    class _MalformedToolCallLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
            self.calls += 1
            if self.calls <= 4:
                return _mobileworld_response(
                    {"action_type": "click"},
                    thought="malformed click missing coordinate",
                )
            return _mobileworld_response(
                {"action_type": "status", "goal_status": "complete"},
                thought="finish task",
            )

    events: list[dict[str, Any]] = []
    recorder = _make_recorder(
        tmp_path,
        "retry malformed tool call",
        events=events,
    )
    llm = _MalformedToolCallLLM()
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        agent_profile="general_e2e",
    )

    result = await agent.run("retry malformed tool call", max_retries=2)

    assert result.success
    assert recorder.path is not None
    assert not any(event["type"] == "attempt_exception" for event in events)
    assert llm.calls == 6


@pytest.mark.asyncio
async def test_retry_uses_clean_default_prompt_after_max_steps(
    tmp_path: Path,
) -> None:
    llm = _RecordingLLM(
        [
            LLMResponse(
                content="wait briefly",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="computer_use",
                        arguments={"action_type": "wait", "duration_ms": 1},
                    )
                ],
            ),
            # Termination summary call (text-only, no tool call) after max_steps hit
            LLMResponse(content="Waited briefly but ran out of steps. Currently on home screen."),
            LLMResponse(
                content="finish task",
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="computer_use",
                        arguments={"action_type": "done", "status": "success"},
                    )
                ],
            ),
        ]
    )
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "retry summary max steps"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
    )

    result = await agent.run("Open Settings", max_retries=2)

    assert result.success
    # calls[0]=attempt1 step, calls[1]=termination summary, calls[2]=attempt2 step
    second_attempt = llm.calls[2]
    retry_text = _last_user_message_text(second_attempt)
    assert retry_text == "Instruction: Open Settings"


@pytest.mark.asyncio
async def test_retry_uses_clean_mobileworld_prompt_after_exception(
    tmp_path: Path,
) -> None:
    class _MalformedThenRecoverLLM:
        def __init__(self) -> None:
            self.calls: list[list[dict]] = []
            self._call_count = 0

        async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
            del tools, tool_choice
            self.calls.append(copy.deepcopy(messages))
            self._call_count += 1
            if self._call_count <= 4:
                return _mobileworld_response(
                    {"action_type": "click"},
                    thought="malformed click missing coordinate",
                )
            return _mobileworld_response(
                {"action_type": "status", "goal_status": "complete"},
                thought="finish task",
            )

    llm = _MalformedThenRecoverLLM()
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "retry summary exception"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        agent_profile="general_e2e",
    )

    result = await agent.run("retry malformed tool call", max_retries=2)

    assert result.success
    second_attempt = llm.calls[5]
    retry_text = _last_user_message_text(second_attempt)
    assert retry_text == "retry malformed tool call"


@pytest.mark.asyncio
async def test_default_agent_uses_compact_history_and_current_image(tmp_path: Path) -> None:
    llm = _RecordingLLM(
        [
            LLMResponse(
                content="wait briefly",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="computer_use",
                        arguments={"action_type": "wait", "duration_ms": 1},
                    )
                ],
            ),
            LLMResponse(
                content="wait again",
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="computer_use",
                        arguments={"action_type": "wait", "duration_ms": 1},
                    )
                ],
            ),
            LLMResponse(
                content="finish task",
                tool_calls=[
                    ToolCall(
                        id="call-3",
                        name="computer_use",
                        arguments={"action_type": "done", "status": "success"},
                    )
                ],
            ),
        ]
    )
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "Open Settings"),
        artifacts_root=tmp_path / "runs",
        max_steps=3,
        history_image_window=2,
    )

    result = await agent.run("Open Settings")

    assert result.success
    assert len(llm.calls) == 3

    third_call = llm.calls[2]
    assert "native tool-calling mechanism" in third_call[0]["content"]
    assert [message["role"] for message in third_call] == ["system", "user"]

    user_text = _message_text(third_call[1])
    assert user_text.startswith("Instruction: Open Settings")
    assert "Recent progress:" in user_text
    assert "wait briefly" in user_text
    assert "wait again" in user_text

    image_blocks = [
        block
        for message in third_call
        for block in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(block, dict) and block.get("type") == "image_url"
    ]
    assert len(image_blocks) == 1


@pytest.mark.asyncio
async def test_agent_uses_mobileworld_raw_response_for_history_and_compact_trace(
    tmp_path: Path,
) -> None:
    llm = _RecordingLLM(
        [
            LLMResponse(
                content=(
                    'Thought: Action: Tap login button\nAction: {"action_type": "click", "coordinate": [500, 250]}'
                ),
                tool_calls=None,
            ),
            LLMResponse(
                content=(
                    'Thought: Action: Finish task\nAction: {"action_type": "status", "goal_status": "complete"}'
                ),
                tool_calls=None,
            ),
        ]
    )
    recorder = _make_recorder(tmp_path, "tool summary")
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        history_image_window=1,
        agent_profile="general_e2e",
    )

    result = await agent.run("Open Login")

    assert result.success
    assert "Action: Finish task" in result.model_summary
    second_call = llm.calls[1]
    assert second_call[2]["content"][0]["text"].startswith("Thought: Action: Tap login button")
    assert "Tool call result: [dry-run] tap at" in _message_text(second_call[3])
    assert any(
        isinstance(block, dict) and block.get("type") == "image_url"
        for block in second_call[3]["content"]
    )

    assert not list((tmp_path / "runs").rglob("trace.jsonl"))
    trajectory = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert trajectory["steps"][0]["model_output"]["content"].startswith(
        "Thought: Action: Tap login button"
    )
    assert trajectory["steps"][1]["model_output"]["content"].startswith(
        "Thought: Action: Finish task"
    )


@pytest.mark.asyncio
async def test_agent_prompt_replays_mobileworld_raw_history(tmp_path: Path) -> None:
    responses = [
        LLMResponse(
            content=f'Thought: Action: Step {index}\nAction: {{"action_type": "wait"}}',
            tool_calls=None,
        )
        for index in range(1, 10)
    ]
    responses.append(
        LLMResponse(
            content=(
                'Thought: Action: Finish task\nAction: {"action_type": "status", "goal_status": "complete"}'
            ),
            tool_calls=None,
        )
    )
    llm = _RecordingLLM(responses)
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "intent window"),
        artifacts_root=tmp_path / "runs",
        max_steps=10,
        history_image_window=1,
        agent_profile="general_e2e",
    )

    result = await agent.run("Open Settings")

    assert result.success
    tenth_call = llm.calls[9]
    assert tenth_call[0]["role"] == "system"
    first_user_text = _message_text(tenth_call[1])
    assert first_user_text.startswith("Open Settings")
    assert "(Previous turn, screen not shown)" in first_user_text
    assistant_texts = [
        message["content"][0]["text"] for message in tenth_call if message["role"] == "assistant"
    ]
    assert len(assistant_texts) == 9
    assert assistant_texts[0].startswith("Thought: Action: Step 1")
    assert assistant_texts[-1].startswith("Thought: Action: Step 9")


@pytest.mark.asyncio
async def test_agent_waits_for_ui_to_settle_before_observing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_sleep = asyncio.sleep

    class _SettlingBackend(DryRunBackend):
        async def execute(self, action, timeout: float = 5.0) -> str:
            events.append(f"execute:{action.action_type}")
            return await super().execute(action, timeout=timeout)

        async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
            events.append(f"observe:{Path(screenshot_path).name}")
            return await super().observe(screenshot_path, timeout=timeout)

    async def fake_sleep(delay: float) -> None:
        events.append(f"sleep:{delay}")
        await original_sleep(0)

    monkeypatch.setattr("guiclaw.agent.asyncio.sleep", fake_sleep)

    agent = GuiAgent(
        _ScriptedLLM(
            [
                LLMResponse(
                    content="tap the screen",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="computer_use",
                            arguments={"action_type": "tap", "x": 10, "y": 20},
                        )
                    ],
                ),
                LLMResponse(
                    content="finish task",
                    tool_calls=[
                        ToolCall(
                            id="call-2",
                            name="computer_use",
                            arguments={"action_type": "done", "status": "success"},
                        )
                    ],
                ),
            ]
        ),
        _SettlingBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "settle"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
    )

    result = await agent.run("Tap once")

    assert result.success
    assert events[:4] == [
        "observe:000_initial.png",
        "execute:tap",
        "sleep:0.5",
        "observe:001_tap.png",
    ]


def test_agent_open_app_settle_seconds_is_five(tmp_path: Path) -> None:
    agent = GuiAgent(
        _ScriptedLLM([]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "open app settle"),
        artifacts_root=tmp_path / "runs",
    )

    assert agent._post_action_settle_seconds(Action(action_type="open_app", text="Calendar")) == 5.0


def test_pyproject_includes_guiclaw_in_build() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    include = pyproject["tool"]["hatch"]["build"]["include"]
    assert "guiclaw/**/*.py" in include

    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert "guiclaw" in wheel["packages"]

    sdist_include = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert "guiclaw/" in sdist_include
