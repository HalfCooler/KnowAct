from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from guiclaw.agent import GuiAgent
from guiclaw.agent_profiles import (
    SUPPORTED_AGENT_PROFILES,
    build_profile_messages,
    canonicalize_agent_profile,
    coordinate_mode_for_profile,
    normalize_profile_response_for_screen,
    parse_profile_action,
    profile_llm_defaults,
    profile_tool_definition,
    profile_uses_native_tools,
)
from guiclaw.backends.dry_run import DryRunBackend
from guiclaw.interfaces import LLMResponse, ToolCall
from guiclaw.observation import Observation
from guiclaw.skills.compact_prompt import CompactPromptParts
from guiclaw.tool_schemas import COMPUTER_USE_TOOL, build_computer_use_tool
from guiclaw.trajectory.recorder import TrajectoryRecorder

EXPECTED_PROFILES = (
    "default",
    "general_e2e",
    "gui_owl",
    "venus",
    "seed",
    "qwen3vl",
    "mai_ui",
    "gelab",
)


def test_supported_profiles_are_strict_public_whitelist() -> None:
    assert SUPPORTED_AGENT_PROFILES == EXPECTED_PROFILES
    assert canonicalize_agent_profile(None) == "default"
    assert canonicalize_agent_profile("") == "default"


@pytest.mark.parametrize(
    "legacy",
    [
        "mobileworld_general_e2e",
        "mobileworld_general_e2e_compact_skill",
        "planner_executor",
        "general",
        "general_e2e_compact_skill",
        "mw-general-e2e",
        "gui_owl_1_5",
        "gui-owl-1.5",
        "ui_venus",
    ],
)
def test_legacy_profile_names_are_rejected(legacy: str) -> None:
    with pytest.raises(ValueError, match="Unsupported agent profile"):
        canonicalize_agent_profile(legacy)


def test_default_is_the_only_native_tool_profile() -> None:
    assert profile_uses_native_tools("default") is True
    for profile in EXPECTED_PROFILES[1:]:
        assert profile_uses_native_tools(profile) is False
    assert profile_tool_definition("default") is COMPUTER_USE_TOOL
    assert profile_tool_definition("general_e2e") is None


def test_default_coordinate_mode_preserves_opencua_model_hints() -> None:
    assert coordinate_mode_for_profile("default", "gpt-4.1") == "absolute"
    assert coordinate_mode_for_profile("default", "qwen-vl-max") == "relative_999"
    assert coordinate_mode_for_profile("default", "gemini-2.5-pro") == "relative_999"
    assert coordinate_mode_for_profile("general_e2e", "qwen-vl-max") == "absolute"


def _observation(path: Path) -> Observation:
    Image.new("RGB", (8, 12), "white").save(path)
    return Observation(
        screenshot_path=str(path),
        screen_width=8,
        screen_height=12,
        foreground_app="Settings",
        platform="android",
    )


def test_default_messages_use_native_opencua_contract(tmp_path: Path) -> None:
    messages = build_profile_messages(
        "default",
        task="Open Settings",
        current_observation=_observation(tmp_path / "screen.png"),
        history=[],
        model_name="gpt-4.1",
        history_image_window=3,
    )

    assert messages[0]["role"] == "system"
    assert "native tool-calling mechanism" in messages[0]["content"]
    assert "computer_use" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["text"] == "Instruction: Open Settings"
    assert messages[1]["content"][-1]["type"] == "image_url"


def test_gui_owl_messages_match_official_history_and_image_contract(tmp_path: Path) -> None:
    observations = [
        _observation(tmp_path / f"screen-{index}.png")
        for index in range(7)
    ]
    history = [
        SimpleNamespace(
            observation=observations[index],
            action_summary=f"Action {index + 1}",
            tool_result_message={"content": f"tool result {index + 1}"},
            raw_response_content=f"raw output {index + 1}",
            assistant_message={"content": f"fallback output {index + 1}"},
        )
        for index in range(6)
    ]

    messages = build_profile_messages(
        "gui_owl",
        task="Open Settings",
        current_observation=observations[-1],
        history=history,
        model_name="mPLUG/GUI-Owl-1.5-8B-Instruct",
        history_image_window=None,
    )

    image_blocks = [
        block
        for message in messages
        for block in message["content"]
        if isinstance(message["content"], list) and block.get("type") == "image_url"
    ]
    assert len(image_blocks) == 5
    first_prompt = messages[1]["content"][0]["text"]
    assert "Today's date is:" in first_prompt
    assert "Step 1: Action 1" in first_prompt
    assert "Step 2: Action 2" in first_prompt
    assert "Tool response:" not in first_prompt
    assert "<tool_response>" not in str(messages)
    assert [message["role"] for message in messages[2::2]] == ["assistant"] * 4
    assert [message["content"][0]["text"] for message in messages[2::2]] == [
        "raw output 3",
        "raw output 4",
        "raw output 5",
        "raw output 6",
    ]
    assert all(
        len(message["content"]) == 1 and message["content"][0]["type"] == "image_url"
        for message in messages[3::2]
    )


@pytest.mark.parametrize("history_image_window", [1, 2, 3, 5])
def test_gui_owl_honors_configured_history_image_window(
    tmp_path: Path,
    history_image_window: int,
) -> None:
    observations = [
        _observation(tmp_path / f"window-{history_image_window}-{index}.png")
        for index in range(7)
    ]
    history = [
        SimpleNamespace(
            observation=observations[index],
            action_summary=f"Action {index + 1}",
            tool_result_message={"content": f"tool result {index + 1}"},
            raw_response_content=f"raw output {index + 1}",
            assistant_message={"content": f"fallback output {index + 1}"},
        )
        for index in range(6)
    ]

    messages = build_profile_messages(
        "gui_owl",
        task="Open Settings",
        current_observation=observations[-1],
        history=history,
        model_name="mPLUG/GUI-Owl-1.5-8B-Instruct",
        history_image_window=history_image_window,
    )

    image_blocks = [
        block
        for message in messages
        if isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "image_url"
    ]
    assert len(image_blocks) == history_image_window


def test_gui_owl_images_use_official_smart_resize(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (101, 203), "white").save(screenshot)
    observation = Observation(
        screenshot_path=str(screenshot),
        screen_width=101,
        screen_height=203,
        foreground_app="Settings",
        platform="android",
    )

    messages = build_profile_messages(
        "gui_owl",
        task="Open Settings",
        current_observation=observation,
        history=[],
        model_name="mPLUG/GUI-Owl-1.5-8B-Instruct",
        history_image_window=3,
    )

    data_url = messages[1]["content"][-1]["image_url"]["url"]
    with Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as image:
        assert image.size == (112, 196)


def test_gui_owl_image_scale_ratio_applies_before_smart_resize(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (1080, 2376), "white").save(screenshot)
    observation = Observation(
        screenshot_path=str(screenshot),
        screen_width=1080,
        screen_height=2376,
        foreground_app="Settings",
        platform="android",
    )

    messages = build_profile_messages(
        "gui_owl",
        task="Open Settings",
        current_observation=observation,
        history=[],
        model_name="mPLUG/GUI-Owl-1.5-8B-Instruct",
        history_image_window=3,
        image_scale_ratio=0.5,
    )

    data_url = messages[1]["content"][-1]["image_url"]["url"]
    with Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as image:
        assert image.size == (532, 1176)


def test_gui_agent_forwards_image_scale_ratio_to_gui_owl_messages(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (101, 203), "white").save(screenshot)
    observation = Observation(
        screenshot_path=str(screenshot),
        screen_width=101,
        screen_height=203,
        foreground_app="Settings",
        platform="android",
    )
    agent = GuiAgent(
        AsyncMock(),
        DryRunBackend(),
        TrajectoryRecorder(output_dir=tmp_path / "traj", task="Open Settings"),
        artifacts_root=tmp_path / "runs",
        agent_profile="gui_owl",
        image_scale_ratio=0.5,
    )

    messages = agent._build_messages(
        task="Open Settings",
        current_observation=observation,
        history=[],
    )

    data_url = messages[1]["content"][-1]["image_url"]["url"]
    with Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as image:
        assert image.size == (56, 112)


def test_legacy_gui_plus_prompt_uses_smart_resized_image_dimensions(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (101, 203), "white").save(screenshot)
    observation = Observation(
        screenshot_path=str(screenshot),
        screen_width=101,
        screen_height=203,
        foreground_app="Settings",
        platform="android",
    )

    legacy_messages = build_profile_messages(
        "gui_owl",
        task="Open Settings",
        current_observation=observation,
        history=[],
        model_name="gui-plus",
        history_image_window=3,
    )
    assert "The screen's resolution is 112x196." in legacy_messages[0]["content"]

    scaled_messages = build_profile_messages(
        "gui_owl",
        task="Open Settings",
        current_observation=observation,
        history=[],
        model_name="gui-plus",
        history_image_window=3,
        image_scale_ratio=0.5,
    )
    assert "The screen's resolution is 56x112." in scaled_messages[0]["content"]
    scaled_data_url = scaled_messages[1]["content"][-1]["image_url"]["url"]
    with Image.open(BytesIO(base64.b64decode(scaled_data_url.split(",", 1)[1]))) as image:
        assert image.size == (56, 112)

    for model_name in ("gui-plus-2026-02-26", "mPLUG/GUI-Owl-1.5-8B-Instruct"):
        messages = build_profile_messages(
            "gui_owl",
            task="Open Settings",
            current_observation=observation,
            history=[],
            model_name=model_name,
            history_image_window=3,
        )
        assert "The screen's resolution is 1000x1000." in messages[0]["content"]


def test_gui_owl_uses_1000_grid_with_bounds_and_raw_coordinate_metadata() -> None:
    content = """
Action: "Tap the center"
<tool_call>
{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,1000]}}
</tool_call>
"""

    payload = parse_profile_action(
        "gui_owl",
        content,
        screen_width=1080,
        screen_height=1920,
    )

    assert payload["x"] == 540
    assert payload["y"] == 1919
    assert payload["params"]["gui_owl_coordinates"] == {"coordinate": [500, 1000]}

    invalid = content.replace("[500,1000]", "[500,1001]")
    with pytest.raises(ValueError, match=r"\[0, 1000\]"):
        parse_profile_action(
            "gui_owl",
            invalid,
            screen_width=1080,
            screen_height=1920,
        )


def test_legacy_gui_plus_maps_smart_resized_pixels_to_device_screen() -> None:
    content = """
<tool_call>
{"name":"mobile_use","arguments":{"action":"swipe","coordinate":[543,2018],"coordinate2":[567,905]}}
</tool_call>
"""

    payload = parse_profile_action(
        "gui_owl",
        content,
        screen_width=1440,
        screen_height=3120,
        model_name="gui-plus",
    )

    assert payload["action_type"] == "drag"
    assert (payload["x"], payload["y"]) == (548, 2026)
    assert (payload["x2"], payload["y2"]) == (572, 908)
    assert payload["params"]["gui_owl_coordinates"] == {
        "coordinate": [543, 2018],
        "coordinate2": [567, 905],
    }

    scaled_payload = parse_profile_action(
        "gui_owl",
        """
<tool_call>
{"name":"mobile_use","arguments":{"action":"click","coordinate":[364,784]}}
</tool_call>
""",
        screen_width=1440,
        screen_height=3120,
        model_name="gui-plus",
        image_scale_ratio=0.5,
    )
    assert (scaled_payload["x"], scaled_payload["y"]) == (720, 1560)

    for model_name in ("gui-plus-2026-02-26", "mPLUG/GUI-Owl-1.5-8B-Instruct"):
        with pytest.raises(ValueError, match=r"\[0, 1000\]"):
            parse_profile_action(
                "gui_owl",
                content,
                screen_width=1440,
                screen_height=3120,
                model_name=model_name,
            )


def test_gui_owl_keeps_2048_step_output_limit() -> None:
    assert profile_llm_defaults("gui_owl") == {"max_tokens": 2048}


def test_default_messages_include_platform_and_foreground_app(tmp_path: Path) -> None:
    observation = _observation(tmp_path / "desktop.png")
    observation.platform = "macos"
    observation.foreground_app = "Google Chrome"

    messages = build_profile_messages(
        "default",
        task="Play a video",
        current_observation=observation,
        history=[],
        model_name="gpt-4.1",
        history_image_window=3,
        available_apps=("Google Chrome", "Safari"),
    )

    system_prompt = messages[0]["content"]
    assert "- Current platform: macos." in system_prompt
    assert "- Current foreground app: Google Chrome." in system_prompt
    assert "adb_command" not in system_prompt
    assert "On Android, use package names" not in system_prompt
    assert "Google Chrome" in system_prompt
    assert "Safari" in system_prompt


@pytest.mark.parametrize("platform", ["macos", "linux", "windows"])
def test_desktop_computer_use_schema_omits_adb_command(platform: str) -> None:
    tool = build_computer_use_tool(platform=platform, available_apps=("Example App",))
    properties = tool["function"]["parameters"]["properties"]

    assert "adb_command" not in properties["action_type"]["enum"]
    assert "command_id" not in properties
    assert "params" not in properties


@pytest.mark.parametrize("platform", ["macos", "windows"])
def test_desktop_schema_exposes_exact_available_apps(platform: str) -> None:
    tool = build_computer_use_tool(
        platform=platform,
        available_apps=("Safari", "Google Chrome", "Safari"),
    )
    properties = tool["function"]["parameters"]["properties"]

    assert "open_app" in properties["action_type"]["enum"]
    assert "Google Chrome" in properties["text"]["description"]
    assert "Safari" in properties["text"]["description"]


@pytest.mark.parametrize("platform", ["macos", "windows"])
def test_desktop_schema_hides_open_app_without_catalog(platform: str) -> None:
    tool = build_computer_use_tool(platform=platform, available_apps=())

    action_types = tool["function"]["parameters"]["properties"]["action_type"]["enum"]
    assert "open_app" not in action_types


def test_linux_schema_hides_open_app_even_with_catalog() -> None:
    tool = build_computer_use_tool(platform="linux", available_apps=("Firefox",))

    action_types = tool["function"]["parameters"]["properties"]["action_type"]["enum"]
    assert "open_app" not in action_types


def test_gui_agent_uses_backend_platform_for_tool_schema() -> None:
    agent = GuiAgent.__new__(GuiAgent)
    agent.backend = SimpleNamespace(platform="macos")
    agent._available_apps = ("Google Chrome",)
    agent._prompt_skills_by_id = {}
    agent._shortcut_tools = []

    tool = agent._build_tools_list()[0]
    action_types = tool["function"]["parameters"]["properties"]["action_type"]["enum"]

    assert "adb_command" not in action_types
    assert "open_app" in action_types


@pytest.mark.asyncio
async def test_gui_agent_loads_available_apps_once() -> None:
    backend = SimpleNamespace(
        platform="macos",
        list_apps=AsyncMock(return_value=["Safari", "Google Chrome", "Safari"]),
    )
    agent = GuiAgent.__new__(GuiAgent)
    agent.backend = backend
    agent._available_apps = None

    await agent._load_available_apps()
    await agent._load_available_apps()

    assert agent._available_apps == ("Google Chrome", "Safari")
    backend.list_apps.assert_awaited_once()


def test_default_qwen_prompt_describes_relative_grid(tmp_path: Path) -> None:
    messages = build_profile_messages(
        "default",
        task="Open Settings",
        current_observation=_observation(tmp_path / "relative.png"),
        history=[],
        model_name="qwen-vl-max",
        history_image_window=3,
    )

    assert "1000x1000 relative coordinate grid" in messages[0]["content"]
    assert "relative=true" in messages[0]["content"]


def test_default_prompt_includes_retrieved_skill_contract(tmp_path: Path) -> None:
    skill_id = "compact:md.obsidian:create_obsidian_note"
    messages = build_profile_messages(
        "default",
        task="Create an Obsidian note",
        current_observation=_observation(tmp_path / "skill.png"),
        history=[],
        model_name="gpt-4.1",
        history_image_window=3,
        compact_prompt_parts=CompactPromptParts(
            compact_skill_instructions=(
                "# Optional Compact GUI Skills\n"
                f"- skill_id={skill_id}; name=create_obsidian_note"
            ),
            skill_ids=(skill_id,),
            catalog=f"- skill_id={skill_id}; name=create_obsidian_note",
        ),
    )

    system_prompt = messages[0]["content"]
    assert skill_id in system_prompt
    assert '"use_skill"' in system_prompt


def test_default_normalization_preserves_native_call_with_text() -> None:
    response = LLMResponse(
        content="Action: Tap Settings",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="computer_use",
                arguments={
                    "action_type": "tap",
                    "x": 100,
                    "y": 200,
                    "intent": "Open Settings",
                    "summary": "Settings icon is visible",
                },
            )
        ],
    )

    assert (
        normalize_profile_response_for_screen(
            "default",
            response,
            screen_width=1080,
            screen_height=1920,
        )
        is response
    )


@pytest.mark.asyncio
async def test_gui_agent_default_uses_required_native_tool_call(tmp_path: Path) -> None:
    class RecordingLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def chat(self, **kwargs):  # noqa: ANN003, ANN202
            self.calls.append(kwargs)
            return LLMResponse(
                content="Action: Finish the completed task",
                tool_calls=[
                    ToolCall(
                        id="done-1",
                        name="computer_use",
                        arguments={
                            "action_type": "done",
                            "status": "success",
                            "text": "Task completed",
                            "intent": "Finish",
                            "summary": "Task is complete",
                        },
                    )
                ],
            )

    llm = RecordingLLM()
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        TrajectoryRecorder(output_dir=tmp_path / "traj", task="native default"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        agent_profile="default",
    )

    result = await agent.run("Finish", max_retries=1)

    assert result.success is True
    assert llm.calls[0]["tool_choice"] == "required"
    assert llm.calls[0]["tools"][0]["function"]["name"] == "computer_use"
    assert "native tool-calling mechanism" in llm.calls[0]["messages"][0]["content"]
