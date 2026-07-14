from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from guiclaw.action import parse_action
from guiclaw.agent import GuiAgent
from guiclaw.agent_profiles import (
    build_profile_messages,
    canonicalize_agent_profile,
    normalize_profile_response_for_screen,
)
from guiclaw.backends.dry_run import DryRunBackend
from guiclaw.interfaces import LLMResponse
from guiclaw.observation import Observation
from guiclaw.trajectory.recorder import TrajectoryRecorder


class _RecordingLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, tool_choice=None, **kwargs):  # noqa: ANN001
        self.calls.append(messages)
        assert tools is None
        assert tool_choice is None
        if not self._responses:
            raise AssertionError("No scripted response left")
        return self._responses.pop(0)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (1, 1), (255, 255, 255, 255)).save(path)


def _observation(path: Path) -> Observation:
    _write_png(path)
    return Observation(
        screenshot_path=str(path),
        screen_width=1080,
        screen_height=1920,
        foreground_app="Settings",
        platform="android",
    )


def test_default_and_general_e2e_are_distinct_profiles() -> None:
    assert canonicalize_agent_profile(None) == "default"
    assert canonicalize_agent_profile("default") == "default"
    assert canonicalize_agent_profile("general_e2e") == "general_e2e"


def test_general_e2e_messages_use_mobileworld_prompt(tmp_path: Path) -> None:
    messages = build_profile_messages(
        "general_e2e",
        task="Open Settings",
        current_observation=_observation(tmp_path / "screen.png"),
        history=[],
        model_name="qwen",
        history_image_window=3,
    )

    assert messages[0]["role"] == "system"
    assert "# Role: Android Phone Operator AI" in messages[0]["content"]
    assert "Thought:" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["text"] == "Open Settings"
    assert messages[1]["content"][1]["type"] == "image_url"


def test_general_e2e_parse_uses_real_screen_dimensions() -> None:
    response = LLMResponse(
        content='Thought: tap it\nAction: {"action_type":"click","coordinate":[500,250]}',
        tool_calls=None,
    )

    normalized = normalize_profile_response_for_screen(
        "general_e2e",
        response,
        screen_width=1080,
        screen_height=1920,
    )

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].arguments["action_type"] == "tap"
    assert normalized.tool_calls[0].arguments["x"] == 540
    assert normalized.tool_calls[0].arguments["y"] == 480
    assert "relative" not in normalized.tool_calls[0].arguments


def test_general_e2e_parse_uses_first_action_when_model_outputs_multiple_actions() -> None:
    response = LLMResponse(
        content=(
            'Thought: tap the search box\nAction: {"action_type":"click","coordinate":[500,250]}\n'
            'Thought: maybe done now\nAction: {"action_type":"status","goal_status":"complete"}'
        ),
        tool_calls=None,
    )

    normalized = normalize_profile_response_for_screen(
        "general_e2e",
        response,
        screen_width=1080,
        screen_height=1920,
    )

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].arguments["action_type"] == "tap"
    assert normalized.tool_calls[0].arguments["x"] == 540
    assert normalized.tool_calls[0].arguments["y"] == 480


def test_general_e2e_parse_accepts_bare_json_list_action_target() -> None:
    response = LLMResponse(
        content='```json\n[{"action":"tap","target":[146,905]}]\n```',
        tool_calls=None,
    )

    normalized = normalize_profile_response_for_screen(
        "general_e2e",
        response,
        screen_width=496,
        screen_height=1080,
    )

    assert normalized.tool_calls is not None
    arguments = normalized.tool_calls[0].arguments
    assert arguments["action_type"] == "tap"
    assert arguments["x"] == 72
    assert arguments["y"] == 977


def test_general_e2e_scroll_adds_guiclaw_default_pixels() -> None:
    response = LLMResponse(
        content='Thought: scroll\nAction: {"action_type":"scroll","direction":"up"}',
        tool_calls=None,
    )

    normalized = normalize_profile_response_for_screen(
        "general_e2e",
        response,
        screen_width=1080,
        screen_height=1920,
    )

    assert normalized.tool_calls is not None
    arguments = normalized.tool_calls[0].arguments
    assert arguments["action_type"] == "scroll"
    assert arguments["direction"] == "up"
    assert arguments["pixels"] == 400
    action = parse_action(arguments)
    assert action.action_type == "scroll"
    assert action.text == "up"
    assert action.pixels == 400


def test_qwen3vl_parse_uses_real_screen_dimensions() -> None:
    response = LLMResponse(
        content=(
            'Thought: tap\nAction: "Tap target"\n'
            '<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,250]}}</tool_call>'
        ),
        tool_calls=None,
    )

    normalized = normalize_profile_response_for_screen(
        "qwen3vl",
        response,
        screen_width=1080,
        screen_height=1920,
    )

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].arguments["action_type"] == "tap"
    assert normalized.tool_calls[0].arguments["x"] == 541
    assert normalized.tool_calls[0].arguments["y"] == 480
    assert "relative" not in normalized.tool_calls[0].arguments


@pytest.mark.asyncio
async def test_gui_agent_general_e2e_uses_text_messages_and_raw_history(tmp_path: Path) -> None:
    first_response = 'Thought: wait\nAction: {"action_type":"wait"}'
    second_response = 'Thought: done\nAction: {"action_type":"status","goal_status":"complete"}'
    llm = _RecordingLLM(
        [
            LLMResponse(content=first_response, tool_calls=None),
            LLMResponse(content=second_response, tool_calls=None),
        ]
    )
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        TrajectoryRecorder(output_dir=tmp_path / "traj", task="mobileworld agent"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        agent_profile="general_e2e",
    )

    result = await agent.run("Wait once and finish", max_retries=1)

    assert result.success is True
    assert len(llm.calls) == 2
    assert "# Role: Android Phone Operator AI" in llm.calls[0][0]["content"]
    assert llm.calls[1][2]["role"] == "assistant"
    assert llm.calls[1][2]["content"][0]["text"] == first_response
    assert llm.calls[1][3]["content"][0]["text"].startswith("Tool call result:")
    assert llm.calls[1][3]["content"][1]["type"] == "image_url"
