"""Test message tool suppress logic for final replies."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path: Path) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")


class _FakeGuiTaskTool(Tool):
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        return "gui_task"

    @property
    def description(self) -> str:
        return "Run a GUI task."

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: object) -> str:
        self.calls.append(dict(kwargs))
        return self.result


class TestMessageToolSuppressLogic:
    """Final reply suppressed only when message tool sends to the same target."""

    @pytest.mark.asyncio
    async def test_suppress_when_sent_to_same_target(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        tool_call = ToolCallRequest(
            id="call1", name="message",
            arguments={"content": "Hello", "channel": "feishu", "chat_id": "chat123"},
        )
        calls = iter([
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="Done", tool_calls=[]),
        ])
        loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])

        sent: list[OutboundMessage] = []
        mt = loop.tools.get("message")
        if isinstance(mt, MessageTool):
            mt.set_send_callback(AsyncMock(side_effect=lambda m: sent.append(m)))

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="Send")
        result = await loop._process_message(msg)

        assert len(sent) == 1
        assert result is None  # suppressed

    @pytest.mark.asyncio
    async def test_not_suppress_when_sent_to_different_target(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        tool_call = ToolCallRequest(
            id="call1", name="message",
            arguments={"content": "Email content", "channel": "email", "chat_id": "user@example.com"},
        )
        calls = iter([
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="I've sent the email.", tool_calls=[]),
        ])
        loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])

        sent: list[OutboundMessage] = []
        mt = loop.tools.get("message")
        if isinstance(mt, MessageTool):
            mt.set_send_callback(AsyncMock(side_effect=lambda m: sent.append(m)))

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="Send email")
        result = await loop._process_message(msg)

        assert len(sent) == 1
        assert sent[0].channel == "email"
        assert result is not None  # not suppressed
        assert result.channel == "feishu"

    @pytest.mark.asyncio
    async def test_not_suppress_when_no_message_tool_used(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="Hello!", tool_calls=[]))
        loop.tools.get_definitions = MagicMock(return_value=[])

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="Hi")
        result = await loop._process_message(msg)

        assert result is not None
        assert "Hello" in result.content

    @pytest.mark.asyncio
    async def test_injected_followup_with_message_tool_does_not_emit_empty_fallback(
        self, tmp_path: Path
    ) -> None:
        loop = _make_loop(tmp_path)
        tool_call = ToolCallRequest(
            id="call1", name="message",
            arguments={"content": "Tool reply", "channel": "feishu", "chat_id": "chat123"},
        )
        calls = iter([
            LLMResponse(content="First answer", tool_calls=[]),
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="", tool_calls=[]),
            LLMResponse(content="", tool_calls=[]),
            LLMResponse(content="", tool_calls=[]),
        ])
        loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])

        sent: list[OutboundMessage] = []
        mt = loop.tools.get("message")
        if isinstance(mt, MessageTool):
            mt.set_send_callback(AsyncMock(side_effect=lambda m: sent.append(m)))

        pending_queue = asyncio.Queue()
        await pending_queue.put(
            InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="follow-up")
        )

        msg = InboundMessage(channel="feishu", sender_id="user1", chat_id="chat123", content="Start")
        result = await loop._process_message(msg, pending_queue=pending_queue)

        assert len(sent) == 1
        assert sent[0].content == "Tool reply"
        assert result is None

    async def test_progress_hides_internal_reasoning(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        tool_call = ToolCallRequest(id="call1", name="read_file", arguments={"path": "foo.txt"})
        calls = iter([
            LLMResponse(
                content="Visible<think>hidden</think>",
                tool_calls=[tool_call],
                reasoning_content="secret reasoning",
                thinking_blocks=[{"signature": "sig", "thought": "secret thought"}],
            ),
            LLMResponse(content="Done", tool_calls=[]),
        ])
        loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])
        loop.tools.execute = AsyncMock(return_value="ok")

        progress: list[tuple[str, bool]] = []

        async def on_progress(content: str, *, tool_hint: bool = False) -> None:
            progress.append((content, tool_hint))

        final_content, _, _, _, _ = await loop._run_agent_loop(
            [], runtime=loop.llm_runtime(), on_progress=on_progress
        )

        assert final_content == "Done"
        assert progress == [
            ("Visible", False),
            ('read foo.txt', True),
        ]

    @pytest.mark.asyncio
    async def test_successful_gui_task_is_followed_by_main_agent_reply(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        gui_call = ToolCallRequest(
            id="call1",
            name="gui_task",
            arguments={"task": "打开 B 站并播放第一个视频"},
        )
        calls = iter([
            LLMResponse(content="", tool_calls=[gui_call]),
            LLMResponse(content="任务已完成，目前停留在视频播放页。", tool_calls=[]),
        ])
        loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
        loop.tools.get_definitions = MagicMock(return_value=[])
        gui_tool = _FakeGuiTaskTool(json.dumps({
            "success": True,
            "summary": "Task completed after 3 step(s).",
            "model_summary": "任务已完成，视频已成功播放",
            "trace_path": None,
            "steps_taken": 3,
            "error": None,
            "post_run_state": {
                "trace_read": True,
                "latest_screenshot_path": "/tmp/gui/latest.png",
                "last_action": {"action_type": "done", "status": "success"},
                "last_action_summary": "视频已开始播放",
                "last_foreground_app": "哔哩哔哩",
                "platform": "android",
                "screen_resolution": "1080x2400",
                "current_state": "Task completed after 3 step(s). Latest visible app: 哔哩哔哩. Screen resolution: 1080x2400.",
                "completion_assessment": "completed",
            },
        }, ensure_ascii=False))
        loop.tools.register(gui_tool)

        msg = InboundMessage(channel="telegram", sender_id="user1", chat_id="chat123", content="播放热门视频")
        result = await loop._process_message(msg)

        assert result is not None
        assert result.content == "任务已完成，目前停留在视频播放页。"
        assert loop.provider.chat_with_retry.await_count == 2
        assert gui_tool.calls == [{"task": "打开 B 站并播放第一个视频"}]


class TestMessageToolTurnTracking:

    def test_sent_in_turn_tracks_same_target(self) -> None:
        tool = MessageTool()
        from nanobot.agent.tools.context import RequestContext, request_context

        with request_context(RequestContext(channel="feishu", chat_id="chat1")):
            assert not tool._sent_in_turn
            tool._sent_in_turn = True
            assert tool._sent_in_turn

    def test_start_turn_resets(self) -> None:
        tool = MessageTool()
        tool._sent_in_turn = True
        tool.start_turn()
        assert not tool._sent_in_turn

    def test_schema_discourages_current_chat_replies(self) -> None:
        tool = MessageTool()

        assert "Do not use this for the normal reply in the current chat" in tool.description
        assert "generate_image creates images in the current chat" in tool.description
        assert (
            "Do not use this for a normal reply in the current chat"
            in tool.parameters["properties"]["content"]["description"]
        )
