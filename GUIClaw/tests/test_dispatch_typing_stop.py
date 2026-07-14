"""Regression tests: _dispatch publishes empty OutboundMessage for all channels when
_process_message returns None, ensuring typing indicators are cleared on Telegram,
Discord, Matrix, and CLI alike.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub tiktoken so test can run without optional runtime deps.
if "tiktoken" not in sys.modules:
    _stub = types.ModuleType("tiktoken")

    class _DummyEncoding:
        @staticmethod
        def encode(text: str) -> list[int]:
            return [1] * len(text)

    _stub.get_encoding = lambda _name: _DummyEncoding()
    _stub.encoding_for_model = lambda _name: _DummyEncoding()
    sys.modules["tiktoken"] = _stub

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage, OutboundMessage


def _make_loop(tmp_path: Path) -> AgentLoop:
    """Build AgentLoop with lightweight mocks so _dispatch logic can be isolated."""
    bus = MagicMock()
    bus.publish_outbound = AsyncMock()
    bus.consume_inbound = AsyncMock()

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock()

    with patch.object(AgentLoop, "_register_default_tools", lambda _self, **_kwargs: None):
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=tmp_path,
            gui_config=None,
        )

    return loop


def _inbound(channel: str, chat_id: str = "chat1") -> InboundMessage:
    return InboundMessage(channel=channel, sender_id="u1", chat_id=chat_id, content="hi")


@pytest.mark.asyncio
async def test_dispatch_publishes_empty_outbound_for_telegram_when_response_none(
    tmp_path: Path,
) -> None:
    """Non-CLI channels must receive an empty OutboundMessage so typing stops."""
    loop = _make_loop(tmp_path)
    msg = _inbound("telegram")

    with patch.object(loop, "_process_message", AsyncMock(return_value=None)):
        await loop._dispatch(msg)

    loop.bus.publish_outbound.assert_called_once()
    published: OutboundMessage = loop.bus.publish_outbound.call_args[0][0]
    assert published.channel == "telegram"
    assert published.chat_id == "chat1"
    assert published.content == ""


@pytest.mark.asyncio
async def test_dispatch_publishes_empty_outbound_for_cli_when_response_none(
    tmp_path: Path,
) -> None:
    """CLI channel must still receive an empty OutboundMessage (preserved behaviour)."""
    loop = _make_loop(tmp_path)
    msg = _inbound("cli")

    with patch.object(loop, "_process_message", AsyncMock(return_value=None)):
        await loop._dispatch(msg)

    loop.bus.publish_outbound.assert_called_once()
    published: OutboundMessage = loop.bus.publish_outbound.call_args[0][0]
    assert published.channel == "cli"
    assert published.content == ""


@pytest.mark.asyncio
async def test_dispatch_publishes_actual_response_when_not_none(
    tmp_path: Path,
) -> None:
    """When _process_message returns a real OutboundMessage it must be forwarded as-is."""
    loop = _make_loop(tmp_path)
    msg = _inbound("telegram")
    real_response = OutboundMessage(channel="telegram", chat_id="chat1", content="hello")

    with patch.object(loop, "_process_message", AsyncMock(return_value=real_response)):
        await loop._dispatch(msg)

    loop.bus.publish_outbound.assert_called_once_with(real_response)
