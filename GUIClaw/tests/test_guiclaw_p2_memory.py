"""Phase 2 memory tests — MEM-05 POLICY always-include behavior."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from guiclaw.agent import GuiAgent
from guiclaw.backends.dry_run import DryRunBackend
from guiclaw.interfaces import LLMResponse, ToolCall
from guiclaw.memory.retrieval import MemoryRetriever
from guiclaw.memory.store import MemoryStore
from guiclaw.memory.types import MemoryEntry, MemoryType
from guiclaw.trajectory.recorder import TrajectoryRecorder


class _FakeEmbedder:
    DIM = 8

    async def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            slot = hash(text) % self.DIM
            vecs[i, slot] = 1.0
        return vecs


class _RecordingLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat(
        self,
        messages,
        tools=None,
        tool_choice=None,
        model=None,
        max_tokens=None,
    ) -> LLMResponse:
        del tools, tool_choice, model, max_tokens
        self.calls.append(copy.deepcopy(messages))
        if not self._responses:
            raise AssertionError("No scripted responses left")
        return self._responses.pop(0)


def _done_response() -> LLMResponse:
    return LLMResponse(
        content='Thought: done\nAction: {"action_type":"status","goal_status":"complete"}',
        tool_calls=[
            ToolCall(
                id="tc_done",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )
        ],
    )


def _make_entry(
    entry_id: str,
    content: str,
    memory_type: MemoryType = MemoryType.APP_GUIDE,
) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        memory_type=memory_type,
        platform="android",
        content=content,
    )


# ---------------------------------------------------------------------------
# MEM-05: POLICY entries always included regardless of relevance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_always_included(tmp_path: Path) -> None:
    """POLICY memory entries should appear in system prompt regardless of relevance score."""
    store = MemoryStore(tmp_path / "mem")
    # Add many non-POLICY entries
    for i in range(6):
        store.add(_make_entry(f"app{i}", f"App guide entry {i}"))
    # Add 1 POLICY entry
    store.add(
        _make_entry("policy1", "Never delete user data without confirmation", MemoryType.POLICY)
    )
    store.save()

    retriever = MemoryRetriever(embedding_provider=_FakeEmbedder())
    await retriever.index(store.list_all())

    llm = _RecordingLLM([_done_response()])
    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task="unrelated query")
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=recorder,
        memory_retriever=retriever,
        memory_top_k=3,
        artifacts_root=tmp_path / "runs",
        max_steps=1,
    )

    result = await agent.run("unrelated query", max_retries=1)
    assert result.success

    system_msg = json.dumps(llm.calls[0], ensure_ascii=False)
    # POLICY always appears
    assert "Never delete user data without confirmation" in system_msg


# ---------------------------------------------------------------------------
# MEM-05: Memory context formatted in system prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_context_formatted_in_system_prompt(tmp_path: Path) -> None:
    """Memory context should be formatted into the active agent system message."""
    store = MemoryStore(tmp_path / "mem")
    store.add(_make_entry("g1", "Swipe up from bottom to go home"))
    store.add(_make_entry("g2", "Long press for app info", MemoryType.OS_GUIDE))
    store.save()

    retriever = MemoryRetriever(embedding_provider=_FakeEmbedder())
    await retriever.index(store.list_all())

    llm = _RecordingLLM([_done_response()])
    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task="go home")
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=recorder,
        memory_retriever=retriever,
        artifacts_root=tmp_path / "runs",
        max_steps=1,
    )

    result = await agent.run("go home", max_retries=1)
    assert result.success

    system_msg = json.dumps(llm.calls[0], ensure_ascii=False)
    # System prompt should contain the "Relevant Knowledge" section
    assert "Relevant Knowledge" in system_msg
    # At least one of our entries should appear
    assert (
        "Swipe up from bottom to go home" in system_msg or "Long press for app info" in system_msg
    )


@pytest.mark.asyncio
async def test_memory_retrieval_is_logged_and_recorded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = MemoryStore(tmp_path / "mem")
    store.add(
        _make_entry(
            "browser1", "Use Command + L to focus the browser address bar", MemoryType.APP_GUIDE
        )
    )
    store.add(_make_entry("os1", "Use Command + Space to open Spotlight", MemoryType.OS_GUIDE))
    store.save()

    retriever = MemoryRetriever(embedding_provider=_FakeEmbedder())
    await retriever.index(store.list_all())

    llm = _RecordingLLM([_done_response()])
    events: list[dict] = []
    recorder = TrajectoryRecorder(
        output_dir=tmp_path / "traj",
        task="focus the browser address bar",
        event_callback=events.append,
    )
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=recorder,
        memory_retriever=retriever,
        artifacts_root=tmp_path / "runs",
        max_steps=1,
    )

    with caplog.at_level(logging.INFO, logger="guiclaw.agent"):
        result = await agent.run("focus the browser address bar", max_retries=1)

    assert result.success
    assert "Memory retrieval: " in caplog.text
    assert "Memory hit id=browser1" in caplog.text

    assert recorder.path is not None
    memory_event = next(event for event in events if event["type"] == "memory_retrieval")
    assert memory_event["hit_count"] >= 1
    assert any(hit["entry_id"] == "browser1" for hit in memory_event["hits"])
    assert "Command + L" in memory_event["context"]
