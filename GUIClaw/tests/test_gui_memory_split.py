"""Regression tests for GUI policy memory injection.

Verifies:
- GuiSubagentTool._load_policy_context initializes and loads POLICY memory
- GuiAgent keeps policy hints independent from retrieved memory
- GuiAgent falls back to memory_retriever when policy_context is None
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Test 1: GuiSubagentTool._load_policy_context reads policy entries
# ---------------------------------------------------------------------------


def test_gui_tool_load_policy_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_policy_context returns formatted policy lines from the memory store."""
    from guiclaw.memory.store import MemoryStore
    from guiclaw.memory.types import MemoryEntry, MemoryType

    # Build a real MemoryStore with known policy entries
    store_dir = tmp_path / "memory"
    store_dir.mkdir()
    store = MemoryStore(store_dir)
    store.add(
        MemoryEntry(
            entry_id="pol-001",
            memory_type=MemoryType.POLICY,
            platform="android",
            content="Never accept calls without user permission.",
        )
    )
    store.add(
        MemoryEntry(
            entry_id="pol-002",
            memory_type=MemoryType.POLICY,
            platform="android",
            content="Do not send messages to unknown contacts.",
        )
    )
    store.save()

    # Monkeypatch DEFAULT_GUICLAW_MEMORY_DIR inside the gui tool module
    import nanobot.agent.tools.gui as gui_module

    monkeypatch.setattr(gui_module, "DEFAULT_GUICLAW_MEMORY_DIR", store_dir)

    # Call _load_policy_context as an unbound method (self is unused in this method)
    result = gui_module.GuiSubagentTool._load_policy_context(object())

    assert result is not None, "Expected non-None policy context"
    assert "Never accept calls without user permission." in result
    assert "Do not send messages to unknown contacts." in result
    assert "- Never accept calls without user permission." in result
    assert "- Do not send messages to unknown contacts." in result


# ---------------------------------------------------------------------------
# Test 5: GuiAgent keeps policy context separate from memory retrieval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gui_agent_policy_does_not_replace_memory_retrieval(tmp_path: Path) -> None:
    """A policy hint must not suppress task-relevant memory retrieval."""
    from guiclaw.agent import GuiAgent
    from guiclaw.backends.dry_run import DryRunBackend
    from guiclaw.trajectory.recorder import TrajectoryRecorder

    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task="test task")
    recorder.start()

    # Provide a mock retriever with no hits.
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])

    agent = GuiAgent(
        llm=MagicMock(),
        backend=DryRunBackend(),
        trajectory_recorder=recorder,
        policy_context="test policy line",
        memory_retriever=mock_retriever,
    )

    result = await agent._retrieve_memory("any task")

    assert result is None
    mock_retriever.search.assert_called()


# ---------------------------------------------------------------------------
# Test 6: GuiAgent falls back to retriever when policy_context is None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gui_agent_falls_back_to_retriever_when_no_policy_context(tmp_path: Path) -> None:
    """GuiAgent._retrieve_memory uses memory_retriever search when policy_context is None."""
    from guiclaw.agent import GuiAgent
    from guiclaw.backends.dry_run import DryRunBackend
    from guiclaw.trajectory.recorder import TrajectoryRecorder

    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task="test task")
    recorder.start()

    # Provide a mock retriever that returns no results (simulates search with no hits)
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])

    agent = GuiAgent(
        llm=MagicMock(),
        backend=DryRunBackend(),
        trajectory_recorder=recorder,
        policy_context=None,
        memory_retriever=mock_retriever,
    )

    result = await agent._retrieve_memory("find wifi settings")

    # With no hits, result should be None and the retriever should have been called
    assert result is None, f"Expected None for no search hits, got: {result!r}"
    mock_retriever.search.assert_called()
