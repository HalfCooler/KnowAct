from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_load_policy_context_initializes_conservative_memory_setup(tmp_path: Path) -> None:
    from guiclaw.memory.policy import DEFAULT_POLICY_ENTRY_ID, load_policy_context
    from guiclaw.memory.store import MemoryStore
    from guiclaw.memory.types import MemoryType

    memory_dir = tmp_path / "memory"

    policy = load_policy_context(memory_dir)

    entries = MemoryStore(memory_dir).list_all(memory_type=MemoryType.POLICY)
    assert len(entries) == 1
    assert entries[0].entry_id == DEFAULT_POLICY_ENTRY_ID
    assert "Deny, Cancel, Not now" in policy
    assert "explicitly requires and authorizes" in policy
    assert (memory_dir / "policy.md").is_file()
    assert not (tmp_path / "policy.md").exists()


def test_load_policy_context_preserves_existing_policy_setup(tmp_path: Path) -> None:
    from guiclaw.memory.policy import DEFAULT_POLICY_ENTRY_ID, load_policy_context
    from guiclaw.memory.store import MemoryStore
    from guiclaw.memory.types import MemoryEntry, MemoryType

    memory_dir = tmp_path / "memory"
    store = MemoryStore(memory_dir)
    store.add(
        MemoryEntry(
            entry_id="custom-policy",
            memory_type=MemoryType.POLICY,
            platform="all",
            content="Prefer the read-only path.",
        )
    )

    policy = load_policy_context(memory_dir)
    entries = MemoryStore(memory_dir).list_all(memory_type=MemoryType.POLICY)

    assert policy == "- Prefer the read-only path."
    assert [entry.entry_id for entry in entries] == ["custom-policy"]
    assert all(entry.entry_id != DEFAULT_POLICY_ENTRY_ID for entry in entries)


@pytest.mark.asyncio
async def test_gui_agent_injects_policy_hint_separately_from_memory(tmp_path: Path) -> None:
    from guiclaw.agent import GuiAgent
    from guiclaw.backends.dry_run import DryRunBackend
    from guiclaw.trajectory.recorder import TrajectoryRecorder

    backend = DryRunBackend()
    agent = GuiAgent(
        llm=MagicMock(),
        backend=backend,
        trajectory_recorder=TrajectoryRecorder(output_dir=tmp_path / "traj", task="test"),
        policy_context="Deny unrequested permissions.",
    )
    observation = await backend.observe(tmp_path / "screen.png")

    messages = agent._build_messages(
        task="Open Contacts",
        current_observation=observation,
        history=[],
        memory_context="Contacts search opens from the toolbar.",
    )
    prompt = repr(messages)

    assert "Advisory Policy Hints" in prompt
    assert "Deny unrequested permissions." in prompt
    assert "Relevant Knowledge" in prompt
    assert "Contacts search opens from the toolbar." in prompt
