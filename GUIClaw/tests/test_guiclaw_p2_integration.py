"""Phase 2 integration tests — agent loop, memory, skills, and trajectory.

Covers: AGENT-05, AGENT-06, SKILL-08, TRAJ-03, TEST-05.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from guiclaw.agent import GuiAgent
from guiclaw.backends.dry_run import DryRunBackend
from guiclaw.interfaces import LLMResponse, ToolCall
from guiclaw.memory.retrieval import MemoryRetriever
from guiclaw.memory.store import MemoryStore
from guiclaw.memory.types import MemoryEntry, MemoryType
from guiclaw.skills.data import Skill, SkillStep
from guiclaw.skills.flat import FlatSkillLibrary
from guiclaw.trajectory.recorder import TrajectoryRecorder

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic embedder: hash each text to a unit vector slot."""

    DIM = 8

    async def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            slot = hash(text) % self.DIM
            vecs[i, slot] = 1.0
        return vecs


class _RecordingLLM:
    """Mock LLM that returns scripted responses and records all calls."""

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


def _done_response(call_id: str = "tc_done") -> LLMResponse:
    return LLMResponse(
        content='Thought: task complete\nAction: {"action_type":"status","goal_status":"complete"}',
        tool_calls=[
            ToolCall(
                id=call_id,
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )
        ],
    )


def _wait_response(call_id: str = "tc_wait") -> LLMResponse:
    return LLMResponse(
        content='Thought: waiting\nAction: {"action_type":"wait"}',
        tool_calls=[
            ToolCall(
                id=call_id,
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 1},
            )
        ],
    )


def _make_recorder(tmp_path: Path, task: str = "test") -> TrajectoryRecorder:
    return TrajectoryRecorder(output_dir=tmp_path / "traj", task=task)


def _make_memory_entry(
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
# AGENT-04 / MEM-05: Memory injected into system prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_injected_into_system_prompt(tmp_path: Path) -> None:
    """GuiAgent.run() should inject memory entries into the system prompt."""
    store = MemoryStore(tmp_path / "mem")
    store.add(_make_memory_entry("m1", "Settings is the gear icon on home screen"))
    store.add(_make_memory_entry("pol1", "Always confirm before deleting", MemoryType.POLICY))
    store.save()

    retriever = MemoryRetriever(embedding_provider=_FakeEmbedder())
    await retriever.index(store.list_all())

    llm = _RecordingLLM([_done_response()])
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "Open Settings"),
        memory_retriever=retriever,
        artifacts_root=tmp_path / "runs",
        max_steps=1,
    )

    result = await agent.run("Open Settings", max_retries=1)
    assert result.success

    # MobileWorld profiles carry memory context in the user instruction payload.
    first_prompt = json.dumps(llm.calls[0], ensure_ascii=False)
    assert "Relevant Knowledge" in first_prompt
    assert "Always confirm before deleting" in first_prompt


# ---------------------------------------------------------------------------
# AGENT-05 / SKILL-08: Skills are prompt-selected, not pre-run matched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_library_does_not_trigger_pre_run_skill_path(tmp_path: Path) -> None:
    """A skill library alone should not execute a skill before the GUI model chooses it."""
    embedder = _FakeEmbedder()
    lib = FlatSkillLibrary(store_dir=tmp_path / "skills", embedding_provider=embedder)
    skill = Skill(
        skill_id="wifi-toggle",
        name="Toggle Wi-Fi",
        description="Toggle Wi-Fi in Settings",
        app="com.android.settings",
        platform="android",
        steps=(
            SkillStep(
                action_type="open_app",
                target="com.android.settings",
                fixed=True,
                fixed_values={"text": "com.android.settings"},
            ),
            SkillStep(action_type="tap", target="Wi-Fi toggle"),
        ),
    )
    lib.add(skill)

    # Mock executor that returns success
    mock_executor = AsyncMock()
    mock_exec_result = AsyncMock()
    mock_exec_result.state.value = "succeeded"
    mock_executor.execute = AsyncMock(return_value=mock_exec_result)

    llm = _RecordingLLM([_done_response()])
    recorder = _make_recorder(tmp_path, "Turn on Wi-Fi")
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=recorder,
        skill_library=lib,
        skill_executor=mock_executor,
        artifacts_root=tmp_path / "runs",
        max_steps=1,
    )

    result = await agent.run("Toggle Wi-Fi", max_retries=1)
    assert result.success
    mock_executor.execute.assert_not_called()

    # No pre-run SKILL phase should be recorded without a prompt-selected use_skill action.
    traj_path = recorder.path
    assert traj_path is not None and traj_path.exists()
    trajectory = json.loads(traj_path.read_text())
    assert not any(step.get("phase") == "skill" for step in trajectory["steps"])


# ---------------------------------------------------------------------------
# AGENT-05: Free explore when no skill match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_free_explore_when_prompt_skill_selection_disabled(tmp_path: Path) -> None:
    """With prompt skill selection disabled, GuiAgent should use free exploration."""
    embedder = _FakeEmbedder()
    lib = FlatSkillLibrary(store_dir=tmp_path / "skills", embedding_provider=embedder)
    # Add an unrelated skill
    lib.add(
        Skill(
            skill_id="unrelated",
            name="Send Email",
            description="Send an email via Gmail",
            app="com.google.android.gm",
            platform="android",
        )
    )

    llm = _RecordingLLM([_done_response()])
    recorder = _make_recorder(tmp_path, "Open calculator")
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=recorder,
        skill_library=lib,
        artifacts_root=tmp_path / "runs",
        max_steps=1,
    )

    result = await agent.run("Open calculator", max_retries=1)
    assert result.success

    # No SKILL phase in trajectory
    trajectory = json.loads(recorder.path.read_text())
    assert not any(step.get("phase") == "skill" for step in trajectory["steps"])


# ---------------------------------------------------------------------------
# AGENT-06 / TRAJ-03: Trajectory recorded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trajectory_recorded_on_run(tmp_path: Path) -> None:
    """Every agent run should produce compact trajectory and result JSON files."""
    llm = _RecordingLLM([_wait_response("w1"), _done_response()])
    recorder = _make_recorder(tmp_path, "Open Settings")
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=5,
    )

    result = await agent.run("Open Settings", max_retries=1)
    assert result.success

    assert recorder.path is not None and recorder.path.exists()
    trajectory = json.loads(recorder.path.read_text())
    result_payload = json.loads((recorder.path.parent / "result.json").read_text())
    assert trajectory["instruction"] == "Open Settings"
    assert [step["action"]["action_type"] for step in trajectory["steps"]] == [
        "wait",
        "done",
    ]
    assert result_payload["run"]["success"] is True


# ---------------------------------------------------------------------------
# TEST-05: Full flow with mock LLM + memory + skills
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_flow_with_mock_llm(tmp_path: Path) -> None:
    """Full flow: DryRunBackend + mock LLM + pre-seeded memory + skill → completion."""
    # Seed memory
    store = MemoryStore(tmp_path / "mem")
    store.add(_make_memory_entry("m1", "Settings is the gear icon"))
    store.add(_make_memory_entry("pol1", "Confirm before destructive actions", MemoryType.POLICY))
    store.save()

    retriever = MemoryRetriever(embedding_provider=_FakeEmbedder())
    await retriever.index(store.list_all())

    # Seed skill library
    embedder = _FakeEmbedder()
    lib = FlatSkillLibrary(store_dir=tmp_path / "skills", embedding_provider=embedder)
    skill = Skill(
        skill_id="open-settings",
        name="Open Settings",
        description="Open the Settings app",
        app="com.android.settings",
        platform="android",
        steps=(SkillStep(action_type="open_app", target="com.android.settings"),),
    )
    lib.add(skill)

    mock_executor = AsyncMock()
    mock_exec_result = AsyncMock()
    mock_exec_result.state.value = "succeeded"
    mock_executor.execute = AsyncMock(return_value=mock_exec_result)

    # GuiAgent LLM: returns done immediately (after skill execution)
    agent_llm = _RecordingLLM([_done_response()])
    recorder = _make_recorder(tmp_path, "Open Settings")

    agent = GuiAgent(
        agent_llm,
        DryRunBackend(),
        trajectory_recorder=recorder,
        memory_retriever=retriever,
        skill_library=lib,
        skill_executor=mock_executor,
        artifacts_root=tmp_path / "runs",
        max_steps=3,
    )

    result = await agent.run("Open Settings", max_retries=1)
    assert result.success

    # Memory appeared in system prompt
    first_prompt = json.dumps(agent_llm.calls[0], ensure_ascii=False)
    assert "Settings is the gear icon" in first_prompt
    assert "Confirm before destructive actions" in first_prompt

    trajectory = json.loads(recorder.path.read_text())
    result_payload = json.loads((recorder.path.parent / "result.json").read_text())
    assert trajectory["instruction"] == "Open Settings"
    assert result_payload["run"]["success"] is True
