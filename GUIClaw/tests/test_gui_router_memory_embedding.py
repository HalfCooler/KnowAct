"""#1 — embedding retrieval for GUI-memory items in GuiRouterMemoryRetriever.

The induced memory items are deliberately abstract / keyword-sparse, which the
keyword scorer under-retrieves.  ``retrieve_async`` ranks them with the framework's
hybrid BM25+FAISS MemoryRetriever; without an embedding provider, or if embedding
fails, it degrades to the keyword path so nothing is silently dropped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import nanobot.agent.tools.gui as gui_tools
from nanobot.agent.tools.gui import (
    GuiRouterContext,
    GuiRouterMemoryEvidence,
    GuiRouterMemoryRetriever,
    GuiWorkflowPlan,
    GuiWorkflowRunner,
    GuiWorkflowSubtask,
)
from scripts.induce_gui_memory import GuiMemoryItem, append_to_memory_bank


class _TopicEmbedder:
    """Maps topic-synonyms to a shared slot so semantically-related (but not
    lexically overlapping) texts embed close together."""

    DIM = 6
    _SLOTS = {
        # slot 0 — composing / messaging
        "compose": 0, "composing": 0, "draft": 0, "new": 0, "message": 0,
        "write": 0, "email": 0, "mail": 0, "editor": 0, "pencil": 0, "recipient": 0,
        # slot 1 — scrolling / feeds
        "scroll": 1, "feed": 1, "list": 1, "older": 1, "entries": 1, "swipe": 1,
    }

    async def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            hit = False
            for token in re.findall(r"[a-z]+", text.lower()):
                slot = self._SLOTS.get(token)
                if slot is not None:
                    vecs[i, slot] += 1.0
                    hit = True
            if not hit:  # keep non-topic texts off the topic axes deterministically
                vecs[i, abs(hash(text)) % self.DIM] = 1.0
        return vecs


class _BoomEmbedder:
    async def embed(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError("embedding endpoint down")


class _CountingEmbedder(_TopicEmbedder):
    """Topic embedder that records the batch size of each embed call."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def embed(self, texts: list[str]) -> np.ndarray:
        self.batch_sizes.append(len(texts))
        return await super().embed(texts)


_COMPOSE = GuiMemoryItem(
    title="Open the compose editor",
    description="Use when starting a new message draft.",
    content="Tap the pencil to open a new message editor and begin composing a draft to the recipient.",
    status="success",
    app="com.gmailclone",
)
_SCROLL = GuiMemoryItem(
    title="Reveal older entries",
    description="Use when the target sits far down a feed.",
    content="Scroll the feed list repeatedly to reach older entries.",
    status="success",
    app="org.joinmastodon.android.mastodon",
)
_INVITE = GuiMemoryItem(
    title="Generate invite links",
    description="Use when generating an expiring invite link.",
    content="Open mastodon server invite controls to generate an invite link with expiry.",
    status="success",
    app="org.joinmastodon.android.mastodon",
)


def _make_bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, items: list[GuiMemoryItem]) -> None:
    bank_dir = tmp_path / "guiclaw_memory"
    bank_dir.mkdir()
    append_to_memory_bank(items, bank_dir / "gui_memory_bank.jsonl")
    monkeypatch.setattr(gui_tools, "DEFAULT_GUICLAW_MEMORY_DIR", bank_dir)


async def test_embedding_ranks_semantic_match_over_lexical_void(tmp_path, monkeypatch):
    """A query that shares no keywords with the relevant item still ranks it first."""
    _make_bank(tmp_path, monkeypatch, [_COMPOSE, _SCROLL])
    retriever = GuiRouterMemoryRetriever(tmp_path / "ws", embedding_provider=_TopicEmbedder())

    # "write an email" shares the *topic* of the compose item but none of its words.
    context = await retriever.retrieve_async("write an email to a teammate", platform="android")

    assert context.evidence, "expected GUI-memory evidence"
    top = context.evidence[0]
    assert "pencil" in top.text or "composing" in top.text  # the compose item, ranked first
    assert top.source.startswith("guiclaw/gui_memory:com.gmailclone")


async def test_no_provider_delegates_to_keyword(tmp_path, monkeypatch):
    """Without an embedding provider, retrieve_async == sync keyword retrieve."""
    _make_bank(tmp_path, monkeypatch, [_INVITE])
    retriever = GuiRouterMemoryRetriever(tmp_path / "ws")  # provider=None
    task = "Generate a one-person invite link that expires in one day."

    async_ctx = await retriever.retrieve_async(task, platform="android")
    sync_ctx = retriever.retrieve(task, platform="android")

    assert async_ctx == sync_ctx
    assert any("invite" in item.text.casefold() for item in async_ctx.evidence)


async def test_embedding_failure_falls_back_to_keyword(tmp_path, monkeypatch):
    """If the embedding call errors, GUI memory still surfaces via the keyword path."""
    _make_bank(tmp_path, monkeypatch, [_INVITE])
    retriever = GuiRouterMemoryRetriever(tmp_path / "ws", embedding_provider=_BoomEmbedder())

    context = await retriever.retrieve_async(
        "Generate a one-person invite link that expires in one day.", platform="android"
    )

    assert any("invite" in item.text.casefold() for item in context.evidence)
    assert any(
        item.source.startswith("guiclaw/gui_memory:org.joinmastodon.android.mastodon")
        for item in context.evidence
    )


async def test_empty_bank_yields_no_evidence(tmp_path, monkeypatch):
    _make_bank(tmp_path, monkeypatch, [])
    retriever = GuiRouterMemoryRetriever(tmp_path / "ws", embedding_provider=_TopicEmbedder())
    context = await retriever.retrieve_async("write an email", platform="android")
    assert context.evidence == ()


async def test_index_is_cached_across_retrievals(tmp_path, monkeypatch):
    """The bank is embedded once; later retrievals reuse the cached index."""
    _make_bank(tmp_path, monkeypatch, [_COMPOSE, _SCROLL])
    emb = _CountingEmbedder()
    retriever = GuiRouterMemoryRetriever(tmp_path / "ws", embedding_provider=emb)

    await retriever.retrieve_async("write an email", platform="android")
    await retriever.retrieve_async("scroll the feed list", platform="android")

    # The 2 bank docs are embedded exactly once (a single batch of size 2);
    # each retrieval only adds its own query embedding (batch of size 1).
    assert emb.batch_sizes.count(2) == 1
    assert emb.batch_sizes.count(1) == 2


# ---------------------------------------------------------------------------
# A+: multi-app subtasks retrieve memory per subtask
# ---------------------------------------------------------------------------


class _RecordingRouter:
    """Stub router that records each retrieval query and returns a naming hint."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve_async(self, task: str, *, platform: str) -> GuiRouterContext:
        self.queries.append(task)
        return GuiRouterContext(
            app_candidates=(),
            evidence=(GuiRouterMemoryEvidence(source="guiclaw/gui_memory:x:success",
                                              text=f"hint-for::{task}"),),
        )


async def test_multi_app_retrieves_memory_per_subtask():
    router = _RecordingRouter()
    run_task = AsyncMock(return_value=json.dumps({
        "success": True, "summary": "ok", "model_summary": None,
        "trace_path": None, "steps_taken": 1, "error": None,
    }))
    runner = GuiWorkflowRunner(
        llm=MagicMock(),
        run_task=run_task,
        load_latest_step_event=MagicMock(),
        router_memory=router,
    )
    plan = GuiWorkflowPlan(
        mode="multi_app",
        subtasks=[
            GuiWorkflowSubtask(task="Open Settings and toggle wifi.", app_hint="Settings"),
            GuiWorkflowSubtask(task="Open Chrome and load a page.", app_hint="Chrome"),
        ],
    )
    backend = SimpleNamespace(platform="android")

    await runner._run_multi_app(backend, "compound task", plan)

    # one retrieval per subtask, each queried with that subtask's OWN goal
    assert router.queries == [
        "Open Settings and toggle wifi.",
        "Open Chrome and load a page.",
    ]
    # each subtask agent prompt carries its own subtask-specific hint
    prompts = [call.args[1] for call in run_task.await_args_list]
    assert "Advisory hints from past GUI memory" in prompts[0]
    assert "hint-for::Open Settings and toggle wifi." in prompts[0]
    assert "hint-for::Open Chrome and load a page." in prompts[1]
    assert "hint-for::Open Settings" not in prompts[1]  # not the other subtask's hint
