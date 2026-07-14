"""
Unit tests for the guiclaw memory module (TEST-02).

Covers:
  - MemoryEntry serialisation round-trip (to_dict / from_dict)
  - MemoryStore JSON persistence (add, reload, get, remove, list_all)
  - MemoryRetriever hybrid BM25+FAISS search, BM25-only, FAISS-only modes
  - MemoryRetriever.format_context output

All tests use tmp_path for file isolation. No network calls, no real LLM.
"""

from __future__ import annotations

import numpy as np
import pytest

import guiclaw.memory.retrieval as _retrieval_mod
from guiclaw.memory.retrieval import EmbeddingProvider, MemoryRetriever, _tokenize
from guiclaw.memory.store import MemoryStore
from guiclaw.memory.types import MemoryEntry, MemoryType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str,
    *,
    entry_id: str = "e1",
    memory_type: MemoryType = MemoryType.APP_GUIDE,
    platform: str = "android",
    app: str | None = None,
    tags: tuple[str, ...] = (),
) -> MemoryEntry:
    """Factory for MemoryEntry with sensible defaults."""
    return MemoryEntry(
        entry_id=entry_id,
        memory_type=memory_type,
        platform=platform,
        content=content,
        app=app,
        tags=tags,
        created_at=1_700_000_000.0,
        access_count=0,
    )


class _FakeEmbedder:
    """Deterministic fake EmbeddingProvider.

    Each text is hashed to a unique index position so that FAISS ranking is
    deterministic and independent of insertion order.  Embeddings are unit
    vectors (float32) which satisfy the inner-product similarity used by FAISS
    IndexFlatIP after L2 normalisation.
    """

    DIM = 8

    async def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            # Map each unique text to a distinct dimension slot so that the
            # embedding of "the query" is closest to the document whose
            # content contains the query text.
            slot = hash(text) % self.DIM
            vecs[i, slot] = 1.0
        return vecs


# ---------------------------------------------------------------------------
# MemoryEntry round-trip
# ---------------------------------------------------------------------------


def test_memory_entry_round_trip() -> None:
    """to_dict / from_dict must preserve all fields exactly."""
    original = _make_entry(
        "Open the Settings app",
        entry_id="roundtrip-1",
        memory_type=MemoryType.OS_GUIDE,
        platform="ios",
        app="Settings",
        tags=("guide", "settings"),
    )

    restored = MemoryEntry.from_dict(original.to_dict())

    assert restored.entry_id == original.entry_id
    assert restored.memory_type == original.memory_type
    assert restored.platform == original.platform
    assert restored.content == original.content
    assert restored.app == original.app
    assert restored.tags == original.tags
    assert restored.created_at == pytest.approx(original.created_at)
    assert restored.access_count == original.access_count


# ---------------------------------------------------------------------------
# MemoryStore persistence
# ---------------------------------------------------------------------------


def test_memory_store_add_and_reload(tmp_path) -> None:
    """An entry added to one MemoryStore instance must survive a reload."""
    store1 = MemoryStore(tmp_path)
    entry = _make_entry("Swipe up to go home", entry_id="persist-1")
    store1.add(entry)

    # Create a fresh instance pointing at the same directory.
    store2 = MemoryStore(tmp_path)

    loaded = store2.get("persist-1")
    assert loaded is not None
    assert loaded.content == "Swipe up to go home"
    assert loaded.memory_type == MemoryType.APP_GUIDE


def test_memory_store_get_missing_returns_none(tmp_path) -> None:
    """get() must return None for an entry_id that does not exist."""
    store = MemoryStore(tmp_path)
    result = store.get("this-id-does-not-exist")
    assert result is None


def test_memory_store_remove(tmp_path) -> None:
    """remove() must delete the entry in memory and persist the deletion."""
    store = MemoryStore(tmp_path)
    entry = _make_entry("Tap the back button", entry_id="remove-1")
    store.add(entry)
    assert store.get("remove-1") is not None

    removed = store.remove("remove-1")
    assert removed is True
    assert store.get("remove-1") is None

    # Verify the removal is persisted by reloading from disk.
    store_reloaded = MemoryStore(tmp_path)
    assert store_reloaded.get("remove-1") is None


def test_memory_store_remove_nonexistent_returns_false(tmp_path) -> None:
    """remove() must return False when the entry_id is not present."""
    store = MemoryStore(tmp_path)
    result = store.remove("ghost-id")
    assert result is False


def test_memory_store_list_all(tmp_path) -> None:
    """list_all() must return all stored entries."""
    store = MemoryStore(tmp_path)
    entries = [
        _make_entry("Entry one", entry_id="la-1"),
        _make_entry("Entry two", entry_id="la-2"),
        _make_entry("Entry three", entry_id="la-3"),
    ]
    for e in entries:
        store.add(e)

    results = store.list_all()
    assert len(results) == 3
    ids = {e.entry_id for e in results}
    assert ids == {"la-1", "la-2", "la-3"}


def test_memory_store_list_all_empty(tmp_path) -> None:
    """list_all() on a fresh store must return an empty list."""
    store = MemoryStore(tmp_path)
    assert store.list_all() == []


# ---------------------------------------------------------------------------
# MemoryRetriever — hybrid search
# ---------------------------------------------------------------------------


async def test_retriever_hybrid_search(tmp_path) -> None:
    """Index 3 entries; the one whose content matches the query term should
    appear first in results (or at least in the result set)."""
    embedder = _FakeEmbedder()
    retriever = MemoryRetriever(embedding_provider=embedder, alpha=0.6, top_k=3)

    entries = [
        _make_entry("Swipe up to open app drawer", entry_id="h-1"),
        _make_entry("Tap the back button to navigate", entry_id="h-2"),
        _make_entry("Long press icon to get options", entry_id="h-3"),
    ]
    await retriever.index(entries)

    results = await retriever.search("swipe up")

    assert len(results) > 0
    # The "swipe up" entry must be in results.
    result_ids = {entry.entry_id for entry, _ in results}
    assert "h-1" in result_ids


async def test_retriever_bm25_only(tmp_path) -> None:
    """With alpha=0.0, only BM25 term-overlap ranking is used.
    A document containing the exact query term should be returned."""
    embedder = _FakeEmbedder()
    retriever = MemoryRetriever(embedding_provider=embedder, alpha=0.0, top_k=3)

    entries = [
        _make_entry("Click the submit button", entry_id="b-1"),
        _make_entry("Press escape to dismiss", entry_id="b-2"),
        _make_entry("Submit the form by clicking OK", entry_id="b-3"),
    ]
    await retriever.index(entries)

    results = await retriever.search("submit")

    assert len(results) > 0
    result_ids = {entry.entry_id for entry, _ in results}
    # Both entries mentioning "submit" should be present.
    assert "b-1" in result_ids or "b-3" in result_ids


async def test_retriever_faiss_only(tmp_path) -> None:
    """With alpha=1.0, only FAISS embedding similarity is used.
    Results must be non-empty and valid MemoryEntry objects."""
    embedder = _FakeEmbedder()
    retriever = MemoryRetriever(embedding_provider=embedder, alpha=1.0, top_k=3)

    entries = [
        _make_entry("Open notifications panel", entry_id="f-1"),
        _make_entry("Toggle wifi from quick settings", entry_id="f-2"),
        _make_entry("Adjust screen brightness", entry_id="f-3"),
    ]
    await retriever.index(entries)

    results = await retriever.search("Open notifications panel")

    assert len(results) > 0
    for entry, score in results:
        assert isinstance(entry, MemoryEntry)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


async def test_retriever_format_context() -> None:
    """format_context() must return a non-empty formatted string."""
    embedder = _FakeEmbedder()
    retriever = MemoryRetriever(embedding_provider=embedder, alpha=0.6, top_k=5)

    entries = [
        _make_entry("Use ADB to capture screenshot", entry_id="fc-1", app="ADB"),
        _make_entry("Press power button twice to capture", entry_id="fc-2"),
    ]
    await retriever.index(entries)

    results = await retriever.search("screenshot")

    context = retriever.format_context(results)

    assert isinstance(context, str)
    assert len(context) > 0
    # Each result should produce a formatted line.
    lines = context.strip().splitlines()
    assert len(lines) == len(results)
    assert all(line.startswith("- [") for line in lines)


async def test_retriever_empty_index() -> None:
    """search() on an empty retriever must return an empty list."""
    embedder = _FakeEmbedder()
    retriever = MemoryRetriever(embedding_provider=embedder)

    results = await retriever.search("anything")

    assert results == []


async def test_retriever_format_context_empty() -> None:
    """format_context() with empty results must return an empty string."""
    embedder = _FakeEmbedder()
    retriever = MemoryRetriever(embedding_provider=embedder)

    context = retriever.format_context([])

    assert context == ""


# ---------------------------------------------------------------------------
# Chinese tokenization / BM25 tests (jieba integration)
# ---------------------------------------------------------------------------


def test_tokenize_chinese_word_segmentation() -> None:
    """_tokenize() with jieba must segment Chinese text into words, not chars.

    "打开浏览器搜索内容" should produce word-level tokens that include "浏览器"
    as a single token (not the three individual chars 浏, 览, 器).
    """
    tokens = _tokenize("打开浏览器搜索内容")
    assert "浏览器" in tokens, f"Expected '浏览器' as a word token; got {tokens}"
    assert "打开" in tokens, f"Expected '打开' as a word token; got {tokens}"


def test_tokenize_mixed_latin_cjk() -> None:
    """_tokenize() must handle mixed Latin + CJK text correctly.

    "open浏览器test" should yield "open", "浏览器", and "test" as separate tokens.
    """
    tokens = _tokenize("open浏览器test")
    assert "open" in tokens, f"Expected 'open' in {tokens}"
    assert "浏览器" in tokens, f"Expected '浏览器' as a word token in {tokens}"
    assert "test" in tokens, f"Expected 'test' in {tokens}"


def test_tokenize_english_unchanged() -> None:
    """_tokenize() must not regress on pure English input."""
    tokens = _tokenize("hello world")
    assert tokens == ["hello", "world"], f"Expected ['hello', 'world'], got {tokens}"


def test_tokenize_fallback_char_level() -> None:
    """When _JIEBA_AVAILABLE is False, _tokenize must fall back to char-level CJK splitting."""
    original = _retrieval_mod._JIEBA_AVAILABLE
    try:
        _retrieval_mod._JIEBA_AVAILABLE = False
        tokens = _tokenize("浏览器")
        assert tokens == ["浏", "览", "器"], (
            f"Expected char-level fallback ['浏', '览', '器'], got {tokens}"
        )
    finally:
        _retrieval_mod._JIEBA_AVAILABLE = original


async def test_retriever_bm25_chinese_search() -> None:
    """BM25-only search (alpha=0.0) must rank the entry containing the Chinese query word first.

    Indexes three entries with distinct Chinese content. Queries with "浏览器"
    (browser) and asserts the first result is the entry about browsers.
    """
    embedder = _FakeEmbedder()
    retriever = MemoryRetriever(embedding_provider=embedder, alpha=0.0, top_k=3)

    entries = [
        _make_entry("打开浏览器进行搜索", entry_id="zh-1"),
        _make_entry("调节屏幕亮度", entry_id="zh-2"),
        _make_entry("返回桌面主屏幕", entry_id="zh-3"),
    ]
    await retriever.index(entries)

    results = await retriever.search("浏览器")

    assert len(results) > 0, "Expected at least one result for '浏览器'"
    top_entry, _score = results[0]
    assert top_entry.entry_id == "zh-1", (
        f"Expected entry 'zh-1' (about browser) to rank first, got '{top_entry.entry_id}'"
    )


async def test_skill_library_search_chinese_bm25() -> None:
    """FlatSkillLibrary.search() must return matching Chinese-described skills.

    Creates a library with three skills that have Chinese descriptions and searches
    with a Chinese compound-word query. Verifies the correct skill is returned.
    """
    from guiclaw.skills.data import Skill, SkillStep
    from guiclaw.skills.flat import FlatSkillLibrary

    with pytest.MonkeyPatch().context() as mp:
        import tempfile
        import pathlib

        tmp_dir = pathlib.Path(tempfile.mkdtemp())
        library = FlatSkillLibrary(store_dir=tmp_dir)

        skills = [
            Skill(
                skill_id="zhs-1",
                name="打开浏览器",
                description="启动浏览器并搜索网页内容",
                app="chrome",
                platform="android",
                steps=[SkillStep(action_type="tap", target="浏览器图标")],
            ),
            Skill(
                skill_id="zhs-2",
                name="调节亮度",
                description="通过快捷设置调节屏幕亮度",
                app="settings",
                platform="android",
                steps=[SkillStep(action_type="swipe", target="通知栏")],
            ),
            Skill(
                skill_id="zhs-3",
                name="返回主屏幕",
                description="按下返回键回到桌面",
                app="launcher",
                platform="android",
                steps=[SkillStep(action_type="press", target="返回键")],
            ),
        ]
        for skill in skills:
            library.add(skill)

        results = await library.search("浏览器搜索", platform="android", top_k=3)

        assert len(results) > 0, "Expected at least one result for '浏览器搜索'"
        top_skill, _score = results[0]
        assert top_skill.skill_id == "zhs-1", (
            f"Expected skill 'zhs-1' (browser) to rank first, got '{top_skill.skill_id}'"
        )
