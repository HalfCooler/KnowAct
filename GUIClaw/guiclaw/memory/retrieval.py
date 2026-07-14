"""
guiclaw.memory.retrieval
========================
Hybrid BM25 + FAISS embedding retrieval for the GUI agent memory store.

Dependencies:
  - ``faiss-cpu`` (or ``faiss-gpu``) for vector similarity search
  - An ``EmbeddingProvider`` implementation for computing embeddings via API
    (e.g. qwen3-vl-embedding through DashScope)
"""

from __future__ import annotations

import math
import re
import typing
from dataclasses import dataclass, field

import numpy as np

from guiclaw.memory.types import MemoryEntry, MemoryType

if typing.TYPE_CHECKING:
    import faiss as _faiss_mod  # noqa: F401

try:
    import jieba as _jieba
    _JIEBA_AVAILABLE = True
except ImportError:
    _jieba = None  # type: ignore[assignment]
    _JIEBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# EmbeddingProvider protocol
# ---------------------------------------------------------------------------

@typing.runtime_checkable
class EmbeddingProvider(typing.Protocol):
    """Protocol for external embedding APIs (e.g. qwen3-vl-embedding)."""

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, dim) float32 array of embeddings for *texts*."""
        ...


# ---------------------------------------------------------------------------
# BM25 (pure-Python, CJK-aware)
# ---------------------------------------------------------------------------

_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _tokenize_charlevel(text: str) -> list[str]:
    """Char-level CJK splitting + Latin word splitting (fallback implementation)."""
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if _is_cjk(ch):
            if buf:
                tokens.append("".join(buf))
                buf.clear()
            tokens.append(ch)
        elif re.match(r"\w", ch):
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf))
                buf.clear()
    if buf:
        tokens.append("".join(buf))
    return tokens


def _tokenize_jieba(text: str) -> list[str]:
    """Word-level tokenization using jieba.lcut(), filtering punctuation segments."""
    raw_tokens: list[str] = _jieba.lcut(text)  # type: ignore[union-attr]
    return [tok for tok in raw_tokens if re.search(r"\w", tok)]


def _tokenize(text: str) -> list[str]:
    """Tokenize *text* with jieba word segmentation (if available) or char-level CJK fallback."""
    text = text.lower()
    if _JIEBA_AVAILABLE:
        return _tokenize_jieba(text)
    return _tokenize_charlevel(text)


@dataclass
class _BM25Index:
    """Lightweight BM25 index over a list of documents."""

    k1: float = 1.5
    b: float = 0.75

    _doc_tokens: list[list[str]] = field(default_factory=list, repr=False)
    _doc_lens: list[int] = field(default_factory=list, repr=False)
    _avgdl: float = 0.0
    _df: dict[str, int] = field(default_factory=dict, repr=False)
    _n: int = 0

    def build(self, documents: list[str]) -> None:
        self._doc_tokens = [_tokenize(d) for d in documents]
        self._doc_lens = [len(t) for t in self._doc_tokens]
        self._n = len(documents)
        self._avgdl = (sum(self._doc_lens) / self._n) if self._n else 0.0
        self._df.clear()
        for tokens in self._doc_tokens:
            seen: set[str] = set()
            for t in tokens:
                if t not in seen:
                    self._df[t] = self._df.get(t, 0) + 1
                    seen.add(t)

    def score(self, query: str) -> list[float]:
        """Return BM25 scores for all indexed documents given *query*."""
        query_tokens = _tokenize(query)
        scores = [0.0] * self._n
        for qt in query_tokens:
            df = self._df.get(qt, 0)
            if df == 0:
                continue
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
            for i, doc_tokens in enumerate(self._doc_tokens):
                tf = doc_tokens.count(qt)
                if tf == 0:
                    continue
                dl = self._doc_lens[i]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores


# ---------------------------------------------------------------------------
# FAISS vector index wrapper
# ---------------------------------------------------------------------------

@dataclass
class _FaissIndex:
    """Thin wrapper around a FAISS flat-IP index."""

    _index: _faiss_mod.IndexFlatIP | None = field(default=None, repr=False)
    _dim: int = 0

    def build(self, embeddings: np.ndarray) -> None:
        """Build (or rebuild) the index from an (N, dim) float32 matrix."""
        import faiss

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self._dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings)

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) arrays of shape (top_k,)."""
        import faiss

        if self._index is None or self._index.ntotal == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        query_vec = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(query_vec)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)
        return scores[0], indices[0]


# ---------------------------------------------------------------------------
# MemoryRetriever — hybrid BM25 + FAISS
# ---------------------------------------------------------------------------

@dataclass
class MemoryRetriever:
    """Hybrid BM25 + embedding retrieval over :class:`MemoryEntry` objects.

    Parameters
    ----------
    embedding_provider:
        An async embedding API conforming to :class:`EmbeddingProvider`.
    alpha:
        Blending weight in ``[0, 1]``.  ``1.0`` = pure embedding,
        ``0.0`` = pure BM25.  Default ``0.6``.
    top_k:
        Number of results to return from :meth:`search`.
    """

    embedding_provider: EmbeddingProvider
    alpha: float = 0.6
    top_k: int = 5

    _entries: list[MemoryEntry] = field(default_factory=list, repr=False)
    _documents: list[str] = field(default_factory=list, repr=False)
    _bm25: _BM25Index = field(default_factory=_BM25Index, repr=False)
    _faiss: _FaissIndex = field(default_factory=_FaissIndex, repr=False)
    _dirty: bool = field(default=True, repr=False)

    async def index(self, entries: list[MemoryEntry]) -> None:
        """(Re)build both BM25 and FAISS indices from *entries*."""
        self._entries = list(entries)
        self._documents = [self._entry_text(e) for e in self._entries]

        # BM25
        self._bm25.build(self._documents)

        # Embeddings via external API → FAISS
        if self._documents:
            embeddings = await self.embedding_provider.embed(self._documents)
            self._last_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            self._faiss.build(embeddings)

        self._dirty = False

    def index_from_embeddings(self, entries: list[MemoryEntry], embeddings: np.ndarray) -> None:
        """(Re)build indices from pre-computed embeddings, skipping the API call.

        *embeddings* must be the raw (unnormalized) float32 array of shape
        ``(len(entries), dim)`` as originally returned by the embedding provider
        and saved via :attr:`_last_embeddings`.
        """
        self._entries = list(entries)
        self._documents = [self._entry_text(e) for e in self._entries]
        self._bm25.build(self._documents)
        if self._documents:
            self._last_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            self._faiss.build(embeddings)
        self._dirty = False

    async def search(
        self,
        query: str,
        *,
        memory_type: MemoryType | None = None,
        platform: str | None = None,
        app: str | None = None,
        top_k: int | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Return up to *top_k* ``(entry, score)`` pairs ranked by hybrid score.

        Pre-filters by *memory_type* / *platform* / *app* before scoring.
        """
        if not self._entries:
            return []

        k = top_k or self.top_k

        # Pre-filter mask
        mask = self._filter_mask(memory_type=memory_type, platform=platform, app=app)
        if not any(mask):
            return []

        # BM25 scores (full corpus, then mask)
        bm25_scores = np.array(self._bm25.score(query), dtype=np.float32)
        bm25_scores[~mask] = -1e9

        # Embedding scores via FAISS
        query_emb = await self.embedding_provider.embed([query])
        faiss_scores_raw, faiss_indices = self._faiss.search(query_emb[0], len(self._entries))

        emb_scores = np.full(len(self._entries), -1e9, dtype=np.float32)
        for score, idx in zip(faiss_scores_raw, faiss_indices):
            if idx >= 0:
                emb_scores[idx] = score
        emb_scores[~mask] = -1e9

        # Normalize to [0, 1] range for blending
        bm25_norm = self._min_max_normalize(bm25_scores, mask)
        emb_norm = self._min_max_normalize(emb_scores, mask)

        # Hybrid score
        hybrid = (1.0 - self.alpha) * bm25_norm + self.alpha * emb_norm

        # Top-k
        ranked_indices = np.argsort(-hybrid)
        results: list[tuple[MemoryEntry, float]] = []
        for idx in ranked_indices:
            if not mask[idx]:
                continue
            if hybrid[idx] <= 0:
                break
            results.append((self._entries[idx], float(hybrid[idx])))
            if len(results) >= k:
                break

        return results

    def format_context(self, results: list[tuple[MemoryEntry, float]]) -> str:
        """Format retrieval results into a text block for prompt injection."""
        if not results:
            return ""
        lines: list[str] = []
        for entry, score in results:
            tag = entry.memory_type.value.upper()
            prefix = f"[{tag}]"
            if entry.app:
                prefix += f" ({entry.app})"
            lines.append(f"- {prefix} {entry.content}")
        return "\n".join(lines)

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _entry_text(entry: MemoryEntry) -> str:
        """Build a searchable text representation of a memory entry."""
        parts = [entry.content]
        if entry.app:
            parts.append(entry.app)
        if entry.tags:
            parts.extend(entry.tags)
        return " ".join(parts)

    def _filter_mask(
        self,
        *,
        memory_type: MemoryType | None,
        platform: str | None,
        app: str | None,
    ) -> np.ndarray:
        mask = np.ones(len(self._entries), dtype=bool)
        for i, entry in enumerate(self._entries):
            if memory_type is not None and entry.memory_type != memory_type:
                mask[i] = False
            elif platform is not None and entry.platform != platform:
                mask[i] = False
            elif app is not None and entry.app != app:
                mask[i] = False
        return mask

    @staticmethod
    def _min_max_normalize(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Min-max normalize scores within the valid (masked) range to [0, 1]."""
        valid = scores[mask]
        if len(valid) == 0:
            return np.zeros_like(scores)
        lo, hi = valid.min(), valid.max()
        if hi - lo < 1e-9:
            # All scores identical — return uniform 0.5 for valid entries
            out = np.zeros_like(scores)
            out[mask] = 0.5
            return out
        return np.where(mask, (scores - lo) / (hi - lo), 0.0)
