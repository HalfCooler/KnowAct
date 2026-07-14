"""Planner-only routing-memory extraction and serialization helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from nanobot.agent.capabilities import CapabilityCatalog
from nanobot.agent.memory import MemoryStore

MAX_HINTS = 5
MAX_HINT_CHARS = 160
MAX_SERIALIZED_HINT_CHARS = 900
_HISTORY_TAIL_CHARS = 4_000
_OUTCOME_KEYWORDS = (
    "worked",
    "succeeded",
    "success",
    "failed",
    "failure",
    "fallback",
)
_ROUTE_ALIAS_STOPWORDS = {"tool", "gui", "mcp", "api", "filesystem", "web"}


@dataclass(frozen=True)
class PlanningMemoryHint:
    """Compact planner-facing routing hint derived from persistent memory."""

    route_id: str
    note: str

    def to_prompt_line(self, max_chars: int = MAX_HINT_CHARS) -> str:
        """Render one bounded line for planner prompts."""
        text = f"{self.route_id}: {self.note.strip()}"
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."


def serialize_memory_hints(
    hints: tuple[PlanningMemoryHint, ...],
    *,
    max_hints: int = MAX_HINTS,
    max_chars: int = MAX_HINT_CHARS,
    max_total_chars: int = MAX_SERIALIZED_HINT_CHARS,
) -> tuple[str, ...]:
    """Serialize bounded planner hints without leaking large memory excerpts."""
    lines: list[str] = []
    total_chars = 0
    for hint in hints[:max_hints]:
        line = hint.to_prompt_line(max_chars=max_chars)
        if lines and total_chars + len(line) > max_total_chars:
            break
        if not lines and len(line) > max_total_chars:
            line = line[: max_total_chars - 3].rstrip() + "..."
        lines.append(line)
        total_chars += len(line)
    return tuple(lines)


class PlanningMemoryHintExtractor:
    """Read persistent memory conservatively and emit only route-relevant hints."""

    def __init__(self, workspace_or_store: Path | MemoryStore):
        if isinstance(workspace_or_store, MemoryStore):
            self._store = workspace_or_store
        else:
            self._store = MemoryStore(workspace_or_store)

    def build(self, task: str, catalog: CapabilityCatalog) -> tuple[PlanningMemoryHint, ...]:
        """Return bounded routing hints derived from existing memory files."""
        del task  # Reserved for later ranking; extraction remains route/outcome focused in Phase 21.
        if not catalog.routes:
            return ()

        hints: list[PlanningMemoryHint] = []
        seen: set[tuple[str, str]] = set()
        for snippet in self._iter_candidate_snippets():
            route_id = self._match_route_id(snippet, catalog)
            if route_id is None:
                continue
            note = self._normalize_snippet(snippet)
            key = (route_id, note.casefold())
            if key in seen:
                continue
            seen.add(key)
            hints.append(PlanningMemoryHint(route_id=route_id, note=note))
            if len(hints) >= MAX_HINTS:
                break
        return tuple(hints)

    def _iter_candidate_snippets(self) -> tuple[str, ...]:
        text_blocks = [self._store.read_memory(), self._read_history_tail()]
        snippets: list[str] = []
        for text in text_blocks:
            if not text:
                continue
            for raw in re.split(r"(?:\n\s*\n|\n)", text):
                snippet = raw.strip(" -*\t")
                if not snippet:
                    continue
                lowered = snippet.casefold()
                if not any(keyword in lowered for keyword in _OUTCOME_KEYWORDS):
                    continue
                snippets.append(snippet)
        return tuple(snippets)

    def _read_history_tail(self) -> str:
        history_file = self._store.history_file
        if not history_file.exists():
            return ""
        return history_file.read_text(encoding="utf-8")[-_HISTORY_TAIL_CHARS:]

    def _match_route_id(self, snippet: str, catalog: CapabilityCatalog) -> str | None:
        lowered = snippet.casefold()
        for route in catalog.routes:
            route_id = route.route_id.casefold()
            if route_id in lowered:
                return route.route_id
            aliases = self._route_aliases(route.route_id, route.kind)
            if any(alias in lowered for alias in aliases):
                return route.route_id
        return None

    @staticmethod
    def _route_aliases(route_id: str, kind: str) -> tuple[str, ...]:
        parts = [part for part in re.split(r"[._-]+", route_id.casefold()) if part and part not in _ROUTE_ALIAS_STOPWORDS]
        aliases = {kind.casefold(), *parts}
        if len(parts) > 1:
            aliases.add(" ".join(parts))
            aliases.add("_".join(parts))
        return tuple(sorted(alias for alias in aliases if len(alias) >= 4))

    @staticmethod
    def _normalize_snippet(snippet: str) -> str:
        compact = " ".join(snippet.split())
        return compact
