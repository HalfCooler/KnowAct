"""
guiclaw.memory.store
====================
Markdown-based persistent memory store.

Each :class:`~guiclaw.memory.types.MemoryType` is stored in its own ``.md``
file under ``{store_dir}/``.  Files are only created for types that have at
least one entry.

File layout::

    {store_dir}/
        os_guide.md
        app_guide.md
        icon_guide.md
        policy.md

Each file is a sequence of H2 sections — one section per
:class:`~guiclaw.memory.types.MemoryEntry`.  The section format is::

    ## {heading}
    id: {entry_id}
    type: {memory_type.value}
    platform: {platform}
    app: {app_or_empty}
    tags: {comma-separated tags, or empty}
    created_at: {float timestamp}
    access_count: {int}

    {content text}

An ``id:`` metadata line is stored explicitly so that the heading text
(which is a human-readable label derived from the content) does not affect
the round-trip identity of an entry.

Migration
---------
If a legacy ``memory.json`` file exists in ``store_dir``, it is automatically
read on first load, its entries are written to the new markdown files, and the
JSON file is deleted.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from guiclaw.memory.types import MemoryEntry, MemoryType

# ---------------------------------------------------------------------------
# Type → filename mapping
# ---------------------------------------------------------------------------

_TYPE_FILENAMES: dict[MemoryType, str] = {
    MemoryType.OS_GUIDE: "os_guide.md",
    MemoryType.APP_GUIDE: "app_guide.md",
    MemoryType.ICON_GUIDE: "icon_guide.md",
    MemoryType.POLICY: "policy.md",
}


def _type_to_filename(memory_type: MemoryType) -> str:
    """Return the markdown filename for *memory_type*."""
    return _TYPE_FILENAMES[memory_type]


# ---------------------------------------------------------------------------
# Per-entry serialisation helpers
# ---------------------------------------------------------------------------

_MAX_HEADING_LEN = 72  # characters


def _entry_heading(entry: MemoryEntry) -> str:
    """Short, single-line heading derived from the first line of content."""
    first_line = entry.content.splitlines()[0] if entry.content else ""
    if len(first_line) > _MAX_HEADING_LEN:
        first_line = first_line[: _MAX_HEADING_LEN - 1] + "…"
    return first_line or entry.entry_id


def _entry_to_section(entry: MemoryEntry) -> str:
    """Serialise *entry* to a markdown H2 section string."""
    heading = _entry_heading(entry)
    tags_str = ", ".join(entry.tags)
    app_str = entry.app if entry.app is not None else ""
    lines = [
        f"## {heading}",
        f"id: {entry.entry_id}",
        f"type: {entry.memory_type.value}",
        f"platform: {entry.platform}",
        f"app: {app_str}",
        f"tags: {tags_str}",
        f"created_at: {entry.created_at}",
        f"access_count: {entry.access_count}",
        "",
        entry.content,
        "",
    ]
    return "\n".join(lines)


def _parse_section(chunk: str) -> MemoryEntry | None:
    """Parse a single markdown chunk (after ``## `` has been stripped) into a
    :class:`MemoryEntry`.

    Returns ``None`` if the chunk is malformed or empty.
    """
    chunk = chunk.strip()
    if not chunk:
        return None

    # Split at first blank line to separate metadata from content
    blank_re = re.compile(r"\n[ \t]*\n", re.MULTILINE)
    m = blank_re.search(chunk)
    if m is None:
        # No blank line separator → entire chunk is metadata, no content body
        meta_block = chunk
        content = ""
    else:
        meta_block = chunk[: m.start()]
        content = chunk[m.end() :]

    # Parse metadata key: value lines (the first line is the heading text,
    # which we skip because the entry_id comes from the explicit ``id:`` line)
    meta: dict[str, str] = {}
    lines = meta_block.splitlines()
    # First line is the heading (may be the remainder after "## " was stripped)
    # which we intentionally skip — identity comes from id: line.
    for line in lines[1:]:
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    entry_id = meta.get("id")
    type_value = meta.get("type")
    platform = meta.get("platform")

    if not entry_id or not type_value or not platform:
        return None

    try:
        memory_type = MemoryType(type_value)
    except ValueError:
        return None

    app_raw = meta.get("app", "")
    app: str | None = app_raw if app_raw else None

    tags_raw = meta.get("tags", "")
    tags: tuple[str, ...] = tuple(t.strip() for t in tags_raw.split(",") if t.strip())

    try:
        created_at = float(meta.get("created_at", "0"))
    except ValueError:
        created_at = 0.0

    try:
        access_count = int(meta.get("access_count", "0"))
    except ValueError:
        access_count = 0

    return MemoryEntry(
        entry_id=entry_id,
        memory_type=memory_type,
        platform=platform,
        content=content.strip(),
        app=app,
        tags=tags,
        created_at=created_at,
        access_count=access_count,
    )


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class MemoryStore:
    """Persistent store for :class:`~guiclaw.memory.types.MemoryEntry` objects.

    Entries are persisted as per-type markdown files under ``{store_dir}/``.
    The public API is identical to the previous JSON-backed implementation so
    that all existing call sites continue to work without changes.
    """

    def __init__(self, store_dir: Path | str) -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, MemoryEntry] = {}
        self.load()

    # -- Public API ----------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._entries)

    def add(self, entry: MemoryEntry) -> None:
        self._entries[entry.entry_id] = entry
        self.save()

    def remove(self, entry_id: str) -> bool:
        if entry_id not in self._entries:
            return False
        del self._entries[entry_id]
        self.save()
        return True

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._entries.get(entry_id)

    def list_all(
        self,
        *,
        memory_type: MemoryType | None = None,
        platform: str | None = None,
        app: str | None = None,
    ) -> list[MemoryEntry]:
        results: list[MemoryEntry] = []
        for entry in self._entries.values():
            if memory_type is not None and entry.memory_type != memory_type:
                continue
            if platform is not None and entry.platform != platform:
                continue
            if app is not None and entry.app != app:
                continue
            results.append(entry)
        return results

    def save(self) -> None:
        """Atomically rewrite all markdown files to reflect the current state."""
        # Group entries by type
        by_type: dict[MemoryType, list[MemoryEntry]] = {t: [] for t in MemoryType}
        for entry in self._entries.values():
            by_type[entry.memory_type].append(entry)

        for memory_type, entries in by_type.items():
            target = self._dir / _type_to_filename(memory_type)
            if not entries:
                # Remove stale file if the type is now empty
                target.unlink(missing_ok=True)
                continue
            content = "".join(_entry_to_section(e) for e in entries)
            self._atomic_write(target, content)

    def load(self) -> None:
        """Load entries from markdown files.

        If a legacy ``memory.json`` exists it is migrated automatically: its
        contents are read, written to markdown, and the JSON file is deleted.
        """
        # --- Legacy JSON migration -------------------------------------------
        legacy_path = self._dir / "memory.json"
        if legacy_path.exists():
            self._load_from_json(legacy_path)
            self.save()
            legacy_path.unlink(missing_ok=True)
            return

        # --- Normal markdown load --------------------------------------------
        self._entries.clear()
        for memory_type in MemoryType:
            md_path = self._dir / _type_to_filename(memory_type)
            if not md_path.exists():
                continue
            self._parse_markdown_file(md_path)

    # -- Private helpers -----------------------------------------------------

    def _parse_markdown_file(self, path: Path) -> None:
        """Parse all H2 sections in *path* and add valid entries to
        ``self._entries``."""
        raw = path.read_text(encoding="utf-8")
        # Split on H2 headings; the first chunk before the first ## is discarded.
        chunks = re.split(r"^## ", raw, flags=re.MULTILINE)
        for chunk in chunks[1:]:  # skip preamble before first ##
            entry = _parse_section(chunk)
            if entry is not None:
                self._entries[entry.entry_id] = entry

    def _load_from_json(self, path: Path) -> None:
        """Populate ``self._entries`` from a legacy ``memory.json`` file."""
        self._entries.clear()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for eid, entry_data in data.get("entries", {}).items():
            self._entries[eid] = MemoryEntry.from_dict(entry_data)

    def _atomic_write(self, target: Path, content: str) -> None:
        """Write *content* to *target* atomically via a temporary file."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            dir=self._dir,
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        )
        try:
            tmp.write(content)
            tmp.close()
            Path(tmp.name).replace(target)
        except BaseException:
            Path(tmp.name).unlink(missing_ok=True)
            raise
