"""Default POLICY setup stored in GUIClaw memory."""

from __future__ import annotations

from pathlib import Path

from guiclaw.memory.store import MemoryStore
from guiclaw.memory.types import MemoryEntry, MemoryType

DEFAULT_POLICY_ENTRY_ID = "guiclaw-default-conservative-permissions"
DEFAULT_POLICY_CONTENT = (
    "Treat permission requests conservatively. Unless the current task explicitly requires "
    "and authorizes a permission, choose Deny, Cancel, Not now, or go back. If the task is "
    "blocked and authorization is unclear, request user intervention instead of granting "
    "the permission."
)


def ensure_default_policy(store: MemoryStore) -> list[MemoryEntry]:
    """Initialize an empty POLICY store without replacing user-defined entries."""
    policy_entries = store.list_all(memory_type=MemoryType.POLICY)
    if policy_entries:
        return policy_entries

    store.add(
        MemoryEntry(
            entry_id=DEFAULT_POLICY_ENTRY_ID,
            memory_type=MemoryType.POLICY,
            platform="all",
            content=DEFAULT_POLICY_CONTENT,
            tags=("default", "permissions", "conservative"),
        )
    )
    return store.list_all(memory_type=MemoryType.POLICY)


def load_policy_context(store_dir: Path | str) -> str:
    """Load all POLICY entries, creating the conservative default when empty."""
    entries = ensure_default_policy(MemoryStore(store_dir))
    return "\n".join(f"- {entry.content}" for entry in entries)
