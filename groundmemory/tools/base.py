"""
Base class and shared utilities for all groundmemory tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from groundmemory.config import groundmemoryConfig


class MemoryToolError(Exception):
    """Raised when a tool call fails with a user-visible error."""


def ok(data: Any) -> dict:
    """Wrap a successful tool result."""
    return {"status": "ok", **data} if isinstance(data, dict) else {"status": "ok", "result": data}


def err(message: str) -> dict:
    """Wrap a tool error result."""
    return {"status": "error", "message": message}


_STANDARD_TIER_NAMES = ("MEMORY.md", "USER.md", "AGENTS.md", "RELATIONS.md")


def is_immutable(file: str, config: "groundmemoryConfig") -> bool:
    """
    Return True if replace/delete (edit) operations on *file* are disallowed.

    Append is never affected by this check - it only governs whether existing
    content in the file may be mutated or removed.

    A specific dated daily log (e.g. 'daily/2026-01-01.md') is always
    immutable history, regardless of config. The bare 'daily' keyword (today's
    live log) and the standard MEMORY.md/USER.md/AGENTS.md/RELATIONS.md tiers
    are mutable only if listed in config.mutable_tiers; any other (custom)
    file falls back to its own CustomFileConfig.mutable flag, defaulting to
    mutable if unrecognised.
    """
    p = Path(file)
    parts = p.parts
    name = p.name

    # A specific dated daily file - always immutable, never a config option.
    if len(parts) >= 2 and parts[-2].lower() == "daily":
        return True

    # The "daily" keyword refers to today's live log.
    if name.lower() == "daily":
        return "daily" not in config.mutable_tiers

    if name in _STANDARD_TIER_NAMES:
        return name not in config.mutable_tiers

    for cf in config.custom_files:
        if cf.name.upper() == name.upper():
            return not cf.mutable

    return False


_IMMUTABLE_MSG = (
    "'{file}' is an append-only memory file and cannot be edited or deleted. "
    "Use memory_write to append new information instead."
)


def _auto_clear_first_run(session) -> None:
    """
    Silently empty FIRST_RUN.md after the first successful write operation.

    This marks onboarding as complete without requiring the model to do it
    explicitly.  Safe to call on every write - does nothing once the file
    is already empty.
    """
    try:
        fr = session.workspace.first_run_file
        if fr.exists() and fr.read_text(encoding="utf-8").strip():
            fr.write_text("", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass  # Never let this interfere with the actual write result


def sync_after_edit(
    session,
    resolved: Path,
    is_relations: bool,
    base_payload: dict,
) -> dict:
    """
    Re-index *resolved* after an in-place edit and return ``ok(base_payload)``.

    * Calls ``sync_file`` to make the updated content immediately searchable.
    * When *is_relations* is True, also calls ``sync_relations_from_file`` and
      appends ``relations_format``, ``format_reminder``, and (if non-empty)
      ``relations_synced`` keys to the payload.
    * On sync failure the exception is swallowed and a ``warning`` key is added
      instead of propagating (the file edit already succeeded).
    """
    from groundmemory.core.sync import sync_file
    from groundmemory.core.relations import sync_relations_from_file, RELATIONS_FORMAT_REMINDER

    relation_sync_result = None
    try:
        sync_file(resolved, session.index, session.provider, session.config.chunking)
        if is_relations:
            relation_sync_result = sync_relations_from_file(resolved, session.index)
    except Exception as exc:  # noqa: BLE001
        base_payload["warning"] = f"Index sync failed: {exc}"
        if is_relations:
            base_payload["relations_format"] = "confirmed"
            base_payload["format_reminder"] = RELATIONS_FORMAT_REMINDER
        return ok(base_payload)

    if is_relations:
        base_payload["relations_format"] = "confirmed"
        base_payload["format_reminder"] = RELATIONS_FORMAT_REMINDER
        if relation_sync_result:
            base_payload["relations_synced"] = relation_sync_result
    _auto_clear_first_run(session)
    return ok(base_payload)
