"""memory_delete tool — tombstone-delete lines from a memory file."""
from __future__ import annotations

from openmemory.tools.base import ok, err, is_immutable, _IMMUTABLE_MSG
from openmemory.core import storage

SCHEMA = {
    "name": "memory_delete",
    "description": (
        "Delete specific lines from a mutable memory file by replacing them with an audit "
        "tombstone comment. The original lines are preserved in an audit trail within the "
        "file so the deletion is always reversible by a human. "
        "Only USER.md and AGENTS.md are editable; MEMORY.md and daily/*.md are "
        "append-only history and cannot be modified."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": (
                    "Relative path to the mutable file to edit, e.g. 'USER.md' or "
                    "'AGENTS.md'. MEMORY.md and daily/*.md are immutable."
                ),
            },
            "start_line": {
                "type": "integer",
                "description": "1-based line number of the first line to delete.",
            },
            "end_line": {
                "type": "integer",
                "description": (
                    "1-based line number of the last line to delete (inclusive). "
                    "Pass the same value as start_line to delete a single line."
                ),
            },
            "reason": {
                "type": "string",
                "description": "Brief human-readable reason for the deletion (stored in audit trail).",
            },
        },
        "required": ["file", "start_line", "end_line"],
    },
}


def run(
    session,
    file: str,
    start_line: int,
    end_line: int,
    reason: str = "deleted by agent",
) -> dict:
    ws = session.workspace

    if is_immutable(file):
        return err(_IMMUTABLE_MSG.format(file=file))

    resolved = ws.resolve_file(file)
    if not resolved.exists():
        return err(f"File not found: {file}")

    # storage.delete_lines uses 0-indexed [start, end) — convert from 1-indexed inclusive
    try:
        result = storage.delete_lines(resolved, start_line - 1, end_line)
    except (ValueError, IOError) as exc:
        return err(str(exc))

    if "error" in result:
        return err(result["error"])

    # Re-index the file so the index reflects the deletion immediately.
    try:
        from openmemory.core.sync import sync_file

        sync_file(resolved, session.index, session.provider, session.config.chunking)
    except Exception as exc:  # noqa: BLE001
        # Non-fatal: the file was deleted correctly; index will catch up on next sync.
        return ok(
            {
                "file": file,
                "deleted_lines": result.get("deleted_lines", f"{start_line}-{end_line}"),
                "warning": f"Index sync failed: {exc}",
            }
        )

    return ok({"file": file, "deleted_lines": result.get("deleted_lines", f"{start_line}-{end_line}")})
