"""memory_replace_text and memory_replace_lines tools — in-place edits to memory files."""
from __future__ import annotations

from openmemory.tools.base import ok, err, is_immutable, _IMMUTABLE_MSG
from openmemory.core import storage

# ---------------------------------------------------------------------------
# memory_replace_text
# ---------------------------------------------------------------------------

SCHEMA_TEXT = {
    "name": "memory_replace_text",
    "description": (
        "Replace the first occurrence of an exact string in a mutable memory file with new text. "
        "Use this to correct or update a specific passage without rewriting the whole file. "
        "The search string must match the file content exactly (including whitespace). "
        "Use memory_get first to read the file and confirm the exact text to replace. "
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
            "search": {
                "type": "string",
                "description": (
                    "Exact string to search for in the file. Must match character-for-character "
                    "including whitespace and newlines. Only the first occurrence is replaced."
                ),
            },
            "replacement": {
                "type": "string",
                "description": "Text to substitute in place of the matched string.",
            },
        },
        "required": ["file", "search", "replacement"],
    },
}


def run_text(session, file: str, search: str, replacement: str) -> dict:
    if is_immutable(file):
        return err(_IMMUTABLE_MSG.format(file=file))

    ws = session.workspace
    resolved = ws.resolve_file(file)

    if not search or not search.strip():
        return err("search cannot be empty")

    result = storage.replace_text(resolved, search, replacement)

    if "error" in result:
        return err(result["error"])

    # Re-index so the updated content is searchable immediately
    try:
        from openmemory.core.sync import sync_file
        sync_file(resolved, session.index, session.provider, session.config.chunking)
    except Exception as exc:  # noqa: BLE001
        return ok({
            "file": file,
            "replaced": True,
            "chars_delta": result.get("chars_delta", 0),
            "warning": f"Index sync failed: {exc}",
        })

    return ok({
        "file": file,
        "replaced": True,
        "chars_delta": result.get("chars_delta", 0),
    })


# ---------------------------------------------------------------------------
# memory_replace_lines
# ---------------------------------------------------------------------------

SCHEMA_LINES = {
    "name": "memory_replace_lines",
    "description": (
        "Replace a range of lines in a mutable memory file with new text. "
        "Use this when you know the line numbers of the content to update. "
        "Call memory_get first to read the file and identify the target line numbers. "
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
                "description": "1-based line number of the first line to replace.",
            },
            "end_line": {
                "type": "integer",
                "description": (
                    "1-based line number of the last line to replace (inclusive). "
                    "Pass the same value as start_line to replace a single line."
                ),
            },
            "replacement": {
                "type": "string",
                "description": "Text that replaces the specified line range.",
            },
        },
        "required": ["file", "start_line", "end_line", "replacement"],
    },
}


def run_lines(
    session,
    file: str,
    start_line: int,
    end_line: int,
    replacement: str,
) -> dict:
    if is_immutable(file):
        return err(_IMMUTABLE_MSG.format(file=file))

    ws = session.workspace
    resolved = ws.resolve_file(file)

    if not resolved.exists():
        return err(f"File not found: {file}")

    result = storage.replace_lines(resolved, start_line, end_line, replacement)

    if "error" in result:
        return err(result["error"])

    # Re-index so the updated content is searchable immediately
    try:
        from openmemory.core.sync import sync_file
        sync_file(resolved, session.index, session.provider, session.config.chunking)
    except Exception as exc:  # noqa: BLE001
        return ok({
            "file": file,
            "replaced_lines": result.get("replaced_lines", f"{start_line}-{end_line}"),
            "replaced_preview": result.get("replaced_preview", ""),
            "chars_delta": result.get("chars_delta", 0),
            "warning": f"Index sync failed: {exc}",
        })

    return ok({
        "file": file,
        "replaced_lines": result.get("replaced_lines", f"{start_line}-{end_line}"),
        "replaced_preview": result.get("replaced_preview", ""),
        "chars_delta": result.get("chars_delta", 0),
    })