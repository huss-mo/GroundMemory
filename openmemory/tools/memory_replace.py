"""memory_replace_text and memory_replace_lines tools — in-place edits to memory files."""
from __future__ import annotations

import re

from openmemory.tools.base import ok, err, is_immutable, _IMMUTABLE_MSG
from openmemory.core import storage
from openmemory.core.graph import RELATION_LINE_RE, sync_relations_from_file

# ---------------------------------------------------------------------------
# RELATIONS.md helpers
# ---------------------------------------------------------------------------

_RELATIONS_FORMAT_REMINDER = (
    "Required format for each line: "
    "- [Subject] --predicate--> [Object] (YYYY-MM-DD) — \"optional note\"\n"
    "Example: - [Alice] --leads--> [Auth Team] (2026-03-20) — \"Sprint planning\""
)


def _validate_relations_replacement(text: str) -> tuple[bool, list[str], list[str]]:
    """
    Validate that every non-blank, non-comment line in *text* matches the
    RELATIONS.md relation format.

    Returns (all_valid, valid_lines, invalid_lines).
    """
    valid: list[str] = []
    invalid: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Allow blank lines and comment / header lines (starting with # or <!--)
        if not stripped or stripped.startswith("#") or stripped.startswith("<!--"):
            continue
        if RELATION_LINE_RE.match(line):
            valid.append(stripped)
        else:
            invalid.append(stripped)
    return len(invalid) == 0, valid, invalid


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
        "Only USER.md, AGENTS.md, and RELATIONS.md are editable; MEMORY.md and daily/*.md are "
        "append-only history and cannot be modified. "
        "When editing RELATIONS.md every replacement line must follow the format: "
        "- [Subject] --predicate--> [Object] (YYYY-MM-DD)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": (
                    "Relative path to the mutable file to edit, e.g. 'USER.md', "
                    "'AGENTS.md', or 'RELATIONS.md'. MEMORY.md and daily/*.md are immutable."
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
                "description": (
                    "Text to substitute in place of the matched string. "
                    "When editing RELATIONS.md, each non-blank line must follow: "
                    "- [Subject] --predicate--> [Object] (YYYY-MM-DD)"
                ),
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

    is_relations = resolved.name.upper() == "RELATIONS.MD"

    # Validate RELATIONS.md replacement format before touching the file
    if is_relations:
        all_valid, valid_lines, invalid_lines = _validate_relations_replacement(replacement)
        if not all_valid:
            return err(
                f"Replacement text contains {len(invalid_lines)} line(s) that do not match "
                f"the required RELATIONS.md format.\n"
                f"Invalid line(s): {invalid_lines}\n"
                f"{_RELATIONS_FORMAT_REMINDER}"
            )

    result = storage.replace_text(resolved, search, replacement)

    if "error" in result:
        return err(result["error"])

    # Re-index so the updated content is searchable immediately
    relation_sync_result = None
    try:
        from openmemory.core.sync import sync_file
        sync_file(resolved, session.index, session.provider, session.config.chunking)
        # sync_file already calls sync_relations_from_file for RELATIONS.md,
        # but we also capture a direct sync result for the response when relevant
        if is_relations:
            relation_sync_result = sync_relations_from_file(resolved, session.index)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "file": file,
            "replaced": True,
            "chars_delta": result.get("chars_delta", 0),
            "warning": f"Index sync failed: {exc}",
        }
        if is_relations:
            payload["relations_format"] = "confirmed"
            payload["format_reminder"] = _RELATIONS_FORMAT_REMINDER
        return ok(payload)

    payload: dict = {
        "file": file,
        "replaced": True,
        "chars_delta": result.get("chars_delta", 0),
    }
    if is_relations:
        payload["relations_format"] = "confirmed"
        payload["format_reminder"] = _RELATIONS_FORMAT_REMINDER
        if relation_sync_result:
            payload["relations_synced"] = relation_sync_result
    return ok(payload)


# ---------------------------------------------------------------------------
# memory_replace_lines
# ---------------------------------------------------------------------------

SCHEMA_LINES = {
    "name": "memory_replace_lines",
    "description": (
        "Replace a range of lines in a mutable memory file with new text. "
        "Use this when you know the line numbers of the content to update. "
        "Call memory_get first to read the file and identify the target line numbers. "
        "Only USER.md, AGENTS.md, and RELATIONS.md are editable; MEMORY.md and daily/*.md are "
        "append-only history and cannot be modified. "
        "When editing RELATIONS.md every replacement line must follow the format: "
        "- [Subject] --predicate--> [Object] (YYYY-MM-DD)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": (
                    "Relative path to the mutable file to edit, e.g. 'USER.md', "
                    "'AGENTS.md', or 'RELATIONS.md'. MEMORY.md and daily/*.md are immutable."
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
                "description": (
                    "Text that replaces the specified line range. "
                    "When editing RELATIONS.md, each non-blank line must follow: "
                    "- [Subject] --predicate--> [Object] (YYYY-MM-DD)"
                ),
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

    is_relations = resolved.name.upper() == "RELATIONS.MD"

    # Validate RELATIONS.md replacement format before touching the file
    if is_relations:
        all_valid, valid_lines, invalid_lines = _validate_relations_replacement(replacement)
        if not all_valid:
            return err(
                f"Replacement text contains {len(invalid_lines)} line(s) that do not match "
                f"the required RELATIONS.md format.\n"
                f"Invalid line(s): {invalid_lines}\n"
                f"{_RELATIONS_FORMAT_REMINDER}"
            )

    result = storage.replace_lines(resolved, start_line, end_line, replacement)

    if "error" in result:
        return err(result["error"])

    # Re-index so the updated content is searchable immediately
    relation_sync_result = None
    try:
        from openmemory.core.sync import sync_file
        sync_file(resolved, session.index, session.provider, session.config.chunking)
        if is_relations:
            relation_sync_result = sync_relations_from_file(resolved, session.index)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "file": file,
            "replaced_lines": result.get("replaced_lines", f"{start_line}-{end_line}"),
            "replaced_preview": result.get("replaced_preview", ""),
            "chars_delta": result.get("chars_delta", 0),
            "warning": f"Index sync failed: {exc}",
        }
        if is_relations:
            payload["relations_format"] = "confirmed"
            payload["format_reminder"] = _RELATIONS_FORMAT_REMINDER
        return ok(payload)

    payload = {
        "file": file,
        "replaced_lines": result.get("replaced_lines", f"{start_line}-{end_line}"),
        "replaced_preview": result.get("replaced_preview", ""),
        "chars_delta": result.get("chars_delta", 0),
    }
    if is_relations:
        payload["relations_format"] = "confirmed"
        payload["format_reminder"] = _RELATIONS_FORMAT_REMINDER
        if relation_sync_result:
            payload["relations_synced"] = relation_sync_result
    return ok(payload)