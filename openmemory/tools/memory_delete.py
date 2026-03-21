"""memory_delete tool — tombstone-delete lines from a memory file."""
from __future__ import annotations

from openmemory.tools.base import ok, err, is_immutable, _IMMUTABLE_MSG
from openmemory.core import storage
from openmemory.core.graph import parse_relations_from_file, _relation_id

SCHEMA = {
    "name": "memory_delete",
    "description": (
        "Delete specific lines from a mutable memory file by replacing them with an audit "
        "tombstone comment. The original lines are preserved in an audit trail within the "
        "file so the deletion is always reversible by a human. "
        "Only USER.md, AGENTS.md, and RELATIONS.md are editable; MEMORY.md and daily/*.md are "
        "append-only history and cannot be modified."
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

    is_relations = resolved.name.upper() == "RELATIONS.MD"

    # Snapshot the relation lines that are about to be deleted BEFORE the edit
    relations_to_delete: list[dict] = []
    if is_relations and resolved.exists():
        all_file_relations = parse_relations_from_file(resolved)
        lines = resolved.read_text(encoding="utf-8").splitlines()
        # Collect lines in the to-be-deleted range (1-indexed inclusive)
        deleted_text = "\n".join(lines[start_line - 1 : end_line])
        # Parse only from the deleted slice to know which triples to remove
        import tempfile, pathlib
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(deleted_text)
            tmp_path = pathlib.Path(tmp.name)
        relations_to_delete = parse_relations_from_file(tmp_path)
        tmp_path.unlink(missing_ok=True)

    # storage.delete_lines uses 0-indexed [start, end) — convert from 1-indexed inclusive
    try:
        result = storage.delete_lines(resolved, start_line - 1, end_line)
    except (ValueError, IOError) as exc:
        return err(str(exc))

    if "error" in result:
        return err(result["error"])

    # Delete the corresponding SQLite relation rows
    relations_deleted: list[str] = []
    if is_relations and relations_to_delete:
        for r in relations_to_delete:
            rid = _relation_id(r["subject"], r["predicate"], r["object"])
            session.index.delete_relation(rid)
            relations_deleted.append(f"[{r['subject']}] --{r['predicate']}--> [{r['object']}]")

    # Re-index the file so the index reflects the deletion immediately.
    try:
        from openmemory.core.sync import sync_file

        sync_file(resolved, session.index, session.provider, session.config.chunking)
    except Exception as exc:  # noqa: BLE001
        # Non-fatal: the file was deleted correctly; index will catch up on next sync.
        payload = {
            "file": file,
            "deleted_lines": result.get("deleted_lines", f"{start_line}-{end_line}"),
            "warning": f"Index sync failed: {exc}",
        }
        if relations_deleted:
            payload["relations_deleted"] = relations_deleted
        return ok(payload)

    payload = {
        "file": file,
        "deleted_lines": result.get("deleted_lines", f"{start_line}-{end_line}"),
    }
    if relations_deleted:
        payload["relations_deleted"] = relations_deleted
    return ok(payload)
