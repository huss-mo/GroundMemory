"""
Relation graph management.

Stores named relationships between entities in:
  1. SQLite (fast lookup, structured queries)
  2. RELATIONS.md (human-readable mirror, injected at bootstrap)

Format in RELATIONS.md:
  - [Alice] --leads--> [Auth Team] (2026-03-20) — "Added during sprint planning"
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from openmemory.core.index import MemoryIndex
from openmemory.core.storage import _atomic_write

if TYPE_CHECKING:
    from openmemory.core.embeddings import EmbeddingProvider

# Regex that matches a valid RELATIONS.md line.
# Groups: subject, predicate, object, date (optional), note (optional)
RELATION_LINE_RE = re.compile(
    r"^\s*-\s+\[([^\]]+)\]\s+--([^->\s][^->]*?)-->\s+\[([^\]]+)\]"
    r"(?:\s+\((\d{4}-\d{2}-\d{2})\))?"
    r"(?:\s+[—–-]\s+\"?(.*?)\"?)?\s*$"
)


def _relation_id(subject: str, predicate: str, object_: str) -> str:
    """Deterministic ID so duplicate relations get upserted, not duplicated."""
    raw = f"{subject.lower()}|{predicate.lower()}|{object_.lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _format_relation_line(
    subject: str,
    predicate: str,
    object_: str,
    note: Optional[str] = None,
) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    line = f"- [{subject}] --{predicate}--> [{object_}] ({date_str})"
    if note:
        line += f' — "{note}"'
    return line


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _find_semantic_duplicate(
    index: MemoryIndex,
    provider: "EmbeddingProvider",
    subject: str,
    predicate: str,
    object_: str,
    threshold: float,
) -> Optional[dict]:
    """
    Embed ``"{subject} {predicate} {object_}"`` and compare cosine similarity
    against all existing relations.

    Returns the first existing relation whose similarity to the new triple
    meets or exceeds *threshold*, or ``None`` if no duplicate is found.

    Returns ``None`` immediately when the provider returns an empty vector
    (e.g. ``NullEmbeddingProvider``), so semantic dedup is a no-op for
    BM25-only sessions.
    """
    new_text = f"{subject} {predicate} {object_}"
    new_vec = provider.embed([new_text])[0]

    # NullEmbeddingProvider returns [] — skip semantic dedup
    if not new_vec:
        return None

    existing = index.get_all_relations()
    for row in existing:
        row_text = f"{row['subject']} {row['predicate']} {row['object']}"
        row_vec = provider.embed([row_text])[0]
        if not row_vec:
            continue
        sim = _cosine_similarity(new_vec, row_vec)
        if sim >= threshold:
            return dict(row)
    return None


def add_relation(
    index: MemoryIndex,
    relations_file: Path,
    subject: str,
    predicate: str,
    object_: str,
    note: Optional[str] = None,
    confidence: float = 1.0,
    provider: Optional["EmbeddingProvider"] = None,
    dedup_threshold: float = 0.92,
) -> dict:
    """
    Record a named relationship between two entities.

    - Writes to SQLite relations table (upsert by deterministic ID).
    - Appends a human-readable line to RELATIONS.md.
    - When *provider* is supplied (and not NullEmbeddingProvider), performs
      semantic deduplication: if an existing relation is cosine-similar above
      *dedup_threshold* the new triple is skipped and the existing one returned.

    Returns a dict describing what was written (or the duplicate if deduped).
    """
    subject = subject.strip()
    predicate = predicate.strip()
    object_ = object_.strip()

    # --- Semantic deduplication (only when a real provider is available) ---
    if provider is not None:
        duplicate = _find_semantic_duplicate(
            index, provider, subject, predicate, object_, dedup_threshold
        )
        if duplicate is not None:
            return {
                "id": duplicate["id"],
                "subject": duplicate["subject"],
                "predicate": duplicate["predicate"],
                "object": duplicate["object"],
                "note": duplicate.get("note"),
                "written_to": str(relations_file),
                "deduplicated": True,
                "duplicate_of": duplicate["id"],
            }

    relation_id = _relation_id(subject, predicate, object_)

    # 1. Upsert into SQLite
    index.insert_relation(
        relation_id=relation_id,
        subject=subject,
        predicate=predicate,
        object_=object_,
        note=note,
        source_file=str(relations_file),
        confidence=confidence,
    )

    # 2. Append to RELATIONS.md (idempotent-ish: check if line already present)
    line = _format_relation_line(subject, predicate, object_, note)
    existing = relations_file.read_text(encoding="utf-8") if relations_file.exists() else ""

    # Only append if this exact triple isn't already in the file
    marker = f"[{subject}] --{predicate}--> [{object_}]"
    if marker not in existing:
        new_content = existing.rstrip() + "\n" + line + "\n"
        _atomic_write(relations_file, new_content)

    return {
        "id": relation_id,
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "note": note,
        "written_to": str(relations_file),
        "deduplicated": False,
    }


def get_relations(index: MemoryIndex, entity: Optional[str] = None) -> list[dict]:
    """
    Return relations as a list of dicts.
    If *entity* is given, filter to relations involving that entity.
    """
    if entity:
        rows = index.get_relations_for_entity(entity)
    else:
        rows = index.get_all_relations()

    return [
        {
            "id": row["id"],
            "subject": row["subject"],
            "predicate": row["predicate"],
            "object": row["object"],
            "note": row["note"],
            "source_file": row["source_file"],
            "created_at": datetime.fromtimestamp(row["created_at"]).isoformat(),
            "confidence": row["confidence"],
        }
        for row in rows
    ]


def format_relations_for_context(relations: list[dict]) -> str:
    """Format a list of relation dicts as a compact Markdown block for injection."""
    if not relations:
        return ""
    lines = ["## Relations\n"]
    for r in relations:
        line = f"- [{r['subject']}] --{r['predicate']}--> [{r['object']}]"
        if r.get("note"):
            line += f" — {r['note']}"
        lines.append(line)
    return "\n".join(lines)


def parse_relations_from_file(relations_file: Path) -> list[dict]:
    """
    Parse all valid relation lines from RELATIONS.md.

    Each returned dict has keys: subject, predicate, object, note (or None).
    Lines that do not match the expected format are silently skipped.
    """
    if not relations_file.exists():
        return []

    results: list[dict] = []
    for line in relations_file.read_text(encoding="utf-8").splitlines():
        m = RELATION_LINE_RE.match(line)
        if m:
            subject, predicate, object_, _date, note = m.groups()
            results.append(
                {
                    "subject": subject.strip(),
                    "predicate": predicate.strip(),
                    "object": object_.strip(),
                    "note": note.strip() if note else None,
                }
            )
    return results


def sync_relations_from_file(relations_file: Path, index: MemoryIndex) -> dict:
    """
    Reconcile the SQLite ``relations`` table with RELATIONS.md (Option B).

    RELATIONS.md is the source of truth.  This function:
      - Parses every valid relation line from the file.
      - Upserts any relation not yet in SQLite.
      - Deletes any SQLite relation whose triple no longer appears in the file.

    Returns a summary dict: {upserted, deleted, total_in_file}.
    """
    file_relations = parse_relations_from_file(relations_file)

    # Build a set of IDs that should exist (derived from file content)
    file_ids: set[str] = set()
    for r in file_relations:
        rid = _relation_id(r["subject"], r["predicate"], r["object"])
        file_ids.add(rid)
        index.insert_relation(
            relation_id=rid,
            subject=r["subject"],
            predicate=r["predicate"],
            object_=r["object"],
            note=r["note"],
            source_file=str(relations_file),
            confidence=1.0,
        )

    # Remove any SQLite rows whose triple is no longer in the file
    existing_rows = index.get_all_relations()
    deleted = 0
    for row in existing_rows:
        if row["id"] not in file_ids:
            index.delete_relation(row["id"])
            deleted += 1

    upserted = len(file_ids)
    return {"upserted": upserted, "deleted": deleted, "total_in_file": len(file_relations)}
