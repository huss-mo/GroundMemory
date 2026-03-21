"""
Base class and shared utilities for all OpenMemory tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MemoryToolError(Exception):
    """Raised when a tool call fails with a user-visible error."""


def ok(data: Any) -> dict:
    """Wrap a successful tool result."""
    return {"status": "ok", **data} if isinstance(data, dict) else {"status": "ok", "result": data}


def err(message: str) -> dict:
    """Wrap a tool error result."""
    return {"status": "error", "message": message}


def is_immutable(file: str) -> bool:
    """
    Return True if *file* refers to an append-only immutable memory tier.

    MEMORY.md and daily/*.md are write-once history files — their existing
    content must never be mutated or deleted by the agent.  Only USER.md,
    AGENTS.md, and any other files are editable.
    """
    p = Path(file)
    # Normalise: strip leading separators so both "daily/x.md" and
    # "/abs/path/.../daily/x.md" are caught.
    parts = p.parts
    name = p.name
    return name == "MEMORY.md" or (len(parts) >= 2 and parts[-2] == "daily")


_IMMUTABLE_MSG = (
    "'{file}' is an append-only memory file and cannot be edited or deleted. "
    "Use memory_write to append new information instead."
)
