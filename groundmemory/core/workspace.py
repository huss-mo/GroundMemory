"""
Workspace management - directory layout, initialization, and file path resolution.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

# Default content for workspace files created on first init
_DEFAULT_MEMORY_MD = """\
# Memory

Curated long-term knowledge: preferences, key decisions, persistent facts.
The agent writes here when something important should be remembered across sessions.
"""

_DEFAULT_USER_MD = """\
# User Profile

Add stable facts about the user here - name, role, working style, preferences.
"""

_DEFAULT_AGENTS_MD = """\
# Agent Instructions

## Memory Tools

You have two tool modes depending on how this server is configured.

**Normal mode** - four separate tools:
- `memory_bootstrap` - call once at session start; loads all memory context
- `memory_read` - search or read a file
- `memory_write` - append, replace, or delete
- `memory_relate` - record entity relationships

**Dispatcher mode** - one tool, `memory_tool(action, args)`:
- `action="bootstrap"` - same as memory_bootstrap
- `action="describe", args={"action":"<name>"}` - get full schema for an action. Use once before invoking each tool to understand its args. Overviews below are not comprehensive.
- `action="read"`, `action="write"`, `action="relate"`, `action="list"` - same as the individual tools

---

## Reading memory overview

**Search** - use when you need to find something:
```
memory_read(query="Alice's current employer")
memory_read(query="auth service architecture", file="daily")
```

**Get a file** - use when you need the full content or a specific range:
```
memory_read(file="USER.md")
memory_read(file="RELATIONS.md", start_line=1, end_line=20)
```

Always search before answering questions that may have been discussed before.

---

## Writing memory overview

### Where to write

| File | Content |
|------|---------|
| `MEMORY.md` | Curated facts, decisions, project knowledge (append-only) |
| `USER.md` | Stable facts about the user: name, role, location, preferences |
| `AGENTS.md` | Behavioural rules for future sessions |
| `daily` | Session notes, task progress, transient context (append-only) |
| `RELATIONS.md` | Entity relationships - edit via memory_write or memory_relate |

### Append (add new content)
```
memory_write(file="MEMORY.md", content="Prefers TypeScript over JavaScript.")
memory_write(file="daily", content="Started refactoring the auth module.")
```
Before appending to MEMORY.md, USER.md, or AGENTS.md, search first to avoid duplicates.

### Replace (update existing content)
```
# By exact text match (first occurrence):
memory_write(file="USER.md", search="Works at Acme Corp.", content="Works at New Corp.")

# By line range (read the file first to find the lines):
memory_write(file="USER.md", start_line=3, end_line=3, content="Works at New Corp.")
```

### Delete lines
```
memory_write(file="USER.md", start_line=5, end_line=7, content="")
```

MEMORY.md and daily logs are append-only; replace and delete are not allowed on them.

---

## Relationships overview

Use `memory_relate` for directional facts between entities (snake_case predicates):
```
memory_relate(subject="Alice", predicate="works_at", object="Acme Corp")
memory_relate(subject="Bob", predicate="manages", object="Alice")
```

When a relation is no longer valid (job change, location change), use `supersedes=True`
to replace the old value instead of accumulating duplicates:
```
memory_relate(subject="Alice", predicate="works_at", object="New Corp", supersedes=True)
```
Do NOT use `supersedes=True` when multiple values are valid simultaneously (e.g. `knows`, `attended`).

### RELATIONS.md format

Every non-blank, non-comment line must follow this exact format:
```
- [Subject] --predicate--> [Object] (YYYY-MM-DD) - "optional note"
```
Lines that do not match will be rejected by the write tools.

---

## Default agent Behavior

- Be warm and direct. Do not override user-defined persona with excessive friendliness.
- Act without narrating: do not announce tool calls before making them.
- After writes (append/replace/delete), give a brief confirmation only if the user asked for the action.
"""

_DEFAULT_RELATIONS_MD = """\
# Relations

Named relationships between entities (people, teams, systems, concepts).
Format: [Subject] --predicate--> [Object] (date) - note

"""


class Workspace:
    """
    Represents a single groundmemory workspace on disk.

    Directory layout:
        <workspace_path>/
        ├── MEMORY.md           long-term curated memory
        ├── USER.md             stable user profile
        ├── AGENTS.md           agent operating instructions
        ├── RELATIONS.md        entity relation graph (human-readable mirror)
        ├── daily/
        │   └── YYYY-MM-DD.md   append-only daily logs
        └── .index/
            └── memory.db       SQLite index (vector + FTS + relations)
    """

    def __init__(self, workspace_path: Path) -> None:
        self.workspace_path = Path(workspace_path)
        self.path = self.workspace_path  # alias for backward-compat
        self._ensure_layout()

    # ------------------------------------------------------------------
    # Directory paths
    # ------------------------------------------------------------------

    @property
    def daily_dir(self) -> Path:
        return self.path / "daily"

    @property
    def index_dir(self) -> Path:
        return self.path / ".index"

    @property
    def db_path(self) -> Path:
        return self.index_dir / "memory.db"

    # ------------------------------------------------------------------
    # File paths
    # ------------------------------------------------------------------

    @property
    def memory_file(self) -> Path:
        return self.path / "MEMORY.md"

    @property
    def user_file(self) -> Path:
        return self.path / "USER.md"

    @property
    def agents_file(self) -> Path:
        return self.path / "AGENTS.md"

    @property
    def relations_file(self) -> Path:
        return self.path / "RELATIONS.md"

    def daily_file(self, day: date | None = None) -> Path:
        """Return the path for the daily log of *day* (defaults to today)."""
        d = day or date.today()
        return self.daily_dir / f"{d.isoformat()}.md"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_layout(self) -> None:
        """Create directories and seed default files if they don't exist."""
        self.path.mkdir(parents=True, exist_ok=True)
        self.daily_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)

        self._seed(self.memory_file, _DEFAULT_MEMORY_MD)
        self._seed(self.user_file, _DEFAULT_USER_MD)
        self._seed(self.agents_file, _DEFAULT_AGENTS_MD)
        self._seed(self.relations_file, _DEFAULT_RELATIONS_MD)

    @staticmethod
    def _seed(path: Path, content: str) -> None:
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def all_memory_files(self) -> list[Path]:
        """Return all Markdown files that should be indexed."""
        files: list[Path] = []
        if self.memory_file.exists():
            files.append(self.memory_file)
        if self.user_file.exists():
            files.append(self.user_file)
        if self.agents_file.exists():
            files.append(self.agents_file)
        if self.relations_file.exists():
            files.append(self.relations_file)
        if self.daily_dir.exists():
            files.extend(sorted(self.daily_dir.glob("*.md")))
        return files

    def resolve_file(self, name: str) -> Path:
        """
        Resolve a user-supplied filename to an absolute path within the workspace.
        Accepts bare names like 'MEMORY.md' or 'daily/2026-03-20.md'.

        Raises ValueError if:
        - the supplied path is absolute (no legitimate tool call needs this), or
        - the resolved path escapes the workspace directory (path traversal).
        """
        p = Path(name)
        if p.is_absolute():
            raise ValueError(
                f"Access denied: absolute paths are not allowed (got '{name}')."
            )
        candidate = (self.path / p).resolve()
        workspace_root = self.path.resolve()
        if not candidate.is_relative_to(workspace_root):
            raise ValueError(
                f"Access denied: '{name}' resolves outside the workspace."
            )
        return candidate
