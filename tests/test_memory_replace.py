"""Tests for the memory_replace_text and memory_replace_lines tools."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed(session, content: str, file: str = "MEMORY.md", tier: str = "long_term") -> None:
    """Write *content* to *file* via memory_write so the file exists."""
    if file == "USER.md":
        tier = "user"
    elif file == "AGENTS.md":
        tier = "agent"
    elif file.startswith("daily/"):
        tier = "daily"
    session.execute_tool("memory_write", content=content, tier=tier)


# ===========================================================================
# memory_replace_text
# ===========================================================================


class TestMemoryReplaceText:
    # --- happy-path ---

    def test_replace_text_returns_ok(self, session):
        _seed(session, "The sky is blue.")
        r = session.execute_tool(
            "memory_replace_text",
            file="MEMORY.md",
            search="The sky is blue.",
            replacement="The sky is clear.",
        )
        assert r["status"] == "ok"

    def test_replace_text_content_updated_in_file(self, session):
        _seed(session, "Alice likes cats.")
        session.execute_tool(
            "memory_replace_text",
            file="MEMORY.md",
            search="Alice likes cats.",
            replacement="Alice likes dogs.",
        )
        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert "Alice likes dogs." in get["content"]
        assert "Alice likes cats." not in get["content"]

    def test_replace_text_returns_replaced_true(self, session):
        _seed(session, "Bob uses Linux.")
        r = session.execute_tool(
            "memory_replace_text",
            file="MEMORY.md",
            search="Bob uses Linux.",
            replacement="Bob uses macOS.",
        )
        assert r["replaced"] is True

    def test_replace_text_returns_chars_delta_positive(self, session):
        _seed(session, "short text.")
        r = session.execute_tool(
            "memory_replace_text",
            file="MEMORY.md",
            search="short text.",
            replacement="longer replacement text.",
        )
        assert r["chars_delta"] > 0

    def test_replace_text_returns_chars_delta_negative(self, session):
        _seed(session, "This is a rather long sentence that will be replaced.")
        r = session.execute_tool(
            "memory_replace_text",
            file="MEMORY.md",
            search="This is a rather long sentence that will be replaced.",
            replacement="Short.",
        )
        assert r["chars_delta"] < 0

    def test_replace_text_returns_chars_delta_zero_for_same_length(self, session):
        _seed(session, "aaa")
        r = session.execute_tool(
            "memory_replace_text",
            file="MEMORY.md",
            search="aaa",
            replacement="bbb",
        )
        assert r["chars_delta"] == 0

    def test_replace_text_only_replaces_first_occurrence(self, session):
        """Two identical occurrences -> only the first one is replaced."""
        # Write both entries (dedup guard means we write different content per call)
        session.execute_tool("memory_write", content="repeated phrase - entry 1", tier="long_term")
        session.execute_tool("memory_write", content="repeated phrase - entry 2", tier="long_term")
        # Patch the file directly to insert a duplicate phrase
        ws = session.workspace
        original = ws.memory_file.read_text(encoding="utf-8")
        ws.memory_file.write_text(
            original.replace("entry 1", "MARKER").replace("entry 2", "MARKER"),
            encoding="utf-8",
        )

        session.execute_tool(
            "memory_replace_text",
            file="MEMORY.md",
            search="MARKER",
            replacement="REPLACED",
        )

        get = session.execute_tool("memory_get", file="MEMORY.md")
        content = get["content"]
        assert content.count("REPLACED") == 1
        assert content.count("MARKER") == 1  # second occurrence untouched

    def test_replace_text_works_on_user_md(self, session):
        _seed(session, "User age is 30.", file="USER.md")
        r = session.execute_tool(
            "memory_replace_text",
            file="USER.md",
            search="User age is 30.",
            replacement="User age is 31.",
        )
        assert r["status"] == "ok"
        get = session.execute_tool("memory_get", file="USER.md")
        assert "User age is 31." in get["content"]

    def test_replace_text_works_on_agents_md(self, session):
        _seed(session, "Always be concise.", file="AGENTS.md")
        r = session.execute_tool(
            "memory_replace_text",
            file="AGENTS.md",
            search="Always be concise.",
            replacement="Always be precise and concise.",
        )
        assert r["status"] == "ok"
        get = session.execute_tool("memory_get", file="AGENTS.md")
        assert "Always be precise and concise." in get["content"]

    def test_replace_text_multiline_search(self, session):
        """Search string may contain newlines."""
        ws = session.workspace
        ws.memory_file.write_text("line one\nline two\nline three\n")
        r = session.execute_tool(
            "memory_replace_text",
            file="MEMORY.md",
            search="line one\nline two",
            replacement="REPLACED_BLOCK",
        )
        assert r["status"] == "ok"
        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert "REPLACED_BLOCK" in get["content"]
        assert "line one" not in get["content"]

    # --- error cases ---

    def test_replace_text_error_when_file_missing(self, session):
        r = session.execute_tool(
            "memory_replace_text",
            file="daily/9999-12-31.md",
            search="anything",
            replacement="something",
        )
        assert r["status"] == "error"

    def test_replace_text_error_when_search_not_found(self, session):
        _seed(session, "Some content here.")
        r = session.execute_tool(
            "memory_replace_text",
            file="MEMORY.md",
            search="text that does not exist in file",
            replacement="irrelevant",
        )
        assert r["status"] == "error"
        assert "not found" in r["message"].lower() or "error" in r

    def test_replace_text_error_when_search_empty(self, session):
        _seed(session, "Content in file.")
        r = session.execute_tool(
            "memory_replace_text",
            file="MEMORY.md",
            search="",
            replacement="something",
        )
        assert r["status"] == "error"

    def test_replace_text_error_when_search_whitespace_only(self, session):
        _seed(session, "Content in file.")
        r = session.execute_tool(
            "memory_replace_text",
            file="MEMORY.md",
            search="   ",
            replacement="something",
        )
        assert r["status"] == "error"


# ===========================================================================
# memory_replace_lines
# ===========================================================================


class TestMemoryReplaceLines:
    # --- happy-path ---

    def test_replace_lines_returns_ok(self, session):
        ws = session.workspace
        ws.memory_file.write_text("line 1\nline 2\nline 3\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=2,
            end_line=2,
            replacement="replaced line 2",
        )
        assert r["status"] == "ok"

    def test_replace_lines_content_updated(self, session):
        ws = session.workspace
        ws.memory_file.write_text("alpha\nbeta\ngamma\n")
        session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=2,
            end_line=2,
            replacement="BETA_REPLACED",
        )
        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert "BETA_REPLACED" in get["content"]
        assert "beta" not in get["content"]

    def test_replace_lines_multi_line_range(self, session):
        ws = session.workspace
        ws.memory_file.write_text("a\nb\nc\nd\ne\n")
        session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=2,
            end_line=4,
            replacement="MIDDLE",
        )
        get = session.execute_tool("memory_get", file="MEMORY.md")
        content = get["content"]
        assert "MIDDLE" in content
        assert "b" not in content
        assert "c" not in content
        assert "d" not in content
        assert "a" in content
        assert "e" in content

    def test_replace_lines_first_line(self, session):
        ws = session.workspace
        ws.memory_file.write_text("first\nsecond\nthird\n")
        session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=1,
            end_line=1,
            replacement="FIRST_REPLACED",
        )
        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert "FIRST_REPLACED" in get["content"]
        assert "first" not in get["content"]

    def test_replace_lines_last_line(self, session):
        ws = session.workspace
        ws.memory_file.write_text("first\nsecond\nthird\n")
        session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=3,
            end_line=3,
            replacement="LAST_REPLACED",
        )
        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert "LAST_REPLACED" in get["content"]
        assert "third" not in get["content"]

    def test_replace_lines_all_lines(self, session):
        ws = session.workspace
        ws.memory_file.write_text("x\ny\nz\n")
        session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=1,
            end_line=3,
            replacement="EVERYTHING",
        )
        get = session.execute_tool("memory_get", file="MEMORY.md")
        content = get["content"]
        assert "EVERYTHING" in content
        assert "x" not in content
        assert "y" not in content
        assert "z" not in content

    def test_replace_lines_returns_replaced_lines_key(self, session):
        ws = session.workspace
        ws.memory_file.write_text("one\ntwo\nthree\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=1,
            end_line=2,
            replacement="NEW",
        )
        assert "replaced_lines" in r
        assert r["replaced_lines"] == "1-2"

    def test_replace_lines_returns_chars_delta(self, session):
        ws = session.workspace
        ws.memory_file.write_text("short\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=1,
            end_line=1,
            replacement="a much longer replacement string here",
        )
        assert "chars_delta" in r
        assert r["chars_delta"] > 0

    def test_replace_lines_returns_replaced_preview(self, session):
        ws = session.workspace
        ws.memory_file.write_text("preview source line\nanother line\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=1,
            end_line=1,
            replacement="new content",
        )
        assert "replaced_preview" in r
        assert "preview source line" in r["replaced_preview"]

    def test_replace_lines_multiline_replacement(self, session):
        """Replacement text may itself span multiple lines."""
        ws = session.workspace
        ws.memory_file.write_text("a\nb\nc\n")
        session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=2,
            end_line=2,
            replacement="line X\nline Y\nline Z",
        )
        get = session.execute_tool("memory_get", file="MEMORY.md")
        content = get["content"]
        assert "line X" in content
        assert "line Y" in content
        assert "line Z" in content

    def test_replace_lines_works_on_user_md(self, session):
        ws = session.workspace
        ws.user_file.write_text("user line 1\nuser line 2\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=1,
            end_line=1,
            replacement="REPLACED USER LINE",
        )
        assert r["status"] == "ok"
        get = session.execute_tool("memory_get", file="USER.md")
        assert "REPLACED USER LINE" in get["content"]

    def test_replace_lines_works_on_agents_md(self, session):
        ws = session.workspace
        ws.agents_file.write_text("rule A\nrule B\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="AGENTS.md",
            start_line=2,
            end_line=2,
            replacement="RULE B UPDATED",
        )
        assert r["status"] == "ok"
        get = session.execute_tool("memory_get", file="AGENTS.md")
        assert "RULE B UPDATED" in get["content"]

    # --- error cases ---

    def test_replace_lines_error_when_file_missing(self, session):
        r = session.execute_tool(
            "memory_replace_lines",
            file="daily/9999-12-31.md",
            start_line=1,
            end_line=1,
            replacement="x",
        )
        assert r["status"] == "error"

    def test_replace_lines_error_start_line_zero(self, session):
        ws = session.workspace
        ws.memory_file.write_text("only one line\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=0,
            end_line=1,
            replacement="x",
        )
        assert r["status"] == "error"

    def test_replace_lines_error_start_line_beyond_file(self, session):
        ws = session.workspace
        ws.memory_file.write_text("line 1\nline 2\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=99,
            end_line=99,
            replacement="x",
        )
        assert r["status"] == "error"

    def test_replace_lines_error_end_line_before_start(self, session):
        ws = session.workspace
        ws.memory_file.write_text("a\nb\nc\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=3,
            end_line=1,
            replacement="x",
        )
        assert r["status"] == "error"

    def test_replace_lines_error_end_line_beyond_file(self, session):
        ws = session.workspace
        ws.memory_file.write_text("a\nb\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=1,
            end_line=5,
            replacement="x",
        )
        assert r["status"] == "error"