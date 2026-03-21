"""Tests for the memory_replace_text and memory_replace_lines tools."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Helpers — write directly to files to avoid dedup / tier restrictions
# ---------------------------------------------------------------------------

def _write_user(session, content: str) -> None:
    session.workspace.user_file.write_text(content, encoding="utf-8")


def _write_agents(session, content: str) -> None:
    session.workspace.agents_file.write_text(content, encoding="utf-8")


# ===========================================================================
# memory_replace_text
# ===========================================================================


class TestMemoryReplaceText:
    # --- happy-path (mutable files only) ---

    def test_replace_text_returns_ok(self, session):
        _write_user(session, "The sky is blue.\n")
        r = session.execute_tool(
            "memory_replace_text",
            file="USER.md",
            search="The sky is blue.",
            replacement="The sky is clear.",
        )
        assert r["status"] == "ok"

    def test_replace_text_content_updated_in_file(self, session):
        _write_user(session, "Alice likes cats.\n")
        session.execute_tool(
            "memory_replace_text",
            file="USER.md",
            search="Alice likes cats.",
            replacement="Alice likes dogs.",
        )
        get = session.execute_tool("memory_get", file="USER.md")
        assert "Alice likes dogs." in get["content"]
        assert "Alice likes cats." not in get["content"]

    def test_replace_text_returns_replaced_true(self, session):
        _write_user(session, "Bob uses Linux.\n")
        r = session.execute_tool(
            "memory_replace_text",
            file="USER.md",
            search="Bob uses Linux.",
            replacement="Bob uses macOS.",
        )
        assert r["replaced"] is True

    def test_replace_text_returns_chars_delta_positive(self, session):
        _write_user(session, "short text.\n")
        r = session.execute_tool(
            "memory_replace_text",
            file="USER.md",
            search="short text.",
            replacement="longer replacement text.",
        )
        assert r["chars_delta"] > 0

    def test_replace_text_returns_chars_delta_negative(self, session):
        _write_user(session, "This is a rather long sentence that will be replaced.\n")
        r = session.execute_tool(
            "memory_replace_text",
            file="USER.md",
            search="This is a rather long sentence that will be replaced.",
            replacement="Short.",
        )
        assert r["chars_delta"] < 0

    def test_replace_text_returns_chars_delta_zero_for_same_length(self, session):
        _write_user(session, "aaa\n")
        r = session.execute_tool(
            "memory_replace_text",
            file="USER.md",
            search="aaa",
            replacement="bbb",
        )
        assert r["chars_delta"] == 0

    def test_replace_text_only_replaces_first_occurrence(self, session):
        """Two identical occurrences -> only the first one is replaced."""
        _write_user(session, "MARKER line one\nMARKER line two\n")
        session.execute_tool(
            "memory_replace_text",
            file="USER.md",
            search="MARKER",
            replacement="REPLACED",
        )
        get = session.execute_tool("memory_get", file="USER.md")
        content = get["content"]
        assert content.count("REPLACED") == 1
        assert content.count("MARKER") == 1  # second occurrence untouched

    def test_replace_text_works_on_user_md(self, session):
        _write_user(session, "User age is 30.\n")
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
        _write_agents(session, "Always be concise.\n")
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
        _write_user(session, "line one\nline two\nline three\n")
        r = session.execute_tool(
            "memory_replace_text",
            file="USER.md",
            search="line one\nline two",
            replacement="REPLACED_BLOCK",
        )
        assert r["status"] == "ok"
        get = session.execute_tool("memory_get", file="USER.md")
        assert "REPLACED_BLOCK" in get["content"]
        assert "line one" not in get["content"]

    # --- error cases ---

    def test_replace_text_error_when_file_missing(self, session):
        r = session.execute_tool(
            "memory_replace_text",
            file="NONEXISTENT.md",
            search="anything",
            replacement="something",
        )
        assert r["status"] == "error"

    def test_replace_text_error_when_search_not_found(self, session):
        _write_user(session, "Some content here.\n")
        r = session.execute_tool(
            "memory_replace_text",
            file="USER.md",
            search="text that does not exist in file",
            replacement="irrelevant",
        )
        assert r["status"] == "error"

    def test_replace_text_error_when_search_empty(self, session):
        _write_user(session, "Content in file.\n")
        r = session.execute_tool(
            "memory_replace_text",
            file="USER.md",
            search="",
            replacement="something",
        )
        assert r["status"] == "error"

    def test_replace_text_error_when_search_whitespace_only(self, session):
        _write_user(session, "Content in file.\n")
        r = session.execute_tool(
            "memory_replace_text",
            file="USER.md",
            search="   ",
            replacement="something",
        )
        assert r["status"] == "error"


class TestMemoryReplaceTextImmutable:
    def test_replace_text_memory_md_returns_error(self, session):
        r = session.execute_tool(
            "memory_replace_text",
            file="MEMORY.md",
            search="anything",
            replacement="something",
        )
        assert r["status"] == "error"
        assert "append-only" in r["message"] or "immutable" in r["message"].lower()

    def test_replace_text_daily_file_returns_error(self, session):
        session.execute_tool("memory_write", content="Daily note.", tier="daily")
        listing = session.execute_tool("memory_list", target="daily")
        daily_name = listing["daily_files"][0]
        r = session.execute_tool(
            "memory_replace_text",
            file=f"daily/{daily_name}",
            search="Daily note.",
            replacement="edited",
        )
        assert r["status"] == "error"
        assert "append-only" in r["message"] or "immutable" in r["message"].lower()


# ===========================================================================
# memory_replace_lines
# ===========================================================================


class TestMemoryReplaceLines:
    # --- happy-path (mutable files only) ---

    def test_replace_lines_returns_ok(self, session):
        _write_user(session, "line 1\nline 2\nline 3\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=2,
            end_line=2,
            replacement="replaced line 2",
        )
        assert r["status"] == "ok"

    def test_replace_lines_content_updated(self, session):
        _write_user(session, "alpha\nbeta\ngamma\n")
        session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=2,
            end_line=2,
            replacement="BETA_REPLACED",
        )
        get = session.execute_tool("memory_get", file="USER.md")
        assert "BETA_REPLACED" in get["content"]
        assert "beta" not in get["content"]

    def test_replace_lines_multi_line_range(self, session):
        _write_user(session, "a\nb\nc\nd\ne\n")
        session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=2,
            end_line=4,
            replacement="MIDDLE",
        )
        get = session.execute_tool("memory_get", file="USER.md")
        content = get["content"]
        assert "MIDDLE" in content
        assert "b" not in content
        assert "c" not in content
        assert "d" not in content
        assert "a" in content
        assert "e" in content

    def test_replace_lines_first_line(self, session):
        _write_user(session, "first\nsecond\nthird\n")
        session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=1,
            end_line=1,
            replacement="FIRST_REPLACED",
        )
        get = session.execute_tool("memory_get", file="USER.md")
        assert "FIRST_REPLACED" in get["content"]
        assert "first" not in get["content"]

    def test_replace_lines_last_line(self, session):
        _write_user(session, "first\nsecond\nthird\n")
        session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=3,
            end_line=3,
            replacement="LAST_REPLACED",
        )
        get = session.execute_tool("memory_get", file="USER.md")
        assert "LAST_REPLACED" in get["content"]
        assert "third" not in get["content"]

    def test_replace_lines_all_lines(self, session):
        _write_user(session, "x\ny\nz\n")
        session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=1,
            end_line=3,
            replacement="EVERYTHING",
        )
        get = session.execute_tool("memory_get", file="USER.md")
        content = get["content"]
        assert "EVERYTHING" in content
        assert "x" not in content
        assert "y" not in content
        assert "z" not in content

    def test_replace_lines_returns_replaced_lines_key(self, session):
        _write_user(session, "one\ntwo\nthree\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=1,
            end_line=2,
            replacement="NEW",
        )
        assert "replaced_lines" in r
        assert r["replaced_lines"] == "1-2"

    def test_replace_lines_returns_chars_delta(self, session):
        _write_user(session, "short\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=1,
            end_line=1,
            replacement="a much longer replacement string here",
        )
        assert "chars_delta" in r
        assert r["chars_delta"] > 0

    def test_replace_lines_returns_replaced_preview(self, session):
        _write_user(session, "preview source line\nanother line\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=1,
            end_line=1,
            replacement="new content",
        )
        assert "replaced_preview" in r
        assert "preview source line" in r["replaced_preview"]

    def test_replace_lines_multiline_replacement(self, session):
        """Replacement text may itself span multiple lines."""
        _write_user(session, "a\nb\nc\n")
        session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=2,
            end_line=2,
            replacement="line X\nline Y\nline Z",
        )
        get = session.execute_tool("memory_get", file="USER.md")
        content = get["content"]
        assert "line X" in content
        assert "line Y" in content
        assert "line Z" in content

    def test_replace_lines_works_on_agents_md(self, session):
        _write_agents(session, "rule A\nrule B\n")
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
            file="NONEXISTENT.md",
            start_line=1,
            end_line=1,
            replacement="x",
        )
        assert r["status"] == "error"

    def test_replace_lines_error_start_line_zero(self, session):
        _write_user(session, "only one line\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=0,
            end_line=1,
            replacement="x",
        )
        assert r["status"] == "error"

    def test_replace_lines_error_start_line_beyond_file(self, session):
        _write_user(session, "line 1\nline 2\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=99,
            end_line=99,
            replacement="x",
        )
        assert r["status"] == "error"

    def test_replace_lines_error_end_line_before_start(self, session):
        _write_user(session, "a\nb\nc\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=3,
            end_line=1,
            replacement="x",
        )
        assert r["status"] == "error"

    def test_replace_lines_error_end_line_beyond_file(self, session):
        _write_user(session, "a\nb\n")
        r = session.execute_tool(
            "memory_replace_lines",
            file="USER.md",
            start_line=1,
            end_line=5,
            replacement="x",
        )
        assert r["status"] == "error"


class TestMemoryReplaceLinesImmutable:
    def test_replace_lines_memory_md_returns_error(self, session):
        r = session.execute_tool(
            "memory_replace_lines",
            file="MEMORY.md",
            start_line=1,
            end_line=1,
            replacement="edited",
        )
        assert r["status"] == "error"
        assert "append-only" in r["message"] or "immutable" in r["message"].lower()

    def test_replace_lines_daily_file_returns_error(self, session):
        session.execute_tool("memory_write", content="Daily note.", tier="daily")
        listing = session.execute_tool("memory_list", target="daily")
        daily_name = listing["daily_files"][0]
        r = session.execute_tool(
            "memory_replace_lines",
            file=f"daily/{daily_name}",
            start_line=1,
            end_line=1,
            replacement="edited",
        )
        assert r["status"] == "error"
        assert "append-only" in r["message"] or "immutable" in r["message"].lower()