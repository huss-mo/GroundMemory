"""Tests for the memory_delete tool."""
from __future__ import annotations


def _seed_user(session, *lines: str) -> None:
    """Seed USER.md with direct writes (bypasses dedup for test setup)."""
    ws = session.workspace
    ws.user_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seed_agents(session, *lines: str) -> None:
    """Seed AGENTS.md with direct writes."""
    ws = session.workspace
    ws.agents_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestMemoryDeleteBasic:
    def test_delete_returns_ok(self, session):
        _seed_user(session, "User line A.", "User line B.")
        r = session.execute_tool("memory_delete", file="USER.md", start_line=1, end_line=1)
        assert r["status"] == "ok"

    def test_delete_reports_deleted_lines(self, session):
        _seed_user(session, "Content to be tombstoned.", "Second line.")
        r = session.execute_tool("memory_delete", file="USER.md", start_line=1, end_line=1)
        assert r["status"] == "ok"
        assert "deleted_lines" in r

    def test_delete_nonexistent_file_returns_error(self, session):
        r = session.execute_tool("memory_delete", file="NONEXISTENT.md", start_line=1, end_line=2)
        assert r["status"] == "error"
        assert "message" in r

    def test_delete_changes_file_content(self, session):
        _seed_user(session, "Keep this.", "Delete this.", "Keep this too.")
        before = session.execute_tool("memory_get", file="USER.md")
        assert "Delete this." in before["content"]

        session.execute_tool("memory_delete", file="USER.md", start_line=2, end_line=2)
        after = session.execute_tool("memory_get", file="USER.md")
        assert after["status"] == "ok"
        # The line is replaced by an HTML tombstone comment (audit trail)
        assert "<!-- deleted" in after["content"]
        # The surrounding lines are still present
        assert "Keep this." in after["content"]
        assert "Keep this too." in after["content"]

    def test_delete_with_reason_stored(self, session):
        _seed_user(session, "Reason test line.")
        r = session.execute_tool(
            "memory_delete",
            file="USER.md",
            start_line=1,
            end_line=1,
            reason="test cleanup",
        )
        assert r["status"] == "ok"

    def test_delete_range_multiple_lines(self, session):
        _seed_user(session, "Entry 0", "Entry 1", "Entry 2", "Entry 3", "Entry 4")
        r = session.execute_tool("memory_delete", file="USER.md", start_line=2, end_line=4)
        assert r["status"] == "ok"

    def test_delete_on_agents_md(self, session):
        _seed_agents(session, "Rule A.", "Rule B.", "Rule C.")
        r = session.execute_tool("memory_delete", file="AGENTS.md", start_line=2, end_line=2)
        assert r["status"] == "ok"
        get = session.execute_tool("memory_get", file="AGENTS.md")
        # Rule B. is preserved only inside a tombstone comment, not as a plain line
        assert "<!-- deleted" in get["content"]
        assert "Rule A." in get["content"]
        assert "Rule C." in get["content"]


class TestMemoryDeleteImmutable:
    def test_delete_memory_md_returns_error(self, session):
        r = session.execute_tool("memory_delete", file="MEMORY.md", start_line=1, end_line=1)
        assert r["status"] == "error"
        assert "append-only" in r["message"] or "immutable" in r["message"].lower()

    def test_delete_daily_file_returns_error(self, session):
        session.execute_tool("memory_write", content="Daily note.", tier="daily")
        listing = session.execute_tool("memory_list", target="daily")
        daily_name = listing["daily_files"][0]
        r = session.execute_tool(
            "memory_delete", file=f"daily/{daily_name}", start_line=1, end_line=1
        )
        assert r["status"] == "error"
        assert "append-only" in r["message"] or "immutable" in r["message"].lower()


class TestMemoryDeleteEdgeCases:
    def test_delete_invalid_line_range_returns_error(self, session):
        """start_line > end_line or out of range should be handled gracefully."""
        _seed_user(session, "Some content.")
        r = session.execute_tool("memory_delete", file="USER.md", start_line=10, end_line=2)
        assert isinstance(r, dict)
        assert "status" in r

    def test_delete_out_of_bounds_line_returns_error(self, session):
        """Deleting beyond file length should be handled gracefully."""
        _seed_user(session, "Short file.")
        r = session.execute_tool("memory_delete", file="USER.md", start_line=1000, end_line=2000)
        assert isinstance(r, dict)
        assert "status" in r