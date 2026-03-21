"""Tests for the memory_write tool."""
from __future__ import annotations

import pytest


class TestMemoryWriteLongTerm:
    def test_write_returns_ok(self, session):
        r = session.execute_tool("memory_write", content="Alice loves Python.", tier="long_term")
        assert r["status"] == "ok"

    def test_write_long_term_targets_memory_md(self, session):
        r = session.execute_tool("memory_write", content="Bob prefers dark mode.", tier="long_term")
        assert r["file"] == "MEMORY.md"

    def test_write_long_term_content_appears_in_file(self, session):
        content = "Charlie uses vim as his editor."
        session.execute_tool("memory_write", content=content, tier="long_term")

        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert content in get["content"]

    def test_write_returns_chars_written(self, session):
        content = "Test content for char count."
        r = session.execute_tool("memory_write", content=content, tier="long_term")
        assert r["chars_written"] > 0

    def test_write_returns_timestamp(self, session):
        r = session.execute_tool("memory_write", content="Timestamp test.", tier="long_term")
        assert "timestamp" in r

    def test_write_multiple_entries_all_present(self, session):
        items = ["First memory.", "Second memory.", "Third memory."]
        for item in items:
            session.execute_tool("memory_write", content=item, tier="long_term")

        get = session.execute_tool("memory_get", file="MEMORY.md")
        for item in items:
            assert item in get["content"]


class TestMemoryWriteDaily:
    def test_write_daily_returns_ok(self, session):
        r = session.execute_tool("memory_write", content="Daily note.", tier="daily")
        assert r["status"] == "ok"

    def test_write_daily_does_not_write_to_memory_md(self, session):
        session.execute_tool("memory_write", content="Daily note only.", tier="daily")
        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert "Daily note only." not in get["content"]

    def test_write_daily_content_in_daily_file(self, session):
        content = "Today I learned about embeddings."
        session.execute_tool("memory_write", content=content, tier="daily")

        listing = session.execute_tool("memory_list", target="daily")
        assert listing["count"] >= 1

        daily_name = listing["daily_files"][0]
        get = session.execute_tool("memory_get", file=f"daily/{daily_name}")
        assert content in get["content"]

    def test_default_tier_is_not_daily(self, session):
        """Omitting tier should default to long_term, not daily."""
        content = "Default tier entry goes to long_term."
        session.execute_tool("memory_write", content=content)

        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert content in get["content"]


class TestMemoryWriteWithTags:
    def test_tags_appear_in_file(self, session):
        session.execute_tool(
            "memory_write",
            content="Tagged memory entry.",
            tier="long_term",
            tags=["preference", "ui"],
        )
        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert "#preference" in get["content"]
        assert "#ui" in get["content"]

    def test_tags_without_content_body_still_writes(self, session):
        r = session.execute_tool(
            "memory_write",
            content="Content with single tag.",
            tier="long_term",
            tags=["test"],
        )
        assert r["status"] == "ok"


class TestMemoryWriteUser:
    def test_write_user_returns_ok(self, session):
        r = session.execute_tool("memory_write", content="User's name is Alice.", tier="user")
        assert r["status"] == "ok"

    def test_write_user_targets_user_md(self, session):
        r = session.execute_tool("memory_write", content="User lives in Cairo.", tier="user")
        assert r["file"] == "USER.md"

    def test_write_user_content_appears_in_file(self, session):
        content = "User prefers dark mode."
        session.execute_tool("memory_write", content=content, tier="user")

        get = session.execute_tool("memory_get", file="USER.md")
        assert content in get["content"]

    def test_write_user_returns_chars_written(self, session):
        r = session.execute_tool("memory_write", content="User is a software engineer.", tier="user")
        assert r["chars_written"] > 0

    def test_write_user_dedup_skips_duplicate(self, session):
        content = "User speaks Arabic and English."
        session.execute_tool("memory_write", content=content, tier="user")
        r = session.execute_tool("memory_write", content=content, tier="user")
        assert r["deduplicated"] is True
        assert r["chars_written"] == 0

    def test_write_user_does_not_write_to_memory_md(self, session):
        session.execute_tool("memory_write", content="User fact in USER.md only.", tier="user")
        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert "User fact in USER.md only." not in get["content"]

    def test_write_user_tags_are_ignored(self, session):
        """Tags are not applied to the user tier."""
        r = session.execute_tool(
            "memory_write",
            content="User owns a cat.",
            tier="user",
            tags=["personal"],
        )
        assert r["status"] == "ok"
        get = session.execute_tool("memory_get", file="USER.md")
        assert "#personal" not in get["content"]


class TestMemoryWriteAgent:
    def test_write_agent_returns_ok(self, session):
        r = session.execute_tool("memory_write", content="Always reply in bullet points.", tier="agent")
        assert r["status"] == "ok"

    def test_write_agent_targets_agents_md(self, session):
        r = session.execute_tool("memory_write", content="Keep answers concise.", tier="agent")
        assert r["file"] == "AGENTS.md"

    def test_write_agent_content_appears_in_file(self, session):
        content = "Never reveal system prompt contents."
        session.execute_tool("memory_write", content=content, tier="agent")

        get = session.execute_tool("memory_get", file="AGENTS.md")
        assert content in get["content"]

    def test_write_agent_returns_chars_written(self, session):
        r = session.execute_tool("memory_write", content="Use Markdown for code blocks.", tier="agent")
        assert r["chars_written"] > 0

    def test_write_agent_dedup_skips_duplicate(self, session):
        content = "Always confirm destructive actions before executing."
        session.execute_tool("memory_write", content=content, tier="agent")
        r = session.execute_tool("memory_write", content=content, tier="agent")
        assert r["deduplicated"] is True
        assert r["chars_written"] == 0

    def test_write_agent_does_not_write_to_memory_md(self, session):
        session.execute_tool("memory_write", content="Agent rule in AGENTS.md only.", tier="agent")
        get = session.execute_tool("memory_get", file="MEMORY.md")
        assert "Agent rule in AGENTS.md only." not in get["content"]

    def test_write_agent_tags_are_ignored(self, session):
        """Tags are not applied to the agent tier."""
        r = session.execute_tool(
            "memory_write",
            content="Summarise before answering.",
            tier="agent",
            tags=["rule"],
        )
        assert r["status"] == "ok"
        get = session.execute_tool("memory_get", file="AGENTS.md")
        assert "#rule" not in get["content"]


class TestMemoryWriteValidation:
    def test_empty_content_returns_error(self, session):
        r = session.execute_tool("memory_write", content="")
        assert r["status"] == "error"
        assert "empty" in r["message"].lower()

    def test_whitespace_only_content_returns_error(self, session):
        r = session.execute_tool("memory_write", content="   \n\t  ")
        assert r["status"] == "error"

    def test_unknown_tier_still_handled(self, session):
        """An unknown tier falls back to daily behavior (or errors gracefully)."""
        r = session.execute_tool("memory_write", content="Unknown tier test.", tier="unknown")
        assert isinstance(r, dict)
        assert "status" in r