"""Tests for configurable mutable-vs-append-only memory tiers (mutable_tiers)."""
from __future__ import annotations

from groundmemory.config import (
    DEFAULT_MUTABLE_TIERS,
    CustomFileConfig,
    EmbeddingConfig,
    groundmemoryConfig,
)
from groundmemory.session import MemorySession

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path, mutable_tiers=None, custom_files=None):
    return groundmemoryConfig(
        root_dir=tmp_path,
        embedding=EmbeddingConfig(provider="none"),
        mutable_tiers=list(DEFAULT_MUTABLE_TIERS) if mutable_tiers is None else mutable_tiers,
        custom_files=custom_files or [],
        expose_memory_list=True,
    )


def _session(tmp_path, mutable_tiers=None, custom_files=None):
    cfg = _cfg(tmp_path, mutable_tiers, custom_files)
    return MemorySession.create("test", config=cfg)


def _immutable_error(r: dict) -> bool:
    msg = r.get("message", "").lower()
    return r["status"] == "error" and ("append-only" in msg or "immutable" in msg)


# ---------------------------------------------------------------------------
# mutable_tiers / CustomFileConfig defaults
# ---------------------------------------------------------------------------

class TestMutableTiersDefaults:
    def test_defaults_match_the_documented_default_split(self):
        assert set(DEFAULT_MUTABLE_TIERS) == {"USER.md", "AGENTS.md", "RELATIONS.md"}
        assert "MEMORY.md" not in DEFAULT_MUTABLE_TIERS
        assert "daily" not in DEFAULT_MUTABLE_TIERS

    def test_config_default_uses_default_mutable_tiers(self, tmp_path):
        cfg = groundmemoryConfig(root_dir=tmp_path, embedding=EmbeddingConfig(provider="none"))
        assert cfg.mutable_tiers == list(DEFAULT_MUTABLE_TIERS)


class TestCustomFileConfigMutableField:
    def test_mutable_defaults_true(self):
        assert CustomFileConfig(name="NOTES.md").mutable is True

    def test_mutable_can_be_set_false(self):
        assert CustomFileConfig(name="NOTES.md", mutable=False).mutable is False


# ---------------------------------------------------------------------------
# Default config: behavior unchanged from the old hardcoded rule
# ---------------------------------------------------------------------------

class TestDefaultConfigBehaviorUnchanged:
    def test_memory_md_replace_rejected_by_default(self, tmp_path):
        s = _session(tmp_path)
        try:
            s.execute_tool("memory_write", file="MEMORY.md", content="Alice loves Python.")
            r = s.execute_tool(
                "memory_write", file="MEMORY.md", search="Alice", content="Bob"
            )
            assert _immutable_error(r)
        finally:
            s.close()

    def test_daily_keyword_replace_rejected_by_default(self, tmp_path):
        s = _session(tmp_path)
        try:
            s.execute_tool("memory_write", file="daily", content="Daily note one.")
            r = s.execute_tool(
                "memory_write", file="daily", search="Daily note one.", content="Edited."
            )
            assert _immutable_error(r)
        finally:
            s.close()

    def test_user_md_replace_allowed_by_default(self, tmp_path):
        s = _session(tmp_path)
        try:
            s.workspace.user_file.write_text("Name: Alice.\n", encoding="utf-8")
            r = s.execute_tool(
                "memory_write", file="USER.md", search="Alice", content="Bob"
            )
            assert r["status"] == "ok"
        finally:
            s.close()


# ---------------------------------------------------------------------------
# Configured overrides: standard tiers
# ---------------------------------------------------------------------------

class TestConfigurableStandardTiers:
    def test_memory_md_in_mutable_tiers_allows_replace(self, tmp_path):
        s = _session(tmp_path, mutable_tiers=["MEMORY.md"])
        try:
            s.execute_tool("memory_write", file="MEMORY.md", content="Alice loves Python.")
            r = s.execute_tool(
                "memory_write", file="MEMORY.md", search="Alice", content="Bob"
            )
            assert r["status"] == "ok"
        finally:
            s.close()

    def test_memory_md_in_mutable_tiers_allows_delete(self, tmp_path):
        s = _session(tmp_path, mutable_tiers=["MEMORY.md"])
        try:
            s.execute_tool("memory_write", file="MEMORY.md", content="Line to delete.")
            r = s.execute_tool(
                "memory_write", file="MEMORY.md", start_line=1, end_line=1, content=""
            )
            assert r["status"] == "ok"
        finally:
            s.close()

    def test_user_md_excluded_from_mutable_tiers_blocks_replace(self, tmp_path):
        s = _session(tmp_path, mutable_tiers=["AGENTS.md", "RELATIONS.md"])
        try:
            s.workspace.user_file.write_text("Name: Alice.\n", encoding="utf-8")
            r = s.execute_tool(
                "memory_write", file="USER.md", search="Alice", content="Bob"
            )
            assert _immutable_error(r)
        finally:
            s.close()

    def test_agents_md_excluded_from_mutable_tiers_blocks_delete(self, tmp_path):
        s = _session(tmp_path, mutable_tiers=["USER.md", "RELATIONS.md"])
        try:
            s.workspace.agents_file.write_text("Rule one.\nRule two.\n", encoding="utf-8")
            r = s.execute_tool(
                "memory_write", file="AGENTS.md", start_line=1, end_line=1, content=""
            )
            assert _immutable_error(r)
        finally:
            s.close()

    def test_relations_md_excluded_from_mutable_tiers_blocks_delete(self, tmp_path):
        s = _session(tmp_path, mutable_tiers=["USER.md", "AGENTS.md"])
        try:
            s.workspace.relations_file.write_text(
                "- [Alice] --leads--> [Team] (2026-01-01)\n", encoding="utf-8"
            )
            r = s.execute_tool(
                "memory_write", file="RELATIONS.md", start_line=1, end_line=1, content=""
            )
            assert _immutable_error(r)
        finally:
            s.close()

    def test_daily_in_mutable_tiers_allows_replace_on_todays_log(self, tmp_path):
        s = _session(tmp_path, mutable_tiers=["daily"])
        try:
            s.execute_tool("memory_write", file="daily", content="Original daily note.")
            r = s.execute_tool(
                "memory_write",
                file="daily",
                search="Original daily note.",
                content="Edited daily note.",
            )
            assert r["status"] == "ok"

            content = s.workspace.daily_file().read_text(encoding="utf-8")
            assert "Edited daily note." in content
            assert "Original daily note." not in content
        finally:
            s.close()

    def test_daily_in_mutable_tiers_still_blocks_specific_dated_file(self, tmp_path):
        """A specific dated daily log is always immutable, even with 'daily' in mutable_tiers."""
        s = _session(tmp_path, mutable_tiers=["daily"])
        try:
            s.execute_tool("memory_write", file="daily", content="Daily note.")
            listing = s.execute_tool("memory_list", target="daily")
            daily_name = listing["daily_files"][0]
            r = s.execute_tool(
                "memory_write",
                file=f"daily/{daily_name}",
                start_line=1,
                end_line=1,
                content="",
            )
            assert _immutable_error(r)
        finally:
            s.close()


# ---------------------------------------------------------------------------
# Configured overrides: custom files
# ---------------------------------------------------------------------------

class TestConfigurableCustomFiles:
    def test_immutable_custom_file_blocks_replace(self, tmp_path):
        s = _session(
            tmp_path,
            custom_files=[CustomFileConfig(name="DECISIONS.md", mutable=False)],
        )
        try:
            s.execute_tool("memory_write", file="DECISIONS.md", content="We chose Postgres.")
            r = s.execute_tool(
                "memory_write", file="DECISIONS.md", search="Postgres", content="MySQL"
            )
            assert _immutable_error(r)
        finally:
            s.close()

    def test_immutable_custom_file_still_allows_append(self, tmp_path):
        s = _session(
            tmp_path,
            custom_files=[CustomFileConfig(name="DECISIONS.md", mutable=False)],
        )
        try:
            r = s.execute_tool("memory_write", file="DECISIONS.md", content="We chose Postgres.")
            assert r["status"] == "ok"
        finally:
            s.close()

    def test_mutable_custom_file_allows_replace(self, tmp_path):
        s = _session(
            tmp_path,
            custom_files=[CustomFileConfig(name="NOTES.md", mutable=True)],
        )
        try:
            s.execute_tool("memory_write", file="NOTES.md", content="Draft idea.")
            r = s.execute_tool(
                "memory_write", file="NOTES.md", search="Draft", content="Final"
            )
            assert r["status"] == "ok"
        finally:
            s.close()
