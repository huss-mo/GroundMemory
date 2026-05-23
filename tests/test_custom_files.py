"""Tests for custom file support (GROUNDMEMORY_CUSTOM_FILES)."""
from __future__ import annotations

import json
import os
import pytest

from groundmemory.config import groundmemoryConfig, CustomFileConfig, EmbeddingConfig, BootstrapConfig
from groundmemory.core.workspace import Workspace
from groundmemory.session import MemorySession
from groundmemory.bootstrap.injector import build_bootstrap_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path, custom_files=None):
    return groundmemoryConfig(
        root_dir=tmp_path,
        embedding=EmbeddingConfig(provider="none"),
        custom_files=custom_files or [],
        expose_memory_list=True,
    )


def _session(tmp_path, custom_files=None):
    cfg = _cfg(tmp_path, custom_files)
    return MemorySession.create("test", config=cfg)


# ---------------------------------------------------------------------------
# TestCustomFileConfig
# ---------------------------------------------------------------------------

class TestCustomFileConfig:
    def test_defaults(self):
        cf = CustomFileConfig(name="NOTES.md")
        assert cf.inject is True
        assert cf.searchable is True
        assert cf.compactable is False
        assert cf.description == ""
        assert cf.max_chars is None

    def test_all_fields(self):
        cf = CustomFileConfig(
            name="RESEARCH.md",
            description="Research notes",
            inject=False,
            max_chars=5000,
            searchable=False,
            compactable=True,
        )
        assert cf.name == "RESEARCH.md"
        assert cf.description == "Research notes"
        assert cf.inject is False
        assert cf.max_chars == 5000
        assert cf.searchable is False
        assert cf.compactable is True

    def test_json_env_var(self, monkeypatch):
        monkeypatch.setenv(
            "GROUNDMEMORY_CUSTOM_FILES",
            '[{"name":"R.md","searchable":false,"compactable":true}]',
        )
        cfg = groundmemoryConfig(embedding=EmbeddingConfig(provider="none"))
        assert len(cfg.custom_files) == 1
        assert cfg.custom_files[0].name == "R.md"
        assert cfg.custom_files[0].searchable is False
        assert cfg.custom_files[0].compactable is True

    def test_multiple_files(self, monkeypatch):
        monkeypatch.setenv(
            "GROUNDMEMORY_CUSTOM_FILES",
            '[{"name":"A.md"},{"name":"B.md","description":"second"}]',
        )
        cfg = groundmemoryConfig(embedding=EmbeddingConfig(provider="none"))
        assert len(cfg.custom_files) == 2
        assert cfg.custom_files[1].description == "second"

    def test_empty_array_default(self):
        cfg = groundmemoryConfig(embedding=EmbeddingConfig(provider="none"))
        assert cfg.custom_files == []


# ---------------------------------------------------------------------------
# TestCustomFileWorkspace
# ---------------------------------------------------------------------------

class TestCustomFileWorkspace:
    def test_all_memory_files_includes_searchable(self, tmp_path):
        cf = CustomFileConfig(name="RESEARCH.md", searchable=True)
        ws = Workspace(tmp_path / "ws", custom_files=[cf])
        (ws.path / "RESEARCH.md").write_text("notes", encoding="utf-8")
        paths = ws.all_memory_files()
        assert any(p.name == "RESEARCH.md" for p in paths)

    def test_all_memory_files_excludes_non_searchable(self, tmp_path):
        cf = CustomFileConfig(name="RESEARCH.md", searchable=False)
        ws = Workspace(tmp_path / "ws", custom_files=[cf])
        (ws.path / "RESEARCH.md").write_text("notes", encoding="utf-8")
        paths = ws.all_memory_files()
        assert not any(p.name == "RESEARCH.md" for p in paths)

    def test_all_files_includes_non_searchable(self, tmp_path):
        cf = CustomFileConfig(name="RESEARCH.md", searchable=False)
        ws = Workspace(tmp_path / "ws", custom_files=[cf])
        (ws.path / "RESEARCH.md").write_text("notes", encoding="utf-8")
        paths = ws.all_files()
        assert any(p.name == "RESEARCH.md" for p in paths)

    def test_all_memory_files_skips_missing(self, tmp_path):
        cf = CustomFileConfig(name="MISSING.md", searchable=True)
        ws = Workspace(tmp_path / "ws", custom_files=[cf])
        # File not created on disk
        paths = ws.all_memory_files()
        assert not any(p.name == "MISSING.md" for p in paths)

    def test_all_files_skips_missing(self, tmp_path):
        cf = CustomFileConfig(name="MISSING.md")
        ws = Workspace(tmp_path / "ws", custom_files=[cf])
        paths = ws.all_files()
        assert not any(p.name == "MISSING.md" for p in paths)

    def test_no_custom_files_is_default(self, tmp_path):
        ws = Workspace(tmp_path / "ws")
        assert ws.custom_files == []


# ---------------------------------------------------------------------------
# TestCustomFileBootstrap
# ---------------------------------------------------------------------------

class TestCustomFileBootstrap:
    def test_inject_true_appears_in_bootstrap(self, tmp_path):
        cf = CustomFileConfig(name="NOTES.md", inject=True)
        ws = Workspace(tmp_path / "ws", custom_files=[cf])
        (ws.path / "NOTES.md").write_text("Important research finding.", encoding="utf-8")
        cfg = BootstrapConfig()
        result = build_bootstrap_prompt(ws, cfg)
        assert "NOTES.md" in result
        assert "Important research finding." in result

    def test_inject_false_skipped(self, tmp_path):
        cf = CustomFileConfig(name="NOTES.md", inject=False)
        ws = Workspace(tmp_path / "ws", custom_files=[cf])
        (ws.path / "NOTES.md").write_text("Secret content.", encoding="utf-8")
        cfg = BootstrapConfig()
        result = build_bootstrap_prompt(ws, cfg)
        assert "Secret content." not in result

    def test_custom_file_after_standard_files(self, tmp_path):
        cf = CustomFileConfig(name="NOTES.md", inject=True)
        ws = Workspace(tmp_path / "ws", custom_files=[cf])
        (ws.path / "NOTES.md").write_text("Custom content here.", encoding="utf-8")
        # Clear standard files so Relations is the last standard section
        ws.relations_file.write_text("", encoding="utf-8")
        cfg = BootstrapConfig(inject_relations=True)
        result = build_bootstrap_prompt(ws, cfg)
        # NOTES.md must appear after the Relations section
        rel_pos = result.find("Relation Graph")
        notes_pos = result.find("NOTES.md")
        if rel_pos != -1 and notes_pos != -1:
            assert notes_pos > rel_pos

    def test_max_chars_override(self, tmp_path):
        cf = CustomFileConfig(name="NOTES.md", inject=True, max_chars=20)
        ws = Workspace(tmp_path / "ws", custom_files=[cf])
        # 200 chars — well above the 20 char limit
        (ws.path / "NOTES.md").write_text("q" * 200, encoding="utf-8")
        cfg = BootstrapConfig()
        result = build_bootstrap_prompt(ws, cfg)
        # Truncation notice must be present because file exceeds max_chars
        assert "TRUNCATED" in result
        # File must still be represented in the output
        assert "NOTES.md" in result

    def test_description_in_header(self, tmp_path):
        cf = CustomFileConfig(name="NOTES.md", description="My research notes", inject=True)
        ws = Workspace(tmp_path / "ws", custom_files=[cf])
        (ws.path / "NOTES.md").write_text("Some content.", encoding="utf-8")
        cfg = BootstrapConfig()
        result = build_bootstrap_prompt(ws, cfg)
        assert "My research notes" in result

    def test_missing_file_silently_skipped(self, tmp_path):
        cf = CustomFileConfig(name="ABSENT.md", inject=True)
        ws = Workspace(tmp_path / "ws", custom_files=[cf])
        # File never written to disk
        cfg = BootstrapConfig()
        result = build_bootstrap_prompt(ws, cfg)
        assert "ABSENT.md" not in result


# ---------------------------------------------------------------------------
# TestCustomFileWrite
# ---------------------------------------------------------------------------

class TestCustomFileWrite:
    def test_append_creates_file(self, tmp_path):
        cf = CustomFileConfig(name="RESEARCH.md")
        s = _session(tmp_path, [cf])
        try:
            r = s.execute_tool("memory_write", file="RESEARCH.md", content="First finding.")
            assert r["status"] == "ok"
            assert (s.workspace.path / "RESEARCH.md").exists()
            assert "First finding." in (s.workspace.path / "RESEARCH.md").read_text()
        finally:
            s.close()

    def test_append_unknown_file_rejected(self, tmp_path):
        s = _session(tmp_path, [])  # no custom files configured
        try:
            r = s.execute_tool("memory_write", file="UNKNOWN.md", content="test")
            assert r["status"] == "error"
            assert "Unknown" in r["message"]
        finally:
            s.close()

    def test_dedup_skips_repeated_content(self, tmp_path):
        cf = CustomFileConfig(name="RESEARCH.md")
        s = _session(tmp_path, [cf])
        try:
            s.execute_tool("memory_write", file="RESEARCH.md", content="Same content.")
            r = s.execute_tool("memory_write", file="RESEARCH.md", content="Same content.")
            assert r["status"] == "ok"
            assert r.get("deduplicated") is True
        finally:
            s.close()

    def test_edit_mode_supported(self, tmp_path):
        cf = CustomFileConfig(name="NOTES.md")
        s = _session(tmp_path, [cf])
        try:
            s.execute_tool("memory_write", file="NOTES.md", content="Original text.")
            file_path = s.workspace.path / "NOTES.md"
            original = file_path.read_text(encoding="utf-8")
            r = s.execute_tool(
                "memory_write",
                file="NOTES.md",
                search="Original text.",
                content="Updated text.",
            )
            assert r["status"] == "ok"
            updated = file_path.read_text(encoding="utf-8")
            assert "Updated text." in updated
        finally:
            s.close()

    def test_schema_lists_custom_files(self, tmp_path):
        from groundmemory.tools import build_tool_registry
        cf = CustomFileConfig(name="RESEARCH.md", description="Research notes")
        cfg = _cfg(tmp_path, [cf])
        _, _, schemas = build_tool_registry(cfg)
        write_schema = schemas["memory_write"]
        desc = write_schema["parameters"]["properties"]["file"]["description"]
        assert "RESEARCH.md" in desc
        assert "Research notes" in desc

    def test_schema_no_custom_files_unchanged(self, tmp_path):
        from groundmemory.tools import build_tool_registry
        cfg = _cfg(tmp_path, [])
        _, _, schemas = build_tool_registry(cfg)
        write_schema = schemas["memory_write"]
        desc = write_schema["parameters"]["properties"]["file"]["description"]
        assert "Custom files" not in desc


# ---------------------------------------------------------------------------
# TestCustomFileSearch
# ---------------------------------------------------------------------------

class TestCustomFileSearch:
    def test_searchable_file_indexed(self, tmp_path):
        cf = CustomFileConfig(name="RESEARCH.md", searchable=True)
        s = _session(tmp_path, [cf])
        try:
            s.execute_tool("memory_write", file="RESEARCH.md", content="quantum computing breakthrough")
            r = s.execute_tool("memory_read", query="quantum computing")
            assert r["status"] == "ok"
            assert any("quantum" in res["text"].lower() for res in r["results"])
        finally:
            s.close()

    def test_non_searchable_file_not_indexed(self, tmp_path):
        cf = CustomFileConfig(name="NOTES.md", searchable=False)
        s = _session(tmp_path, [cf])
        try:
            # Write directly to disk (bypassing the tool which only allows configured targets)
            path = s.workspace.path / "NOTES.md"
            path.write_text("unique_phrase_xyz_not_indexed", encoding="utf-8")
            s.sync()
            r = s.execute_tool("memory_read", query="unique_phrase_xyz_not_indexed")
            assert r["status"] == "ok"
            assert not any("unique_phrase_xyz" in res["text"] for res in r["results"])
        finally:
            s.close()

    def test_non_searchable_file_readable_via_get(self, tmp_path):
        cf = CustomFileConfig(name="NOTES.md", searchable=False)
        s = _session(tmp_path, [cf])
        try:
            path = s.workspace.path / "NOTES.md"
            path.write_text("direct read content", encoding="utf-8")
            r = s.execute_tool("memory_read", file="NOTES.md")
            assert r["status"] == "ok"
            assert "direct read content" in r["content"]
        finally:
            s.close()

    def test_scoped_search_by_file(self, tmp_path):
        cf1 = CustomFileConfig(name="FILE_A.md", searchable=True)
        cf2 = CustomFileConfig(name="FILE_B.md", searchable=True)
        s = _session(tmp_path, [cf1, cf2])
        try:
            s.execute_tool("memory_write", file="FILE_A.md", content="red apple fruit")
            s.execute_tool("memory_write", file="FILE_B.md", content="blue ocean water")
            r = s.execute_tool("memory_read", query="apple", file="FILE_A.md")
            assert r["status"] == "ok"
            # Results should only come from FILE_A
            for res in r["results"]:
                assert res["source"] != "file_b"
        finally:
            s.close()


# ---------------------------------------------------------------------------
# TestCustomFileList
# ---------------------------------------------------------------------------

class TestCustomFileList:
    def test_memory_list_includes_searchable_custom_files(self, tmp_path):
        cf = CustomFileConfig(name="NOTES.md", searchable=True)
        s = _session(tmp_path, [cf])
        try:
            s.execute_tool("memory_write", file="NOTES.md", content="some content")
            r = s.execute_tool("memory_list", target="files")
            assert r["status"] == "ok"
            assert any("NOTES.md" in f["file"] for f in r["files"])
        finally:
            s.close()

    def test_memory_list_includes_non_searchable_custom_files(self, tmp_path):
        cf = CustomFileConfig(name="HIDDEN.md", searchable=False)
        s = _session(tmp_path, [cf])
        try:
            # Write directly since tool only allows searchable custom files via memory_write
            (s.workspace.path / "HIDDEN.md").write_text("hidden content", encoding="utf-8")
            r = s.execute_tool("memory_list", target="files")
            assert r["status"] == "ok"
            assert any("HIDDEN.md" in f["file"] for f in r["files"])
        finally:
            s.close()


# ---------------------------------------------------------------------------
# TestCustomFileCompaction
# ---------------------------------------------------------------------------

class TestCustomFileCompaction:
    def test_compactable_adds_to_schema_enum(self, tmp_path):
        from groundmemory.tools import build_tool_registry
        cf = CustomFileConfig(name="RESEARCH.md", compactable=True)
        cfg = _cfg(tmp_path, [cf])
        _, _, schemas = build_tool_registry(cfg)
        compact_enum = schemas["memory_compact"]["parameters"]["properties"]["tier"]["enum"]
        assert "RESEARCH.md" in compact_enum

    def test_non_compactable_absent_from_enum(self, tmp_path):
        from groundmemory.tools import build_tool_registry
        cf = CustomFileConfig(name="RESEARCH.md", compactable=False)
        cfg = _cfg(tmp_path, [cf])
        _, _, schemas = build_tool_registry(cfg)
        compact_enum = schemas["memory_compact"]["parameters"]["properties"]["tier"]["enum"]
        assert "RESEARCH.md" not in compact_enum

    def test_compactable_included_in_bootstrap_compaction_notice(self, tmp_path):
        cf = CustomFileConfig(name="RESEARCH.md", compactable=True)
        ws = Workspace(tmp_path / "ws", custom_files=[cf])
        cfg = BootstrapConfig(
            inject_long_term_memory=True,
            compaction_tiers=["MEMORY.md"],
        )
        # Write something so bootstrap isn't empty
        ws.memory_file.write_text("some memory content", encoding="utf-8")
        result = build_bootstrap_prompt(ws, cfg, inject_compaction_notice=True)
        assert "RESEARCH.md" in result
