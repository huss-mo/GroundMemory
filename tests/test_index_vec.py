"""
Unit tests for the sqlite-vec fast-path in MemoryIndex.vector_search().

These tests exercise the vec0 KNN path directly — they bypass the MCP layer
and tool system and operate on MemoryIndex in isolation.

All tests gracefully skip when sqlite-vec is not installed, so the suite
stays green in environments without the optional dependency.
"""
from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from groundmemory.core.chunker import Chunk
from groundmemory.core.index import MemoryIndex, _SQLITE_VEC_MODULE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(path: str, chunk_id: str, text: str, source: str = "memory") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        path=path,
        source=source,
        start_line=1,
        end_line=1,
        content_hash=chunk_id,
        text=text,
    )


def _rand_unit(dim: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def idx(tmp_path: Path) -> MemoryIndex:
    return MemoryIndex(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Skip marker
# ---------------------------------------------------------------------------

vec_available = pytest.mark.skipif(
    _SQLITE_VEC_MODULE is None,
    reason="sqlite-vec not installed",
)


# ---------------------------------------------------------------------------
# Tests: extension loading
# ---------------------------------------------------------------------------

class TestVecLoading:
    def test_vec_available_flag_set(self, idx: MemoryIndex) -> None:
        """When sqlite-vec is installed, _vec_available should be True."""
        if _SQLITE_VEC_MODULE is None:
            pytest.skip("sqlite-vec not installed")
        assert idx._vec_available is True

    def test_vec_version_callable(self, idx: MemoryIndex) -> None:
        """vec_version() must return a version string."""
        if _SQLITE_VEC_MODULE is None:
            pytest.skip("sqlite-vec not installed")
        row = idx._conn.execute("SELECT vec_version()").fetchone()
        assert row is not None
        assert row[0].startswith("v")


# ---------------------------------------------------------------------------
# Tests: vec_chunks table lifecycle
# ---------------------------------------------------------------------------

class TestVecTable:
    @vec_available
    def test_vec_table_created_on_first_upsert(self, idx: MemoryIndex, tmp_path: Path) -> None:
        """vec_chunks virtual table must be created after the first upsert."""
        # Before any upsert, vec_chunks should not exist.
        tbl = idx._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
        ).fetchone()
        assert tbl is None

        # Upsert one chunk with a 4-dim embedding.
        path = str(tmp_path / "MEMORY.md")
        idx.upsert_file(path, "memory", "h1", time.time(), 10)
        chunk = _make_chunk(path, "c1", "hello world")
        idx.upsert_chunks([chunk], [[0.1, 0.2, 0.3, 0.4]], "test-model")

        tbl = idx._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
        ).fetchone()
        assert tbl is not None

    @vec_available
    def test_meta_dim_stored(self, idx: MemoryIndex, tmp_path: Path) -> None:
        """Embedding dimension is persisted in the meta table."""
        path = str(tmp_path / "MEMORY.md")
        idx.upsert_file(path, "memory", "h1", time.time(), 10)
        chunk = _make_chunk(path, "c1", "hello")
        idx.upsert_chunks([chunk], [[0.1, 0.2, 0.3, 0.4]], "test-model")

        row = idx._conn.execute("SELECT value FROM meta WHERE key='vec_dim'").fetchone()
        assert row is not None
        assert int(row["value"]) == 4

    @vec_available
    def test_vec_table_rebuilt_on_dim_change(self, idx: MemoryIndex, tmp_path: Path) -> None:
        """When embedding dimension changes, vec_chunks is rebuilt."""
        path = str(tmp_path / "MEMORY.md")
        idx.upsert_file(path, "memory", "h1", time.time(), 10)

        chunk_a = _make_chunk(path, "c1", "dim4")
        idx.upsert_chunks([chunk_a], [[0.1, 0.2, 0.3, 0.4]], "model-a")

        # Now upsert with a different dimension — should rebuild without error.
        chunk_b = _make_chunk(path, "c2", "dim3")
        idx.upsert_chunks([chunk_b], [[0.1, 0.2, 0.3]], "model-b")

        row = idx._conn.execute("SELECT value FROM meta WHERE key='vec_dim'").fetchone()
        assert int(row["value"]) == 3


# ---------------------------------------------------------------------------
# Tests: zero-dim embeddings (NullEmbeddingProvider)
# ---------------------------------------------------------------------------

class TestZeroDimEmbeddings:
    def test_null_embeddings_do_not_create_vec_table(self, idx: MemoryIndex, tmp_path: Path) -> None:
        """Zero-dimension embeddings must not trigger vec_chunks creation."""
        path = str(tmp_path / "MEMORY.md")
        idx.upsert_file(path, "memory", "h1", time.time(), 10)
        chunk = _make_chunk(path, "c1", "hello")
        idx.upsert_chunks([chunk], [[]], "null")  # empty embedding

        tbl = idx._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
        ).fetchone()
        assert tbl is None

    def test_null_embeddings_search_returns_empty(self, idx: MemoryIndex, tmp_path: Path) -> None:
        """vector_search with empty query embedding returns []."""
        path = str(tmp_path / "MEMORY.md")
        idx.upsert_file(path, "memory", "h1", time.time(), 10)
        chunk = _make_chunk(path, "c1", "hello")
        idx.upsert_chunks([chunk], [[]], "null")

        results = idx.vector_search([], top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: KNN correctness
# ---------------------------------------------------------------------------

class TestVecSearchCorrectness:
    @vec_available
    def test_identical_query_scores_1(self, idx: MemoryIndex, tmp_path: Path) -> None:
        """Querying with the same vector as a stored chunk must yield score == 1.0."""
        path = str(tmp_path / "MEMORY.md")
        idx.upsert_file(path, "memory", "h1", time.time(), 10)
        emb = _rand_unit(16, seed=0)
        chunk = _make_chunk(path, "c1", "exact match")
        idx.upsert_chunks([chunk], [emb], "model")

        results = idx.vector_search(emb, top_k=1)
        assert len(results) == 1
        assert math.isclose(results[0]["vector_score"], 1.0, abs_tol=1e-5)

    @vec_available
    def test_ranking_order_correct(self, idx: MemoryIndex, tmp_path: Path) -> None:
        """The most similar chunk must rank first."""
        path = str(tmp_path / "MEMORY.md")
        idx.upsert_file(path, "memory", "h1", time.time(), 10)

        query = _rand_unit(32, seed=42)
        # c1: identical to query (cosine=1.0)
        # c2: orthogonal (cosine=0.0)
        # c3: opposite (cosine=-1.0)
        q = np.array(query, dtype=np.float32)
        ortho = np.zeros(32, dtype=np.float32)
        ortho[1] = 1.0 if abs(q[0]) < 0.9 else 0.0
        ortho[0] = -q[1] / q[0] if q[0] != 0 else 1.0
        ortho = ortho / np.linalg.norm(ortho)
        opposite = (-q).tolist()

        chunks = [
            _make_chunk(path, "c1", "identical"),
            _make_chunk(path, "c2", "orthogonal"),
            _make_chunk(path, "c3", "opposite"),
        ]
        embeddings = [query, ortho.tolist(), opposite]
        idx.upsert_chunks(chunks, embeddings, "model")

        results = idx.vector_search(query, top_k=3)
        ids = [r["chunk_id"] for r in results]
        scores = [r["vector_score"] for r in results]

        assert ids[0] == "c1", f"Expected c1 first, got {ids}"
        assert scores[0] > scores[1] >= scores[2], f"Scores not descending: {scores}"

    @vec_available
    def test_top_k_limits_results(self, idx: MemoryIndex, tmp_path: Path) -> None:
        """top_k must cap the number of returned results."""
        path = str(tmp_path / "MEMORY.md")
        idx.upsert_file(path, "memory", "h1", time.time(), 10)

        chunks = [_make_chunk(path, f"c{i}", f"chunk {i}") for i in range(10)]
        embs = [_rand_unit(8, seed=i) for i in range(10)]
        idx.upsert_chunks(chunks, embs, "model")

        results = idx.vector_search(_rand_unit(8, seed=99), top_k=3)
        assert len(results) <= 3

    @vec_available
    def test_source_filter_applied(self, idx: MemoryIndex, tmp_path: Path) -> None:
        """source_filter must exclude chunks from other sources."""
        path_mem = str(tmp_path / "MEMORY.md")
        path_usr = str(tmp_path / "USER.md")
        idx.upsert_file(path_mem, "memory", "h1", time.time(), 10)
        idx.upsert_file(path_usr, "user", "h2", time.time(), 10)

        emb = _rand_unit(8, seed=7)
        chunk_mem = _make_chunk(path_mem, "c_mem", "memory chunk", source="memory")
        chunk_usr = _make_chunk(path_usr, "c_usr", "user chunk", source="user")
        idx.upsert_chunks([chunk_mem], [emb], "model")
        idx.upsert_chunks([chunk_usr], [emb], "model")

        results = idx.vector_search(emb, top_k=5, source_filter="memory")
        sources = {r["source"] for r in results}
        assert sources == {"memory"}, f"Unexpected sources: {sources}"

    @vec_available
    def test_cosine_score_in_result_dict(self, idx: MemoryIndex, tmp_path: Path) -> None:
        """Result dicts must have vector_score, text_score, and score keys."""
        path = str(tmp_path / "MEMORY.md")
        idx.upsert_file(path, "memory", "h1", time.time(), 10)
        emb = _rand_unit(8, seed=1)
        chunk = _make_chunk(path, "c1", "test chunk")
        idx.upsert_chunks([chunk], [emb], "model")

        results = idx.vector_search(emb, top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "vector_score" in r
        assert "text_score" in r
        assert "score" in r
        assert r["text_score"] == 0.0
        assert -1.0 <= r["vector_score"] <= 1.0

    @vec_available
    def test_empty_index_returns_empty(self, idx: MemoryIndex) -> None:
        """Searching an empty index (no vec_chunks table yet) must return []."""
        query = _rand_unit(8, seed=0)
        results = idx.vector_search(query, top_k=5)
        assert results == []

    @vec_available
    def test_l2_to_cosine_conversion(self, idx: MemoryIndex, tmp_path: Path) -> None:
        """
        For unit vectors, cosine_similarity = 1 - L2^2/2.
        Verify the conversion produces values consistent with NumPy reference.
        """
        path = str(tmp_path / "MEMORY.md")
        idx.upsert_file(path, "memory", "h1", time.time(), 10)
        dim = 16
        seeds = [10, 20, 30]
        chunks = [_make_chunk(path, f"c{s}", f"chunk {s}") for s in seeds]
        embs = [_rand_unit(dim, seed=s) for s in seeds]
        idx.upsert_chunks(chunks, embs, "model")

        query = _rand_unit(dim, seed=99)
        q = np.array(query, dtype=np.float32)

        # Reference cosines from NumPy
        ref = {}
        for chunk, emb in zip(chunks, embs):
            e = np.array(emb, dtype=np.float32)
            ref[chunk.chunk_id] = float(np.dot(q, e))  # both unit-length

        results = idx.vector_search(query, top_k=len(seeds))
        for r in results:
            expected = ref[r["chunk_id"]]
            assert math.isclose(r["vector_score"], expected, abs_tol=1e-4), (
                f"chunk {r['chunk_id']}: got {r['vector_score']:.6f}, expected {expected:.6f}"
            )


# ---------------------------------------------------------------------------
# Tests: fallback path consistency
# ---------------------------------------------------------------------------

class TestFallbackConsistency:
    @vec_available
    def test_vec_and_numpy_agree(self, tmp_path: Path) -> None:
        """
        vec path and NumPy fallback must return results in the same order
        with scores that agree to within floating-point tolerance.
        """
        idx_vec = MemoryIndex(tmp_path / "vec.db")
        idx_np = MemoryIndex(tmp_path / "np.db")
        # Force NumPy path on idx_np
        idx_np._vec_available = False

        dim = 32
        path = str(tmp_path / "MEMORY.md")
        for idx_inst in (idx_vec, idx_np):
            idx_inst.upsert_file(path, "memory", "h1", time.time(), 10)

        n_chunks = 20
        chunks = [_make_chunk(path, f"c{i}", f"text {i}") for i in range(n_chunks)]
        embs = [_rand_unit(dim, seed=i) for i in range(n_chunks)]

        idx_vec.upsert_chunks(chunks, embs, "model")
        idx_np.upsert_chunks(chunks, embs, "model")

        query = _rand_unit(dim, seed=999)
        top_k = 5

        res_vec = idx_vec.vector_search(query, top_k=top_k)
        res_np = idx_np.vector_search(query, top_k=top_k)

        assert len(res_vec) == len(res_np), (
            f"Different result counts: vec={len(res_vec)}, numpy={len(res_np)}"
        )
        for rv, rn in zip(res_vec, res_np):
            assert rv["chunk_id"] == rn["chunk_id"], (
                f"Order mismatch: vec={rv['chunk_id']}, numpy={rn['chunk_id']}"
            )
            assert math.isclose(rv["vector_score"], rn["vector_score"], abs_tol=1e-4), (
                f"Score mismatch for {rv['chunk_id']}: "
                f"vec={rv['vector_score']:.6f}, numpy={rn['vector_score']:.6f}"
            )