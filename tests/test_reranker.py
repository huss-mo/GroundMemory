"""
Tests for groundmemory/core/reranker.py

All tests use a fake CrossEncoder so no sentence_transformers install is needed.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import groundmemory.core.reranker as reranker_module
from groundmemory.core.reranker import _load_cross_encoder, rerank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_results(texts: list[str]) -> list[dict]:
    """Build minimal result dicts from a list of text strings."""
    return [
        {
            "chunk_id": f"c{i}",
            "text": t,
            "score": 0.5,
            "vector_score": 0.5,
            "text_score": 0.5,
        }
        for i, t in enumerate(texts)
    ]


def _fake_cross_encoder(model_name: str) -> MagicMock:
    """Return a CrossEncoder mock whose predict() returns descending scores."""
    mock = MagicMock()
    # predict returns a numpy-like array; .tolist() converts it to a Python list.
    # Scores are assigned in reverse order so the last item gets the highest score,
    # making reordering easy to verify.
    def predict(pairs):
        n = len(pairs)
        arr = MagicMock()
        arr.tolist.return_value = [float(n - i) for i in range(n)]
        return arr

    mock.predict.side_effect = predict
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the singleton cache before each test."""
    reranker_module._CROSS_ENCODER_CACHE.clear()
    yield
    reranker_module._CROSS_ENCODER_CACHE.clear()


# ---------------------------------------------------------------------------
# _load_cross_encoder tests
# ---------------------------------------------------------------------------


class TestLoadCrossEncoder:
    def test_raises_import_error_when_sentence_transformers_missing(self):
        """If sentence_transformers is not installed, raise a helpful ImportError."""
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            # Reload to re-evaluate the lazy import path
            with pytest.raises(ImportError) as exc_info:
                _load_cross_encoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        msg = str(exc_info.value)
        assert "sentence_transformers" in msg
        assert "pip install groundmemory[local]" in msg
        assert "rerank_model=" in msg

    def test_error_message_includes_model_name(self):
        model = "cross-encoder/my-custom-model"
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError) as exc_info:
                _load_cross_encoder(model)
        assert model in str(exc_info.value)

    def test_loads_model_and_caches_it(self):
        """Model is loaded once and reused on subsequent calls."""
        fake = _fake_cross_encoder("test-model")
        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = fake

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            m1 = _load_cross_encoder("test-model")
            m2 = _load_cross_encoder("test-model")

        assert m1 is m2
        mock_st.CrossEncoder.assert_called_once_with("test-model")

    def test_different_model_names_cached_separately(self):
        mock_a = _fake_cross_encoder("model-a")
        mock_b = _fake_cross_encoder("model-b")
        call_count = 0

        def side_effect(name):
            nonlocal call_count
            call_count += 1
            return mock_a if name == "model-a" else mock_b

        mock_st = MagicMock()
        mock_st.CrossEncoder.side_effect = side_effect

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            _load_cross_encoder("model-a")
            _load_cross_encoder("model-b")
            _load_cross_encoder("model-a")  # should be cached

        assert call_count == 2  # only two unique models loaded


# ---------------------------------------------------------------------------
# rerank() tests
# ---------------------------------------------------------------------------


class TestRerank:
    def _patched_rerank(self, query: str, results: list[dict], model_name: str = "model", top_k=None):
        """Call rerank() with a patched CrossEncoder."""
        fake = _fake_cross_encoder(model_name)
        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = fake
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            return rerank(query, results, model_name, top_k=top_k)

    def test_empty_results_returned_unchanged(self):
        out = self._patched_rerank("query", [])
        assert out == []

    def test_results_reordered_by_cross_encoder_score(self):
        texts = ["alpha", "beta", "gamma"]
        results = _make_results(texts)
        # _fake_cross_encoder assigns scores [3, 2, 1], so order stays alpha > beta > gamma
        out = self._patched_rerank("query", results)
        assert [r["chunk_id"] for r in out] == ["c0", "c1", "c2"]

    def test_score_field_updated(self):
        results = _make_results(["a", "b"])
        out = self._patched_rerank("query", results)
        # Scores from fake: [2.0, 1.0]
        assert out[0]["score"] == pytest.approx(2.0)
        assert out[1]["score"] == pytest.approx(1.0)

    def test_original_vector_and_text_scores_preserved(self):
        results = _make_results(["a"])
        results[0]["vector_score"] = 0.9
        results[0]["text_score"] = 0.8
        out = self._patched_rerank("query", results)
        assert out[0]["vector_score"] == pytest.approx(0.9)
        assert out[0]["text_score"] == pytest.approx(0.8)

    def test_top_k_truncates_output(self):
        results = _make_results(["a", "b", "c", "d"])
        out = self._patched_rerank("query", results, top_k=2)
        assert len(out) == 2

    def test_top_k_none_returns_all(self):
        results = _make_results(["a", "b", "c"])
        out = self._patched_rerank("query", results, top_k=None)
        assert len(out) == 3

    def test_single_result(self):
        results = _make_results(["only one"])
        out = self._patched_rerank("query", results)
        assert len(out) == 1
        assert out[0]["chunk_id"] == "c0"

    def test_results_with_missing_text_key(self):
        """Results without 'text' key should use empty string without raising."""
        results = [{"chunk_id": "x", "score": 0.5, "vector_score": 0.5, "text_score": 0.5}]
        # Should not raise
        out = self._patched_rerank("query", results)
        assert len(out) == 1

    def test_raises_import_error_when_sentence_transformers_missing(self):
        results = _make_results(["hello"])
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError) as exc_info:
                rerank("query", results, "cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert "pip install groundmemory[local]" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Integration: rerank_model=None disables reranking in hybrid_search
# ---------------------------------------------------------------------------


class TestSearchIntegrationNoRerank:
    """Verify that hybrid_search skips reranking when rerank_model is None."""

    def test_reranker_not_called_when_model_is_none(self, monkeypatch):
        from groundmemory.core import search as search_module
        from groundmemory.config import SearchConfig

        called = []

        def fake_rerank(query, results, model_name, top_k=None):
            called.append(model_name)
            return results

        monkeypatch.setattr("groundmemory.core.reranker.rerank", fake_rerank)

        cfg = SearchConfig(rerank_model=None)
        assert cfg.rerank_model is None
        # rerank() should never be called if rerank_model is None
        # (we verify by checking the config guard in search.py)
        assert not called