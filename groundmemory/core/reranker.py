"""
Cross-encoder reranking for GroundMemory hybrid search.

Usage
-----
Call ``rerank(query, results, model_name)`` to rescore a candidate list with a
cross-encoder.  The model is loaded once per process and cached as a singleton
so repeated search calls don't reload it.

Requirements
------------
Reranking is an **optional** feature that requires the ``[local]`` extra::

    pip install groundmemory[local]
    # or: uv sync --extra local

If ``rerank_model`` is set in config but ``sentence_transformers`` is not
installed, a clear ``ImportError`` is raised at call time (not at import time).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Only imported for type hints; never executed at runtime.
    from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Lazy singleton cache: model_name → CrossEncoder instance
# ---------------------------------------------------------------------------

_CROSS_ENCODER_CACHE: dict[str, "CrossEncoder"] = {}


def _load_cross_encoder(model_name: str) -> "CrossEncoder":
    """Return a cached CrossEncoder, loading it on first call.

    Args:
        model_name: HuggingFace model identifier, e.g.
                    ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``.

    Raises:
        ImportError: If ``sentence_transformers`` is not installed.
    """
    if model_name in _CROSS_ENCODER_CACHE:
        return _CROSS_ENCODER_CACHE[model_name]

    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            f"Cross-encoder reranking requires the 'sentence_transformers' package, "
            f"which is not installed.\n\n"
            f"Install it with:\n"
            f"    pip install groundmemory[local]\n"
            f"or:\n"
            f"    uv sync --extra local\n\n"
            f"You set rerank_model={model_name!r} in your config. "
            f"Either install the dependency above or remove rerank_model from your config "
            f"to disable reranking."
        ) from exc

    model = CrossEncoder(model_name)
    _CROSS_ENCODER_CACHE[model_name] = model
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    results: list[dict],
    model_name: str,
    top_k: Optional[int] = None,
) -> list[dict]:
    """Rescore *results* with a cross-encoder and return them sorted by new score.

    The cross-encoder receives ``(query, chunk_text)`` pairs and produces a
    relevance logit for each.  The logit replaces ``result["score"]`` so that
    downstream steps (graph expansion, final slicing) see the reranked order.
    Original ``vector_score`` and ``text_score`` are preserved unchanged.

    Args:
        query:      The search query string.
        results:    List of merged result dicts (as produced by ``_merge_results``).
                    Each dict must have a ``"text"`` key.
        model_name: HuggingFace cross-encoder model identifier.
        top_k:      If given, return only the top *top_k* results after reranking.
                    Pass ``None`` to return all results (preserving the full list).

    Returns:
        Results sorted by descending cross-encoder score, optionally truncated to
        *top_k*.

    Raises:
        ImportError: If ``sentence_transformers`` is not installed (with a
                     clear message explaining how to install the dependency).
    """
    if not results:
        return results

    model = _load_cross_encoder(model_name)

    pairs = [(query, r.get("text", "")) for r in results]
    scores: list[float] = model.predict(pairs).tolist()  # type: ignore[union-attr]

    for result, score in zip(results, scores):
        result["score"] = float(score)

    results.sort(key=lambda r: r["score"], reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results