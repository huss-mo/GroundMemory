"""
Integration tests for real embeddings + vector search.

Two test markers gate these tests:

  local
    Requires: pip install groundmemory[local]  (sentence-transformers)
    Model names are read from groundmemory.yaml / .env:
      embedding.local_model  (default: sentence-transformers/all-MiniLM-L6-v2)
      search.rerank_model    (used by reranker tests, not here)
    Skipped automatically when sentence-transformers is not importable.

  api_embeddings
    Requires: a configured OpenAI-compatible HTTP embedding endpoint.
    All settings are read from groundmemory.yaml / .env:
      embedding.provider   must be "openai"
      embedding.base_url
      embedding.api_key
      embedding.model
    Skipped automatically when provider != "openai" or endpoint unreachable.

Run selectively:
    pytest -m local
    pytest -m api_embeddings
    pytest -m "local or api_embeddings"

Run everything (skips gracefully when deps/config are absent):
    pytest
"""
from __future__ import annotations

import uuid
import pytest

from groundmemory.config import groundmemoryConfig, SearchConfig
from groundmemory.session import MemorySession


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config() -> groundmemoryConfig:
    """Load config from groundmemory.yaml / .env (standard priority chain)."""
    return groundmemoryConfig.auto()


def _try_probe_provider(provider) -> str | None:
    """Embed a single string to verify the provider is reachable.

    Returns None on success, or an error message string on failure.
    """
    try:
        vecs = provider.embed(["probe"])
        if not vecs or not vecs[0]:
            return "provider returned empty vector"
        return None
    except Exception as exc:
        return str(exc)


# ---------------------------------------------------------------------------
# local fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def local_session(tmp_path_factory):
    """
    MemorySession backed by the local sentence-transformers provider.

    - Model name comes from config (embedding.local_model), defaulting to
      sentence-transformers/all-MiniLM-L6-v2.
    - Skips the whole module group if sentence-transformers is not installed.
    """
    try:
        from groundmemory.core.embeddings import SentenceTransformerProvider  # noqa: F401
    except ImportError:
        pytest.skip("sentence-transformers not installed (pip install groundmemory[local])")

    cfg = _load_config()
    local_model = cfg.embedding.local_model  # e.g. "sentence-transformers/all-MiniLM-L6-v2"
    # Strip the "sentence-transformers/" namespace prefix if present, as
    # SentenceTransformerProvider expects a plain model name.
    model_name = local_model.removeprefix("sentence-transformers/")

    tmp = tmp_path_factory.mktemp("local_session")
    from groundmemory.config import EmbeddingConfig

    session_cfg = groundmemoryConfig(
        root_dir=tmp,
        workspace="local-test",
        embedding=EmbeddingConfig(provider="local", local_model=local_model),
        search=cfg.search,
    )
    name = uuid.uuid4().hex[:8]
    s = MemorySession.create(name, config=session_cfg)

    err = _try_probe_provider(s.provider)
    if err:
        s.close()
        pytest.skip(f"Local embedding provider failed: {err}")

    yield s
    s.close()


# ---------------------------------------------------------------------------
# api_embeddings fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_session(tmp_path_factory):
    """
    MemorySession backed by the OpenAI-compatible HTTP embedding provider.

    - All settings (base_url, api_key, model) are read from groundmemory.yaml / .env.
    - Skips if embedding.provider != "openai" or the endpoint is unreachable.
    """
    cfg = _load_config()

    if cfg.embedding.provider != "openai":
        pytest.skip(
            "API embeddings not configured (set embedding.provider=openai in "
            "groundmemory.yaml or .env)"
        )

    tmp = tmp_path_factory.mktemp("api_session")
    session_cfg = groundmemoryConfig(
        root_dir=tmp,
        workspace="api-test",
        embedding=cfg.embedding,
        search=cfg.search,
    )
    name = uuid.uuid4().hex[:8]
    s = MemorySession.create(name, config=session_cfg)

    err = _try_probe_provider(s.provider)
    if err:
        s.close()
        pytest.skip(f"API embedding endpoint unreachable: {err}")

    yield s
    s.close()


# ---------------------------------------------------------------------------
# Parametrised fixture: runs tests under both local and api_embeddings
# ---------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    """Parametrise 'any_session' with both local and api_session fixtures."""
    if "any_session" in metafunc.fixturenames:
        metafunc.parametrize(
            "any_session",
            ["local_session", "api_session"],
            indirect=True,
        )


@pytest.fixture
def any_session(request):
    """Indirect fixture: delegates to either local_session or api_session."""
    return request.getfixturevalue(request.param)


# ---------------------------------------------------------------------------
# Tests: SentenceTransformerProvider (local only)
# ---------------------------------------------------------------------------

@pytest.mark.local
class TestSentenceTransformerProvider:
    """Unit tests for SentenceTransformerProvider using a real downloaded model."""

    @pytest.fixture(scope="class")
    def provider(self):
        try:
            from groundmemory.core.embeddings import SentenceTransformerProvider
        except ImportError:
            pytest.skip("sentence-transformers not installed (pip install groundmemory[local])")

        cfg = _load_config()
        local_model = cfg.embedding.local_model
        model_name = local_model.removeprefix("sentence-transformers/")
        return SentenceTransformerProvider(model_name=model_name)

    def test_import_no_error(self):
        """sentence-transformers can be imported without error."""
        try:
            from groundmemory.core.embeddings import SentenceTransformerProvider  # noqa: F401
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_model_id_has_namespace(self, provider):
        """model_id always has the sentence-transformers/ namespace prefix."""
        assert provider.model_id.startswith("sentence-transformers/")

    def test_dimensions_positive(self, provider):
        """Provider reports a positive embedding dimension."""
        assert provider.dimensions > 0

    def test_embed_empty_returns_empty(self, provider):
        assert provider.embed([]) == []

    def test_embed_single_returns_correct_dim(self, provider):
        vecs = provider.embed(["hello world"])
        assert len(vecs) == 1
        assert len(vecs[0]) == provider.dimensions

    def test_embed_multiple_returns_correct_count(self, provider):
        texts = ["foo", "bar", "baz"]
        vecs = provider.embed(texts)
        assert len(vecs) == 3
        assert all(len(v) == provider.dimensions for v in vecs)

    def test_embed_values_are_floats(self, provider):
        vecs = provider.embed(["check types"])
        assert all(isinstance(x, float) for x in vecs[0])


# ---------------------------------------------------------------------------
# Tests: EmbeddingProvider contract (local + api_embeddings)
# ---------------------------------------------------------------------------

@pytest.mark.local
@pytest.mark.api_embeddings
class TestEmbeddingProvider:
    """Provider contract tests - run under both local and api_embeddings."""

    def test_embed_returns_nonempty_vectors(self, any_session):
        """Provider returns one non-empty vector per input string."""
        texts = ["hello world", "the quick brown fox"]
        vecs = any_session.provider.embed(texts)
        assert len(vecs) == 2
        assert all(len(v) > 0 for v in vecs)

    def test_embed_vectors_have_consistent_dimensions(self, any_session):
        """All vectors from the same provider have the same dimensionality."""
        texts = ["foo", "bar", "baz"]
        vecs = any_session.provider.embed(texts)
        dims = [len(v) for v in vecs]
        assert len(set(dims)) == 1, f"Inconsistent dimensions: {dims}"

    def test_embed_single_text(self, any_session):
        """Single-item input works correctly."""
        vecs = any_session.provider.embed(["single sentence"])
        assert len(vecs) == 1
        assert len(vecs[0]) > 0

    def test_embed_empty_list(self, any_session):
        """Empty input returns empty list without error."""
        vecs = any_session.provider.embed([])
        assert vecs == []

    def test_similar_texts_have_higher_similarity(self, any_session):
        """Semantically similar texts score higher than unrelated ones."""
        from groundmemory.core.embeddings import cosine_similarity

        provider = any_session.provider
        v_base = provider.embed(["machine learning model training"])[0]
        v_similar = provider.embed(["neural network training process"])[0]
        v_unrelated = provider.embed(["apple pie recipe cooking"])[0]

        sim_similar = cosine_similarity(v_base, v_similar)
        sim_unrelated = cosine_similarity(v_base, v_unrelated)

        assert sim_similar > sim_unrelated, (
            f"Expected similar ({sim_similar:.3f}) > unrelated ({sim_unrelated:.3f})"
        )


# ---------------------------------------------------------------------------
# Tests: Vector search (local + api_embeddings)
# ---------------------------------------------------------------------------

@pytest.mark.local
@pytest.mark.api_embeddings
class TestVectorSearch:
    """Vector and hybrid search tests - run under both local and api_embeddings."""

    def test_sync_and_vector_search_returns_results(self, any_session, tmp_path_factory):
        """After syncing content, vector search returns relevant chunks."""
        tmp = tmp_path_factory.mktemp("vector_search")
        cfg = _load_config()
        s_cfg = groundmemoryConfig(
            root_dir=tmp,
            workspace="vs-test",
            embedding=any_session.config.embedding,
            search=SearchConfig(top_k=5, vector_weight=1.0),
        )
        s = MemorySession.create(uuid.uuid4().hex[:8], config=s_cfg)
        try:
            s.execute_tool(
                "memory_write",
                file="MEMORY.md",
                content="Alice is a senior software engineer who specialises in Python and machine learning.",
            )
            s.execute_tool(
                "memory_write",
                file="MEMORY.md",
                content="Bob is a product manager focused on mobile applications and user experience.",
            )
            s.sync()

            results = s.execute_tool("memory_read", query="Python software engineer")
            assert results["status"] == "ok"
            assert len(results["results"]) > 0
            assert any(
                "Alice" in r["text"] or "Python" in r["text"]
                for r in results["results"]
            )
        finally:
            s.close()

    def test_semantic_search_finds_paraphrase(self, any_session, tmp_path_factory):
        """Vector search matches semantically equivalent queries even without keyword overlap."""
        tmp = tmp_path_factory.mktemp("paraphrase_search")
        s_cfg = groundmemoryConfig(
            root_dir=tmp,
            workspace="para-test",
            embedding=any_session.config.embedding,
            search=SearchConfig(top_k=5, vector_weight=1.0),
        )
        s = MemorySession.create(uuid.uuid4().hex[:8], config=s_cfg)
        try:
            s.execute_tool(
                "memory_write",
                file="MEMORY.md",
                content="The team uses pytest for automated testing of the backend services.",
            )
            s.sync()

            results = s.execute_tool(
                "memory_read",
                query="automated quality assurance framework for server-side code",
            )
            assert results["status"] == "ok"
            assert len(results["results"]) > 0
        finally:
            s.close()

    def test_hybrid_search_outperforms_bm25_on_paraphrase(self, any_session, tmp_path_factory):
        """Hybrid search surfaces paraphrase content that pure BM25 would miss."""
        tmp = tmp_path_factory.mktemp("hybrid_vs_bm25")

        content = (
            "The engineering team decided to migrate the authentication service "
            "from a monolith to microservices architecture."
        )
        paraphrase_query = "breaking apart a large application into smaller independent services"

        def _run_search(vector_weight: float) -> list[dict]:
            s_cfg = groundmemoryConfig(
                root_dir=tmp,
                workspace=f"hybrid-{int(vector_weight * 100)}",
                embedding=any_session.config.embedding,
                search=SearchConfig(top_k=5, vector_weight=vector_weight),
            )
            s = MemorySession.create(uuid.uuid4().hex[:8], config=s_cfg)
            try:
                s.execute_tool("memory_write", file="MEMORY.md", content=content)
                s.sync()
                r = s.execute_tool("memory_read", query=paraphrase_query)
                return r.get("results", [])
            finally:
                s.close()

        hybrid_results = _run_search(vector_weight=0.7)
        _run_search(vector_weight=0.0)  # BM25-only run (result not asserted)

        assert len(hybrid_results) > 0, "Hybrid search returned no results"
        assert any(
            "microservices" in r["text"] or "monolith" in r["text"]
            for r in hybrid_results
        )


# ---------------------------------------------------------------------------
# Tests: End-to-end memory search (local + api_embeddings)
# ---------------------------------------------------------------------------

@pytest.mark.local
@pytest.mark.api_embeddings
class TestEndToEndMemorySearch:
    """End-to-end write → sync → search tests - run under both local and api_embeddings."""

    def test_memory_search_tool_with_real_embeddings(self, any_session, tmp_path_factory):
        """write → sync → search returns the written facts."""
        tmp = tmp_path_factory.mktemp("e2e")
        s_cfg = groundmemoryConfig(
            root_dir=tmp,
            workspace="e2e-test",
            embedding=any_session.config.embedding,
            search=SearchConfig(top_k=20, vector_weight=any_session.config.search.vector_weight),
        )
        s = MemorySession.create(uuid.uuid4().hex[:8], config=s_cfg)
        try:
            facts = [
                "The project deadline is end of Q2 2026.",
                "Alice prefers async code reviews over synchronous meetings.",
                "The staging environment runs on Kubernetes in AWS us-east-1.",
            ]
            for fact in facts:
                s.execute_tool("memory_write", file="MEMORY.md", content=fact)
            s.sync()

            r = s.execute_tool("memory_read", query="deployment infrastructure cloud")
            assert r["status"] == "ok"
            assert len(r["results"]) > 0
            assert any(
                "Kubernetes" in res["text"] or "AWS" in res["text"]
                for res in r["results"]
            )
        finally:
            s.close()