"""End-to-end: profile renders into the answer prompt that the orchestrator builds."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_db_singleton():
    import hrag.db.connection as _conn_mod  # noqa: PLC0415

    _conn_mod._db_singleton = None
    yield
    _conn_mod._db_singleton = None


def _make_orch(sample_config, monkeypatch, fake_llm, fake_embedder):
    from hrag.orchestrator import Orchestrator  # noqa: PLC0415

    monkeypatch.setattr(
        "hrag.providers.llm.get_llm_provider", lambda cfg: fake_llm
    )
    monkeypatch.setattr(
        "hrag.providers.embeddings.get_embedding_provider", lambda cfg: fake_embedder
    )
    # Disable the cross-encoder reranker so Orchestrator boots without
    # sentence_transformers.CrossEncoder. The conftest stubs cover the
    # vector store, embedder, and ollama paths but not CrossEncoder.
    sample_config.retrieval.rerank_enabled = False
    sample_config.retrieval.doc_scope_enabled = False
    return Orchestrator(sample_config)


def test_chat_renders_empty_profile_when_no_prefs(
    sample_config, monkeypatch, fake_llm, fake_embedder
):
    orch = _make_orch(sample_config, monkeypatch, fake_llm, fake_embedder)
    # Inject a no-op retriever so we don't depend on Chroma.
    orch.retriever = type("NoOpR", (), {"retrieve": lambda self, *a, **kw: []})()
    orch.reranker = None
    orch.mst_organizer = None

    result = orch.chat("Hello?", "default")
    assert "(no profile yet)" in result.prompt
    orch.close()


def test_chat_renders_seeded_profile_into_prompt(
    sample_config, monkeypatch, fake_llm, fake_embedder
):
    orch = _make_orch(sample_config, monkeypatch, fake_llm, fake_embedder)
    orch.retriever = type("NoOpR", (), {"retrieve": lambda self, *a, **kw: []})()
    orch.reranker = None
    orch.mst_organizer = None

    orch.profile_store.upsert(
        "default", "fact", "occupation", "data engineer", confidence=0.95
    )
    orch.profile_store.upsert(
        "default", "style", "response length", "shorter answers", confidence=0.9
    )

    result = orch.chat("What are good Python libraries?", "default")
    assert "Facts: occupation: data engineer" in result.prompt
    assert "Style preferences: response length: shorter answers" in result.prompt
    orch.close()


def test_auto_extractor_off_by_default(
    sample_config, monkeypatch, fake_llm, fake_embedder
):
    orch = _make_orch(sample_config, monkeypatch, fake_llm, fake_embedder)
    assert orch.auto_extractor is None
    orch.close()


def test_auto_extractor_constructed_when_enabled(
    sample_config, monkeypatch, fake_llm, fake_embedder
):
    sample_config.memory.auto_extract = True
    orch = _make_orch(sample_config, monkeypatch, fake_llm, fake_embedder)
    assert orch.auto_extractor is not None
    orch.close()
