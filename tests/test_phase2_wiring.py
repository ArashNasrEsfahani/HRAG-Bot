"""Integration tests for Phase 2 wiring.

Verifies that factory.py, orchestrator.py, and cli.py correctly plumb
the KG layer without requiring real Leiden / scipy / networkx in the
lightweight paths. Heavy tests are gated with pytest.importorskip.
"""

from __future__ import annotations

from typing import Optional

import pytest

from tests.conftest import FakeLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_db_singleton() -> None:
    import hrag.db.connection as _conn_mod

    _conn_mod._db_singleton = None


def _make_orchestrator(cfg):
    """Build an Orchestrator from *cfg*, resetting the DB singleton first.

    Disables the cross-encoder reranker to avoid needing the real
    sentence_transformers.CrossEncoder (the conftest stub lacks it).
    Also disables the intent gate — these tests patch ``orch.llm`` after
    construction, but the cached ``intent_classifier`` keeps the original
    LLM reference; the simplest fix is to bypass intent routing here so
    every query takes the FACTUAL path (which is what these tests exercise).
    """
    _reset_db_singleton()
    cfg.retrieval.rerank_enabled = False
    cfg.intent.enabled = False
    from hrag.orchestrator import Orchestrator

    return Orchestrator(cfg)


# ---------------------------------------------------------------------------
# 1. Default config → KG disabled → all KG attrs are None
# ---------------------------------------------------------------------------


def test_orchestrator_kg_disabled_default(sample_config):
    """Default Config has kg.enabled=False → kg_store/community_store/mst_organizer are None."""
    orch = _make_orchestrator(sample_config)
    try:
        assert orch.kg_store is None
        assert orch.community_store is None
        assert orch.mst_organizer is None
    finally:
        orch.close()
        _reset_db_singleton()


# ---------------------------------------------------------------------------
# 2. kg.enabled=True → stores are constructed (requires networkx + scipy)
# ---------------------------------------------------------------------------


def test_orchestrator_kg_enabled_constructs_stores(sample_config):
    """With kg.enabled=True, Orchestrator builds KGStore + CommunityStore + MSTOrganizer."""
    pytest.importorskip("networkx")
    pytest.importorskip("scipy.sparse")

    sample_config.kg.enabled = True
    # Ensure the kg_path directory exists (KGStore creates it internally but
    # we need the parent to exist for resolve() to work correctly).
    kg_path = sample_config.resolve(sample_config.storage.kg_path)
    kg_path.mkdir(parents=True, exist_ok=True)

    orch = _make_orchestrator(sample_config)
    try:
        assert orch.kg_store is not None, "kg_store should be built when kg.enabled=True"
        assert orch.community_store is not None, "community_store should be built when kg.enabled=True"
        assert orch.mst_organizer is not None, "mst_organizer should be built when kg.enabled=True"
    finally:
        orch.close()
        _reset_db_singleton()


# ---------------------------------------------------------------------------
# 3. build_retriever("kg_ppr") without required deps raises ValueError
# ---------------------------------------------------------------------------


def test_build_retriever_kg_ppr_requires_deps(sample_config, tmp_db, fake_embedder):
    """build_retriever(retriever='kg_ppr') without kg_store raises ValueError."""
    from hrag.retrieval.factory import build_retriever
    from hrag.retrieval.vector import VectorStore

    chroma_path = sample_config.resolve(sample_config.storage.chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    vs = VectorStore(chroma_path, fake_embedder.dim)

    sample_config.retrieval.retriever = "kg_ppr"

    with pytest.raises(ValueError, match="kg_ppr"):
        build_retriever(
            sample_config.retrieval,
            tmp_db,
            vs,
            fake_embedder,
            llm=None,        # missing
            kg_store=None,   # missing
            kg_cfg=None,     # missing
        )


# ---------------------------------------------------------------------------
# 4. build_retriever("router") with only vector deps → QueryRouter with
#    kg_ppr_retriever=None and community_retriever=None
# ---------------------------------------------------------------------------


def test_build_retriever_router_works_with_partial_deps(sample_config, tmp_db, fake_embedder):
    """retriever='router' with only vector_store falls back to single-retriever passthrough."""
    from hrag.retrieval.factory import build_retriever
    from hrag.retrieval.router import QueryRouter
    from hrag.retrieval.vector import VectorStore

    chroma_path = sample_config.resolve(sample_config.storage.chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    vs = VectorStore(chroma_path, fake_embedder.dim)

    fake_llm = FakeLLM()
    sample_config.retrieval.retriever = "router"

    retriever = build_retriever(
        sample_config.retrieval,
        tmp_db,
        vs,
        fake_embedder,
        llm=fake_llm,
        kg_store=None,        # not available
        community_store=None, # not available
        kg_cfg=sample_config.kg,
    )

    # The factory now wraps the built retriever in DocScopedRetriever by
    # default. Peel the wrapper before checking the inner type.
    from hrag.retrieval.doc_scope import DocScopedRetriever

    inner = retriever._wrapped if isinstance(retriever, DocScopedRetriever) else retriever
    assert isinstance(inner, QueryRouter)
    # With no KG/community deps, sub-retrievers should be None.
    assert inner._kg_ppr is None
    assert inner._community is None
    assert inner._vector is not None


# ---------------------------------------------------------------------------
# 5. build_mst_organizer returns None when KG is disabled
# ---------------------------------------------------------------------------


def test_build_mst_organizer_returns_none_when_disabled(sample_config):
    """build_mst_organizer with kg.enabled=False returns None."""
    from hrag.retrieval.factory import build_mst_organizer

    assert sample_config.kg.enabled is False
    result = build_mst_organizer(sample_config.kg, kg_store=None)
    assert result is None


def test_build_mst_organizer_returns_none_when_store_missing(sample_config):
    """build_mst_organizer with kg.enabled=True but kg_store=None returns None."""
    from hrag.retrieval.factory import build_mst_organizer

    sample_config.kg.enabled = True
    result = build_mst_organizer(sample_config.kg, kg_store=None)
    assert result is None


# ---------------------------------------------------------------------------
# 6. chat() invokes mst_organizer.organize when mst_organizer is set
# ---------------------------------------------------------------------------


class _SpyOrganizer:
    """Minimal stand-in for MSTOrganizer that records calls."""

    name = "spy_organizer"
    call_count = 0
    last_input: Optional[list] = None

    def organize(self, results: list) -> list:
        _SpyOrganizer.call_count += 1
        _SpyOrganizer.last_input = results
        return results  # pass-through


def test_orchestrator_chat_calls_mst_when_present(sample_config):
    """When orch.mst_organizer is not None, chat() invokes it after rerank."""
    # Build orchestrator with KG disabled (so no heavy deps needed)
    assert sample_config.kg.enabled is False

    orch = _make_orchestrator(sample_config)
    try:
        # Patch in a spy organizer directly
        spy = _SpyOrganizer()
        _SpyOrganizer.call_count = 0
        orch.mst_organizer = spy

        # Also patch LLM so no real inference happens
        orch.llm = FakeLLM()

        orch.chat("test question", user_id="default")
        assert _SpyOrganizer.call_count == 1, (
            "mst_organizer.organize should have been called exactly once"
        )
    finally:
        orch.close()
        _reset_db_singleton()


# ---------------------------------------------------------------------------
# 7. build_retriever("community") without community_store raises ValueError
# ---------------------------------------------------------------------------


def test_build_retriever_community_requires_store(sample_config, tmp_db, fake_embedder):
    """build_retriever(retriever='community') without community_store raises ValueError."""
    from hrag.retrieval.factory import build_retriever
    from hrag.retrieval.vector import VectorStore

    chroma_path = sample_config.resolve(sample_config.storage.chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    vs = VectorStore(chroma_path, fake_embedder.dim)

    sample_config.retrieval.retriever = "community"

    with pytest.raises(ValueError, match="community"):
        build_retriever(
            sample_config.retrieval,
            tmp_db,
            vs,
            fake_embedder,
            community_store=None,  # missing
        )


# ---------------------------------------------------------------------------
# 8. build_retriever("router") without llm raises ValueError
# ---------------------------------------------------------------------------


def test_build_retriever_router_requires_llm(sample_config, tmp_db, fake_embedder):
    """build_retriever(retriever='router') without llm raises ValueError."""
    from hrag.retrieval.factory import build_retriever
    from hrag.retrieval.vector import VectorStore

    chroma_path = sample_config.resolve(sample_config.storage.chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    vs = VectorStore(chroma_path, fake_embedder.dim)

    sample_config.retrieval.retriever = "router"

    with pytest.raises(ValueError, match="router"):
        build_retriever(
            sample_config.retrieval,
            tmp_db,
            vs,
            fake_embedder,
            llm=None,  # missing
        )


# ---------------------------------------------------------------------------
# 9. Orchestrator ingest pipeline receives kg_store keyword arg
# ---------------------------------------------------------------------------


def test_orchestrator_ingest_pipeline_has_kg_store_slot(sample_config):
    """IngestPipeline on the orchestrator exposes .kg_store (None when KG disabled)."""
    orch = _make_orchestrator(sample_config)
    try:
        # When KG is disabled, kg_store on the pipeline should be None.
        assert hasattr(orch.ingest, "kg_store")
        assert orch.ingest.kg_store is None
    finally:
        orch.close()
        _reset_db_singleton()


# ---------------------------------------------------------------------------
# 10. organize_done progress event is emitted when mst_organizer present
# ---------------------------------------------------------------------------


def test_orchestrator_chat_emits_organize_done_event(sample_config):
    """When mst_organizer is set, chat() emits the 'organize_done' progress event."""
    orch = _make_orchestrator(sample_config)
    try:
        spy = _SpyOrganizer()
        _SpyOrganizer.call_count = 0
        orch.mst_organizer = spy
        orch.llm = FakeLLM()

        events: list[tuple[str, dict]] = []

        def _cb(name: str, payload: dict) -> None:
            events.append((name, payload))

        orch.chat("test question", user_id="default", progress=_cb)

        org_events = [p for n, p in events if n == "organize_done"]
        assert len(org_events) == 1, "Expected exactly one 'organize_done' event"
        ev = org_events[0]
        assert "input" in ev
        assert "output" in ev
        assert "dropped" in ev
        assert ev["dropped"] == ev["input"] - ev["output"]
    finally:
        orch.close()
        _reset_db_singleton()
