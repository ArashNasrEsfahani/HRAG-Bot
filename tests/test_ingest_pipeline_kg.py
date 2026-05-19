"""Tests for the KG-layer integration in IngestPipeline (Phase 2 wiring).

These tests cover the new step 5b added to ingest_document:
  - kg.enabled=False  → no extraction, no KG rows
  - kg.enabled=True, llm=None → warning logged, ingest still succeeds
  - kg.enabled=True, full deps  → triples extracted, SQLite rows created
  - Re-ingest idempotency: counts don't double on second ingest of same doc
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

# KGStore + networkx are heavy deps — skip when absent.
# NOTE: we still import the pipeline itself (which imports nothing heavy at
# module top) — those tests run anywhere.
networkx = pytest.importorskip("networkx")
pytest.importorskip("numpy")

from hrag.kg.store import KGStore  # noqa: E402  — only reachable when nx present


# ---------------------------------------------------------------------------
# Minimal stub LLM for KG tests
# ---------------------------------------------------------------------------


class _StubLLM:
    """LLM that always returns a fixed JSON triple list."""

    name = "stub_kg"

    def __init__(self, output: str = '[{"head":"A","relation":"describes","tail":"B"}]') -> None:
        self._output = output
        self.calls: list[str] = []

    def complete(self, prompt: str, **_kwargs: Any) -> str:  # noqa: D401
        self.calls.append(prompt)
        return self._output

    # generate() is not used by TripleExtractor, but include it for interface
    # completeness in case of future introspection.
    def generate(self, request: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_fake_vector_store():
    """Return a minimal VectorStore duck-type that records calls."""

    class _FakeVS:
        def __init__(self):
            self.added: list = []
            self.deleted: list = []

        def add_chunks(self, user_id, chunks, embeddings):
            self.added.append((user_id, chunks, embeddings))

        def delete_doc(self, user_id, doc_id):
            self.deleted.append((user_id, doc_id))

    return _FakeVS()


def _build_pipeline(sample_config, tmp_db, fake_embedder, *, kg_enabled: bool, llm=None, kg_store=None):
    """Convenience: build IngestPipeline with the desired KG config.

    Disables the quality filter so the short sample_md fixture isn't silently
    dropped — we want chunks to reach the KG extraction step.
    """
    from hrag.ingest.pipeline import IngestPipeline

    sample_config.kg.enabled = kg_enabled
    sample_config.chunking.quality.enabled = False  # don't drop the short sample chunks
    vs = _make_fake_vector_store()
    return IngestPipeline(
        sample_config,
        tmp_db,
        fake_embedder,
        vs,
        llm=llm,
        kg_store=kg_store,
    )


def _kg_node_count(db) -> int:
    cur = db.execute("SELECT COUNT(*) AS cnt FROM kg_nodes")
    return cur.fetchone()["cnt"]


def _kg_edge_count(db) -> int:
    cur = db.execute("SELECT COUNT(*) AS cnt FROM kg_edges")
    return cur.fetchone()["cnt"]


# ---------------------------------------------------------------------------
# Test 1: kg.enabled=False — no KG extraction at all
# ---------------------------------------------------------------------------


def test_kg_disabled_no_rows(tmp_db, fake_embedder, sample_config, sample_md_path):
    """When kg.enabled=False the pipeline must not touch kg_nodes or kg_edges."""
    pipeline = _build_pipeline(sample_config, tmp_db, fake_embedder, kg_enabled=False)
    pipeline.ingest_path(sample_md_path, "default")

    assert _kg_node_count(tmp_db) == 0
    assert _kg_edge_count(tmp_db) == 0


# ---------------------------------------------------------------------------
# Test 2: kg.enabled=True but llm=None → warning + ingest succeeds
# ---------------------------------------------------------------------------


def test_kg_enabled_no_llm_warns_and_succeeds(
    tmp_db, fake_embedder, sample_config, sample_md_path, caplog
):
    """Pipeline should log a warning and continue without crashing."""
    pipeline = _build_pipeline(
        sample_config, tmp_db, fake_embedder, kg_enabled=True, llm=None, kg_store=None
    )

    with caplog.at_level(logging.WARNING, logger="hrag.ingest.pipeline"):
        pipeline.ingest_path(sample_md_path, "default")

    # Warning was emitted
    assert any("kg.enabled=True" in r.message and "llm or kg_store is None" in r.message
               for r in caplog.records), (
        f"Expected warning not found in log records: {[r.message for r in caplog.records]}"
    )

    # Vector store got populated (ingest did succeed)
    cur = tmp_db.execute("SELECT COUNT(*) AS cnt FROM chunks")
    assert cur.fetchone()["cnt"] > 0

    # No KG rows written
    assert _kg_node_count(tmp_db) == 0


# ---------------------------------------------------------------------------
# Test 3: kg.enabled=True with full deps → rows created
# ---------------------------------------------------------------------------


def test_kg_enabled_full_deps_creates_rows(
    tmp_db, fake_embedder, sample_config, sample_md_path, tmp_path
):
    """Full KG path: triples extracted and mirrored into kg_nodes / kg_edges."""
    tmp_db.ensure_user("default")
    tmp_db.commit()

    stub_llm = _StubLLM('[{"head":"Introduction","relation":"describes","tail":"Background"}]')
    kg_store = KGStore(
        db=tmp_db,
        embedder=fake_embedder,
        kg_path=tmp_path / "kg",
        synonym_threshold=0.99,  # high threshold → no synonym merging surprises
    )

    pipeline = _build_pipeline(
        sample_config,
        tmp_db,
        fake_embedder,
        kg_enabled=True,
        llm=stub_llm,
        kg_store=kg_store,
    )
    pipeline.ingest_path(sample_md_path, "default")

    # At least one node and one edge must have been written
    assert _kg_node_count(tmp_db) > 0, "Expected kg_nodes rows after ingest"
    assert _kg_edge_count(tmp_db) > 0, "Expected kg_edges rows after ingest"

    # LLM was actually called (at least once per chunk)
    assert len(stub_llm.calls) > 0


# ---------------------------------------------------------------------------
# Test 4: Re-ingest is idempotent for KG
# ---------------------------------------------------------------------------


def test_kg_reupsert_is_idempotent(
    tmp_db, fake_embedder, sample_config, sample_md_path, tmp_path
):
    """Ingesting the same doc twice must not double the KG rows."""
    tmp_db.ensure_user("default")
    tmp_db.commit()

    stub_llm = _StubLLM('[{"head":"A","relation":"describes","tail":"B"}]')
    kg_store = KGStore(
        db=tmp_db,
        embedder=fake_embedder,
        kg_path=tmp_path / "kg",
        synonym_threshold=0.99,
    )

    pipeline = _build_pipeline(
        sample_config,
        tmp_db,
        fake_embedder,
        kg_enabled=True,
        llm=stub_llm,
        kg_store=kg_store,
    )

    # First ingest
    pipeline.ingest_path(sample_md_path, "default")
    nodes_after_first = _kg_node_count(tmp_db)
    edges_after_first = _kg_edge_count(tmp_db)

    assert nodes_after_first > 0, "Sanity: first ingest should create KG rows"

    # Second ingest of the same file
    pipeline.ingest_path(sample_md_path, "default")
    nodes_after_second = _kg_node_count(tmp_db)
    edges_after_second = _kg_edge_count(tmp_db)

    # Counts must not have doubled (upsert_triples wipes-then-adds per doc)
    assert nodes_after_second == nodes_after_first, (
        f"kg_nodes doubled: {nodes_after_first} → {nodes_after_second}"
    )
    assert edges_after_second == edges_after_first, (
        f"kg_edges doubled: {edges_after_first} → {edges_after_second}"
    )


# ---------------------------------------------------------------------------
# Test 5: backward-compat — positional-only constructor still works
# ---------------------------------------------------------------------------


def test_pipeline_positional_args_still_work(tmp_db, fake_embedder, sample_config):
    """IngestPipeline(config, db, embedder, vs) must keep working (no llm/kg_store)."""
    from hrag.ingest.pipeline import IngestPipeline

    vs = _make_fake_vector_store()
    # No keyword args for llm / kg_store — should not raise
    pipeline = IngestPipeline(sample_config, tmp_db, fake_embedder, vs)
    assert pipeline.llm is None
    assert pipeline.kg_store is None
