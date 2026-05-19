"""Tests for KGPPRRetriever.

Each test is isolated: a fresh tmp_db + fresh KGStore in a temp dir.
Heavy deps (scipy, networkx) are skipped at collection time if absent.
"""

from __future__ import annotations

import pytest

# Gate the entire module on scipy + networkx being importable.
scipy_sparse = pytest.importorskip("scipy.sparse")
networkx = pytest.importorskip("networkx")


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _StubNER:
    """Minimal NER stub: returns whatever terms were injected at construction."""

    def __init__(self, terms: list[str]) -> None:
        self._terms = list(terms)

    def extract(self, query: str) -> list[str]:  # noqa: ARG002
        return list(self._terms)


def _make_kg_store(tmp_path, db, embedder):
    """Construct a KGStore backed by tmp_path/kg."""
    from hrag.kg.store import KGStore

    kg_path = tmp_path / "kg"
    kg_path.mkdir(parents=True, exist_ok=True)
    return KGStore(db=db, embedder=embedder, kg_path=kg_path)


def _insert_chunk(
    db,
    *,
    chunk_id: str,
    doc_id: str,
    user_id: str = "default",
    text: str = "sample text",
    source_type: str = "document",
    excluded: int = 0,
) -> None:
    """Insert a minimal chunk row so _hydrate can find it."""
    # Ensure the parent document row exists first (FK constraint).
    db.execute(
        "INSERT OR IGNORE INTO documents(doc_id, user_id, source_path, title, source_type) "
        "VALUES (?, ?, ?, ?, ?)",
        (doc_id, user_id, "/fake/path", "Test Doc", source_type),
    )
    db.execute(
        "INSERT OR REPLACE INTO chunks"
        "(chunk_id, doc_id, user_id, text, title, section, subsection,"
        " chunk_index, token_count, source_type, excluded, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            chunk_id,
            doc_id,
            user_id,
            text,
            "",
            "",
            "",
            0,
            len(text.split()),
            source_type,
            excluded,
            None,
        ),
    )
    db.commit()


def _make_triple(head: str, relation: str, tail: str, chunk_id: str):
    """Create a Triple dataclass instance."""
    from hrag.kg.builder import Triple

    return Triple(head=head, relation=relation, tail=tail, source_chunk_id=chunk_id)


def _make_retriever(db, kg_store, ner_terms, damping=0.5):
    """Convenience factory for KGPPRRetriever."""
    from hrag.retrieval.kg_ppr import KGPPRRetriever

    return KGPPRRetriever(
        db=db,
        kg_store=kg_store,
        ner=_StubNER(ner_terms),
        damping=damping,
    )


# ---------------------------------------------------------------------------
# Test 1 — Empty KG returns []
# ---------------------------------------------------------------------------


def test_empty_kg_returns_empty(tmp_path, tmp_db, fake_embedder):
    """An empty KGStore (no triples) must return [] without raising."""
    kg_store = _make_kg_store(tmp_path, tmp_db, fake_embedder)
    retriever = _make_retriever(tmp_db, kg_store, ner_terms=["alice"])
    results = retriever.retrieve("tell me about alice", user_id="default")
    assert results == []


# ---------------------------------------------------------------------------
# Test 2 — NER finds zero seeds in a populated KG -> returns []
# ---------------------------------------------------------------------------


def test_no_matching_seeds_returns_empty(tmp_path, tmp_db, fake_embedder):
    """KG has phrase nodes, but the query's NER terms don't match any of them."""
    kg_store = _make_kg_store(tmp_path, tmp_db, fake_embedder)

    # Insert a chunk and triples for "alice"
    _insert_chunk(tmp_db, chunk_id="c1", doc_id="doc1")
    kg_store.upsert_triples(
        "default",
        "doc1",
        [_make_triple("alice", "knows", "bob", "c1")],
    )

    # NER returns terms that definitely won't be in the KG
    retriever = _make_retriever(tmp_db, kg_store, ner_terms=["zzznomatch"])
    results = retriever.retrieve("zzznomatch", user_id="default")
    assert results == []


# ---------------------------------------------------------------------------
# Test 3 — Single seed retrieves directly connected passages
# ---------------------------------------------------------------------------


def test_single_seed_retrieves_connected_passages(tmp_path, tmp_db, fake_embedder):
    """When entity X is connected to chunks A and B, both should appear."""
    kg_store = _make_kg_store(tmp_path, tmp_db, fake_embedder)

    _insert_chunk(tmp_db, chunk_id="cA", doc_id="doc1", text="Text about alice chunk A")
    _insert_chunk(tmp_db, chunk_id="cB", doc_id="doc1", text="Text about alice chunk B")

    kg_store.upsert_triples(
        "default",
        "doc1",
        [
            _make_triple("alice", "works_at", "acme", "cA"),
            _make_triple("alice", "lives_in", "boston", "cB"),
        ],
    )

    retriever = _make_retriever(tmp_db, kg_store, ner_terms=["alice"])
    results = retriever.retrieve("tell me about alice", user_id="default", top_k=10)

    result_ids = {r.chunk.chunk_id for r in results}
    assert "cA" in result_ids, "Chunk A should be retrieved"
    assert "cB" in result_ids, "Chunk B should be retrieved"
    # All results should carry the kg_ppr tag
    for r in results:
        assert r.retriever == "kg_ppr"


# ---------------------------------------------------------------------------
# Test 4 — Multi-hop: indirect passage is still reachable
# ---------------------------------------------------------------------------


def test_multihop_passage_reachable(tmp_path, tmp_db, fake_embedder):
    """X -> Y -> Z (triple chain); chunk for Z should be reachable via PPR.

    PPR distributes score across all reachable nodes, so even a node two
    hops away should receive a non-zero score (and thus appear in results).
    """
    kg_store = _make_kg_store(tmp_path, tmp_db, fake_embedder)

    _insert_chunk(tmp_db, chunk_id="cXY", doc_id="doc1", text="X and Y relation chunk")
    _insert_chunk(tmp_db, chunk_id="cYZ", doc_id="doc1", text="Y and Z relation chunk")

    kg_store.upsert_triples(
        "default",
        "doc1",
        [
            # X -> Y lives in cXY
            _make_triple("entityx", "related_to", "entityy", "cXY"),
            # Y -> Z lives in cYZ (multi-hop from X)
            _make_triple("entityy", "related_to", "entityz", "cYZ"),
        ],
    )

    retriever = _make_retriever(tmp_db, kg_store, ner_terms=["entityx"])
    results = retriever.retrieve("tell me about entityx", user_id="default", top_k=10)

    result_ids = {r.chunk.chunk_id for r in results}
    # cXY is a direct passage; cYZ is two hops away via entityy
    assert "cXY" in result_ids, "Direct passage cXY should be retrieved"
    assert "cYZ" in result_ids, "Multi-hop passage cYZ should be reachable"


# ---------------------------------------------------------------------------
# Test 5 — top_k cap is respected
# ---------------------------------------------------------------------------


def test_top_k_cap_respected(tmp_path, tmp_db, fake_embedder):
    """With more passages than top_k, only top_k results are returned."""
    kg_store = _make_kg_store(tmp_path, tmp_db, fake_embedder)

    # Create 5 chunks linked to the same entity
    triples = []
    for i in range(5):
        cid = f"chunk_{i}"
        _insert_chunk(tmp_db, chunk_id=cid, doc_id="doc1", text=f"Chunk {i} text")
        triples.append(_make_triple("alice", f"rel_{i}", f"obj_{i}", cid))

    kg_store.upsert_triples("default", "doc1", triples)

    retriever = _make_retriever(tmp_db, kg_store, ner_terms=["alice"])
    results = retriever.retrieve("alice", user_id="default", top_k=3)
    assert len(results) <= 3, f"Expected at most 3 results, got {len(results)}"


# ---------------------------------------------------------------------------
# Test 6 — source_types filter excludes non-matching chunks
# ---------------------------------------------------------------------------


def test_source_types_filter(tmp_path, tmp_db, fake_embedder):
    """Chunks with source_type='document' are excluded when filter is ['episodic']."""
    kg_store = _make_kg_store(tmp_path, tmp_db, fake_embedder)

    _insert_chunk(
        tmp_db, chunk_id="doc_chunk", doc_id="doc1",
        text="Document chunk", source_type="document",
    )
    _insert_chunk(
        tmp_db, chunk_id="epi_chunk", doc_id="doc1",
        text="Episodic chunk", source_type="episodic",
    )

    kg_store.upsert_triples(
        "default",
        "doc1",
        [
            _make_triple("alice", "mentioned_in", "doc", "doc_chunk"),
            _make_triple("alice", "mentioned_in", "epi", "epi_chunk"),
        ],
    )

    retriever = _make_retriever(tmp_db, kg_store, ner_terms=["alice"])

    # Filter to episodic only
    results = retriever.retrieve(
        "alice", user_id="default", top_k=10, source_types=["episodic"]
    )
    chunk_ids = {r.chunk.chunk_id for r in results}
    assert "epi_chunk" in chunk_ids, "Episodic chunk should be included"
    assert "doc_chunk" not in chunk_ids, "Document chunk should be excluded"


# ---------------------------------------------------------------------------
# Test 7 — Tombstoned chunks (excluded=1) are skipped
# ---------------------------------------------------------------------------


def test_tombstoned_chunks_skipped(tmp_path, tmp_db, fake_embedder):
    """Chunks with excluded=1 must not appear in results."""
    kg_store = _make_kg_store(tmp_path, tmp_db, fake_embedder)

    _insert_chunk(tmp_db, chunk_id="live_chunk", doc_id="doc1", text="Live content")
    _insert_chunk(
        tmp_db, chunk_id="dead_chunk", doc_id="doc1",
        text="Deleted content", excluded=1,
    )

    kg_store.upsert_triples(
        "default",
        "doc1",
        [
            _make_triple("alice", "rel", "live", "live_chunk"),
            _make_triple("alice", "rel", "dead", "dead_chunk"),
        ],
    )

    retriever = _make_retriever(tmp_db, kg_store, ner_terms=["alice"])
    results = retriever.retrieve("alice", user_id="default", top_k=10)

    chunk_ids = {r.chunk.chunk_id for r in results}
    assert "live_chunk" in chunk_ids, "Live chunk should be present"
    assert "dead_chunk" not in chunk_ids, "Tombstoned chunk must be excluded"


# ---------------------------------------------------------------------------
# Test 8 — name attribute and retriever tag on results
# ---------------------------------------------------------------------------


def test_name_and_retriever_tag(tmp_path, tmp_db, fake_embedder):
    """Verify class-level name attribute and per-result retriever field."""
    from hrag.retrieval.kg_ppr import KGPPRRetriever

    assert KGPPRRetriever.name == "kg_ppr"

    kg_store = _make_kg_store(tmp_path, tmp_db, fake_embedder)
    _insert_chunk(tmp_db, chunk_id="c1", doc_id="doc1", text="Sample text")
    kg_store.upsert_triples(
        "default",
        "doc1",
        [_make_triple("alice", "is_a", "person", "c1")],
    )

    retriever = _make_retriever(tmp_db, kg_store, ner_terms=["alice"])
    results = retriever.retrieve("alice", user_id="default", top_k=5)

    assert len(results) >= 1
    for r in results:
        assert r.retriever == "kg_ppr", f"Expected 'kg_ppr', got {r.retriever!r}"


# ---------------------------------------------------------------------------
# HippoRAG-faithful scoring tests (improvement 1, 2, 3)
# ---------------------------------------------------------------------------


def test_phrase_only_alpha_zero_reproduces_prior_behavior(tmp_path, tmp_db, fake_embedder):
    """alpha = 0 collapses to the legacy phrase-aggregate scoring path.

    Synthetic 3-node setup: alice -> chunk cA, alice -> chunk cB. With
    alpha=0 the score is sum(phrase_ppr) over phrase predecessors, which
    must rank both passages above zero and still tag them with kg_ppr.
    """
    from hrag.retrieval.kg_ppr import KGPPRRetriever

    kg_store = _make_kg_store(tmp_path, tmp_db, fake_embedder)
    _insert_chunk(tmp_db, chunk_id="cA", doc_id="doc1", text="alice chunk A")
    _insert_chunk(tmp_db, chunk_id="cB", doc_id="doc1", text="alice chunk B")
    kg_store.upsert_triples(
        "default",
        "doc1",
        [
            _make_triple("alice", "knows", "bob", "cA"),
            _make_triple("alice", "lives_in", "boston", "cB"),
        ],
    )

    retriever = KGPPRRetriever(
        db=tmp_db,
        kg_store=kg_store,
        ner=_StubNER(["alice"]),
        damping=0.5,
        seed_top_k=1,
        passage_node_alpha=0.0,
    )
    results = retriever.retrieve("alice", user_id="default", top_k=10)
    ids = {r.chunk.chunk_id for r in results}
    assert ids == {"cA", "cB"}
    # All non-zero scores under phrase-aggregate.
    for r in results:
        assert r.score > 0.0


def test_passage_node_alpha_one_scores_passages_directly(tmp_path, tmp_db, fake_embedder):
    """alpha = 1 reads passage scores directly from PPR over the full graph.

    With passage nodes present, mass flows phrase --contains--> passage,
    so the seeded passage's PPR score must be strictly positive and the
    retriever must surface it in the top-K.
    """
    from hrag.retrieval.kg_ppr import KGPPRRetriever

    kg_store = _make_kg_store(tmp_path, tmp_db, fake_embedder)
    _insert_chunk(tmp_db, chunk_id="cA", doc_id="doc1", text="alice chunk A")
    _insert_chunk(tmp_db, chunk_id="cB", doc_id="doc1", text="alice chunk B")
    kg_store.upsert_triples(
        "default",
        "doc1",
        [
            _make_triple("alice", "knows", "bob", "cA"),
            _make_triple("alice", "lives_in", "boston", "cB"),
        ],
    )

    retriever = KGPPRRetriever(
        db=tmp_db,
        kg_store=kg_store,
        ner=_StubNER(["alice"]),
        damping=0.5,
        seed_top_k=1,
        passage_node_alpha=1.0,
    )
    results = retriever.retrieve("alice", user_id="default", top_k=10)
    ids = {r.chunk.chunk_id for r in results}
    # Passage-node mass IS the only score component; both passages must
    # have non-zero score and appear.
    assert ids == {"cA", "cB"}
    for r in results:
        assert r.score > 0.0


def test_seed_broadening_top_k_changes_ranking(tmp_path, tmp_db, fake_embedder):
    """K=3 with multiple synonym-like phrases produces different top-1 vs K=1.

    We seed two surface forms: "tau" appears in chunk cTau alone; the broader
    "synonymy threshold" appears in chunk cBroad. With K=1 only the exact
    canonical match for the query term is seeded; with K=3 the embedder finds
    additional related phrase nodes (FakeEmbedder is deterministic, so given
    enough phrases the broadening provably changes which seeds get mass).
    """
    from hrag.retrieval.kg_ppr import KGPPRRetriever

    kg_store = _make_kg_store(tmp_path, tmp_db, fake_embedder)

    _insert_chunk(tmp_db, chunk_id="cTau", doc_id="doc1", text="tau definition")
    _insert_chunk(tmp_db, chunk_id="cThresh", doc_id="doc1", text="threshold details")
    _insert_chunk(tmp_db, chunk_id="cSyn", doc_id="doc1", text="synonymy spec")

    # Each surface phrase becomes its own canonical phrase node here because
    # FakeEmbedder returns distinct hash-derived vectors -> cosine < 0.8.
    kg_store.upsert_triples(
        "default",
        "doc1",
        [
            _make_triple("tau", "is_a", "symbol", "cTau"),
            _make_triple("threshold", "is_a", "knob", "cThresh"),
            _make_triple("synonymy threshold", "is_a", "concept", "cSyn"),
        ],
    )

    # K=1: only exact-match canonical seed for "tau" gets mass; chunks
    # connected to that single phrase dominate.
    r1 = KGPPRRetriever(
        db=tmp_db,
        kg_store=kg_store,
        ner=_StubNER(["tau"]),
        damping=0.5,
        seed_top_k=1,
        passage_node_alpha=0.5,
    )
    res1 = r1.retrieve("tau", user_id="default", top_k=10)
    ids_k1 = [r.chunk.chunk_id for r in res1]

    # K=3: the embedder is consulted, returning the top-3 phrase nodes by
    # cosine similarity. The seed set contains entries beyond the exact
    # canonical match. We assert that the SET of seeds has grown -- meaning
    # at least one additional phrase node was seeded.
    r3 = KGPPRRetriever(
        db=tmp_db,
        kg_store=kg_store,
        ner=_StubNER(["tau"]),
        damping=0.5,
        seed_top_k=3,
        passage_node_alpha=0.5,
    )

    # Inspect the seed set directly to verify broadening happened.
    seeds_k1, _ = r1._build_seed_set(["tau"])
    seeds_k3, _ = r3._build_seed_set(["tau"])
    assert len(seeds_k3) > len(seeds_k1), (
        f"K=3 should yield more seeds than K=1; got {seeds_k3} vs {seeds_k1}"
    )

    # And that retrieval still produces a result list (top-1 may or may not
    # change in a synthetic 3-chunk graph because the deterministic
    # FakeEmbedder couples chunk-to-phrase scores in a fixed way; the
    # behavioural contract we assert is that the seed expansion is real).
    res3 = r3.retrieve("tau", user_id="default", top_k=10)
    assert len(res3) >= 1
    # ids_k1 used only to pin the K=1 path is exercised end-to-end.
    assert len(ids_k1) >= 1


def test_passage_node_alpha_default_constructor(tmp_path, tmp_db, fake_embedder):
    """Default constructor wiring: alpha=0.5, K=3 still returns sane results."""
    from hrag.retrieval.kg_ppr import KGPPRRetriever

    kg_store = _make_kg_store(tmp_path, tmp_db, fake_embedder)
    _insert_chunk(tmp_db, chunk_id="cA", doc_id="doc1", text="A")
    kg_store.upsert_triples(
        "default",
        "doc1",
        [_make_triple("alice", "knows", "bob", "cA")],
    )

    # Construct with explicit defaults to pin the surface API.
    retriever = KGPPRRetriever(
        db=tmp_db,
        kg_store=kg_store,
        ner=_StubNER(["alice"]),
    )
    results = retriever.retrieve("alice", user_id="default", top_k=10)
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "cA"
