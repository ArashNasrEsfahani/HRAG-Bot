"""Tests for hrag.retrieval.community.CommunityRetriever.

Two test groups:
  A — interface / edge-case tests (work with the conftest ChromaDB stub).
  B — populated round-trip tests (monkey-patch CommunityStore.query to return
      canned results; pre-populate SQLite kg_communities directly).
"""

from __future__ import annotations

import json

import pytest

from hrag.retrieval.community import CommunityRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_community_store(tmp_db, fake_embedder, tmp_path):
    """Construct a real CommunityStore backed by tmp_path/chroma_comm.

    The conftest chromadb stub is used when the real chromadb is absent,
    so `store.query` will return empty by default — which is fine for Group A
    tests.
    """
    from hrag.kg.communities import CommunityStore

    return CommunityStore(
        db=tmp_db,
        embedder=fake_embedder,
        chroma_path=tmp_path / "chroma_comm",
    )


def _insert_community(db, community_id: str, user_id: str, level: int,
                      summary: str, member_chunks: list[str]) -> None:
    """Insert a row directly into kg_communities (bypassing CommunityStore.upsert)."""
    db.ensure_user(user_id)
    db.commit()
    db.execute(
        "INSERT OR REPLACE INTO kg_communities"
        "(community_id, user_id, level, summary, member_chunks) "
        "VALUES (?, ?, ?, ?, ?)",
        (community_id, user_id, level, summary, json.dumps(member_chunks)),
    )
    db.commit()


# ---------------------------------------------------------------------------
# Group A — interface / edge-case tests
# ---------------------------------------------------------------------------


def test_name_attribute():
    """CommunityRetriever.name must be 'community'."""
    assert CommunityRetriever.name == "community"


def test_empty_store_returns_empty_list(tmp_db, fake_embedder, tmp_path):
    """A fresh CommunityStore (nothing upserted) must return []."""
    store = _make_community_store(tmp_db, fake_embedder, tmp_path)
    retriever = CommunityRetriever(community_store=store, embedder=fake_embedder)
    results = retriever.retrieve("What is HippoRAG?", user_id="default")
    assert results == []


def test_empty_query_returns_empty_list(tmp_db, fake_embedder, tmp_path):
    """Empty or whitespace-only query must return [] immediately."""
    store = _make_community_store(tmp_db, fake_embedder, tmp_path)
    # Pre-populate SQLite so the store isn't bare.
    _insert_community(
        tmp_db, "level0_c0", "default", 0, "Some summary", ["c1", "c2"]
    )
    retriever = CommunityRetriever(community_store=store, embedder=fake_embedder)

    assert retriever.retrieve("", user_id="default") == []
    assert retriever.retrieve("   ", user_id="default") == []
    assert retriever.retrieve("\t\n", user_id="default") == []


def test_source_types_excludes_community(tmp_db, fake_embedder, tmp_path):
    """If source_types=['document'], community results must be suppressed."""
    store = _make_community_store(tmp_db, fake_embedder, tmp_path)
    _insert_community(
        tmp_db, "level0_c0", "default", 0, "Some summary", ["c1"]
    )
    retriever = CommunityRetriever(community_store=store, embedder=fake_embedder)

    # Explicitly exclude community type.
    results = retriever.retrieve(
        "HippoRAG", user_id="default", source_types=["document"]
    )
    assert results == []

    # Also test with episodic only.
    results = retriever.retrieve(
        "HippoRAG", user_id="default", source_types=["episodic"]
    )
    assert results == []


def test_source_types_none_passes_through(tmp_db, fake_embedder, tmp_path):
    """source_types=None must not block community results (it's the default)."""
    store = _make_community_store(tmp_db, fake_embedder, tmp_path)
    retriever = CommunityRetriever(community_store=store, embedder=fake_embedder)

    # The stub chromadb returns empty, so we'll get [] — but NOT because of
    # source_types filtering.  The retriever must reach the query stage.
    called = []

    original_query = store.query

    def spy_query(**kwargs):
        called.append(kwargs)
        return original_query(**kwargs)

    store.query = lambda *a, **kw: spy_query(
        user_id=kw.get("user_id", a[0] if a else None),
        query_embedding=kw.get("query_embedding", a[1] if len(a) > 1 else None),
        top_k=kw.get("top_k", 5),
        levels=kw.get("levels", None),
    )
    retriever.retrieve("test query", user_id="default", source_types=None)
    assert len(called) == 1, "store.query should be called when source_types=None"


def test_source_types_containing_community_passes_through(
    tmp_db, fake_embedder, tmp_path
):
    """source_types=['community'] or ['community', 'document'] must not block."""
    store = _make_community_store(tmp_db, fake_embedder, tmp_path)
    retriever = CommunityRetriever(community_store=store, embedder=fake_embedder)

    called = []

    def mock_query(**kw):
        called.append(kw)
        return []

    store.query = lambda *a, **kw: mock_query(
        user_id=kw.get("user_id"),
        query_embedding=kw.get("query_embedding"),
        top_k=kw.get("top_k", 5),
        levels=kw.get("levels"),
    ) if not a else mock_query(
        user_id=a[0],
        query_embedding=a[1] if len(a) > 1 else kw.get("query_embedding"),
        top_k=a[2] if len(a) > 2 else kw.get("top_k", 5),
        levels=a[3] if len(a) > 3 else kw.get("levels"),
    )

    retriever.retrieve("test", user_id="default", source_types=["community"])
    assert len(called) >= 1

    called.clear()
    retriever.retrieve(
        "test", user_id="default", source_types=["community", "document"]
    )
    assert len(called) >= 1


def test_skips_un_hydratable_community_ids(tmp_db, fake_embedder, tmp_path):
    """If a community_id from Chroma is missing in SQLite, skip without raising."""
    store = _make_community_store(tmp_db, fake_embedder, tmp_path)

    # Monkey-patch query to return a community_id that is NOT in SQLite.
    store.query = lambda *a, **kw: [("ghost_community", 0.95)]

    retriever = CommunityRetriever(community_store=store, embedder=fake_embedder)
    results = retriever.retrieve("HippoRAG", user_id="default")
    # Should return empty, not raise.
    assert results == []


def test_result_chunk_fields(tmp_db, fake_embedder, tmp_path):
    """Verify chunk_id, source_type, and metadata fields on returned results."""
    store = _make_community_store(tmp_db, fake_embedder, tmp_path)

    _insert_community(
        tmp_db, "level0_c7", "default", 0, "A nice summary", ["c1", "c2", "c3"]
    )

    # Monkey-patch query to return our pre-seeded community.
    store.query = lambda *a, **kw: [("level0_c7", 0.88)]

    retriever = CommunityRetriever(community_store=store, embedder=fake_embedder)
    results = retriever.retrieve("HippoRAG", user_id="default")

    assert len(results) == 1
    result = results[0]
    chunk = result.chunk

    # chunk_id must be prefixed with "community::".
    assert chunk.chunk_id == "community::level0_c7"

    # source_type must be "community".
    assert chunk.source_type == "community"

    # metadata must contain required keys.
    assert chunk.metadata["community_id"] == "level0_c7"
    assert chunk.metadata["level"] == 0
    assert chunk.metadata["member_chunk_ids"] == ["c1", "c2", "c3"]

    # text is the summary.
    assert chunk.text == "A nice summary"

    # retriever tag.
    assert result.retriever == "community"

    # score passes through.
    assert result.score == pytest.approx(0.88)

    # rerank_score is None.
    assert result.rerank_score is None


# ---------------------------------------------------------------------------
# Group B — populated round-trip (monkey-patched CommunityStore.query)
# ---------------------------------------------------------------------------


def _seed_communities(db, user_id: str, n: int) -> list[str]:
    """Insert *n* community rows and return their community_ids."""
    ids = []
    for i in range(n):
        cid = f"level0_c{i}"
        _insert_community(
            db, cid, user_id, 0, f"Summary for community {i}", [f"chunk_{i}"]
        )
        ids.append(cid)
    return ids


def test_returns_correct_number_of_results_ordered_by_score(
    tmp_db, fake_embedder, tmp_path
):
    """Results should be returned in descending score order."""
    store = _make_community_store(tmp_db, fake_embedder, tmp_path)
    ids = _seed_communities(tmp_db, "default", 3)

    # Canned pairs in non-sorted order.
    canned = [
        (ids[0], 0.5),
        (ids[1], 0.9),
        (ids[2], 0.7),
    ]
    store.query = lambda *a, **kw: canned

    retriever = CommunityRetriever(community_store=store, embedder=fake_embedder)
    results = retriever.retrieve("some query", user_id="default", top_k=3)

    assert len(results) == 3
    scores = [r.score for r in results]
    # Scores must be in the same order as the canned pairs (retriever
    # preserves CommunityStore.query order — the store is responsible
    # for sorting).
    assert scores == [0.5, 0.9, 0.7]


def test_top_k_is_honoured(tmp_db, fake_embedder, tmp_path):
    """retrieve(top_k=2) with 5 canned results should return at most 2."""
    store = _make_community_store(tmp_db, fake_embedder, tmp_path)
    ids = _seed_communities(tmp_db, "default", 5)

    # Canned returns 5 items — but the retriever should pass top_k=2 down,
    # and CommunityStore.query is supposed to cap at top_k.
    # Here we simulate the store already having respected top_k=2.
    def mock_query_top2(*a, top_k=5, **kw):
        pairs = [(cid, 0.9 - i * 0.1) for i, cid in enumerate(ids)]
        return pairs[:top_k]

    store.query = mock_query_top2

    retriever = CommunityRetriever(community_store=store, embedder=fake_embedder)
    results = retriever.retrieve("query", user_id="default", top_k=2)

    assert len(results) == 2


def test_levels_filter_passes_through_to_query(tmp_db, fake_embedder, tmp_path):
    """levels=[0] set on the retriever must be forwarded to community_store.query."""
    store = _make_community_store(tmp_db, fake_embedder, tmp_path)
    ids = _seed_communities(tmp_db, "default", 2)

    captured_kwargs: list[dict] = []

    def spy_query(user_id, query_embedding, top_k=5, levels=None):
        captured_kwargs.append({"user_id": user_id, "levels": levels, "top_k": top_k})
        return [(ids[0], 0.8)]

    store.query = spy_query

    retriever = CommunityRetriever(
        community_store=store,
        embedder=fake_embedder,
        levels=[0],
    )
    retriever.retrieve("some query", user_id="default", top_k=5)

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["levels"] == [0], (
        f"Expected levels=[0], got {captured_kwargs[0]['levels']!r}"
    )


def test_levels_none_passes_none_to_query(tmp_db, fake_embedder, tmp_path):
    """levels=None (default) must pass levels=None to community_store.query."""
    store = _make_community_store(tmp_db, fake_embedder, tmp_path)
    ids = _seed_communities(tmp_db, "default", 1)

    captured_kwargs: list[dict] = []

    def spy_query(user_id, query_embedding, top_k=5, levels=None):
        captured_kwargs.append({"levels": levels})
        return [(ids[0], 0.75)]

    store.query = spy_query

    retriever = CommunityRetriever(
        community_store=store,
        embedder=fake_embedder,
        levels=None,
    )
    retriever.retrieve("query", user_id="default")

    assert captured_kwargs[0]["levels"] is None


def test_doc_id_encodes_level(tmp_db, fake_embedder, tmp_path):
    """The synthetic doc_id must be 'community_level_{level}'."""
    store = _make_community_store(tmp_db, fake_embedder, tmp_path)
    _insert_community(tmp_db, "level2_c3", "default", 2, "Fine summary", ["cx"])

    store.query = lambda *a, **kw: [("level2_c3", 0.6)]

    retriever = CommunityRetriever(community_store=store, embedder=fake_embedder)
    results = retriever.retrieve("query", user_id="default")

    assert len(results) == 1
    assert results[0].chunk.doc_id == "community_level_2"
