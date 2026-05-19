"""Tests for hrag.kg.communities — Leiden clustering + summarisation."""

from __future__ import annotations

import json
import threading
from typing import Sequence

import pytest

# Heavy deps required at runtime for these tests.
pytest.importorskip("networkx")
pytest.importorskip("numpy")
pytest.importorskip("scipy.sparse")

from hrag.kg.builder import Triple  # noqa: E402
from hrag.kg.communities import (  # noqa: E402
    Community,
    CommunityDetector,
    CommunityStore,
    CommunitySummarizer,
    _format_member_passages,
    _phrase_subgraph,
    detect_and_summarize,
)
from hrag.kg.store import KGStore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Embedder:
    """Tiny deterministic embedder good enough for KGStore + community tests."""

    name = "fake"
    _DIM = 8

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(b / 255.0) * 2.0 - 1.0 for b in digest[: self._DIM]]

    @property
    def dim(self) -> int:
        return self._DIM


def _seed_chunk_row(db, chunk_id: str, doc_id: str, user_id: str, text: str,
                     title: str = "", section: str = "") -> None:
    """Insert a chunks row needed for summarizer hydration."""
    db.execute(
        "INSERT OR REPLACE INTO documents(doc_id, user_id, source_path, title) "
        "VALUES (?, ?, ?, ?)",
        (doc_id, user_id, f"/tmp/{doc_id}", title or doc_id),
    )
    db.execute(
        "INSERT OR REPLACE INTO chunks"
        "(chunk_id, doc_id, user_id, text, title, section, chunk_index, token_count) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
        (chunk_id, doc_id, user_id, text, title, section),
    )
    db.commit()


def _build_two_cluster_kg(tmp_db, tmp_path):
    """Build a KG with two well-separated phrase clusters and seed chunks.

    Each cluster's triples are spread across ``c1a/c1b`` (cluster A) and
    ``c2a/c2b`` (cluster B), so each community ends up with >=2 supporting
    chunks — i.e. above the default ``min_chunk_members`` threshold.
    """
    tmp_db.ensure_user("u1")
    tmp_db.commit()
    store = KGStore(
        db=tmp_db,
        embedder=_Embedder(),
        kg_path=tmp_path / "kg",
        synonym_threshold=0.99,  # keep all phrases distinct
    )

    # Cluster A: HippoRAG entities densely connected, spread across 2 chunks.
    triples_a = [
        Triple(head="HippoRAG", relation="uses", tail="PageRank",
               source_chunk_id="c1a"),
        Triple(head="HippoRAG", relation="models", tail="Hippocampus",
               source_chunk_id="c1a"),
        Triple(head="PageRank", relation="is", tail="Algorithm",
               source_chunk_id="c1b"),
        Triple(head="Hippocampus", relation="is", tail="BrainRegion",
               source_chunk_id="c1b"),
        Triple(head="PageRank", relation="supports", tail="HippoRAG",
               source_chunk_id="c1b"),
    ]

    # Cluster B: RAGate entities densely connected, spread across 2 chunks.
    triples_b = [
        Triple(head="RAGate", relation="extends", tail="RAG",
               source_chunk_id="c2a"),
        Triple(head="RAGate", relation="uses", tail="GatingNetwork",
               source_chunk_id="c2a"),
        Triple(head="RAG", relation="uses", tail="Retriever",
               source_chunk_id="c2b"),
        Triple(head="GatingNetwork", relation="controls", tail="Retriever",
               source_chunk_id="c2b"),
        Triple(head="RAGate", relation="augments", tail="GatingNetwork",
               source_chunk_id="c2b"),
    ]

    store.upsert_triples("u1", "docA", triples_a)
    store.upsert_triples("u1", "docB", triples_b)

    # Seed chunks rows so the summarizer can hydrate them.
    _seed_chunk_row(
        tmp_db, "c1a", "docA", "u1",
        "HippoRAG uses PageRank to traverse a hippocampus-inspired index.",
        title="HippoRAG paper", section="Method",
    )
    _seed_chunk_row(
        tmp_db, "c1b", "docA", "u1",
        "PageRank is an algorithm; the hippocampus is a brain region.",
        title="HippoRAG paper", section="Background",
    )
    _seed_chunk_row(
        tmp_db, "c2a", "docB", "u1",
        "RAGate extends RAG with a gating network deciding when to retrieve.",
        title="RAGate paper", section="Method",
    )
    _seed_chunk_row(
        tmp_db, "c2b", "docB", "u1",
        "The gating network controls a retriever inside the RAG pipeline.",
        title="RAGate paper", section="Architecture",
    )

    return store


# ---------------------------------------------------------------------------
# CommunityDetector
# ---------------------------------------------------------------------------


def test_detector_returns_empty_when_graph_too_small(tmp_db, tmp_path):
    tmp_db.ensure_user("u1")
    tmp_db.commit()
    store = KGStore(
        db=tmp_db,
        embedder=_Embedder(),
        kg_path=tmp_path / "kg",
        synonym_threshold=0.99,
    )
    # Only 2 phrase nodes — below the _MIN_CLUSTER_SIZE threshold.
    store.upsert_triples(
        "u1", "doc1",
        [Triple(head="A", relation="r", tail="B", source_chunk_id="c1")],
    )
    detector = CommunityDetector(store, levels=[0])
    assert detector.detect("u1") == []


def test_detect_two_clusters(tmp_db, tmp_path):
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")

    store = _build_two_cluster_kg(tmp_db, tmp_path)
    detector = CommunityDetector(store, levels=[0])
    communities = detector.detect("u1")

    # Expect at least 2 communities at level 0 (the two disconnected clusters).
    level0 = [c for c in communities if c.level == 0]
    assert len(level0) >= 2
    for c in level0:
        assert c.community_id.startswith("level0_c")
        assert len(c.member_phrase_node_ids) >= 3


def test_communities_have_chunk_ids(tmp_db, tmp_path):
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")

    store = _build_two_cluster_kg(tmp_db, tmp_path)
    detector = CommunityDetector(store, levels=[0])
    communities = detector.detect("u1")

    assert any(c.member_chunk_ids for c in communities)
    found_chunks: set[str] = set()
    for c in communities:
        found_chunks.update(c.member_chunk_ids)
    assert found_chunks & {"c1a", "c1b", "c2a", "c2b"}


def _build_chunk_singleton_kg(tmp_db, tmp_path, n_chunks: int):
    """Build a KG where one Leiden cluster (>=4 phrase nodes) only references
    *n_chunks* distinct chunks via 'contains' edges.
    """
    tmp_db.ensure_user("u1")
    tmp_db.commit()
    store = KGStore(
        db=tmp_db,
        embedder=_Embedder(),
        kg_path=tmp_path / "kg",
        synonym_threshold=0.99,
    )

    # Cluster: 4 phrase nodes densely connected. Distribute the triples
    # across exactly *n_chunks* chunk ids so the resulting community has
    # `n_chunks` supporting passages.
    triples = [
        ("Alpha", "rel", "Beta"),
        ("Beta", "rel", "Gamma"),
        ("Gamma", "rel", "Delta"),
        ("Delta", "rel", "Alpha"),
        ("Alpha", "rel", "Gamma"),
        ("Beta", "rel", "Delta"),
    ]
    chunk_ids = [f"chunk_{i}" for i in range(n_chunks)]
    triple_objs = [
        Triple(
            head=h,
            relation=r,
            tail=t,
            source_chunk_id=chunk_ids[i % n_chunks],
        )
        for i, (h, r, t) in enumerate(triples)
    ]
    store.upsert_triples("u1", "docX", triple_objs)

    for cid in chunk_ids:
        _seed_chunk_row(
            tmp_db, cid, "docX", "u1",
            f"text for {cid}",
            title="X paper", section="S",
        )

    return store


def test_drops_chunk_singletons(tmp_db, tmp_path):
    """A community with only 1 supporting chunk is filtered out by default."""
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")

    store = _build_chunk_singleton_kg(tmp_db, tmp_path, n_chunks=1)
    detector = CommunityDetector(store, levels=[0])
    communities = detector.detect("u1")

    # With only 1 supporting chunk and the default min_chunk_members=2,
    # the cluster is dropped.
    assert communities == []


def test_keeps_when_above_threshold(tmp_db, tmp_path):
    """A community with >= min_chunk_members supporting chunks survives."""
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")

    store = _build_chunk_singleton_kg(tmp_db, tmp_path, n_chunks=3)
    detector = CommunityDetector(store, levels=[0])
    communities = detector.detect("u1")

    assert len(communities) >= 1
    for c in communities:
        assert len(c.member_chunk_ids) >= 2


def test_min_chunk_members_constructor_arg(tmp_db, tmp_path):
    """Setting min_chunk_members=1 retains chunk-singletons (back-compat)."""
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")

    store = _build_chunk_singleton_kg(tmp_db, tmp_path, n_chunks=1)
    detector = CommunityDetector(store, levels=[0], min_chunk_members=1)
    communities = detector.detect("u1")

    # With the lenient threshold the singleton cluster survives.
    assert len(communities) >= 1
    assert any(len(c.member_chunk_ids) == 1 for c in communities)


def test_summarizer_call_count_drops_with_filter(tmp_db, tmp_path, fake_llm,
                                                  fake_embedder):
    """End-to-end: stricter filtering => fewer summarizer invocations."""
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")

    from hrag.config import KGConfig
    from hrag.kg import communities as comm_mod

    store = _build_chunk_singleton_kg(tmp_db, tmp_path, n_chunks=1)

    # Stub CommunitySummarizer.summarize_all to count calls.
    call_log: list[int] = []
    real_summarize_all = comm_mod.CommunitySummarizer.summarize_all

    def _counting_summarize_all(self, communities):
        call_log.append(len(communities))
        for c in communities:
            c.summary = "stub"
        return communities

    comm_mod.CommunitySummarizer.summarize_all = _counting_summarize_all
    try:
        cfg = KGConfig(parallel_workers=1, leiden_seed=42, community_levels=[0])
        # With default min_chunk_members=2 the singleton cluster is dropped
        # before the summarizer is even constructed (detect returns []).
        out = comm_mod.detect_and_summarize(
            kg_store=store,
            llm=fake_llm,
            db=tmp_db,
            embedder=fake_embedder,
            chroma_path=tmp_path / "chroma_comm",
            user_id="u1",
            cfg=cfg,
        )
        assert out == []
        # summarize_all was never called because there were no communities.
        assert call_log == []

        # Sanity check: with the lenient threshold via constructor, the
        # community DOES survive and the summarizer DOES run.
        detector = CommunityDetector(store, levels=[0], min_chunk_members=1)
        lenient = detector.detect("u1")
        assert len(lenient) >= 1

        summarizer = comm_mod.CommunitySummarizer(
            llm=fake_llm, db=tmp_db, max_workers=1
        )
        summarizer.summarize_all(lenient)
        # The stub recorded exactly one invocation with the lenient list.
        assert call_log == [len(lenient)]
    finally:
        comm_mod.CommunitySummarizer.summarize_all = real_summarize_all


def test_phrase_subgraph_excludes_passages(tmp_db, tmp_path):
    store = _build_two_cluster_kg(tmp_db, tmp_path)
    sub = _phrase_subgraph(store)
    # No chunk_id (passage) nodes should be present.
    for nid, data in sub.nodes(data=True):
        assert data.get("node_type") != "passage"
        assert nid not in {"c1a", "c1b", "c2a", "c2b"}


# ---------------------------------------------------------------------------
# CommunitySummarizer
# ---------------------------------------------------------------------------


def test_summarizer_runs_and_sets_summaries(tmp_db, fake_llm):
    # 5 communities each pointing at chunk c1.
    tmp_db.ensure_user("u1")
    _seed_chunk_row(tmp_db, "c1", "doc1", "u1",
                    "Sample passage about HippoRAG and PageRank.",
                    title="paper", section="intro")
    communities = [
        Community(
            community_id=f"level0_c{i}",
            level=0,
            member_phrase_node_ids=["a", "b", "c"],
            member_chunk_ids=["c1"],
        )
        for i in range(5)
    ]

    summarizer = CommunitySummarizer(llm=fake_llm, db=tmp_db, max_workers=4)
    out = summarizer.summarize_all(communities)

    assert out is communities
    assert len(fake_llm.calls) == 5
    for c in communities:
        assert c.summary == fake_llm.CANNED_ANSWER


def test_summarizer_tolerates_failures(tmp_db):
    tmp_db.ensure_user("u1")
    _seed_chunk_row(tmp_db, "c1", "doc1", "u1", "passage", title="t", section="s")

    class _FlakyLLM:
        name = "flaky"
        CANNED_ANSWER = "ok summary"

        def __init__(self) -> None:
            self.calls: list[str] = []
            self._lock = threading.Lock()

        def complete(self, prompt: str, **_kw) -> str:
            with self._lock:
                idx = len(self.calls)
                self.calls.append(prompt)
            # Fail on the second call (whichever community gets there second).
            if idx == 1:
                raise RuntimeError("simulated provider blow-up")
            return self.CANNED_ANSWER

    communities = [
        Community(
            community_id=f"level0_c{i}",
            level=0,
            member_phrase_node_ids=["a", "b", "c"],
            member_chunk_ids=["c1"],
        )
        for i in range(3)
    ]
    # Force serial execution so "second call" is deterministic.
    summarizer = CommunitySummarizer(llm=_FlakyLLM(), db=tmp_db, max_workers=1)
    summarizer.summarize_all(communities)

    summaries = [c.summary for c in communities]
    assert "<summary unavailable>" in summaries
    assert "ok summary" in summaries
    # Two succeeded, one failed.
    assert sum(s == "ok summary" for s in summaries) == 2
    assert sum(s == "<summary unavailable>" for s in summaries) == 1


def test_format_member_passages_truncates_and_numbers():
    rows = [
        {"chunk_id": "c1", "title": "T", "section": "S",
         "text": "x" * 1000},
        {"chunk_id": "c2", "title": "T2", "section": "S2",
         "text": "short"},
    ]
    rendered = _format_member_passages(rows)
    assert "[1]" in rendered
    assert "[2]" in rendered
    # First chunk text should be truncated to 800 chars.
    body_1 = rendered.split("\n\n")[0]
    assert body_1.count("x") == 800
    assert "short" in rendered


# ---------------------------------------------------------------------------
# CommunityStore
# ---------------------------------------------------------------------------


def test_store_upsert_writes_sqlite(tmp_db, tmp_path, fake_embedder):
    tmp_db.ensure_user("u1")
    tmp_db.commit()
    store = CommunityStore(
        db=tmp_db, embedder=fake_embedder, chroma_path=tmp_path / "chroma_comm"
    )
    communities = [
        Community(
            community_id="level0_c0",
            level=0,
            member_phrase_node_ids=["a", "b"],
            member_chunk_ids=["c1", "c2"],
            summary="alpha summary",
        ),
        Community(
            community_id="level0_c1",
            level=0,
            member_phrase_node_ids=["x", "y"],
            member_chunk_ids=["c3"],
            summary="beta summary",
        ),
    ]
    store.upsert("u1", communities)

    cur = tmp_db.execute(
        "SELECT COUNT(*) AS cnt FROM kg_communities WHERE user_id = ?", ("u1",)
    )
    assert cur.fetchone()["cnt"] == 2

    cur = tmp_db.execute(
        "SELECT member_chunks FROM kg_communities WHERE community_id = ?",
        ("level0_c0",),
    )
    decoded = json.loads(cur.fetchone()["member_chunks"])
    assert decoded == ["c1", "c2"]


def test_store_query_filters_by_user_and_level(tmp_db, tmp_path, fake_embedder):
    """Insert with two users + two levels, verify SQLite slicing.

    Note: the conftest stub returns empty Chroma results, so we assert the
    SQLite mirror's contents (which the upsert is responsible for) and that
    `query` doesn't blow up. With real Chroma installed, the query would
    return the matching ids.
    """
    tmp_db.ensure_user("u1")
    tmp_db.ensure_user("u2")
    tmp_db.commit()
    store = CommunityStore(
        db=tmp_db, embedder=fake_embedder, chroma_path=tmp_path / "chroma_comm"
    )

    store.upsert(
        "u1",
        [
            Community(community_id="u1_l0_c0", level=0,
                      member_phrase_node_ids=["a"], member_chunk_ids=["c1"],
                      summary="u1 level 0"),
            Community(community_id="u1_l1_c0", level=1,
                      member_phrase_node_ids=["a"], member_chunk_ids=["c1"],
                      summary="u1 level 1"),
        ],
    )
    store.upsert(
        "u2",
        [
            Community(community_id="u2_l0_c0", level=0,
                      member_phrase_node_ids=["a"], member_chunk_ids=["c2"],
                      summary="u2 level 0"),
        ],
    )

    # SQLite slicing: only u1, level 1.
    cur = tmp_db.execute(
        "SELECT community_id FROM kg_communities WHERE user_id = ? AND level = ?",
        ("u1", 1),
    )
    rows = [r["community_id"] for r in cur.fetchall()]
    assert rows == ["u1_l1_c0"]

    # query() should return without error (results may be empty under stub).
    embedding = fake_embedder.embed_one("query")
    out = store.query("u1", embedding, top_k=5, levels=[1])
    assert isinstance(out, list)


def test_store_delete_user_wipes_sqlite(tmp_db, tmp_path, fake_embedder):
    tmp_db.ensure_user("u1")
    tmp_db.commit()
    store = CommunityStore(
        db=tmp_db, embedder=fake_embedder, chroma_path=tmp_path / "chroma_comm"
    )
    store.upsert(
        "u1",
        [
            Community(community_id="x_0", level=0,
                      member_phrase_node_ids=["a"], member_chunk_ids=["c1"],
                      summary="hi"),
        ],
    )
    cur = tmp_db.execute(
        "SELECT COUNT(*) AS cnt FROM kg_communities WHERE user_id = ?", ("u1",)
    )
    assert cur.fetchone()["cnt"] == 1

    store.delete_user("u1")
    cur = tmp_db.execute(
        "SELECT COUNT(*) AS cnt FROM kg_communities WHERE user_id = ?", ("u1",)
    )
    assert cur.fetchone()["cnt"] == 0

    embedding = fake_embedder.embed_one("query")
    assert store.query("u1", embedding, top_k=5) == []


def test_store_get_community_hydrates(tmp_db, tmp_path, fake_embedder):
    tmp_db.ensure_user("u1")
    tmp_db.commit()
    store = CommunityStore(
        db=tmp_db, embedder=fake_embedder, chroma_path=tmp_path / "chroma_comm"
    )
    store.upsert(
        "u1",
        [
            Community(community_id="lvl0_cidx0", level=0,
                      member_phrase_node_ids=["a", "b"],
                      member_chunk_ids=["c1", "c2", "c3"],
                      summary="hello world"),
        ],
    )

    out = store.get_community("lvl0_cidx0")
    assert out is not None
    assert out["community_id"] == "lvl0_cidx0"
    assert out["level"] == 0
    assert out["summary"] == "hello world"
    assert out["member_chunks"] == ["c1", "c2", "c3"]

    assert store.get_community("missing") is None


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def test_detect_and_summarize_end_to_end(
    tmp_db, tmp_path, fake_llm, fake_embedder
):
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")

    from hrag.config import KGConfig

    store = _build_two_cluster_kg(tmp_db, tmp_path)

    cfg = KGConfig(
        parallel_workers=2,
        leiden_seed=42,
        community_levels=[0],
    )
    communities = detect_and_summarize(
        kg_store=store,
        llm=fake_llm,
        db=tmp_db,
        embedder=fake_embedder,
        chroma_path=tmp_path / "chroma_comm",
        user_id="u1",
        cfg=cfg,
    )

    assert len(communities) >= 2
    for c in communities:
        assert c.summary  # non-empty
        assert c.summary == fake_llm.CANNED_ANSWER

    # SQLite mirror must reflect what we wrote.
    cur = tmp_db.execute(
        "SELECT COUNT(*) AS cnt FROM kg_communities WHERE user_id = ?", ("u1",)
    )
    assert cur.fetchone()["cnt"] == len(communities)
