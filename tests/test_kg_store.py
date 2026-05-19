"""Tests for hrag.kg.store.KGStore."""

from __future__ import annotations

from typing import Sequence

import pytest

# These tests need real networkx and scipy/numpy. Skip cleanly if absent.
pytest.importorskip("networkx")
pytest.importorskip("numpy")
pytest.importorskip("scipy.sparse")

from hrag.kg.builder import Triple  # noqa: E402
from hrag.kg.store import KGStore  # noqa: E402


# ---------------------------------------------------------------------------
# Custom embedder: deterministic vectors via dict lookup, with a fallback.
# ---------------------------------------------------------------------------


class DictEmbedder:
    """Embedder that maps specific phrases to specific vectors.

    Lets us craft tests where two strings have a known cosine similarity.
    Anything not in the dict gets a fallback orthogonal-ish vector based on
    string hash so unrelated phrases don't accidentally collide.
    """

    name = "dict"
    _DIM = 8

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self._map = dict(mapping or {})
        self._fallback_counter = 0

    def add(self, text: str, vec: list[float]) -> None:
        self._map[text] = list(vec)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        if text in self._map:
            return list(self._map[text])
        # Stable-ish fallback: derive from hash, normalize.
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        floats = [(b / 255.0) * 2.0 - 1.0 for b in digest[: self._DIM]]
        return floats

    @property
    def dim(self) -> int:
        return self._DIM


# Helper: pick two unit vectors with a chosen cosine similarity. We build a
# 2D pair then pad with zeros to _DIM.
def _vecs_with_cosine(target: float, dim: int = 8) -> tuple[list[float], list[float]]:
    import math

    a = [1.0] + [0.0] * (dim - 1)
    # We want b such that cos(a, b) = target.
    # Choose b = [target, sqrt(1-target^2), 0, 0, ...] which is unit norm.
    rest = math.sqrt(max(0.0, 1.0 - target * target))
    b = [target, rest] + [0.0] * (dim - 2)
    return a, b


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kg_store(tmp_db, tmp_path):
    tmp_db.ensure_user("u1")
    tmp_db.commit()
    embedder = DictEmbedder()
    store = KGStore(
        db=tmp_db,
        embedder=embedder,
        kg_path=tmp_path / "kg",
        synonym_threshold=0.8,
    )
    return store


@pytest.fixture()
def kg_with_embedder(tmp_db, tmp_path):
    """Returns a function (mapping, threshold) -> KGStore for tests that
    want to inject specific vectors."""
    tmp_db.ensure_user("u1")
    tmp_db.commit()

    def _make(mapping: dict[str, list[float]] | None = None, threshold: float = 0.8) -> KGStore:
        embedder = DictEmbedder(mapping)
        return KGStore(
            db=tmp_db,
            embedder=embedder,
            kg_path=tmp_path / "kg",
            synonym_threshold=threshold,
        )

    return _make


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_graph(kg_store: KGStore) -> None:
    assert kg_store.num_phrase_nodes() == 0
    assert kg_store.num_passage_nodes() == 0
    assert kg_store.num_edges() == 0


def test_single_triple_inserts_two_phrases_and_passage(kg_store: KGStore) -> None:
    triples = [
        Triple(head="HippoRAG", relation="uses", tail="PageRank", source_chunk_id="c1"),
    ]
    kg_store.upsert_triples("u1", "doc1", triples)

    assert kg_store.num_phrase_nodes() == 2
    assert kg_store.num_passage_nodes() == 1
    # 1 phrase->phrase + 2 phrase->contains->passage = 3 edges total
    assert kg_store.num_edges() == 3

    # Canonical keys are lowercase.
    assert "hipporag" in {n for n, d in kg_store._graph.nodes(data=True) if d.get("node_type") == "phrase"}
    assert "pagerank" in {n for n, d in kg_store._graph.nodes(data=True) if d.get("node_type") == "phrase"}


def test_synonym_merge_above_threshold(kg_with_embedder) -> None:
    a, b = _vecs_with_cosine(0.9)
    store = kg_with_embedder({"hipporag": a, "hipporag framework": b}, threshold=0.8)

    triples_1 = [
        Triple(head="HippoRAG", relation="is", tail="Method", source_chunk_id="c1"),
    ]
    store.upsert_triples("u1", "doc1", triples_1)

    triples_2 = [
        Triple(
            head="HippoRAG framework",
            relation="cites",
            tail="OtherWork",
            source_chunk_id="c2",
        ),
    ]
    store.upsert_triples("u1", "doc2", triples_2)

    phrase_nodes = [n for n, d in store._graph.nodes(data=True) if d.get("node_type") == "phrase"]
    # Should be: hipporag (merged target), method, otherwork
    assert "hipporag" in phrase_nodes
    assert "hipporag framework" not in phrase_nodes
    aliases = store._graph.nodes["hipporag"]["aliases"]
    assert "HippoRAG framework" in aliases or "hipporag framework" in aliases


def test_no_merge_below_threshold(kg_with_embedder) -> None:
    # Orthogonal vectors -> cosine 0
    a, b = _vecs_with_cosine(0.0)
    store = kg_with_embedder({"alpha": a, "beta": b}, threshold=0.8)

    triples = [
        Triple(head="Alpha", relation="r", tail="Beta", source_chunk_id="c1"),
    ]
    store.upsert_triples("u1", "doc1", triples)

    assert store.num_phrase_nodes() == 2


def test_idempotent_reupsert(kg_store: KGStore) -> None:
    triples = [
        Triple(head="A", relation="r", tail="B", source_chunk_id="c1"),
        Triple(head="A", relation="r2", tail="C", source_chunk_id="c1"),
    ]
    kg_store.upsert_triples("u1", "doc1", triples)
    n_phrase_1 = kg_store.num_phrase_nodes()
    n_passage_1 = kg_store.num_passage_nodes()
    n_edges_1 = kg_store.num_edges()

    kg_store.upsert_triples("u1", "doc1", triples)
    assert kg_store.num_phrase_nodes() == n_phrase_1
    assert kg_store.num_passage_nodes() == n_passage_1
    assert kg_store.num_edges() == n_edges_1


def test_per_doc_deletion_preserves_shared_phrase(kg_store: KGStore) -> None:
    triples_a = [
        Triple(head="Shared", relation="r", tail="OnlyA", source_chunk_id="cA1"),
    ]
    triples_b = [
        Triple(head="Shared", relation="r", tail="OnlyB", source_chunk_id="cB1"),
    ]
    kg_store.upsert_triples("u1", "docA", triples_a)
    kg_store.upsert_triples("u1", "docB", triples_b)

    # Sanity: 3 phrase nodes (shared, onlya, onlyb) + 2 passages
    assert kg_store.num_phrase_nodes() == 3
    assert kg_store.num_passage_nodes() == 2

    kg_store.delete_doc("u1", "docA")

    # Shared phrase still present.
    phrase_ids = {
        n for n, d in kg_store._graph.nodes(data=True) if d.get("node_type") == "phrase"
    }
    assert "shared" in phrase_ids
    assert "onlyb" in phrase_ids

    # docA's passage and orphan phrase->phrase edge gone.
    assert "cA1" not in kg_store._graph
    assert "cB1" in kg_store._graph

    # The shared->onlya edge should be gone (not supported by any chunk now);
    # but onlya may or may not still exist (we never delete phrase nodes).
    # Just verify docB's edge still works:
    assert kg_store.num_passage_nodes() == 1


def test_persistence_round_trip(tmp_db, tmp_path) -> None:
    tmp_db.ensure_user("u1")
    tmp_db.commit()
    embedder = DictEmbedder()
    store1 = KGStore(
        db=tmp_db,
        embedder=embedder,
        kg_path=tmp_path / "kg",
        synonym_threshold=0.8,
    )
    triples = [
        Triple(head="A", relation="r", tail="B", source_chunk_id="c1"),
    ]
    store1.upsert_triples("u1", "doc1", triples)

    n_phrase = store1.num_phrase_nodes()
    n_passage = store1.num_passage_nodes()
    n_edges = store1.num_edges()

    # New instance pointing at same kg_path → graph must reload from pickle.
    store2 = KGStore(
        db=tmp_db,
        embedder=DictEmbedder(),
        kg_path=tmp_path / "kg",
        synonym_threshold=0.8,
    )
    assert store2.num_phrase_nodes() == n_phrase
    assert store2.num_passage_nodes() == n_passage
    assert store2.num_edges() == n_edges


def test_find_phrase_nodes_case_insensitive(kg_store: KGStore) -> None:
    triples = [
        Triple(head="Alpha", relation="r", tail="Beta", source_chunk_id="c1"),
    ]
    kg_store.upsert_triples("u1", "doc1", triples)

    found = kg_store.find_phrase_nodes(["ALPHA", " beta ", "missing"])
    assert "alpha" in found
    assert "beta" in found
    assert "missing" not in found


def test_passage_nodes_for(kg_store: KGStore) -> None:
    triples = [
        Triple(head="Alpha", relation="r", tail="Beta", source_chunk_id="cX"),
        Triple(head="Gamma", relation="r", tail="Delta", source_chunk_id="cY"),
    ]
    kg_store.upsert_triples("u1", "doc1", triples)

    passages = kg_store.passage_nodes_for(["alpha"])
    assert passages == {"cX"}

    passages = kg_store.passage_nodes_for(["alpha", "gamma"])
    assert passages == {"cX", "cY"}

    passages = kg_store.passage_nodes_for(["nonexistent"])
    assert passages == set()


def test_to_sparse_adjacency(kg_store: KGStore) -> None:
    triples = [
        Triple(head="A", relation="r", tail="B", source_chunk_id="c1"),
    ]
    kg_store.upsert_triples("u1", "doc1", triples)

    mat, node_ids = kg_store.to_sparse_adjacency()
    n = mat.shape[0]
    assert n == len(node_ids)
    assert n == kg_store.num_phrase_nodes() + kg_store.num_passage_nodes()
    # Non-zero entries should equal the number of edges.
    assert mat.nnz == kg_store.num_edges()


def test_to_sparse_adjacency_empty(kg_store: KGStore) -> None:
    mat, node_ids = kg_store.to_sparse_adjacency()
    assert mat.shape == (0, 0)
    assert node_ids == []


def test_neighbors_depth_1_and_2(kg_store: KGStore) -> None:
    # A -> B -> C  (chain)
    triples = [
        Triple(head="A", relation="r1", tail="B", source_chunk_id="c1"),
        Triple(head="B", relation="r2", tail="C", source_chunk_id="c2"),
    ]
    kg_store.upsert_triples("u1", "doc1", triples)

    n1 = kg_store.neighbors("a", depth=1)
    # Direct neighbors of A: B (via r1), c1 (via contains)
    assert "b" in n1
    assert "c1" in n1
    # 'c' is 2 hops away
    assert "c" not in n1

    n2 = kg_store.neighbors("a", depth=2)
    assert "c" in n2 or "b" in n2  # c reached via B at depth 2
    assert "c" in n2


def test_corrupt_pickle_starts_fresh(tmp_db, tmp_path) -> None:
    kg_dir = tmp_path / "kg"
    kg_dir.mkdir()
    (kg_dir / "graph.pkl").write_bytes(b"this is not a pickle")

    with pytest.warns(UserWarning, match="failed to load graph"):
        store = KGStore(
            db=tmp_db,
            embedder=DictEmbedder(),
            kg_path=kg_dir,
            synonym_threshold=0.8,
        )
    assert store.num_phrase_nodes() == 0
    assert store.num_passage_nodes() == 0


def test_sqlite_mirror_consistent(kg_store: KGStore, tmp_db) -> None:
    triples = [
        Triple(head="A", relation="r", tail="B", source_chunk_id="c1"),
        Triple(head="C", relation="r2", tail="D", source_chunk_id="c2"),
    ]
    kg_store.upsert_triples("u1", "doc1", triples)

    cur = tmp_db.execute(
        "SELECT COUNT(*) AS cnt FROM kg_nodes WHERE user_id = ?", ("u1",)
    )
    cnt = cur.fetchone()["cnt"]
    assert cnt == kg_store.num_phrase_nodes() + kg_store.num_passage_nodes()


def test_sqlite_mirror_phrase_has_null_doc_id(kg_store: KGStore, tmp_db) -> None:
    triples = [
        Triple(head="A", relation="r", tail="B", source_chunk_id="c1"),
    ]
    kg_store.upsert_triples("u1", "doc1", triples)

    cur = tmp_db.execute(
        "SELECT node_id, doc_id, node_type FROM kg_nodes WHERE user_id = ?", ("u1",)
    )
    rows = list(cur.fetchall())
    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r["node_type"], []).append(r)

    for r in by_type.get("phrase", []):
        assert r["doc_id"] is None
    for r in by_type.get("passage", []):
        assert r["doc_id"] == "doc1"


def test_chunk_id_to_doc_id_override(kg_store: KGStore) -> None:
    """If the optional mapping is provided, the passage node uses *that*
    doc_id rather than the upsert's nominal doc_id."""
    triples = [
        Triple(head="A", relation="r", tail="B", source_chunk_id="cFromOther"),
    ]
    # Map cFromOther -> doc_other even though we're nominally upserting doc1.
    kg_store.upsert_triples(
        "u1",
        "doc1",
        triples,
        chunk_id_to_doc_id={"cFromOther": "doc_other"},
    )
    assert kg_store._graph.nodes["cFromOther"]["doc_id"] == "doc_other"
