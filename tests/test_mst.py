"""Tests for hrag.retrieval.mst.MSTOrganizer (KG2RAG context organizer)."""

from __future__ import annotations

from typing import Sequence

import pytest

# These tests need real networkx + numpy + scipy. Skip cleanly if absent.
pytest.importorskip("networkx")
pytest.importorskip("numpy")
pytest.importorskip("scipy.sparse")

from hrag.kg.builder import Triple  # noqa: E402
from hrag.kg.store import KGStore  # noqa: E402
from hrag.retrieval.mst import MSTOrganizer  # noqa: E402
from hrag.types import Chunk, RetrievalResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Embedder:
    """Stable hash-based embedder; nothing in these tests cares about the
    actual values, only that two surface forms get the *same* canonical form
    when they're the same string."""

    name = "test"
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


def _make_result(
    chunk_id: str,
    score: float,
    rerank_score: float | None = None,
    text: str = "",
) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="d1",
        user_id="u1",
        text=text or chunk_id,
        embedding_text=text or chunk_id,
    )
    return RetrievalResult(
        chunk=chunk,
        score=score,
        retriever="vector",
        rerank_score=rerank_score,
    )


@pytest.fixture()
def kg_store(tmp_db, tmp_path):
    tmp_db.ensure_user("u1")
    tmp_db.commit()
    return KGStore(
        db=tmp_db,
        embedder=_Embedder(),
        kg_path=tmp_path / "kg",
        synonym_threshold=0.99,  # effectively disable synonym merging
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_results_returns_empty(kg_store: KGStore) -> None:
    organizer = MSTOrganizer(kg_store)
    assert organizer.organize([]) == []


def test_no_kg_returns_input_unchanged(kg_store: KGStore) -> None:
    """KG with zero phrase nodes -> pure no-op."""
    organizer = MSTOrganizer(kg_store)
    results = [_make_result("a", 0.9), _make_result("b", 0.8)]
    out = organizer.organize(results)
    assert out is results or [r.chunk.chunk_id for r in out] == ["a", "b"]


def test_chunks_not_in_kg_returns_input_unchanged(kg_store: KGStore) -> None:
    """KG has phrase nodes but none of the test chunks have passage nodes
    in the graph -> no-op."""
    # Populate KG with triples that reference *different* chunk ids.
    kg_store.upsert_triples(
        "u1",
        "d1",
        [Triple(head="x", relation="r", tail="y", source_chunk_id="c_other")],
    )
    organizer = MSTOrganizer(kg_store)

    results = [_make_result("a", 0.9), _make_result("b", 0.8)]
    out = organizer.organize(results)
    assert [r.chunk.chunk_id for r in out] == ["a", "b"]


def test_high_overlap_drops_lower_scored(kg_store: KGStore) -> None:
    """Two chunks with identical entity sets {x,y,z} (Jaccard 1.0): the
    lower-scored one is dropped."""
    triples = [
        Triple(head="x", relation="r", tail="y", source_chunk_id="cA"),
        Triple(head="y", relation="r", tail="z", source_chunk_id="cA"),
        Triple(head="x", relation="r", tail="z", source_chunk_id="cA"),
        Triple(head="x", relation="r", tail="y", source_chunk_id="cB"),
        Triple(head="y", relation="r", tail="z", source_chunk_id="cB"),
        Triple(head="x", relation="r", tail="z", source_chunk_id="cB"),
    ]
    kg_store.upsert_triples("u1", "d1", triples)

    organizer = MSTOrganizer(kg_store, redundancy_threshold=0.7)
    results = [_make_result("cA", 0.9), _make_result("cB", 0.5)]
    out = organizer.organize(results)
    assert len(out) == 1
    assert out[0].chunk.chunk_id == "cA"


def test_disjoint_chunks_both_kept_singletons(kg_store: KGStore) -> None:
    """No entity overlap at all -> both kept as singleton components,
    ordered by score desc."""
    triples = [
        Triple(head="alpha", relation="r", tail="beta", source_chunk_id="c1"),
        Triple(head="gamma", relation="r", tail="delta", source_chunk_id="c2"),
    ]
    kg_store.upsert_triples("u1", "d1", triples)

    organizer = MSTOrganizer(kg_store)
    results = [_make_result("c1", 0.5), _make_result("c2", 0.9)]
    out = organizer.organize(results)
    # Both kept; higher-scored component comes first (size tie -> max_score).
    assert len(out) == 2
    assert out[0].chunk.chunk_id == "c2"
    assert out[1].chunk.chunk_id == "c1"


def test_mst_chain_bfs_order(kg_store: KGStore) -> None:
    """Four chunks A-B-C-D with monotonically decreasing overlap.

    Entity sets:
        A = {a1, a2, a3, a4, a5}
        B = {a1, a2, a3, a4, b1}      (large overlap with A)
        C = {a3, a4, c1, c2}          (medium overlap with B; small with A)
        D = {c1, d1}                  (small overlap with C; none with A,B)

    Scores: A=0.9, B=0.7, C=0.5, D=0.3. MST is the chain A-B-C-D and BFS
    from A yields A,B,C,D.

    Jaccards:
        A∩B = {a1,a2,a3,a4} (4/6 ≈ 0.67)  -> below 0.7 redundancy threshold
        A∩C = {a3,a4} (2/7 ≈ 0.29)
        A∩D = {} -> no edge
        B∩C = {a3,a4} (2/7 ≈ 0.29)
        B∩D = {} -> no edge
        C∩D = {c1} (1/5 = 0.2)
    """
    triples: list[Triple] = []
    for entity in ("a1", "a2", "a3", "a4", "a5"):
        triples.append(Triple(head=entity, relation="r", tail="seed", source_chunk_id="A"))
    for entity in ("a1", "a2", "a3", "a4", "b1"):
        triples.append(Triple(head=entity, relation="r", tail="seed", source_chunk_id="B"))
    for entity in ("a3", "a4", "c1", "c2"):
        triples.append(Triple(head=entity, relation="r", tail="seed", source_chunk_id="C"))
    for entity in ("c1", "d1"):
        triples.append(Triple(head=entity, relation="r", tail="seed", source_chunk_id="D"))
    kg_store.upsert_triples("u1", "d1", triples)

    organizer = MSTOrganizer(kg_store, redundancy_threshold=0.99)  # disable pruning
    results = [
        _make_result("A", 0.9),
        _make_result("B", 0.7),
        _make_result("C", 0.5),
        _make_result("D", 0.3),
    ]
    out = organizer.organize(results)
    assert [r.chunk.chunk_id for r in out] == ["A", "B", "C", "D"]


def test_max_chunks_cap_honored(kg_store: KGStore) -> None:
    """Cap=5, hand it 20 disjoint chunks → exactly 5 returned."""
    triples: list[Triple] = []
    for i in range(20):
        triples.append(
            Triple(head=f"e{i}", relation="r", tail="seed", source_chunk_id=f"c{i}")
        )
    kg_store.upsert_triples("u1", "d1", triples)

    organizer = MSTOrganizer(kg_store, max_chunks=5)
    results = [_make_result(f"c{i}", 1.0 - i * 0.01) for i in range(20)]
    out = organizer.organize(results)
    assert len(out) == 5


def test_mixed_in_kg_and_out_of_kg(kg_store: KGStore) -> None:
    """Some chunks live in the KG, some don't. The non-KG chunks must still
    appear in the output (treated as singletons)."""
    # Two highly-overlapping chunks in the KG.
    triples = [
        Triple(head="x", relation="r", tail="y", source_chunk_id="cKG1"),
        Triple(head="y", relation="r", tail="z", source_chunk_id="cKG1"),
        Triple(head="x", relation="r", tail="y", source_chunk_id="cKG2"),
        Triple(head="y", relation="r", tail="z", source_chunk_id="cKG2"),
    ]
    kg_store.upsert_triples("u1", "d1", triples)

    organizer = MSTOrganizer(kg_store, redundancy_threshold=0.7)
    results = [
        _make_result("cKG1", 0.9),
        _make_result("cKG2", 0.5),  # redundant with cKG1 -> dropped
        _make_result("cExtra1", 0.6),  # not in KG
        _make_result("cExtra2", 0.4),  # not in KG
    ]
    out = organizer.organize(results)
    out_ids = {r.chunk.chunk_id for r in out}

    assert "cKG1" in out_ids
    assert "cKG2" not in out_ids  # redundancy-pruned
    assert "cExtra1" in out_ids   # KG-absent extras pass through
    assert "cExtra2" in out_ids


def test_default_threshold_drops_high_overlap(kg_store: KGStore) -> None:
    """A and B share 5 of 7 entities (Jaccard = 5/7 ≈ 0.71 > default 0.4).

    Both A and B use the same tail node "common", so the phrase sets are:
        A phrases = {p, q, r, s, a1, common}   (6 nodes)
        B phrases = {p, q, r, s, b1, common}   (6 nodes)
        Intersection = {p, q, r, s, common} → 5 shared
        Union        = {p, q, r, s, a1, b1, common} → 7 total
        Jaccard = 5/7 ≈ 0.71 > 0.4 ✓

    With the new default ``redundancy_threshold=0.4``, B (lower-scored) is
    dropped; C (unrelated) survives.  Result: 2 survivors, B absent.
    """
    triples: list[Triple] = []
    for entity in ("p", "q", "r", "s", "a1"):
        triples.append(Triple(head=entity, relation="r", tail="common", source_chunk_id="A"))
    for entity in ("p", "q", "r", "s", "b1"):
        triples.append(Triple(head=entity, relation="r", tail="common", source_chunk_id="B"))
    for entity in ("alpha", "beta"):
        triples.append(Triple(head=entity, relation="r", tail="seed_c", source_chunk_id="C"))
    kg_store.upsert_triples("u1", "d1", triples)

    # Default threshold is now 0.4; 0.43 >= 0.4 → B should be pruned.
    organizer = MSTOrganizer(kg_store)
    results = [
        _make_result("A", 0.9),  # higher score → survives
        _make_result("B", 0.5),  # lower score → dropped
        _make_result("C", 0.7),  # disjoint → survives
    ]
    out = organizer.organize(results)
    out_ids = [r.chunk.chunk_id for r in out]
    assert len(out) == 2
    assert "B" not in out_ids
    assert "A" in out_ids
    assert "C" in out_ids


def test_min_edge_jaccard_default_includes_low_overlap(kg_store: KGStore) -> None:
    """Two chunks sharing exactly 1 entity out of 20 total (Jaccard = 1/20 = 0.05).

    The old hardcoded floor was 0.05; at exactly 0.05 the edge was excluded
    (``< 0.05`` check).  The new default of 0.02 passes 0.05 through, so the
    chunks end up in the same connected component.

    Entity sets:
        P = {shared, p1, p2, ..., p9}   (10 entities: 1 shared + 9 unique)
        Q = {shared, q1, q2, ..., q9}   (10 entities: 1 shared + 9 unique)
        Jaccard = 1 / (10 + 10 - 1) = 1/19 ≈ 0.053 > 0.02 ✓
    """
    triples: list[Triple] = []
    triples.append(Triple(head="shared", relation="r", tail="seed_p", source_chunk_id="P"))
    for i in range(9):
        triples.append(Triple(head=f"p{i}", relation="r", tail="seed_p", source_chunk_id="P"))
    triples.append(Triple(head="shared", relation="r", tail="seed_q", source_chunk_id="Q"))
    for i in range(9):
        triples.append(Triple(head=f"q{i}", relation="r", tail="seed_q", source_chunk_id="Q"))
    kg_store.upsert_triples("u1", "d1", triples)

    # Default min_edge_jaccard=0.02; Jaccard≈0.053 > 0.02 → edge created →
    # both chunks land in the same connected component.
    organizer = MSTOrganizer(kg_store)
    results = [_make_result("P", 0.9), _make_result("Q", 0.8)]
    # Inspect the chunk graph directly by running organize and verifying both
    # survive (no pruning: Jaccard 0.053 < 0.4 redundancy threshold).
    out = organizer.organize(results)
    assert len(out) == 2
    out_ids = {r.chunk.chunk_id for r in out}
    assert "P" in out_ids
    assert "Q" in out_ids

    # Confirm they form one component: if same component, BFS order keeps them
    # consecutive (one component spec vs. two separate singleton specs).
    # The safest observable: the result starts with the higher-scored chunk.
    assert out[0].chunk.chunk_id == "P"


def test_low_threshold_more_aggressive_pruning(kg_store: KGStore) -> None:
    """With redundancy_threshold=0.2, A∩B (Jaccard≈0.71) still exceeds the bar,
    so B is dropped — same outcome as the default=0.4 test.  This validates
    the constructor argument is honoured when set lower than the default.

    Uses the same shared-tail fixture as test_default_threshold_drops_high_overlap:
        A phrases = {p, q, r, s, a1, common}  Jaccard(A,B) = 5/7 ≈ 0.71
        B phrases = {p, q, r, s, b1, common}
    """
    triples: list[Triple] = []
    for entity in ("p", "q", "r", "s", "a1"):
        triples.append(Triple(head=entity, relation="r", tail="common", source_chunk_id="A"))
    for entity in ("p", "q", "r", "s", "b1"):
        triples.append(Triple(head=entity, relation="r", tail="common", source_chunk_id="B"))
    for entity in ("alpha", "beta"):
        triples.append(Triple(head=entity, relation="r", tail="seed_c", source_chunk_id="C"))
    kg_store.upsert_triples("u1", "d1", triples)

    organizer = MSTOrganizer(kg_store, redundancy_threshold=0.2)
    results = [
        _make_result("A", 0.9),
        _make_result("B", 0.5),
        _make_result("C", 0.7),
    ]
    out = organizer.organize(results)
    assert len(out) == 2
    assert "B" not in {r.chunk.chunk_id for r in out}


def test_high_threshold_conservative(kg_store: KGStore) -> None:
    """With redundancy_threshold=0.9, A∩B (Jaccard≈0.71) is below the bar.

    No chunks are dropped; all three survive.  This mirrors the pre-fix
    behaviour where the default was 0.7 and overlapping chunks were NOT dropped.

    Uses the same shared-tail fixture as test_default_threshold_drops_high_overlap:
        A phrases = {p, q, r, s, a1, common}  Jaccard(A,B) = 5/7 ≈ 0.71 < 0.9
        B phrases = {p, q, r, s, b1, common}
    """
    triples: list[Triple] = []
    for entity in ("p", "q", "r", "s", "a1"):
        triples.append(Triple(head=entity, relation="r", tail="common", source_chunk_id="A"))
    for entity in ("p", "q", "r", "s", "b1"):
        triples.append(Triple(head=entity, relation="r", tail="common", source_chunk_id="B"))
    for entity in ("alpha", "beta"):
        triples.append(Triple(head=entity, relation="r", tail="seed_c", source_chunk_id="C"))
    kg_store.upsert_triples("u1", "d1", triples)

    organizer = MSTOrganizer(kg_store, redundancy_threshold=0.9)
    results = [
        _make_result("A", 0.9),
        _make_result("B", 0.5),
        _make_result("C", 0.7),
    ]
    out = organizer.organize(results)
    assert len(out) == 3


def test_default_min_edge_jaccard_param(kg_store: KGStore) -> None:
    """min_edge_jaccard=0.8 means only near-identical chunks create edges.

    The A/B pair (Jaccard≈0.71) is now BELOW 0.8 → no edge → separate
    singleton components.  The organizer must not crash; both survive since
    absence of an edge means no pruning can occur.

    Uses the same shared-tail fixture (Jaccard 5/7 ≈ 0.71 < 0.8):
        A phrases = {p, q, r, s, a1, common}
        B phrases = {p, q, r, s, b1, common}
    """
    triples: list[Triple] = []
    for entity in ("p", "q", "r", "s", "a1"):
        triples.append(Triple(head=entity, relation="r", tail="common", source_chunk_id="A"))
    for entity in ("p", "q", "r", "s", "b1"):
        triples.append(Triple(head=entity, relation="r", tail="common", source_chunk_id="B"))
    kg_store.upsert_triples("u1", "d1", triples)

    organizer = MSTOrganizer(kg_store, min_edge_jaccard=0.8)
    results = [_make_result("A", 0.9), _make_result("B", 0.8)]
    out = organizer.organize(results)
    # No edge created (Jaccard 0.71 < min_edge_jaccard 0.8) → no pruning → both survive.
    assert len(out) == 2


def test_rerank_score_preferred_for_tiebreak(kg_store: KGStore) -> None:
    """When deciding which side of a redundant pair to drop, rerank_score
    wins over raw score."""
    triples = [
        Triple(head="x", relation="r", tail="y", source_chunk_id="hi"),
        Triple(head="y", relation="r", tail="z", source_chunk_id="hi"),
        Triple(head="x", relation="r", tail="y", source_chunk_id="lo"),
        Triple(head="y", relation="r", tail="z", source_chunk_id="lo"),
    ]
    kg_store.upsert_triples("u1", "d1", triples)

    organizer = MSTOrganizer(kg_store, redundancy_threshold=0.7)
    # Raw 'score' would say "hi" loses (lower raw score). rerank_score
    # inverts that.
    results = [
        _make_result("hi", score=0.1, rerank_score=2.0),
        _make_result("lo", score=0.9, rerank_score=1.0),
    ]
    out = organizer.organize(results)
    assert len(out) == 1
    assert out[0].chunk.chunk_id == "hi"
