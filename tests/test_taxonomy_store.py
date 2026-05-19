"""Tests for hrag.taxonomy.store.TaxonomyStore — CRUD + beam descend."""

from __future__ import annotations

import hashlib
from typing import Sequence

import pytest

from hrag.db.connection import Database
from hrag.taxonomy.store import TaxonomyStore


# ---------------------------------------------------------------------------
# Deterministic embedder
# ---------------------------------------------------------------------------


class DictEmbedder:
    """Hash-derived deterministic embedder, with a mapping override."""

    name = "dict"
    _DIM = 8

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self._map = dict(mapping or {})

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        if text in self._map:
            return list(self._map[text])
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(b / 255.0) * 2.0 - 1.0 for b in digest[: self._DIM]]

    @property
    def dim(self) -> int:
        return self._DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.sqlite"
    db = Database(path)
    db.init_schema()
    db.ensure_user("u1")
    yield db
    db.close()


@pytest.fixture
def store(db):
    return TaxonomyStore(db, DictEmbedder())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ensure_root_is_idempotent(store):
    r1 = store.ensure_root("u1")
    r2 = store.ensure_root("u1")
    assert r1.node_id == r2.node_id
    assert r1.parent_id is None
    assert r1.depth == 0


def test_add_node_increments_depth(store):
    root = store.ensure_root("u1")
    child = store.add_node("u1", root.node_id, "Robotics")
    grand = store.add_node("u1", child.node_id, "Manipulation", is_leaf=True)
    assert child.depth == 1
    assert grand.depth == 2
    assert grand.parent_id == child.node_id


def test_list_nodes_returns_all(store):
    root = store.ensure_root("u1")
    store.add_node("u1", root.node_id, "A")
    store.add_node("u1", root.node_id, "B")
    nodes = store.list_nodes("u1")
    assert len(nodes) == 3  # root + A + B
    labels = sorted(n.label for n in nodes)
    assert labels == ["A", "B", "root"]


def test_assign_doc_requires_leaf(db, store):
    root = store.ensure_root("u1")
    internal = store.add_node("u1", root.node_id, "Internal", is_leaf=False)
    # Insert a fake doc row so the FK is satisfied.
    db.execute(
        "INSERT INTO documents(doc_id, user_id, source_path, title) VALUES (?,?,?,?)",
        ("doc1", "u1", "/tmp/d.pdf", "D"),
    )
    db.commit()
    with pytest.raises(ValueError):
        store.assign_doc("u1", "doc1", internal.node_id)


def test_assign_and_get_docs_at(db, store):
    root = store.ensure_root("u1")
    leaf = store.add_node("u1", root.node_id, "Leaf", is_leaf=True)
    db.execute(
        "INSERT INTO documents(doc_id, user_id, source_path, title) VALUES (?,?,?,?)",
        ("doc1", "u1", "/tmp/d.pdf", "D"),
    )
    db.commit()
    store.assign_doc("u1", "doc1", leaf.node_id, score=0.9)
    assert store.get_docs_at(leaf.node_id) == ["doc1"]


def test_get_docs_at_with_descendants(db, store):
    root = store.ensure_root("u1")
    mid = store.add_node("u1", root.node_id, "Mid", is_leaf=False)
    leaf_a = store.add_node("u1", mid.node_id, "A", is_leaf=True)
    leaf_b = store.add_node("u1", mid.node_id, "B", is_leaf=True)
    for did in ("d1", "d2"):
        db.execute(
            "INSERT INTO documents(doc_id, user_id, source_path, title) VALUES (?,?,?,?)",
            (did, "u1", f"/tmp/{did}.pdf", did),
        )
    db.commit()
    store.assign_doc("u1", "d1", leaf_a.node_id)
    store.assign_doc("u1", "d2", leaf_b.node_id)
    docs = sorted(store.get_docs_at(mid.node_id, include_descendants=True))
    assert docs == ["d1", "d2"]


def test_beam_descend_picks_best_leaf(db, store):
    """Build a small tree, give it controlled centroids, verify beam picks the
    leaf whose centroid is closest to the query."""
    emb = DictEmbedder()
    # Carefully chosen pseudo-embeddings:
    target_vec = [1.0] + [0.0] * 7      # query direction
    far_vec = [0.0, 1.0] + [0.0] * 6     # orthogonal
    emb._map["query"] = target_vec
    emb._map["target"] = target_vec
    emb._map["far"] = far_vec

    store_emb = TaxonomyStore(db, emb)

    root = store_emb.ensure_root("u1")
    # Two top-level branches, each with one leaf.
    target_branch = store_emb.add_node("u1", root.node_id, "Target", is_leaf=False)
    far_branch = store_emb.add_node("u1", root.node_id, "Far", is_leaf=False)
    target_leaf = store_emb.add_node("u1", target_branch.node_id, "TargetLeaf", is_leaf=True)
    far_leaf = store_emb.add_node("u1", far_branch.node_id, "FarLeaf", is_leaf=True)

    # Insert docs and assign to leaves with explicit centroids.
    db.execute(
        "INSERT INTO documents(doc_id, user_id, source_path, title) VALUES (?,?,?,?)",
        ("dt", "u1", "/tmp/dt.pdf", "DT"),
    )
    db.execute(
        "INSERT INTO documents(doc_id, user_id, source_path, title) VALUES (?,?,?,?)",
        ("df", "u1", "/tmp/df.pdf", "DF"),
    )
    db.commit()
    store_emb.upsert_doc_meta("u1", "dt", "near", target_vec)
    store_emb.upsert_doc_meta("u1", "df", "far", far_vec)
    store_emb.assign_doc("u1", "dt", target_leaf.node_id)
    store_emb.assign_doc("u1", "df", far_leaf.node_id)
    store_emb.recompute_all_centroids("u1")

    result = store_emb.beam_descend(
        "u1", target_vec, beam_width=1, max_depth=4, min_score=-1.0
    )
    assert len(result.leaves) == 1
    assert result.leaves[0].node.label == "TargetLeaf"
    # Trace records both descend levels.
    assert len(result.trace) == 2


def test_beam_descend_dominance_gap_narrows_beam(db, store):
    """A clearly-dominant top score should narrow the beam to 1 even when
    beam_width=3, so obviously-unrelated branches are not descended."""
    emb = DictEmbedder()
    # Three top-level branches:
    #   - "Personal"  with score ~ 0.95 against the query (the clear winner)
    #   - "AI"        with score ~ 0.20
    #   - "NLP"       with score ~ 0.10
    # The 0.95 → 0.20 gap (>= 0.10) should chop the beam to 1.
    q_vec = [1.0] + [0.0] * 7

    def vec(x: float, y: float = 0.0, z: float = 0.0) -> list[float]:
        out = [x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0]
        # leave un-normalized; _cosine renormalizes internally
        return out

    emb._map["query"] = q_vec
    emb._map["personal"] = vec(0.95, 0.05)
    emb._map["ai"]       = vec(0.20, 0.90)
    emb._map["nlp"]      = vec(0.10, 0.99)

    s = TaxonomyStore(db, emb)
    root = s.ensure_root("u1")
    personal_internal = s.add_node("u1", root.node_id, "Personal")
    ai_internal       = s.add_node("u1", root.node_id, "AI")
    nlp_internal      = s.add_node("u1", root.node_id, "NLP")
    personal_leaf = s.add_node("u1", personal_internal.node_id, "PLeaf", is_leaf=True)
    ai_leaf       = s.add_node("u1", ai_internal.node_id,       "ALeaf", is_leaf=True)
    nlp_leaf      = s.add_node("u1", nlp_internal.node_id,      "NLeaf", is_leaf=True)

    for did, key in (("p", "personal"), ("a", "ai"), ("n", "nlp")):
        db.execute(
            "INSERT INTO documents(doc_id, user_id, source_path, title) VALUES (?,?,?,?)",
            (did, "u1", f"/tmp/{did}.pdf", did),
        )
        s.upsert_doc_meta("u1", did, key, emb.embed_one(key))
    s.assign_doc("u1", "p", personal_leaf.node_id)
    s.assign_doc("u1", "a", ai_leaf.node_id)
    s.assign_doc("u1", "n", nlp_leaf.node_id)
    db.commit()
    s.recompute_all_centroids("u1")

    # Without gap pruning: beam_width=3 keeps all three branches at level 0.
    wide = s.beam_descend(
        "u1", q_vec, beam_width=3, max_depth=4, min_score=-1.0, dominance_gap=0.0
    )
    assert len(wide.trace[0].kept) == 3

    # With dominance_gap=0.10 the 0.95→~0.20 gap triggers, beam → 1.
    narrow = s.beam_descend(
        "u1", q_vec, beam_width=3, max_depth=4, min_score=-1.0, dominance_gap=0.10
    )
    assert len(narrow.trace[0].kept) == 1
    assert narrow.trace[0].kept[0].node.label == "Personal"
    # And only the personal leaf should surface.
    assert len(narrow.leaves) == 1
    assert narrow.leaves[0].node.label == "PLeaf"


def test_beam_descend_min_top_score_floor_narrows_to_one(db, store):
    """If even the best candidate at any level is below ``min_top_score_floor``,
    the descend should narrow the beam at that level to 1 — protecting
    against 'open 23 of 24 docs' on a low-confidence query like "hey!"."""
    emb = DictEmbedder()

    # The query is orthogonal to everything in the tree; every branch will
    # score weakly. The top one (Personal at 0.18) is still below 0.30.
    q_vec = [1.0] + [0.0] * 7
    emb._map["query"] = q_vec
    emb._map["personal"] = [0.18, 0.98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    emb._map["ai"]       = [0.15, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0]
    emb._map["nlp"]      = [0.10, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0]

    s = TaxonomyStore(db, emb)
    root = s.ensure_root("u1")
    p_int = s.add_node("u1", root.node_id, "Personal")
    a_int = s.add_node("u1", root.node_id, "AI")
    n_int = s.add_node("u1", root.node_id, "NLP")
    p_leaf = s.add_node("u1", p_int.node_id, "PLeaf", is_leaf=True)
    a_leaf = s.add_node("u1", a_int.node_id, "ALeaf", is_leaf=True)
    n_leaf = s.add_node("u1", n_int.node_id, "NLeaf", is_leaf=True)

    for did, key in (("p", "personal"), ("a", "ai"), ("n", "nlp")):
        db.execute(
            "INSERT INTO documents(doc_id, user_id, source_path, title) VALUES (?,?,?,?)",
            (did, "u1", f"/tmp/{did}.pdf", did),
        )
        s.upsert_doc_meta("u1", did, key, emb.embed_one(key))
    s.assign_doc("u1", "p", p_leaf.node_id)
    s.assign_doc("u1", "a", a_leaf.node_id)
    s.assign_doc("u1", "n", n_leaf.node_id)
    db.commit()
    s.recompute_all_centroids("u1")

    # Without the floor, the gaps between 0.18 / 0.15 / 0.10 are tiny —
    # dominance_gap=0.10 won't trigger, so all three get kept.
    no_floor = s.beam_descend(
        "u1", q_vec, beam_width=3, max_depth=4, min_score=-1.0,
        dominance_gap=0.10, min_top_score_floor=0.0,
    )
    assert len(no_floor.trace[0].kept) == 3

    # With min_top_score_floor=0.30, the top score (0.18) is below the
    # floor → root beam collapses to 1.
    floored = s.beam_descend(
        "u1", q_vec, beam_width=3, max_depth=4, min_score=-1.0,
        dominance_gap=0.10, min_top_score_floor=0.30,
    )
    assert len(floored.trace[0].kept) == 1
    assert floored.trace[0].kept[0].node.label == "Personal"


def test_update_node(store):
    root = store.ensure_root("u1")
    n = store.add_node("u1", root.node_id, "Old Label")
    store.update_node(n.node_id, label="New Label", description="updated")
    refreshed = store.get_node(n.node_id)
    assert refreshed.label == "New Label"
    assert refreshed.description == "updated"


def test_clear_wipes_user_tree(db, store):
    root = store.ensure_root("u1")
    store.add_node("u1", root.node_id, "A")
    assert len(store.list_nodes("u1")) == 2
    store.clear("u1")
    assert store.list_nodes("u1") == []


def test_get_doc_nodes(db, store):
    root = store.ensure_root("u1")
    leaf = store.add_node("u1", root.node_id, "L", is_leaf=True)
    db.execute(
        "INSERT INTO documents(doc_id, user_id, source_path, title) VALUES (?,?,?,?)",
        ("d", "u1", "/tmp/d.pdf", "D"),
    )
    db.commit()
    store.assign_doc("u1", "d", leaf.node_id)
    nodes = store.get_doc_nodes("u1", "d")
    assert len(nodes) == 1 and nodes[0].node_id == leaf.node_id
