"""Phase 12 — hybrid keyword routing for the taxonomy.

Covers the pure keyword module (bilingual EN/FA, ZWNJ), the store's keywords
CRUD + cache invalidation, and the hybrid beam-descend scoring (no-op at
weight 0; reorders a planted tie when keywords decide it).
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import pytest

from hrag.db.connection import Database
from hrag.taxonomy import keywords as kw
from hrag.taxonomy.store import TaxonomyStore


# ---------------------------------------------------------------------------
# 1. Pure keyword module
# ---------------------------------------------------------------------------


def test_tokenize_english_drops_stopwords():
    toks = kw.tokenize("What is the reinforcement learning for robots?")
    assert "reinforcement" in toks and "learning" in toks and "robots" in toks
    assert "the" not in toks and "is" not in toks and "for" not in toks


def test_tokenize_persian_strips_zwnj():
    # "robots" written with a ZWNJ (ربات‌ها) must normalize to the joined form.
    toks = kw.tokenize("یادگیری تقویتی برای ربات‌ها")
    assert "ربات‌ها" not in toks  # ZWNJ form must not survive
    assert "رباتها" in toks
    assert "یادگیری" in toks and "تقویتی" in toks


def test_extract_keywords_surfaces_phrase():
    kws = kw.extract_keywords(
        [
            "Soft actor-critic reinforcement learning for dexterous robotic manipulation",
            "Deep reinforcement learning policy gradients for robot control",
        ],
        top_k=6,
    )
    # The shared salient bigram should rank at/near the top.
    assert "reinforcement learning" in kws
    # Pure function: empty input → empty output.
    assert kw.extract_keywords([]) == []


def test_keyword_overlap_bounds_and_values():
    assert kw.keyword_overlap([], ["a"]) == 0.0
    assert kw.keyword_overlap(["a"], []) == 0.0
    # node has tokens {reinforcement, learning, robot, control}; query supplies
    # reinforcement, learning, robot → 3/4.
    ov = kw.keyword_overlap(
        ["reinforcement", "learning", "robot"],
        ["reinforcement learning", "robot control"],
    )
    assert ov == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Store fixtures
# ---------------------------------------------------------------------------


class DictEmbedder:
    name = "dict"
    _DIM = 8

    def __init__(self, mapping=None):
        self._map = dict(mapping or {})

    def embed(self, texts: Sequence[str]):
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str):
        if text in self._map:
            return list(self._map[text])
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(b / 255.0) * 2.0 - 1.0 for b in digest[: self._DIM]]

    @property
    def dim(self) -> int:
        return self._DIM


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.sqlite"
    d = Database(path)
    d.init_schema()
    d.ensure_user("u1")
    yield d
    d.close()


@pytest.fixture
def store(db):
    return TaxonomyStore(db, DictEmbedder())


# ---------------------------------------------------------------------------
# 2. Store keywords CRUD
# ---------------------------------------------------------------------------


def test_add_node_persists_keywords(store):
    root = store.ensure_root("u1")
    node = store.add_node("u1", root.node_id, "RL", keywords=["reinforcement", "policy"])
    reloaded = store.get_node(node.node_id)
    assert reloaded.keywords == ["reinforcement", "policy"]


def test_set_node_keywords_round_trip_and_cache_invalidation(store):
    root = store.ensure_root("u1")
    node = store.add_node("u1", root.node_id, "RL")
    assert store.get_node(node.node_id).keywords == []
    # Warm the per-user cache, then mutate, then confirm the read is fresh.
    _ = store.list_nodes("u1")
    store.set_node_keywords("u1", node.node_id, ["reward", "agent"])
    fresh = [n for n in store.list_nodes("u1") if n.node_id == node.node_id][0]
    assert fresh.keywords == ["reward", "agent"]


def test_missing_keywords_decode_to_empty(store):
    root = store.ensure_root("u1")
    n = store.add_node("u1", root.node_id, "X")  # no keywords arg
    assert store.get_node(n.node_id).keywords == []


# ---------------------------------------------------------------------------
# 3. Hybrid beam descend
# ---------------------------------------------------------------------------


def _planted_tie_tree(db):
    """Root → two leaves with IDENTICAL centroids but different keywords."""
    emb = DictEmbedder()
    q_vec = [1.0] + [0.0] * 7
    s = TaxonomyStore(db, emb)
    root = s.ensure_root("u1")
    # Leaves directly under root. Same centroid (via the same assigned doc vec).
    leaf_a = s.add_node("u1", root.node_id, "Alpha", is_leaf=True,
                        keywords=["finance", "market", "stock"])
    leaf_b = s.add_node("u1", root.node_id, "Beta", is_leaf=True,
                        keywords=["reinforcement", "learning", "robot"])
    for did, leaf in (("da", leaf_a), ("db", leaf_b)):
        db.execute(
            "INSERT INTO documents(doc_id, user_id, source_path, title) VALUES (?,?,?,?)",
            (did, "u1", f"/tmp/{did}.pdf", did),
        )
    db.commit()
    s.upsert_doc_meta("u1", "da", "a", q_vec)
    s.upsert_doc_meta("u1", "db", "b", q_vec)
    s.assign_doc("u1", "da", leaf_a.node_id)
    s.assign_doc("u1", "db", leaf_b.node_id)
    s.recompute_all_centroids("u1")
    return s, q_vec


def test_keyword_weight_zero_is_noop(db):
    s, q_vec = _planted_tie_tree(db)
    res = s.beam_descend("u1", q_vec, beam_width=1, max_depth=4, min_score=-1.0,
                         query_keywords=["reinforcement", "robot"], keyword_weight=0.0)
    # With weight 0 the keyword signal is ignored — tie broken by label order
    # ("Alpha" first), and every considered node reports keyword_score 0.
    assert res.leaves[0].node.label == "Alpha"
    for level in res.trace:
        for ns in level.considered:
            assert ns.keyword_score == 0.0


def test_keyword_weight_reorders_tie(db):
    s, q_vec = _planted_tie_tree(db)
    # Query keywords match Beta's keywords → Beta should win despite the cosine
    # tie and despite "Alpha" sorting first alphabetically.
    res = s.beam_descend("u1", q_vec, beam_width=1, max_depth=4, min_score=-1.0,
                         query_keywords=["reinforcement", "learning", "robot"],
                         keyword_weight=0.5)
    assert res.leaves[0].node.label == "Beta"
    # The winning node carries a positive keyword contribution in the trace.
    beta = [ns for lvl in res.trace for ns in lvl.considered
            if ns.node.label == "Beta"][0]
    assert beta.keyword_score > 0.0
