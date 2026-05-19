"""Verify TaxonomyRetriever respects the max_docs_pct safety cap.

When the taxonomy is imbalanced — one leaf owns most of the corpus — a
low-confidence query can still pick that leaf and end up retrieving from
most of the library. The cap drops the weakest leaves until the union of
their docs is within the configured share of the user's total corpus.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import pytest

from hrag.config import TaxonomyConfig
from hrag.db.connection import Database
from hrag.retrieval.taxonomy import TaxonomyRetriever
from hrag.taxonomy.store import TaxonomyStore


class _DictEmbedder:
    """Deterministic 8-dim embedder with a name→vector override map."""

    name = "dict"
    _DIM = 8

    def __init__(self, mapping=None) -> None:
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


class _SpyVectorStore:
    """Captures the doc_ids allow-list passed by the retriever.

    Phase 8.2: the retriever now does a SECOND vector query with no
    ``doc_ids`` to surface global episodic memories (memories filed under
    a sibling leaf the beam did not pick). Capture only the FIRST call —
    that's the leaf-doc allow-list this fixture was designed to inspect.
    """

    def __init__(self) -> None:
        self.last_doc_ids: list[str] | None = None
        self._captured = False

    def query(self, *, user_id, query_embedding, top_k, source_types=None, doc_ids=None, where=None):
        if not self._captured:
            self.last_doc_ids = list(doc_ids) if doc_ids else []
            self._captured = True
        return []


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.sqlite"
    db = Database(path)
    db.init_schema()
    db.ensure_user("u1")
    yield db
    db.close()


def test_max_docs_pct_trims_imbalanced_tree(db) -> None:
    """A tree where one leaf holds 14/20 docs (70%) must NOT open that leaf
    in full when ``max_docs_pct`` is set to 0.40 — the retriever should
    drop it (and any subsequent low-score leaves) so the union stays at
    or below the cap."""
    target_vec = [1.0] + [0.0] * 7
    second_vec = [0.95, 0.31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    big_vec    = [0.85, 0.0, 0.53, 0.0, 0.0, 0.0, 0.0, 0.0]  # high-score "big" leaf

    emb = _DictEmbedder({
        "query":  target_vec,
        "small":  target_vec,
        "second": second_vec,
        "big":    big_vec,
    })

    store = TaxonomyStore(db, emb)
    root = store.ensure_root("u1")
    small_leaf  = store.add_node("u1", root.node_id, "SmallLeaf",  is_leaf=True)
    second_leaf = store.add_node("u1", root.node_id, "SecondLeaf", is_leaf=True)
    big_leaf    = store.add_node("u1", root.node_id, "BigLeaf",    is_leaf=True)

    # Seed: 3 docs on small, 3 on second, 14 on big. Total = 20.
    def _seed(prefix: str, n: int, leaf_id: str, key: str) -> None:
        for i in range(n):
            did = f"{prefix}{i}"
            db.execute(
                "INSERT INTO documents(doc_id, user_id, source_path, title) VALUES (?,?,?,?)",
                (did, "u1", f"/tmp/{did}.pdf", did),
            )
            store.upsert_doc_meta("u1", did, key, emb.embed_one(key))
            store.assign_doc("u1", did, leaf_id)

    _seed("s",  3,  small_leaf.node_id,  "small")
    _seed("c",  3,  second_leaf.node_id, "second")
    _seed("b",  14, big_leaf.node_id,    "big")
    db.commit()
    store.recompute_all_centroids("u1")

    cfg = TaxonomyConfig(
        enabled=True,
        beam_width=3,
        max_depth=4,
        min_node_score=-1.0,        # let all three leaves through the floor
        beam_dominance_gap=0.0,     # disable gap pruning to isolate the cap
        min_top_score_floor=0.0,    # disable confidence floor
        max_docs_pct=0.40,          # cap at 40% of 20 = 8 docs
    )

    spy = _SpyVectorStore()
    retriever = TaxonomyRetriever(
        db=db,
        vector_store=spy,                     # type: ignore[arg-type]
        embedder=emb,                         # type: ignore[arg-type]
        taxonomy_store=store,
        cfg=cfg,
        fallback=None,                        # type: ignore[arg-type] — not used
    )

    retriever.retrieve("query", "u1", top_k=10)

    # The cap is 8 docs (40% of 20). small=3 fits → add. second=3, union=6 → add.
    # big=14, union would be 20 → reject. So we expect 6 docs in the allow-list.
    assert spy.last_doc_ids is not None
    assert len(spy.last_doc_ids) == 6
    # And the picked leaves reflect what actually drove retrieval — big is gone.
    picked_labels = {lf.node.label for lf in retriever.last_descend.leaves}
    assert "BigLeaf" not in picked_labels
    assert {"SmallLeaf", "SecondLeaf"}.issubset(picked_labels)


# NOTE: tests for `short_query_force_top1_words` were removed when that knob
# was deprecated. Short queries like "hey" never reach retrieval at all now —
# they're routed by the intent classifier (see tests/test_intent.py).


def test_max_docs_pct_disabled_keeps_all_leaves(db) -> None:
    """With ``max_docs_pct=1.0`` (disabled) every picked leaf survives,
    even on imbalanced trees."""
    target_vec = [1.0] + [0.0] * 7
    emb = _DictEmbedder({"query": target_vec, "k": target_vec})

    store = TaxonomyStore(db, emb)
    root = store.ensure_root("u1")
    a = store.add_node("u1", root.node_id, "A", is_leaf=True)
    b = store.add_node("u1", root.node_id, "B", is_leaf=True)

    for did, leaf in (("d1", a), ("d2", b), ("d3", b)):
        db.execute(
            "INSERT INTO documents(doc_id, user_id, source_path, title) VALUES (?,?,?,?)",
            (did, "u1", f"/tmp/{did}.pdf", did),
        )
        store.upsert_doc_meta("u1", did, "k", target_vec)
        store.assign_doc("u1", did, leaf.node_id)
    db.commit()
    store.recompute_all_centroids("u1")

    cfg = TaxonomyConfig(
        enabled=True, beam_width=3, max_depth=4,
        min_node_score=-1.0, beam_dominance_gap=0.0,
        min_top_score_floor=0.0,
        max_docs_pct=1.0,    # disabled
    )

    spy = _SpyVectorStore()
    retriever = TaxonomyRetriever(
        db=db, vector_store=spy, embedder=emb,        # type: ignore[arg-type]
        taxonomy_store=store, cfg=cfg, fallback=None, # type: ignore[arg-type]
    )

    retriever.retrieve("query", "u1", top_k=10)
    assert spy.last_doc_ids is not None
    assert sorted(spy.last_doc_ids) == ["d1", "d2", "d3"]
