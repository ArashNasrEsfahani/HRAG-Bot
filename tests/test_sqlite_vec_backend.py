"""Tests for :class:`SqliteVecBackend` — the sqlite-vec vector backend.

The whole module is skipped when the ``sqlite_vec`` package is unavailable
(via :func:`pytest.importorskip`) so CI without the optional dep stays green.
"""

from __future__ import annotations

import math

import pytest

# Skip the whole module if the optional dep is absent.
pytest.importorskip("sqlite_vec")

from hrag.retrieval.backends.sqlite_vec import SqliteVecBackend  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v: list[float]) -> list[float]:
    """Return ``v`` scaled to unit length (L2-normalised)."""
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return v
    return [x / n for x in v]


def _make_backend(tmp_path) -> SqliteVecBackend:
    return SqliteVecBackend(tmp_path / "vec")


# Three orthonormal-ish probe vectors used across tests. The whole pipeline
# treats embeddings as L2-normalised (Phase-3 contract #6) so we honour that.
E_X = _unit([1.0, 0.0, 0.0])
E_Y = _unit([0.0, 1.0, 0.0])
E_NEAR_X = _unit([0.95, 0.05, 0.0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_count(tmp_path):
    """Fresh backend reports 0 records and a query returns no rows."""
    b = _make_backend(tmp_path)
    assert b.count() == 0
    ids, dists = b.query_one(E_X, top_k=5, where=None)
    assert ids == []
    assert dists == []


def test_upsert_and_count(tmp_path):
    b = _make_backend(tmp_path)
    b.upsert(
        ids=["a", "b"],
        embeddings=[E_X, E_Y],
        documents=["doc a", "doc b"],
        metadatas=[
            {"user_id": "u1", "doc_id": "d1", "source_type": "document", "excluded": 0},
            {"user_id": "u1", "doc_id": "d2", "source_type": "document", "excluded": 0},
        ],
    )
    assert b.count() == 2

    # Re-upsert the same id should NOT duplicate.
    b.upsert(
        ids=["a"],
        embeddings=[E_X],
        documents=["doc a v2"],
        metadatas=[{"user_id": "u1", "doc_id": "d1", "source_type": "document", "excluded": 0}],
    )
    assert b.count() == 2


def test_query_one_returns_nearest(tmp_path):
    """The query vector is closest to its near twin and farthest from the orthogonal vector."""
    b = _make_backend(tmp_path)
    b.upsert(
        ids=["near", "far"],
        embeddings=[E_NEAR_X, E_Y],
        documents=["near to x", "orthogonal"],
        metadatas=[
            {"user_id": "u1", "source_type": "document", "excluded": 0},
            {"user_id": "u1", "source_type": "document", "excluded": 0},
        ],
    )
    ids, dists = b.query_one(E_X, top_k=2, where=None)
    assert ids[0] == "near"
    assert ids[1] == "far"
    # cosine distances must be ascending; nearest is < far.
    assert dists[0] < dists[1]
    # Sanity: orthogonal vectors give cosine distance ~ 1.
    assert dists[1] == pytest.approx(1.0, abs=1e-3)
    # Near twin has cosine distance ~ 1 - 0.95 = 0.05 (approximate).
    assert dists[0] < 0.1


def test_where_filter_narrows_results(tmp_path):
    """Equality + $eq + $in filters all narrow result set."""
    b = _make_backend(tmp_path)
    b.upsert(
        ids=["u1a", "u1b", "u2a"],
        embeddings=[E_X, E_NEAR_X, E_X],
        documents=["u1 first", "u1 second", "u2 first"],
        metadatas=[
            {"user_id": "u1", "doc_id": "d1", "source_type": "document", "excluded": 0},
            {"user_id": "u1", "doc_id": "d2", "source_type": "episodic", "excluded": 0},
            {"user_id": "u2", "doc_id": "d1", "source_type": "document", "excluded": 0},
        ],
    )

    # Flat equality.
    ids, _ = b.query_one(E_X, top_k=10, where={"user_id": "u1"})
    assert set(ids) == {"u1a", "u1b"}

    # $eq operator.
    ids, _ = b.query_one(E_X, top_k=10, where={"user_id": {"$eq": "u2"}})
    assert ids == ["u2a"]

    # $and with $eq.
    where = {"$and": [{"user_id": {"$eq": "u1"}}, {"source_type": {"$eq": "episodic"}}]}
    ids, _ = b.query_one(E_X, top_k=10, where=where)
    assert ids == ["u1b"]

    # $in list.
    ids, _ = b.query_one(E_X, top_k=10, where={"doc_id": {"$in": ["d1"]}})
    assert set(ids) == {"u1a", "u2a"}

    # Excluded tombstone filter — $eq=0.
    ids, _ = b.query_one(E_X, top_k=10, where={"excluded": {"$eq": 0}})
    assert set(ids) == {"u1a", "u1b", "u2a"}


def test_delete_where(tmp_path):
    b = _make_backend(tmp_path)
    b.upsert(
        ids=["a", "b", "c"],
        embeddings=[E_X, E_NEAR_X, E_Y],
        documents=["A", "B", "C"],
        metadatas=[
            {"user_id": "u1", "doc_id": "d1", "source_type": "document", "excluded": 0},
            {"user_id": "u1", "doc_id": "d1", "source_type": "document", "excluded": 0},
            {"user_id": "u1", "doc_id": "d2", "source_type": "document", "excluded": 0},
        ],
    )
    assert b.count() == 3

    # Delete everything from d1.
    b.delete_where({"$and": [{"user_id": {"$eq": "u1"}}, {"doc_id": {"$eq": "d1"}}]})
    assert b.count() == 1
    ids, _ = b.query_one(E_X, top_k=10, where=None)
    assert ids == ["c"]

    # Deleting with an unmatched filter is a no-op.
    b.delete_where({"doc_id": {"$eq": "does-not-exist"}})
    assert b.count() == 1


def test_update_metadata_changes_filterable_field(tmp_path):
    """update_metadata can flip ``excluded`` so a row is hidden by filter."""
    b = _make_backend(tmp_path)
    b.upsert(
        ids=["a"],
        embeddings=[E_X],
        documents=["A"],
        metadatas=[{"user_id": "u1", "source_type": "document", "excluded": 0}],
    )

    ids, _ = b.query_one(E_X, top_k=10, where={"excluded": {"$eq": 0}})
    assert ids == ["a"]

    # Tombstone the row.
    b.update_metadata(["a"], [{"excluded": 1}])

    ids, _ = b.query_one(E_X, top_k=10, where={"excluded": {"$eq": 0}})
    assert ids == []

    ids, _ = b.query_one(E_X, top_k=10, where={"excluded": {"$eq": 1}})
    assert ids == ["a"]

    # update on an unknown id is a no-op, not an error.
    b.update_metadata(["nope"], [{"excluded": 0}])
    assert b.count() == 1


def test_persistence_across_reopen(tmp_path):
    """Closing and re-opening the backend preserves all rows and the dim."""
    b1 = _make_backend(tmp_path)
    b1.upsert(
        ids=["a", "b"],
        embeddings=[E_X, E_Y],
        documents=["A", "B"],
        metadatas=[
            {"user_id": "u1", "source_type": "document", "excluded": 0},
            {"user_id": "u1", "source_type": "document", "excluded": 0},
        ],
    )
    assert b1.count() == 2
    b1.close()

    b2 = SqliteVecBackend(tmp_path / "vec")
    assert b2.count() == 2
    ids, _ = b2.query_one(E_X, top_k=2, where=None)
    assert ids[0] == "a"  # nearest to E_X
    assert set(ids) == {"a", "b"}
    b2.close()


def test_dim_mismatch_raises(tmp_path):
    """Upserting a vector with the wrong dim after the table is created raises."""
    b = _make_backend(tmp_path)
    b.upsert(
        ids=["a"],
        embeddings=[E_X],  # dim=3
        documents=["A"],
        metadatas=[{"user_id": "u1", "source_type": "document", "excluded": 0}],
    )
    with pytest.raises(ValueError):
        b.upsert(
            ids=["b"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],  # dim=4
            documents=["B"],
            metadatas=[{"user_id": "u1", "source_type": "document", "excluded": 0}],
        )


def test_empty_inputs_are_safe(tmp_path):
    """Upserts and updates on empty input lists are no-ops, not errors."""
    b = _make_backend(tmp_path)
    b.upsert(ids=[], embeddings=[], documents=[], metadatas=[])
    b.update_metadata([], [])
    assert b.count() == 0
    # delete_where with an empty dict must be a safe no-op (NOT a wipe).
    b.upsert(
        ids=["a"],
        embeddings=[E_X],
        documents=["A"],
        metadatas=[{"user_id": "u1", "source_type": "document", "excluded": 0}],
    )
    b.delete_where({})
    assert b.count() == 1
