"""Live-Neo4j tests for :class:`hrag.kg.backends.neo4j.Neo4jBackend`.

The whole module is double-guarded so it skips cleanly on developer laptops
that don't have either the ``neo4j`` driver installed or a reachable server:

1. ``pytest.importorskip("neo4j")`` — module-level skip when the driver is
   missing.
2. Env-var guard — module-level skip when ``NEO4J_URI`` is not set.

Each test gets a fresh, isolated set of node ids prefixed with the per-session
token ``_SESSION_PREFIX`` and the test name, so concurrent test runs against
the same Neo4j instance can't collide. The session-scoped fixture wipes any
node whose id starts with ``_SESSION_PREFIX`` once at startup and again at
teardown.

To run these tests:

.. code-block:: bash

    export NEO4J_URI=bolt://localhost:7687
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=test
    pytest tests/test_neo4j_backend.py -q
"""

from __future__ import annotations

import os
import uuid

import pytest

# Skip the whole module when the driver isn't installed.
pytest.importorskip("neo4j")

# Skip the whole module when no server is configured.
if not os.environ.get("NEO4J_URI"):
    pytest.skip("NEO4J_URI not set", allow_module_level=True)


# Per-run id prefix so we never touch unrelated nodes in a shared Neo4j.
_SESSION_PREFIX = f"hrag_test_{uuid.uuid4().hex[:8]}__"


def _pid(name: str) -> str:
    """Build a session-scoped node id from a short logical name."""
    return f"{_SESSION_PREFIX}{name}"


@pytest.fixture(scope="module")
def backend():
    """Module-scoped Neo4jBackend. We wipe our prefix at module setup so
    rerunning the same test file from a crashed prior run starts clean."""
    from hrag.kg.backends.neo4j import Neo4jBackend

    be = Neo4jBackend()
    # Pre-clean any leftover nodes from a previous aborted run (in case the
    # process died before the per-test teardown ran).
    be._run(  # noqa: SLF001 - testing internals is fine here
        "MATCH (n:Node) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
        prefix=_SESSION_PREFIX,
    )
    yield be
    # Module teardown: nuke everything we created.
    be._run(  # noqa: SLF001
        "MATCH (n:Node) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
        prefix=_SESSION_PREFIX,
    )
    be.close()


@pytest.fixture(autouse=True)
def _clean_between_tests(backend):
    """Wipe just our prefix before each test for hermetic state."""
    backend._run(  # noqa: SLF001
        "MATCH (n:Node) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
        prefix=_SESSION_PREFIX,
    )
    yield


# ----------------------------------------------------------------------
# Node-level tests
# ----------------------------------------------------------------------


def test_add_node_then_has_node(backend):
    nid = _pid("phrase_a")
    assert backend.has_node(nid) is False
    backend.add_node(nid, label="transformer", kind="phrase")
    assert backend.has_node(nid) is True


def test_get_node_data_roundtrip_primitive_attrs(backend):
    nid = _pid("phrase_b")
    backend.add_node(nid, label="attention", weight=0.5, count=3, is_seed=True)
    data = backend.get_node_data(nid)
    assert data["label"] == "attention"
    assert data["weight"] == pytest.approx(0.5)
    assert data["count"] == 3
    assert data["is_seed"] is True
    # The merge key ``id`` is not surfaced as a node attribute.
    assert "id" not in data


def test_get_node_data_roundtrip_non_primitive_attrs(backend):
    """Attrs that aren't Neo4j primitives must go through the __json_attrs
    sidecar and come back intact."""
    nid = _pid("phrase_c")
    backend.add_node(
        nid,
        label="bert",
        # dict → JSON sidecar
        provenance={"doc": "abc", "page": 7},
        # set → JSON sidecar (will round-trip as a sorted list)
        aliases={"BERT", "bert"},
        # list of primitives → stays a native list
        token_ids=[101, 102, 103],
    )
    data = backend.get_node_data(nid)
    assert data["label"] == "bert"
    assert data["provenance"] == {"doc": "abc", "page": 7}
    assert sorted(data["aliases"]) == ["BERT", "bert"]
    assert data["token_ids"] == [101, 102, 103]


def test_get_node_data_missing_raises_keyerror(backend):
    with pytest.raises(KeyError):
        backend.get_node_data(_pid("does_not_exist"))


def test_remove_node_is_idempotent(backend):
    nid = _pid("phrase_d")
    backend.add_node(nid, label="x")
    assert backend.has_node(nid)
    backend.remove_node(nid)
    assert not backend.has_node(nid)
    # Removing again must not raise.
    backend.remove_node(nid)


def test_iter_nodes_and_number_of_nodes(backend):
    ids = [_pid(f"n{i}") for i in range(5)]
    for i, nid in enumerate(ids):
        backend.add_node(nid, idx=i)

    # Filter to just our session prefix — other sessions may exist on shared DBs.
    seen = {nid: attrs for nid, attrs in backend.iter_nodes() if nid.startswith(_SESSION_PREFIX)}
    assert set(seen) == set(ids)
    for i, nid in enumerate(ids):
        assert seen[nid]["idx"] == i

    # number_of_nodes is a global count; we assert >= len(ids) because the DB
    # may hold unrelated data, but the count over our prefix matches exactly.
    assert backend.number_of_nodes() >= len(ids)


# ----------------------------------------------------------------------
# Edge-level tests
# ----------------------------------------------------------------------


def test_add_edge_with_multiple_keys_between_same_pair(backend):
    src = _pid("e_src")
    dst = _pid("e_dst")
    backend.add_node(src)
    backend.add_node(dst)

    k1 = backend.add_edge(src, dst, relation="cites", weight=1.0)
    k2 = backend.add_edge(src, dst, relation="extends", weight=2.0)
    k3 = backend.add_edge(src, dst, relation="cites", weight=3.0)
    assert len({k1, k2, k3}) == 3, "edge keys must be unique"

    data = backend.get_edge_data(src, dst)
    assert data is not None
    assert set(data.keys()) == {k1, k2, k3}
    assert data[k1]["relation"] == "cites"
    assert data[k2]["relation"] == "extends"
    assert data[k3]["weight"] == pytest.approx(3.0)


def test_get_edge_data_missing_returns_none(backend):
    assert backend.get_edge_data(_pid("no_src"), _pid("no_dst")) is None


def test_remove_edge_with_key(backend):
    src = _pid("r_src")
    dst = _pid("r_dst")
    k1 = backend.add_edge(src, dst, relation="a")
    k2 = backend.add_edge(src, dst, relation="b")
    backend.remove_edge(src, dst, key=k1)
    data = backend.get_edge_data(src, dst)
    assert data is not None
    assert set(data.keys()) == {k2}


def test_iter_edges_and_number_of_edges(backend):
    src = _pid("ie_src")
    dst1 = _pid("ie_d1")
    dst2 = _pid("ie_d2")
    backend.add_edge(src, dst1, relation="r1")
    backend.add_edge(src, dst2, relation="r2", weight=0.7)

    edges = [
        e for e in backend.iter_edges(data=True)
        if e[0].startswith(_SESSION_PREFIX)
    ]
    assert len(edges) == 2
    rels = {(u, v): d.get("relation") for u, v, d in edges}
    assert rels[(src, dst1)] == "r1"
    assert rels[(src, dst2)] == "r2"

    edges_k = [
        e for e in backend.iter_edges(keys=True, data=True)
        if e[0].startswith(_SESSION_PREFIX)
    ]
    assert len(edges_k) == 2
    assert all(isinstance(e[2], str) for e in edges_k)  # key is a hex string

    # Bare 2-tuples
    edges_bare = [
        e for e in backend.iter_edges(keys=False, data=False)
        if e[0].startswith(_SESSION_PREFIX)
    ]
    assert all(len(e) == 2 for e in edges_bare)

    assert backend.number_of_edges() >= 2


def test_successors_predecessors_in_edges(backend):
    a = _pid("trav_a")
    b = _pid("trav_b")
    c = _pid("trav_c")
    backend.add_edge(a, b, relation="r1")
    backend.add_edge(a, b, relation="r2")  # parallel edge — DISTINCT in successors
    backend.add_edge(c, b, relation="r3")

    succs = list(backend.successors(a))
    assert succs == [b]  # DISTINCT collapses the parallel edges

    preds = sorted(backend.predecessors(b))
    assert preds == sorted([a, c])

    in_edges = list(backend.in_edges(b, data=True))
    src_set = {src for src, _dst, _d in in_edges}
    assert src_set == {a, c}
    # parallel edges show up as separate rows
    assert len([e for e in in_edges if e[0] == a]) == 2

    in_edges_bare = list(backend.in_edges(b, data=False))
    assert all(len(e) == 2 for e in in_edges_bare)


# ----------------------------------------------------------------------
# Bulk export tests
# ----------------------------------------------------------------------


def test_to_sparse_adjacency_tiny_graph(backend):
    """A -> B (weight 2), A -> C (default 1), B -> C (default 1)."""
    import numpy as np

    a = _pid("adj_a")
    b = _pid("adj_b")
    c = _pid("adj_c")
    backend.add_node(a)
    backend.add_node(b)
    backend.add_node(c)
    backend.add_edge(a, b, weight=2.0)
    backend.add_edge(a, c)
    backend.add_edge(b, c)

    mat, ids = backend.to_sparse_adjacency()
    # Filter to our session — sparse adjacency is over all nodes in the DB.
    our_idx = {ids[i]: i for i in range(len(ids)) if ids[i].startswith(_SESSION_PREFIX)}
    assert set(our_idx) == {a, b, c}

    dense = mat.toarray()
    assert dense[our_idx[a], our_idx[b]] == pytest.approx(2.0)
    assert dense[our_idx[a], our_idx[c]] == pytest.approx(1.0)
    assert dense[our_idx[b], our_idx[c]] == pytest.approx(1.0)
    # No reverse edges.
    assert dense[our_idx[b], our_idx[a]] == 0.0
    # Total weight sum over our 3 edges = 4.0
    sub = dense[np.ix_([our_idx[a], our_idx[b], our_idx[c]],
                       [our_idx[a], our_idx[b], our_idx[c]])]
    assert sub.sum() == pytest.approx(4.0)


def test_to_networkx_roundtrip(backend):
    import networkx as nx

    a = _pid("nx_a")
    b = _pid("nx_b")
    backend.add_node(a, label="hello", aliases={"hi", "hey"})
    backend.add_node(b, label="world")
    k1 = backend.add_edge(a, b, relation="r1", weight=0.5)
    k2 = backend.add_edge(a, b, relation="r2")

    g = backend.to_networkx()
    assert isinstance(g, nx.MultiDiGraph)
    # The graph contains all DB nodes; check ours are present + correct.
    assert a in g
    assert b in g
    assert g.nodes[a]["label"] == "hello"
    assert sorted(g.nodes[a]["aliases"]) == ["hey", "hi"]

    edges_ab = g.get_edge_data(a, b)
    assert edges_ab is not None
    assert set(edges_ab.keys()) == {k1, k2}
    assert edges_ab[k1]["relation"] == "r1"
    assert edges_ab[k1]["weight"] == pytest.approx(0.5)
    assert edges_ab[k2]["relation"] == "r2"


def test_save_load_are_noops(backend, tmp_path):
    # Should not raise — server-backed store has no on-disk artefact.
    backend.save(tmp_path / "irrelevant.pkl")
    backend.load(tmp_path / "also_irrelevant.pkl")
