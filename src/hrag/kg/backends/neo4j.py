"""Neo4j-backed :class:`KGBackend` implementation.

This backend stores the HRAG knowledge graph in a running Neo4j server using
the official ``neo4j`` Python driver. Every protocol method maps to a single
parameterised Cypher query against a single label ``:Node`` and a single
relationship type ``:LINK``.

Design notes
------------
* The merge key is ``n.id`` (the HRAG node id, e.g. ``"phrase:transformer"``
  or ``"passage:abc123"``). All other attributes go on as native properties
  when they are primitives (or lists of primitives), and are JSON-serialised
  into a single ``__json_attrs`` string property otherwise — Neo4j does not
  support nested dicts as node/edge properties.
* Multi-edges between the same pair are supported via a per-edge ``r.key``
  (a Python ``uuid.uuid4().hex``); this mirrors networkx ``MultiDiGraph``
  semantics. ``add_edge`` returns the generated key.
* :meth:`save` and :meth:`load` are no-ops — Neo4j is the durable store and
  every Cypher write auto-commits within its transaction.
* :meth:`to_networkx` does an O(V+E) sweep over Cypher and rebuilds a
  ``networkx.MultiDiGraph`` in memory. This is the slow path; it is only
  used by the community detector and the MST organiser.

Setup
-----
The class import is zero-cost (no ``neo4j`` driver import at module load).
The driver is imported in :meth:`__init__`; if either the driver is missing
or no URI is configured (env var ``NEO4J_URI`` or constructor kwarg), the
constructor raises ``RuntimeError`` with a clear setup hint.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx
    import scipy.sparse as sp


logger = logging.getLogger(__name__)


# Properties Neo4j accepts natively as node / relationship properties:
# bool, int, float, str, and homogeneous lists of those. Anything else
# (dict, set, tuple containing nested structures, etc.) is funnelled into
# the ``__json_attrs`` sidecar property.
_PRIMITIVE_TYPES = (bool, int, float, str)


def _split_attrs(attrs: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Partition *attrs* into Neo4j-native primitives and a JSON sidecar.

    Returns ``(primitive_attrs, json_attrs_str_or_None)``.
    """
    primitive: dict[str, Any] = {}
    needs_json: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, _PRIMITIVE_TYPES):
            # bool is a subclass of int — both fall into _PRIMITIVE_TYPES.
            primitive[key] = value
            continue
        if isinstance(value, list) and all(isinstance(x, _PRIMITIVE_TYPES) for x in value):
            primitive[key] = value
            continue
        # Sets, dicts, tuples, lists-of-non-primitives, custom objects → JSON.
        try:
            # ``default=list`` lets sets round-trip as lists (caller can convert
            # back if it wants set semantics).
            needs_json[key] = value if not isinstance(value, set) else sorted(value)
        except TypeError:
            needs_json[key] = repr(value)
    json_attrs = json.dumps(needs_json, default=str) if needs_json else None
    return primitive, json_attrs


def _unpack_props(props: dict[str, Any]) -> dict[str, Any]:
    """Reverse of :func:`_split_attrs`: merge ``__json_attrs`` back into the
    flat attribute dict. The sidecar key is stripped from the result."""
    out = dict(props)
    js = out.pop("__json_attrs", None)
    if js:
        try:
            out.update(json.loads(js))
        except json.JSONDecodeError:
            logger.warning("Neo4jBackend: __json_attrs is not valid JSON; ignoring.")
    return out


class Neo4jBackend:
    """:class:`KGBackend` backed by a Neo4j server.

    Connection parameters are taken from constructor kwargs, falling back to
    the env vars ``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD`` /
    ``NEO4J_DATABASE``. The driver and the bolt connection are opened in
    ``__init__``; closing happens in :meth:`close` and on garbage collection.
    """

    name = "neo4j"

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        **kwargs: Any,  # noqa: ARG002 - reserved for future driver options
    ) -> None:
        try:
            from neo4j import GraphDatabase  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised manually
            raise RuntimeError(
                "Neo4jBackend requires a running Neo4j server. "
                "Set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD env vars or pass "
                "them to the constructor."
            ) from exc

        resolved_uri = uri or os.environ.get("NEO4J_URI")
        if not resolved_uri:
            raise RuntimeError(
                "Neo4jBackend requires a running Neo4j server. "
                "Set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD env vars or pass "
                "them to the constructor."
            )

        resolved_user = user or os.environ.get("NEO4J_USER", "neo4j")
        resolved_password = password or os.environ.get("NEO4J_PASSWORD", "")
        self._db = database or os.environ.get("NEO4J_DATABASE", "neo4j")

        self._driver = GraphDatabase.driver(
            resolved_uri, auth=(resolved_user, resolved_password)
        )

        # Ensure a uniqueness constraint on Node.id so MERGE is fast and safe.
        # If the constraint already exists, Neo4j is a no-op.
        try:
            with self._driver.session(database=self._db) as session:
                session.run(
                    "CREATE CONSTRAINT hrag_node_id_unique IF NOT EXISTS "
                    "FOR (n:Node) REQUIRE n.id IS UNIQUE"
                )
        except Exception as exc:  # noqa: BLE001 - best-effort, log and continue
            logger.warning("Neo4jBackend: failed to create id constraint: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying bolt driver. Safe to call multiple times."""
        drv = getattr(self, "_driver", None)
        if drv is not None:
            try:
                drv.close()
            except Exception:  # noqa: BLE001
                pass
            self._driver = None  # type: ignore[assignment]

    def __del__(self) -> None:  # pragma: no cover - GC timing
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # Internal helper -------------------------------------------------------

    def _run(self, cypher: str, **params: Any):
        """Execute *cypher* in a fresh session and return the result records.

        We materialise into a list immediately so the session can be closed
        before the caller iterates.
        """
        with self._driver.session(database=self._db) as session:
            result = session.run(cypher, **params)
            return list(result)

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, **attrs: Any) -> None:
        primitive, json_attrs = _split_attrs(attrs)
        # MERGE on id; then SET native primitives + the JSON sidecar. We use
        # two SET clauses because ``n += $map`` requires a map argument, while
        # the JSON property is set separately for cleanliness (and so callers
        # who pass no non-primitive attrs don't leave a stale __json_attrs).
        if json_attrs is None:
            self._run(
                "MERGE (n:Node {id: $node_id}) SET n += $primitive_attrs "
                "REMOVE n.__json_attrs",
                node_id=node_id,
                primitive_attrs=primitive,
            )
        else:
            self._run(
                "MERGE (n:Node {id: $node_id}) SET n += $primitive_attrs, "
                "n.__json_attrs = $json_attrs",
                node_id=node_id,
                primitive_attrs=primitive,
                json_attrs=json_attrs,
            )

    def remove_node(self, node_id: str) -> None:
        self._run(
            "MATCH (n:Node {id: $node_id}) DETACH DELETE n",
            node_id=node_id,
        )

    def has_node(self, node_id: str) -> bool:
        records = self._run(
            "MATCH (n:Node {id: $node_id}) RETURN count(n) > 0 AS present",
            node_id=node_id,
        )
        return bool(records[0]["present"]) if records else False

    def get_node_data(self, node_id: str) -> dict[str, Any]:
        records = self._run(
            "MATCH (n:Node {id: $node_id}) RETURN properties(n) AS props",
            node_id=node_id,
        )
        if not records:
            raise KeyError(node_id)
        props = dict(records[0]["props"])
        # ``id`` is the merge key; networkx semantics don't surface it as a
        # node attribute, so drop it from the returned dict.
        props.pop("id", None)
        return _unpack_props(props)

    def iter_nodes(self) -> Iterator[tuple[str, dict[str, Any]]]:
        records = self._run(
            "MATCH (n:Node) RETURN n.id AS id, properties(n) AS props"
        )
        for rec in records:
            props = dict(rec["props"])
            props.pop("id", None)
            yield rec["id"], _unpack_props(props)

    def number_of_nodes(self) -> int:
        records = self._run("MATCH (n:Node) RETURN count(n) AS n")
        return int(records[0]["n"]) if records else 0

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, src: str, dst: str, **attrs: Any) -> Any:
        primitive, json_attrs = _split_attrs(attrs)
        key = uuid.uuid4().hex
        # Make endpoints exist so the call is idempotent (matches networkx,
        # which auto-creates nodes referenced by ``add_edge``).
        if json_attrs is None:
            self._run(
                "MERGE (s:Node {id: $src}) "
                "MERGE (t:Node {id: $dst}) "
                "CREATE (s)-[r:LINK {key: $key}]->(t) "
                "SET r += $primitive_attrs",
                src=src,
                dst=dst,
                key=key,
                primitive_attrs=primitive,
            )
        else:
            self._run(
                "MERGE (s:Node {id: $src}) "
                "MERGE (t:Node {id: $dst}) "
                "CREATE (s)-[r:LINK {key: $key}]->(t) "
                "SET r += $primitive_attrs, r.__json_attrs = $json_attrs",
                src=src,
                dst=dst,
                key=key,
                primitive_attrs=primitive,
                json_attrs=json_attrs,
            )
        return key

    def remove_edge(self, src: str, dst: str, key: Any = None) -> None:
        if key is None:
            # Delete one arbitrary parallel edge (matches networkx semantics).
            self._run(
                "MATCH (s:Node {id: $src})-[r:LINK]->(t:Node {id: $dst}) "
                "WITH r LIMIT 1 DELETE r",
                src=src,
                dst=dst,
            )
        else:
            self._run(
                "MATCH (s:Node {id: $src})-[r:LINK]->(t:Node {id: $dst}) "
                "WHERE r.key = $key DELETE r",
                src=src,
                dst=dst,
                key=key,
            )

    def get_edge_data(self, src: str, dst: str) -> dict[Any, dict[str, Any]] | None:
        records = self._run(
            "MATCH (s:Node {id: $src})-[r:LINK]->(t:Node {id: $dst}) "
            "RETURN r.key AS key, properties(r) AS props",
            src=src,
            dst=dst,
        )
        if not records:
            return None
        out: dict[Any, dict[str, Any]] = {}
        for rec in records:
            props = dict(rec["props"])
            # The key lives both as a top-level RETURN column AND inside props;
            # strip it from the props so the returned dict matches networkx
            # (which keeps the key out of the per-edge attr dict).
            props.pop("key", None)
            out[rec["key"]] = _unpack_props(props)
        return out

    def iter_edges(
        self, *, keys: bool = False, data: bool = True
    ) -> Iterator[tuple]:
        records = self._run(
            "MATCH (s:Node)-[r:LINK]->(t:Node) "
            "RETURN s.id AS src, t.id AS dst, r.key AS key, properties(r) AS props"
        )
        for rec in records:
            props = dict(rec["props"])
            props.pop("key", None)
            attrs = _unpack_props(props)
            if keys and data:
                yield rec["src"], rec["dst"], rec["key"], attrs
            elif keys:
                yield rec["src"], rec["dst"], rec["key"]
            elif data:
                yield rec["src"], rec["dst"], attrs
            else:
                yield rec["src"], rec["dst"]

    def number_of_edges(self) -> int:
        records = self._run("MATCH ()-[r:LINK]->() RETURN count(r) AS n")
        return int(records[0]["n"]) if records else 0

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def successors(self, node_id: str) -> Iterator[str]:
        records = self._run(
            "MATCH (n:Node {id: $node_id})-[:LINK]->(m:Node) "
            "RETURN DISTINCT m.id AS id",
            node_id=node_id,
        )
        for rec in records:
            yield rec["id"]

    def predecessors(self, node_id: str) -> Iterator[str]:
        records = self._run(
            "MATCH (n:Node {id: $node_id})<-[:LINK]-(m:Node) "
            "RETURN DISTINCT m.id AS id",
            node_id=node_id,
        )
        for rec in records:
            yield rec["id"]

    def in_edges(
        self, node_id: str, *, data: bool = True
    ) -> Iterator[tuple]:
        records = self._run(
            "MATCH (s:Node)-[r:LINK]->(t:Node {id: $node_id}) "
            "RETURN s.id AS src, t.id AS dst, properties(r) AS props",
            node_id=node_id,
        )
        for rec in records:
            if data:
                props = dict(rec["props"])
                props.pop("key", None)
                yield rec["src"], rec["dst"], _unpack_props(props)
            else:
                yield rec["src"], rec["dst"]

    # ------------------------------------------------------------------
    # Bulk export
    # ------------------------------------------------------------------

    def to_sparse_adjacency(self) -> tuple["sp.csr_matrix", list[str]]:
        """Pull all nodes + edges over Cypher and assemble a CSR matrix.

        Mirrors :meth:`NetworkXBackend.to_sparse_adjacency` exactly: edge
        weight defaults to 1.0, parallel edges between the same pair are
        summed.
        """
        import numpy as np  # noqa: PLC0415
        import scipy.sparse as sp  # noqa: PLC0415

        # Deterministic id order — Cypher's ordering is not guaranteed across
        # versions without ORDER BY, so we sort lexicographically.
        node_records = self._run("MATCH (n:Node) RETURN n.id AS id ORDER BY n.id")
        node_ids: list[str] = [rec["id"] for rec in node_records]
        n = len(node_ids)
        if n == 0:
            return sp.csr_matrix((0, 0), dtype=float), node_ids

        index_of = {nid: i for i, nid in enumerate(node_ids)}

        edge_records = self._run(
            "MATCH (s:Node)-[r:LINK]->(t:Node) "
            "RETURN s.id AS src, t.id AS dst, properties(r) AS props"
        )
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for rec in edge_records:
            props = dict(rec["props"])
            weight = props.get("weight", 1.0)
            try:
                w = float(weight)
            except (TypeError, ValueError):
                w = 1.0
            rows.append(index_of[rec["src"]])
            cols.append(index_of[rec["dst"]])
            data.append(w)

        mat = sp.coo_matrix(
            (
                np.array(data, dtype=float),
                (np.array(rows, dtype=int), np.array(cols, dtype=int)),
            ),
            shape=(n, n),
        ).tocsr()
        return mat, node_ids

    def to_networkx(self) -> "nx.MultiDiGraph":
        """Reconstruct a ``networkx.MultiDiGraph`` from the Neo4j contents.

        This is the slow path — O(V+E) over Cypher with two queries (one for
        nodes, one for edges). Used by the community detector and the MST
        organiser, both of which consume an in-memory NetworkX graph.
        """
        import networkx as nx  # noqa: PLC0415

        g: nx.MultiDiGraph = nx.MultiDiGraph()
        for node_id, attrs in self.iter_nodes():
            g.add_node(node_id, **attrs)
        for src, dst, key, attrs in self.iter_edges(keys=True, data=True):
            g.add_edge(src, dst, key=key, **attrs)
        return g

    # ------------------------------------------------------------------
    # Persistence (no-op for a server-backed store)
    # ------------------------------------------------------------------

    def save(self, path: Any) -> None:
        # No-op: a real Neo4j backend persists to the database server, not to
        # ``path``. The KGStore upsert path calls ``save()`` after every
        # batch; we accept the call but do nothing — each Cypher write has
        # already auto-committed within its own transaction.
        return

    def load(self, path: Any) -> None:
        # No-op: there is no on-disk artefact to restore from. The Neo4j
        # server is the durable store.
        return
