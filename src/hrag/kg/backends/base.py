"""``KGBackend`` Protocol — graph-engine abstraction for ``KGStore``.

``KGStore`` (the user-facing API) is engine-agnostic. The actual graph storage
sits behind a :class:`KGBackend` Protocol with two concrete implementations:

* :class:`hrag.kg.backends.networkx.NetworkXBackend` — in-memory
  ``networkx.MultiDiGraph``; the default and fully covered by the acceptance
  test-suite.
* :class:`hrag.kg.backends.neo4j.Neo4jBackend` — Neo4j-backed implementation
  (Phase 6 Track B); requires the optional ``neo4j`` driver and a reachable
  server. Drop-in alternative to the NetworkX backend.

Selecting a backend
-------------------
The backend is picked at :class:`KGStore` construction time from
``config.kg.backend`` (``"networkx"`` by default). Call-sites do not need to
know which backend is in use.

Mutation semantics
------------------
All mutation methods (``add_node``, ``add_edge``, ``remove_node``,
``remove_edge``) are idempotent or follow MultiDiGraph semantics — multiple
``add_edge`` calls between the same pair create multi-edges, and
``remove_edge`` operates on a specific ``key``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx
    import scipy.sparse as sp


@runtime_checkable
class KGBackend(Protocol):
    """Engine-agnostic interface for the HRAG knowledge graph store.

    Every method maps 1-to-1 to a ``networkx.MultiDiGraph`` operation that
    :class:`KGStore` currently performs. New backends (Neo4j, sqlite-vec, etc.)
    must surface the same operations to remain a drop-in replacement.
    """

    name: str

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, **attrs: Any) -> None:
        """Insert a node with the given id and attribute dict. Idempotent: if
        the node exists, its attributes are updated in place (networkx
        semantics)."""
        ...

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all incident edges. No-op if absent."""
        ...

    def has_node(self, node_id: str) -> bool:
        """Return True iff a node with this id exists."""
        ...

    def get_node_data(self, node_id: str) -> dict[str, Any]:
        """Return the mutable attribute dict for *node_id*. Raises ``KeyError``
        if the node is absent — matches ``MultiDiGraph.nodes[node_id]``."""
        ...

    def iter_nodes(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Iterate ``(node_id, attr_dict)`` pairs over every node."""
        ...

    def number_of_nodes(self) -> int:
        """Total node count (phrase + passage)."""
        ...

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, src: str, dst: str, **attrs: Any) -> Any:
        """Add an edge ``src -> dst`` with the given attributes. Returns the
        edge key (MultiDiGraph allows multiple edges between the same pair)."""
        ...

    def remove_edge(self, src: str, dst: str, key: Any = None) -> None:
        """Remove a specific edge keyed by ``key``. If ``key`` is None, the
        backend removes an arbitrary parallel edge (matches networkx)."""
        ...

    def get_edge_data(self, src: str, dst: str) -> dict[Any, dict[str, Any]] | None:
        """Return the mapping ``{key: attr_dict}`` for all edges ``src -> dst``,
        or ``None`` if no such edge exists. Mirrors
        ``MultiDiGraph.get_edge_data``."""
        ...

    def iter_edges(
        self, *, keys: bool = False, data: bool = True
    ) -> Iterator[tuple]:
        """Iterate edges. With ``keys=False`` yields ``(u, v, data)``; with
        ``keys=True`` yields ``(u, v, key, data)``. ``data`` is the live
        attribute dict (mutations on it persist)."""
        ...

    def number_of_edges(self) -> int:
        """Total edge count across all multi-edge keys."""
        ...

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def successors(self, node_id: str) -> Iterator[str]:
        """Out-neighbours of ``node_id`` (each successor yielded once even when
        multiple parallel edges connect the pair)."""
        ...

    def predecessors(self, node_id: str) -> Iterator[str]:
        """In-neighbours of ``node_id`` (each predecessor yielded once)."""
        ...

    def in_edges(
        self, node_id: str, *, data: bool = True
    ) -> Iterator[tuple]:
        """Iterate edges that terminate at ``node_id``. With ``data=True``
        yields ``(src, dst, attrs)``; otherwise ``(src, dst)``."""
        ...

    # ------------------------------------------------------------------
    # Bulk export helpers
    # ------------------------------------------------------------------

    def to_sparse_adjacency(self) -> tuple["sp.csr_matrix", list[str]]:
        """Build a ``(csr_matrix, node_ids)`` adjacency view over ALL nodes.

        Edge weight defaults to 1.0; the ``weight`` attribute is used if set.
        Parallel edges between the same pair add together. Used by the PPR
        layer (``hrag.kg.ppr``)."""
        ...

    def to_networkx(self) -> "nx.MultiDiGraph":
        """Return a ``networkx.MultiDiGraph`` view of the backend.

        For NetworkX-backed stores this is the live underlying graph (zero
        copy). For non-NetworkX backends a materialised copy is acceptable; the
        Leiden community detector and the MST organiser consume this view.
        """
        ...

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Any) -> None:
        """Persist the graph to *path* (engine-specific format). Called by
        :meth:`KGStore.upsert_triples` after each batch."""
        ...

    def load(self, path: Any) -> None:
        """Restore the graph from *path*. Silently no-op when the artefact is
        missing — empty graph is a valid starting state."""
        ...
