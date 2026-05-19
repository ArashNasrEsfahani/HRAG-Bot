"""NetworkX-backed :class:`KGBackend` implementation.

This is the default and currently the only fully-functional KG backend. It
wraps a :class:`networkx.MultiDiGraph` and forwards every protocol method
straight onto the underlying graph.

The class is intentionally thin: business logic (synonym merging, source-chunk
tracking, SQLite mirror) lives in :class:`hrag.kg.store.KGStore` so it stays
backend-agnostic. The only state owned by this class is the in-memory graph
and the pickle path used by :meth:`save` / :meth:`load`.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx
    import scipy.sparse as sp


logger = logging.getLogger(__name__)


class NetworkXBackend:
    """In-memory :class:`KGBackend` using ``networkx.MultiDiGraph``."""

    name = "networkx"

    def __init__(self) -> None:
        import networkx as nx  # noqa: PLC0415

        self._graph: "nx.MultiDiGraph" = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self._graph.add_node(node_id, **attrs)

    def remove_node(self, node_id: str) -> None:
        self._graph.remove_node(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._graph

    def get_node_data(self, node_id: str) -> dict[str, Any]:
        return self._graph.nodes[node_id]

    def iter_nodes(self) -> Iterator[tuple[str, dict[str, Any]]]:
        return iter(self._graph.nodes(data=True))

    def number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, src: str, dst: str, **attrs: Any) -> Any:
        return self._graph.add_edge(src, dst, **attrs)

    def remove_edge(self, src: str, dst: str, key: Any = None) -> None:
        if key is None:
            self._graph.remove_edge(src, dst)
        else:
            self._graph.remove_edge(src, dst, key=key)

    def get_edge_data(self, src: str, dst: str) -> dict[Any, dict[str, Any]] | None:
        return self._graph.get_edge_data(src, dst)

    def iter_edges(
        self, *, keys: bool = False, data: bool = True
    ) -> Iterator[tuple]:
        if keys:
            return iter(self._graph.edges(keys=True, data=data))
        return iter(self._graph.edges(data=data))

    def number_of_edges(self) -> int:
        return self._graph.number_of_edges()

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def successors(self, node_id: str) -> Iterator[str]:
        return self._graph.successors(node_id)

    def predecessors(self, node_id: str) -> Iterator[str]:
        return self._graph.predecessors(node_id)

    def in_edges(
        self, node_id: str, *, data: bool = True
    ) -> Iterator[tuple]:
        return iter(self._graph.in_edges(node_id, data=data))

    # ------------------------------------------------------------------
    # Bulk export
    # ------------------------------------------------------------------

    def to_sparse_adjacency(self) -> tuple["sp.csr_matrix", list[str]]:
        import numpy as np  # noqa: PLC0415
        import scipy.sparse as sp  # noqa: PLC0415

        node_ids: list[str] = list(self._graph.nodes())
        n = len(node_ids)
        if n == 0:
            return sp.csr_matrix((0, 0), dtype=float), node_ids

        index_of = {nid: i for i, nid in enumerate(node_ids)}

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for u, v, edata in self._graph.edges(data=True):
            rows.append(index_of[u])
            cols.append(index_of[v])
            data.append(float(edata.get("weight", 1.0)))

        mat = sp.coo_matrix(
            (
                np.array(data, dtype=float),
                (np.array(rows, dtype=int), np.array(cols, dtype=int)),
            ),
            shape=(n, n),
        ).tocsr()
        return mat, node_ids

    def to_networkx(self) -> "nx.MultiDiGraph":
        """Return the live underlying ``MultiDiGraph`` (zero-copy).

        Mutations on the returned graph are visible to this backend — kept
        deliberate so the community detector and MST organiser (which read
        the graph directly) avoid an O(V+E) copy on every call.
        """
        return self._graph

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self._graph, fh)

    def load(self, path: Any) -> None:
        """Restore the pickled graph from *path*. Silently no-op when missing;
        warns and resets to empty on corruption."""
        import networkx as nx  # noqa: PLC0415

        p = Path(path)
        if not p.exists():
            return
        try:
            with p.open("rb") as fh:
                loaded = pickle.load(fh)
            if not isinstance(loaded, nx.MultiDiGraph):
                raise TypeError(
                    f"Pickled object is {type(loaded).__name__}, "
                    f"expected MultiDiGraph"
                )
            self._graph = loaded
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"NetworkXBackend: failed to load graph at {p}: "
                f"{exc}. Starting with a fresh empty graph.",
                UserWarning,
                stacklevel=2,
            )
            self._graph = nx.MultiDiGraph()
