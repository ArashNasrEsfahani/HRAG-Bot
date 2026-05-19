"""KG backend implementations.

The :class:`KGBackend` Protocol lives in :mod:`hrag.kg.backends.base`. Two
concrete implementations are exported here:

* :class:`NetworkXBackend` — default; in-memory :class:`networkx.MultiDiGraph`.
* :class:`Neo4jBackend` — real Neo4j-backed implementation (Phase 6 Track B);
  requires the optional ``neo4j`` driver and a reachable server (URI via
  the ``NEO4J_URI`` env var or constructor kwarg).

:class:`hrag.kg.store.KGStore` resolves the backend from
``config.kg.backend`` (``"networkx"`` by default).
"""

from __future__ import annotations

from .base import KGBackend
from .networkx import NetworkXBackend
from .neo4j import Neo4jBackend

__all__ = ["KGBackend", "NetworkXBackend", "Neo4jBackend"]
