"""Knowledge graph store + SQLite mirror.

A pluggable :class:`KGBackend` (default: NetworkX ``MultiDiGraph``) is the
source of truth at query time (fast in-memory traversal). The SQLite mirror
in ``kg_nodes`` / ``kg_edges`` enables cross-process inspection and per-doc
wipe on re-ingest.

Two node kinds live in the graph:

* ``phrase`` nodes — entities. ``node_id`` is the canonical lowercase form
  (deterministic). Attributes: ``label`` (canonical surface form),
  ``aliases`` (set of seen surface forms), ``embedding`` (list[float]).
* ``passage`` nodes — chunks. ``node_id`` is the chunk_id.
  Attributes: ``chunk_id``, ``doc_id``.

Edges:

* ``phrase --relation--> phrase`` for each triple. The relation is stored as
  the edge attribute ``relation``. Multiple distinct relations between two
  phrases are allowed (hence ``MultiDiGraph``). The edge attribute
  ``source_chunk_ids`` is a ``set[str]`` of chunks that support that edge.
* ``phrase --contains--> passage`` for each triple's source chunk.

Backend selection
-----------------
The backend is chosen at construction. Pass ``backend=NetworkXBackend()``
explicitly, or use the :meth:`KGStore.from_config` factory which reads
``config.kg.backend``. When ``backend is None`` the constructor defaults to
:class:`NetworkXBackend` — preserves the original behaviour.

Heavy deps (``networkx``, ``numpy``, ``scipy``) are lazy-imported inside the
backend implementations so this module is importable when those packages
are absent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from hrag.kg.backends import KGBackend, NetworkXBackend, Neo4jBackend

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx

    from hrag.config import KGConfig
    from hrag.db.connection import Database
    from hrag.kg.builder import Triple
    from hrag.providers.embeddings import EmbeddingProvider


logger = logging.getLogger(__name__)


def _canon(text: str) -> str:
    """Canonical form for a surface phrase: lowercase + strip."""
    return text.strip().lower()


def _cosine(a, b) -> float:
    """Cosine similarity with numpy. Returns 0.0 on zero-norm vectors."""
    import numpy as np  # noqa: PLC0415

    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def _build_backend_from_name(name: str) -> KGBackend:
    """Resolve a ``config.kg.backend`` string to a backend instance."""
    n = (name or "").strip().lower()
    if n == "networkx" or n == "":
        return NetworkXBackend()
    if n == "neo4j":
        return Neo4jBackend()
    raise ValueError(
        f"Unknown kg.backend {name!r}; expected 'networkx' or 'neo4j'."
    )


class KGStore:
    """Knowledge graph store backed by a pluggable :class:`KGBackend`.

    The store is engine-agnostic: synonym merging, source-chunk tracking, and
    the SQLite mirror live here; node/edge primitives are delegated to the
    backend.
    """

    name = "kg_store"

    def __init__(
        self,
        db: "Database",
        embedder: "EmbeddingProvider",
        kg_path: str | Path,
        synonym_threshold: float = 0.8,
        backend: KGBackend | None = None,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._kg_path = Path(kg_path)
        self._synonym_threshold = float(synonym_threshold)
        self._graph_path = self._kg_path / "graph.pkl"

        # Default to NetworkX for backward compatibility.
        self._backend: KGBackend = backend if backend is not None else NetworkXBackend()

        # Restore prior state from disk (no-op when the artefact is missing).
        self._backend.load(self._graph_path)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        db: "Database",
        embedder: "EmbeddingProvider",
        kg_path: str | Path,
        cfg: "KGConfig",
    ) -> "KGStore":
        """Build a :class:`KGStore` whose backend is selected from
        ``cfg.backend`` (``"networkx"`` by default)."""
        backend = _build_backend_from_name(cfg.backend)
        return cls(
            db=db,
            embedder=embedder,
            kg_path=kg_path,
            synonym_threshold=cfg.synonym_threshold,
            backend=backend,
        )

    # ------------------------------------------------------------------
    # Back-compat: legacy `_graph` attribute
    # ------------------------------------------------------------------

    @property
    def _graph(self) -> "nx.MultiDiGraph":
        """Return the underlying NetworkX graph (back-compat shim).

        Older callers and tests reach into ``kg_store._graph`` directly. This
        property forwards to ``backend.to_networkx()`` — zero-copy for
        NetworkXBackend; raises ``NotImplementedError`` for Neo4jBackend
        (matches the stub-everywhere contract).
        """
        return self._backend.to_networkx()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_triples(
        self,
        user_id: str,
        doc_id: str,
        triples: list["Triple"],
        chunk_id_to_doc_id: dict[str, str] | None = None,
    ) -> None:
        """Idempotent: wipe-then-add for this (user, doc).

        Sequence:
        1. ``delete_doc(user_id, doc_id)`` — removes prior contributions.
        2. For each triple, canonicalise head/tail (synonym merge), add phrase
           nodes if needed, add the relation edge, and add the contains edge
           to the passage node.
        3. Persist: pickle the graph + write SQLite mirror.
        4. Commit DB.
        """
        # 1. Wipe prior contributions for this doc.
        self.delete_doc(user_id, doc_id)

        # Touched nodes/edges that need to be mirrored at the end.
        touched_phrase_ids: set[str] = set()
        touched_passage_ids: set[str] = set()
        new_edge_rows: list[tuple] = []

        chunk_id_to_doc_id = chunk_id_to_doc_id or {}

        for triple in triples:
            head_id = self._canonicalize_phrase(triple.head)
            tail_id = self._canonicalize_phrase(triple.tail)

            # Mirror canonical keys back onto the triple so callers can see
            # what we ended up doing.
            triple.head_canonical = head_id
            triple.tail_canonical = tail_id

            touched_phrase_ids.add(head_id)
            touched_phrase_ids.add(tail_id)

            # phrase --relation--> phrase
            self._add_or_extend_phrase_edge(
                head_id,
                tail_id,
                triple.relation,
                triple.source_chunk_id,
            )
            new_edge_rows.append(
                (user_id, None, head_id, triple.relation, tail_id, 1.0)
            )

            # passage node
            chunk_id = triple.source_chunk_id
            passage_doc = chunk_id_to_doc_id.get(chunk_id, doc_id)
            self._ensure_passage_node(chunk_id, passage_doc)
            touched_passage_ids.add(chunk_id)

            # phrase --contains--> passage (head + tail both contain the chunk)
            for src in (head_id, tail_id):
                self._add_or_extend_contains_edge(src, chunk_id)
                new_edge_rows.append(
                    (user_id, passage_doc, src, "contains", chunk_id, 1.0)
                )

        # 2. Persist graph to disk.
        self._save_graph()

        # 3. SQLite mirror.
        self._mirror_phrase_nodes(user_id, touched_phrase_ids)
        self._mirror_passage_nodes(user_id, touched_passage_ids)

        if new_edge_rows:
            self._db.executemany(
                "INSERT INTO kg_edges(user_id, doc_id, src, relation, dst, weight) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                new_edge_rows,
            )

        self._db.commit()

    def delete_doc(self, user_id: str, doc_id: str) -> None:
        """Remove all data contributed by (user_id, doc_id) from both the
        in-memory graph and the SQLite mirror.

        Phrase nodes themselves are NEVER deleted: they may be cited by other
        docs and re-canonicalisation is expensive. Phrase->phrase edges are
        pruned when their ``source_chunk_ids`` set becomes empty after
        removing chunks from this doc.
        """
        # ------------------------------------------------------------------
        # In-memory graph mutation
        # ------------------------------------------------------------------
        # Find passages that belong to this doc.
        passage_ids_to_remove: list[str] = []
        for node_id, data in list(self._backend.iter_nodes()):
            if (
                data.get("node_type") == "passage"
                and data.get("doc_id") == doc_id
            ):
                passage_ids_to_remove.append(node_id)

        passage_set = set(passage_ids_to_remove)

        # Remove those passage nodes (this also removes their incident
        # 'contains' edges automatically).
        for pid in passage_ids_to_remove:
            self._backend.remove_node(pid)

        # Prune phrase->phrase edges whose support is now exhausted.
        edges_to_remove: list[tuple] = []
        for u, v, key, data in list(self._backend.iter_edges(keys=True, data=True)):
            # Skip 'contains' edges (they already vanished with the passage
            # node) and any edge that doesn't carry source_chunk_ids.
            if data.get("relation") == "contains":
                continue
            sources = data.get("source_chunk_ids")
            if not isinstance(sources, set):
                continue
            sources -= passage_set
            data["source_chunk_ids"] = sources
            if not sources:
                edges_to_remove.append((u, v, key))

        for u, v, key in edges_to_remove:
            self._backend.remove_edge(u, v, key=key)

        # ------------------------------------------------------------------
        # SQLite mirror
        # ------------------------------------------------------------------
        # Delete passage rows + edge rows for this doc. Phrase rows have
        # NULL doc_id and are intentionally untouched.
        self._db.execute(
            "DELETE FROM kg_nodes WHERE user_id = ? AND doc_id = ?",
            (user_id, doc_id),
        )
        self._db.execute(
            "DELETE FROM kg_edges WHERE user_id = ? AND doc_id = ?",
            (user_id, doc_id),
        )
        # Phrase->phrase edges are stored with doc_id NULL but were inserted
        # during the upsert that contributed *this* doc's triples. They are
        # the ones whose source_chunk_ids referenced chunks of this doc.
        # Rather than try to track that in SQL, we wipe all NULL-doc edges
        # whose endpoints reference removed passages — which is impossible to
        # do precisely without cross-table joins. The pragmatic solution
        # adopted here: phrase->phrase edges in SQLite mirror what's in the
        # graph after this method completes, so we re-mirror them at the end
        # of the next upsert cycle. To keep the mirror consistent right now
        # (in case nothing follows), wipe all phrase-edges contributed by
        # this doc — they were inserted with the doc_id we just deleted.
        # (The upsert path inserts phrase->phrase edges with doc_id=NULL,
        # so a more robust strategy is to also delete those orphans by
        # recomputing from the graph.)
        # We re-emit the phrase-edge rows from the live graph so the mirror
        # stays in sync without depending on which doc inserted them.
        self._rewrite_phrase_edges(user_id)

        self._db.commit()

    def neighbors(self, node_id: str, depth: int = 1) -> set[str]:
        """BFS up to *depth* hops. Returns the set of reachable node_ids,
        excluding the seed itself. Direction-agnostic on a MultiDiGraph:
        we follow both successors and predecessors.
        """
        if not self._backend.has_node(node_id) or depth < 1:
            return set()

        seen: set[str] = {node_id}
        frontier: set[str] = {node_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for n in frontier:
                for succ in self._backend.successors(n):
                    if succ not in seen:
                        seen.add(succ)
                        next_frontier.add(succ)
                for pred in self._backend.predecessors(n):
                    if pred not in seen:
                        seen.add(pred)
                        next_frontier.add(pred)
            if not next_frontier:
                break
            frontier = next_frontier

        seen.discard(node_id)
        return seen

    def passage_nodes_for(self, phrase_node_ids: list[str]) -> set[str]:
        """Return all passage node_ids reachable via 'contains' edges from any
        of the given phrase nodes."""
        out: set[str] = set()
        for pid in phrase_node_ids:
            if not self._backend.has_node(pid):
                continue
            for succ in self._backend.successors(pid):
                succ_data = self._backend.get_node_data(succ)
                if succ_data.get("node_type") == "passage":
                    out.add(succ)
        return out

    def find_phrase_nodes(self, surface_forms: list[str]) -> list[str]:
        """Look up canonical node_ids for a list of surface forms.

        Lookup is lower-cased + stripped. Missing terms are filtered out.
        Order follows input; duplicates removed.
        """
        seen: dict[str, None] = {}
        for sf in surface_forms:
            key = _canon(sf)
            if not key or key in seen:
                continue
            if (
                self._backend.has_node(key)
                and self._backend.get_node_data(key).get("node_type") == "phrase"
            ):
                seen[key] = None
        return list(seen.keys())

    def to_sparse_adjacency(self):
        """Build a sparse adjacency matrix over ALL nodes.

        Returns (csr_matrix shape (n, n), node_ids list[str] of length n).
        Edge weight defaults to 1.0; the ``weight`` attribute is used if set.
        Multi-edges between the same pair add up.
        """
        return self._backend.to_sparse_adjacency()

    def num_phrase_nodes(self) -> int:
        return sum(
            1 for _, d in self._backend.iter_nodes() if d.get("node_type") == "phrase"
        )

    def num_passage_nodes(self) -> int:
        return sum(
            1 for _, d in self._backend.iter_nodes() if d.get("node_type") == "passage"
        )

    def num_edges(self) -> int:
        return self._backend.number_of_edges()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _canonicalize_phrase(self, surface: str) -> str:
        """Map a surface phrase to a canonical phrase node_id, creating or
        merging as needed.

        Returns the canonical node_id (lowercase form of either the original
        surface phrase or a synonym we merged into).
        """
        canon = _canon(surface)
        if not canon:
            # Defensive: empty string would corrupt the graph. Use a sentinel
            # but log it; upstream cleanup should prevent this.
            canon = "<empty>"

        if (
            self._backend.has_node(canon)
            and self._backend.get_node_data(canon).get("node_type") == "phrase"
        ):
            # Already exists; just record the surface form as an alias.
            self._backend.get_node_data(canon).setdefault("aliases", set()).add(surface)
            return canon

        # New phrase: embed it and look for synonyms.
        new_emb = self._embedder.embed_one(canon)

        best_id: str | None = None
        best_sim = -1.0
        for nid, data in self._backend.iter_nodes():
            if data.get("node_type") != "phrase":
                continue
            existing_emb = data.get("embedding")
            if existing_emb is None:
                continue
            sim = _cosine(new_emb, existing_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = nid

        if best_id is not None and best_sim >= self._synonym_threshold:
            # Merge: redirect to the matched canonical node.
            existing_attrs = self._backend.get_node_data(best_id)
            existing_attrs.setdefault("aliases", set()).add(surface)
            existing_attrs["aliases"].add(canon)
            return best_id

        # Fresh node.
        self._backend.add_node(
            canon,
            node_type="phrase",
            label=surface,
            aliases={surface, canon},
            embedding=new_emb,
        )
        return canon

    def _ensure_passage_node(self, chunk_id: str, doc_id: str) -> None:
        if (
            self._backend.has_node(chunk_id)
            and self._backend.get_node_data(chunk_id).get("node_type") == "passage"
        ):
            # Update doc_id in case it had been missing (defensive).
            self._backend.get_node_data(chunk_id)["doc_id"] = doc_id
            return
        self._backend.add_node(
            chunk_id,
            node_type="passage",
            chunk_id=chunk_id,
            doc_id=doc_id,
        )

    def _add_or_extend_phrase_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        source_chunk_id: str,
    ) -> None:
        """Either find an existing edge (src --relation--> dst) and extend its
        source_chunk_ids set, or create a new one."""
        # Search existing keyed edges between this pair for the same relation.
        existing = self._backend.get_edge_data(src, dst)
        if existing:
            for _key, data in existing.items():
                if data.get("relation") == relation:
                    sources = data.setdefault("source_chunk_ids", set())
                    sources.add(source_chunk_id)
                    return

        self._backend.add_edge(
            src,
            dst,
            relation=relation,
            source_chunk_ids={source_chunk_id},
            weight=1.0,
        )

    def _add_or_extend_contains_edge(self, phrase_id: str, passage_id: str) -> None:
        existing = self._backend.get_edge_data(phrase_id, passage_id)
        if existing:
            for _key, data in existing.items():
                if data.get("relation") == "contains":
                    return  # already there
        self._backend.add_edge(
            phrase_id,
            passage_id,
            relation="contains",
            weight=1.0,
        )

    def _save_graph(self) -> None:
        self._backend.save(self._graph_path)

    # ----- SQLite mirror ----------------------------------------------------

    def _mirror_phrase_nodes(self, user_id: str, phrase_ids: set[str]) -> None:
        if not phrase_ids:
            return
        rows = []
        for pid in phrase_ids:
            data = self._backend.get_node_data(pid)
            label = data.get("label", pid)
            aliases = data.get("aliases")
            if isinstance(aliases, set):
                aliases_list = sorted(aliases)
            else:
                aliases_list = []
            metadata = json.dumps({"aliases": aliases_list})
            rows.append((pid, user_id, None, label, "phrase", metadata))
        self._db.executemany(
            "INSERT OR REPLACE INTO kg_nodes(node_id, user_id, doc_id, label, node_type, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    def _mirror_passage_nodes(self, user_id: str, passage_ids: set[str]) -> None:
        if not passage_ids:
            return
        rows = []
        for pid in passage_ids:
            data = self._backend.get_node_data(pid)
            doc_id = data.get("doc_id")
            label = data.get("label", pid)
            rows.append((pid, user_id, doc_id, label, "passage", None))
        self._db.executemany(
            "INSERT OR REPLACE INTO kg_nodes(node_id, user_id, doc_id, label, node_type, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    def _rewrite_phrase_edges(self, user_id: str) -> None:
        """Re-emit phrase->phrase edge rows from the live graph for *user_id*.

        We delete all phrase-edge rows that have ``doc_id IS NULL`` for this
        user, then re-insert from the in-memory graph. This keeps the mirror
        consistent after partial-doc deletions without needing to track which
        doc inserted each phrase-edge row.
        """
        self._db.execute(
            "DELETE FROM kg_edges WHERE user_id = ? AND doc_id IS NULL",
            (user_id,),
        )
        rows = []
        for u, v, edata in self._backend.iter_edges(data=True):
            relation = edata.get("relation")
            if relation == "contains":
                # 'contains' edges live with a real doc_id, handled elsewhere.
                continue
            weight = float(edata.get("weight", 1.0))
            rows.append((user_id, None, u, relation, v, weight))
        if rows:
            self._db.executemany(
                "INSERT INTO kg_edges(user_id, doc_id, src, relation, dst, weight) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
