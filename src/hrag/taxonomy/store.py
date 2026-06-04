"""SQLite-backed taxonomy store: CRUD, centroid math, beam descend.

The tree is persisted entirely in SQLite (``kg_taxonomy_nodes``,
``kg_taxonomy_assignments``, ``kg_taxonomy_doc_meta``). No in-memory cache
lives across calls — the store is recreated per request and pulls just what
it needs. Beam descend is the one hot path and loads the user's full node
set with a single SELECT to avoid N+1 SQL traffic during the descent.

Centroids are L2-normalized float lists (the embedder normalizes them); we
store them packed as little-endian float32 in the ``centroid`` BLOB column
with the parallel ``centroid_dim`` so reads can verify packing. Cosine on
normalized vectors is just the dot product, but ``_cosine`` re-normalizes
defensively so externally-supplied vectors don't silently mis-score.

Three categories of public API live on :class:`TaxonomyStore`:

* Tree CRUD — ``ensure_root``, ``add_node``, ``update_node``, ``move_node``,
  ``delete_node``, plus read helpers (``get_node``, ``get_children``,
  ``list_nodes``, ``get_tree``, ``clear``).
* Assignments — ``assign_doc``, ``unassign_doc``, ``get_docs_at``,
  ``get_doc_nodes``.
* Doc-meta cache + centroid recompute — ``upsert_doc_meta``,
  ``get_doc_meta``, ``recompute_node_centroid``, ``recompute_all_centroids``.

And the hot path: ``beam_descend``.
"""

from __future__ import annotations

import json
import logging
import math
import secrets
import struct
import time
import warnings
from typing import TYPE_CHECKING, Optional

from hrag.taxonomy.keywords import keyword_overlap
from hrag.taxonomy.types import (
    DescendResult,
    LevelTrace,
    NodeScore,
    TaxonomyNode,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hrag.db.connection import Database
    from hrag.providers.embeddings import EmbeddingProvider


logger = logging.getLogger(__name__)


class TaxonomyStore:
    """SQLite-backed taxonomy: CRUD + centroid math + beam descend."""

    name: str = "taxonomy_store"

    def __init__(self, db: "Database", embedder: "EmbeddingProvider") -> None:
        self._db = db
        self._embedder = embedder
        # Phase 12 — in-memory per-user node cache for the beam-descend hot
        # path. Each entry is (timestamp, nodes). Invalidated immediately on
        # any write through this instance and after ``cache_ttl_s`` seconds so
        # an external writer (e.g. a CLI rebuild) is eventually picked up.
        # ``cache_ttl_s <= 0`` disables caching entirely.
        self.cache_ttl_s: float = 30.0
        self._node_cache: dict[str, tuple[float, list[TaxonomyNode]]] = {}

    def _invalidate_cache(self, user_id: Optional[str] = None) -> None:
        """Drop cached nodes for *user_id* (or everyone when None)."""
        if user_id is None:
            self._node_cache.clear()
        else:
            self._node_cache.pop(user_id, None)

    def _commit(self) -> None:
        """Commit a write and invalidate the node cache.

        Every node-mutating method commits through here so a write is always
        reflected on the next ``list_nodes`` / ``beam_descend``. The cache is
        small, so a full clear on each (infrequent) write is cheap.
        """
        self._db.commit()
        self._invalidate_cache()

    # ------------------------------------------------------------------
    # Packing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pack(vec: list[float]) -> bytes:
        return struct.pack(f"<{len(vec)}f", *vec)

    @staticmethod
    def _unpack(blob: bytes, dim: int) -> list[float]:
        return list(struct.unpack(f"<{dim}f", blob))

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        # Defensive re-normalization for imported non-unit vectors.
        return dot / (math.sqrt(na) * math.sqrt(nb))

    # ------------------------------------------------------------------
    # Row -> dataclass
    # ------------------------------------------------------------------

    def _row_to_node(self, row) -> TaxonomyNode:
        centroid: Optional[list[float]] = None
        blob = row["centroid"]
        dim = row["centroid_dim"]
        if blob is not None and dim:
            centroid = self._unpack(blob, int(dim))
        return TaxonomyNode(
            node_id=row["node_id"],
            user_id=row["user_id"],
            parent_id=row["parent_id"],
            label=row["label"],
            description=row["description"] or "",
            depth=int(row["depth"]),
            is_leaf=bool(row["is_leaf"]),
            centroid=centroid,
            doc_count=int(row["doc_count"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            keywords=_decode_keywords(_row_get(row, "keywords")),
        )

    # ------------------------------------------------------------------
    # Tree CRUD
    # ------------------------------------------------------------------

    def ensure_root(self, user_id: str) -> TaxonomyNode:
        root_id = f"root::{user_id}"
        existing = self.get_node(root_id)
        if existing is not None:
            return existing
        with self._db.conn:
            self._db.execute(
                "INSERT INTO kg_taxonomy_nodes("
                "node_id, user_id, parent_id, label, description, depth, is_leaf, doc_count"
                ") VALUES (?, ?, NULL, ?, ?, 0, 0, 0)",
                (root_id, user_id, "root", ""),
            )
        self._commit()
        node = self.get_node(root_id)
        assert node is not None  # just inserted
        return node

    def add_node(
        self,
        user_id: str,
        parent_id: str,
        label: str,
        description: str = "",
        is_leaf: bool = False,
        keywords: Optional[list[str]] = None,
    ) -> TaxonomyNode:
        parent = self.get_node(parent_id)
        if parent is None:
            raise ValueError(f"Parent node {parent_id!r} does not exist")
        if parent.user_id != user_id:
            raise ValueError(
                f"Parent node {parent_id!r} belongs to user {parent.user_id!r}, "
                f"not {user_id!r}"
            )

        node_id = f"tx_{secrets.token_hex(6)}"
        depth = parent.depth + 1

        # If the parent had docs filed directly on it (it was a leaf), we
        # cannot lose those assignments — move them to an auto-created
        # "unsorted" child so the parent can become internal cleanly.
        unsorted_id: Optional[str] = None
        if parent.is_leaf:
            parent_assignments = self._db.execute(
                "SELECT COUNT(*) AS n FROM kg_taxonomy_assignments WHERE node_id = ?",
                (parent_id,),
            ).fetchone()
            if parent_assignments and int(parent_assignments["n"]) > 0:
                unsorted_id = f"tx_{secrets.token_hex(6)}"
                warnings.warn(
                    f"add_node: parent {parent_id!r} had direct assignments; "
                    f"moved to auto-created 'unsorted' leaf {unsorted_id!r}.",
                    UserWarning,
                    stacklevel=2,
                )

        with self._db.conn:
            self._db.execute(
                "INSERT INTO kg_taxonomy_nodes("
                "node_id, user_id, parent_id, label, description, depth, is_leaf, doc_count, keywords"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (node_id, user_id, parent_id, label, description, depth,
                 1 if is_leaf else 0, _encode_keywords(keywords)),
            )
            if unsorted_id is not None:
                self._db.execute(
                    "INSERT INTO kg_taxonomy_nodes("
                    "node_id, user_id, parent_id, label, description, depth, is_leaf, doc_count"
                    ") VALUES (?, ?, ?, ?, ?, ?, 1, 0)",
                    (unsorted_id, user_id, parent_id, "unsorted", "", depth),
                )
                self._db.execute(
                    "UPDATE kg_taxonomy_assignments SET node_id = ? WHERE node_id = ?",
                    (unsorted_id, parent_id),
                )
            # Parent that gains a child stops being a leaf.
            self._db.execute(
                "UPDATE kg_taxonomy_nodes "
                "SET is_leaf = 0, updated_at = datetime('now') "
                "WHERE node_id = ?",
                (parent_id,),
            )
        self._commit()
        node = self.get_node(node_id)
        assert node is not None
        return node

    def update_node(
        self,
        node_id: str,
        *,
        label: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        existing = self.get_node(node_id)
        if existing is None:
            raise ValueError(f"Node {node_id!r} does not exist")
        if existing.parent_id is None:
            raise ValueError("Cannot rename the root node")

        sets: list[str] = []
        params: list = []
        if label is not None:
            sets.append("label = ?")
            params.append(label)
        if description is not None:
            sets.append("description = ?")
            params.append(description)
        if not sets:
            return
        sets.append("updated_at = datetime('now')")
        params.append(node_id)

        with self._db.conn:
            self._db.execute(
                f"UPDATE kg_taxonomy_nodes SET {', '.join(sets)} WHERE node_id = ?",
                params,
            )
        self._commit()

    def move_node(self, node_id: str, new_parent_id: str) -> None:
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} does not exist")
        if node.parent_id is None:
            raise ValueError("Cannot move the root node")
        new_parent = self.get_node(new_parent_id)
        if new_parent is None:
            raise ValueError(f"New parent {new_parent_id!r} does not exist")
        if new_parent.user_id != node.user_id:
            raise ValueError("Cross-user move is not allowed")
        if new_parent_id == node_id:
            raise ValueError("Cannot make a node its own parent")

        # Cycle check: walk from new_parent up to the root; node must not appear.
        cursor_id: Optional[str] = new_parent_id
        while cursor_id is not None:
            if cursor_id == node_id:
                raise ValueError(
                    f"Refusing cycle: {new_parent_id!r} is a descendant of {node_id!r}"
                )
            row = self._db.execute(
                "SELECT parent_id FROM kg_taxonomy_nodes WHERE node_id = ?",
                (cursor_id,),
            ).fetchone()
            cursor_id = row["parent_id"] if row else None

        old_parent_id = node.parent_id
        new_depth = new_parent.depth + 1
        depth_delta = new_depth - node.depth

        # Collect subtree (BFS) so we can shift depth in bulk.
        subtree_ids: list[str] = [node_id]
        frontier = [node_id]
        while frontier:
            placeholders = ",".join("?" for _ in frontier)
            rows = self._db.execute(
                f"SELECT node_id FROM kg_taxonomy_nodes WHERE parent_id IN ({placeholders})",
                frontier,
            ).fetchall()
            children = [r["node_id"] for r in rows]
            subtree_ids.extend(children)
            frontier = children

        with self._db.conn:
            self._db.execute(
                "UPDATE kg_taxonomy_nodes "
                "SET parent_id = ?, updated_at = datetime('now') "
                "WHERE node_id = ?",
                (new_parent_id, node_id),
            )
            if depth_delta != 0:
                placeholders = ",".join("?" for _ in subtree_ids)
                self._db.execute(
                    f"UPDATE kg_taxonomy_nodes "
                    f"SET depth = depth + ?, updated_at = datetime('now') "
                    f"WHERE node_id IN ({placeholders})",
                    [depth_delta, *subtree_ids],
                )
            # New parent gains a child → not a leaf anymore.
            self._db.execute(
                "UPDATE kg_taxonomy_nodes "
                "SET is_leaf = 0, updated_at = datetime('now') "
                "WHERE node_id = ?",
                (new_parent_id,),
            )
            # Old parent may have become childless; we don't auto-flip it back
            # to a leaf because that would invite silent semantics changes.
            # Callers that want that should call recompute_all_centroids and
            # inspect children separately.
            _ = old_parent_id
        self._commit()

    def delete_node(
        self,
        node_id: str,
        *,
        reassign_docs_to: Optional[str] = None,
    ) -> None:
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} does not exist")
        if node.parent_id is None:
            raise ValueError("Cannot delete the root node")

        if reassign_docs_to is not None:
            target = self.get_node(reassign_docs_to)
            if target is None:
                raise ValueError(f"Reassign target {reassign_docs_to!r} does not exist")
            if not target.is_leaf:
                raise ValueError(
                    f"Reassign target {reassign_docs_to!r} must be a leaf"
                )
            # Find all node_ids in the doomed subtree, then move their
            # assignments to the target before FK CASCADE wipes them.
            subtree_ids: list[str] = [node_id]
            frontier = [node_id]
            while frontier:
                placeholders = ",".join("?" for _ in frontier)
                rows = self._db.execute(
                    f"SELECT node_id FROM kg_taxonomy_nodes WHERE parent_id IN ({placeholders})",
                    frontier,
                ).fetchall()
                children = [r["node_id"] for r in rows]
                subtree_ids.extend(children)
                frontier = children
            with self._db.conn:
                placeholders = ",".join("?" for _ in subtree_ids)
                self._db.execute(
                    f"UPDATE kg_taxonomy_assignments SET node_id = ? "
                    f"WHERE node_id IN ({placeholders})",
                    [reassign_docs_to, *subtree_ids],
                )

        with self._db.conn:
            self._db.execute(
                "DELETE FROM kg_taxonomy_nodes WHERE node_id = ?",
                (node_id,),
            )
        self._commit()

    def get_node(self, node_id: str) -> Optional[TaxonomyNode]:
        row = self._db.execute(
            "SELECT node_id, user_id, parent_id, label, description, depth, "
            "is_leaf, centroid, centroid_dim, doc_count, created_at, updated_at, keywords "
            "FROM kg_taxonomy_nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        return self._row_to_node(row) if row else None

    def get_children(self, node_id: str) -> list[TaxonomyNode]:
        rows = self._db.execute(
            "SELECT node_id, user_id, parent_id, label, description, depth, "
            "is_leaf, centroid, centroid_dim, doc_count, created_at, updated_at, keywords "
            "FROM kg_taxonomy_nodes WHERE parent_id = ? ORDER BY label",
            (node_id,),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def list_nodes(self, user_id: str) -> list[TaxonomyNode]:
        # Phase 12 — serve from the per-user cache when fresh. Beam descend
        # calls this on every retrieval; caching skips the full SELECT + unpack
        # for bursts of queries. Results are identical to a fresh read.
        if self.cache_ttl_s > 0.0:
            cached = self._node_cache.get(user_id)
            if cached is not None and (time.monotonic() - cached[0]) < self.cache_ttl_s:
                return cached[1]
        rows = self._db.execute(
            "SELECT node_id, user_id, parent_id, label, description, depth, "
            "is_leaf, centroid, centroid_dim, doc_count, created_at, updated_at, keywords "
            "FROM kg_taxonomy_nodes WHERE user_id = ? ORDER BY depth, label",
            (user_id,),
        ).fetchall()
        nodes = [self._row_to_node(r) for r in rows]
        if self.cache_ttl_s > 0.0:
            self._node_cache[user_id] = (time.monotonic(), nodes)
        return nodes

    def set_node_keywords(
        self, user_id: str, node_id: str, keywords: list[str]
    ) -> None:
        """Persist *keywords* (JSON list) on a node. Invalidates the cache."""
        with self._db.conn:
            self._db.execute(
                "UPDATE kg_taxonomy_nodes "
                "SET keywords = ?, updated_at = datetime('now') "
                "WHERE node_id = ? AND user_id = ?",
                (_encode_keywords(keywords), node_id, user_id),
            )
        self._commit()
        self._invalidate_cache(user_id)

    def get_tree(self, user_id: str) -> dict[str, list[str]]:
        rows = self._db.execute(
            "SELECT node_id, parent_id FROM kg_taxonomy_nodes "
            "WHERE user_id = ? ORDER BY depth, label",
            (user_id,),
        ).fetchall()
        tree: dict[str, list[str]] = {r["node_id"]: [] for r in rows}
        for r in rows:
            parent = r["parent_id"]
            if parent is not None and parent in tree:
                tree[parent].append(r["node_id"])
        return tree

    def clear(self, user_id: str, *, wipe_doc_meta: bool = False) -> None:
        """Drop the user's taxonomy tree (nodes + assignments via FK cascade).

        ``doc_meta`` is a per-doc summary+centroid cache that survives tree
        rebuilds by default — wiping it would force expensive re-summarization
        and re-embedding for every doc. Set ``wipe_doc_meta=True`` to force a
        complete reset (e.g. for the `taxonomy clear` CLI command).
        """
        with self._db.conn:
            self._db.execute(
                "DELETE FROM kg_taxonomy_nodes WHERE user_id = ?",
                (user_id,),
            )
            if wipe_doc_meta:
                self._db.execute(
                    "DELETE FROM kg_taxonomy_doc_meta WHERE user_id = ?",
                    (user_id,),
                )
        self._commit()

    # ------------------------------------------------------------------
    # Assignments
    # ------------------------------------------------------------------

    def assign_doc(
        self,
        user_id: str,
        doc_id: str,
        node_id: str,
        score: float = 1.0,
        is_primary: bool = True,
    ) -> None:
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} does not exist")
        if not node.is_leaf:
            raise ValueError(
                f"Cannot assign doc to internal node {node_id!r}; assignments must target leaves"
            )
        if node.user_id != user_id:
            raise ValueError(
                f"Node {node_id!r} belongs to user {node.user_id!r}, not {user_id!r}"
            )

        with self._db.conn:
            # Upsert by the unique index (user_id, doc_id, node_id).
            self._db.execute(
                "INSERT INTO kg_taxonomy_assignments(user_id, doc_id, node_id, score, is_primary) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, doc_id, node_id) DO UPDATE SET "
                "score = excluded.score, is_primary = excluded.is_primary",
                (user_id, doc_id, node_id, float(score), 1 if is_primary else 0),
            )
        self._commit()

    def unassign_doc(
        self,
        user_id: str,
        doc_id: str,
        node_id: Optional[str] = None,
    ) -> int:
        if node_id is None:
            cur = self._db.execute(
                "DELETE FROM kg_taxonomy_assignments WHERE user_id = ? AND doc_id = ?",
                (user_id, doc_id),
            )
        else:
            cur = self._db.execute(
                "DELETE FROM kg_taxonomy_assignments "
                "WHERE user_id = ? AND doc_id = ? AND node_id = ?",
                (user_id, doc_id, node_id),
            )
        removed = cur.rowcount or 0
        self._commit()
        return int(removed)

    def get_docs_at(
        self,
        node_id: str,
        *,
        include_descendants: bool = False,
    ) -> list[str]:
        if not include_descendants:
            rows = self._db.execute(
                "SELECT DISTINCT doc_id FROM kg_taxonomy_assignments "
                "WHERE node_id = ? ORDER BY doc_id",
                (node_id,),
            ).fetchall()
            return [r["doc_id"] for r in rows]

        # BFS the subtree, then bulk-select assignments.
        subtree_ids: list[str] = [node_id]
        frontier = [node_id]
        while frontier:
            placeholders = ",".join("?" for _ in frontier)
            rows = self._db.execute(
                f"SELECT node_id FROM kg_taxonomy_nodes WHERE parent_id IN ({placeholders})",
                frontier,
            ).fetchall()
            children = [r["node_id"] for r in rows]
            subtree_ids.extend(children)
            frontier = children
        placeholders = ",".join("?" for _ in subtree_ids)
        rows = self._db.execute(
            f"SELECT DISTINCT doc_id FROM kg_taxonomy_assignments "
            f"WHERE node_id IN ({placeholders}) ORDER BY doc_id",
            subtree_ids,
        ).fetchall()
        return [r["doc_id"] for r in rows]

    def get_doc_nodes(self, user_id: str, doc_id: str) -> list[TaxonomyNode]:
        rows = self._db.execute(
            "SELECT n.node_id, n.user_id, n.parent_id, n.label, n.description, "
            "n.depth, n.is_leaf, n.centroid, n.centroid_dim, n.doc_count, "
            "n.created_at, n.updated_at "
            "FROM kg_taxonomy_nodes n "
            "JOIN kg_taxonomy_assignments a ON a.node_id = n.node_id "
            "WHERE a.user_id = ? AND a.doc_id = ? "
            "ORDER BY a.is_primary DESC, n.label",
            (user_id, doc_id),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    # ------------------------------------------------------------------
    # Doc-meta cache
    # ------------------------------------------------------------------

    def upsert_doc_meta(
        self,
        user_id: str,
        doc_id: str,
        summary: Optional[str],
        centroid: list[float],
    ) -> None:
        blob: Optional[bytes] = None
        dim: Optional[int] = None
        if centroid:
            blob = self._pack(centroid)
            dim = len(centroid)
        with self._db.conn:
            self._db.execute(
                "INSERT INTO kg_taxonomy_doc_meta("
                "user_id, doc_id, summary, centroid, centroid_dim, updated_at"
                ") VALUES (?, ?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT(user_id, doc_id) DO UPDATE SET "
                "summary = excluded.summary, "
                "centroid = excluded.centroid, "
                "centroid_dim = excluded.centroid_dim, "
                "updated_at = excluded.updated_at",
                (user_id, doc_id, summary, blob, dim),
            )
        self._commit()

    def get_doc_meta(self, user_id: str, doc_id: str) -> Optional[dict]:
        row = self._db.execute(
            "SELECT summary, centroid, centroid_dim "
            "FROM kg_taxonomy_doc_meta WHERE user_id = ? AND doc_id = ?",
            (user_id, doc_id),
        ).fetchone()
        if row is None:
            return None
        centroid: Optional[list[float]] = None
        if row["centroid"] is not None and row["centroid_dim"]:
            centroid = self._unpack(row["centroid"], int(row["centroid_dim"]))
        return {"summary": row["summary"], "centroid": centroid}

    # ------------------------------------------------------------------
    # Centroid recompute
    # ------------------------------------------------------------------

    def recompute_node_centroid(self, user_id: str, node_id: str) -> None:
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} does not exist")

        new_centroid: Optional[list[float]] = None
        new_doc_count = 0

        if node.is_leaf:
            rows = self._db.execute(
                "SELECT m.centroid, m.centroid_dim "
                "FROM kg_taxonomy_assignments a "
                "JOIN kg_taxonomy_doc_meta m "
                "  ON m.user_id = a.user_id AND m.doc_id = a.doc_id "
                "WHERE a.node_id = ? AND a.user_id = ?",
                (node_id, user_id),
            ).fetchall()
            count_row = self._db.execute(
                "SELECT COUNT(*) AS n FROM kg_taxonomy_assignments "
                "WHERE node_id = ? AND user_id = ?",
                (node_id, user_id),
            ).fetchone()
            new_doc_count = int(count_row["n"]) if count_row else 0

            vecs: list[list[float]] = []
            for r in rows:
                if r["centroid"] is None or not r["centroid_dim"]:
                    continue
                vecs.append(self._unpack(r["centroid"], int(r["centroid_dim"])))
            if vecs:
                new_centroid = _mean(vecs)
        else:
            child_rows = self._db.execute(
                "SELECT centroid, centroid_dim, doc_count "
                "FROM kg_taxonomy_nodes WHERE parent_id = ?",
                (node_id,),
            ).fetchall()
            vecs = []
            for r in child_rows:
                new_doc_count += int(r["doc_count"] or 0)
                if r["centroid"] is None or not r["centroid_dim"]:
                    continue
                vecs.append(self._unpack(r["centroid"], int(r["centroid_dim"])))
            if vecs:
                new_centroid = _mean(vecs)

        blob: Optional[bytes] = self._pack(new_centroid) if new_centroid else None
        dim: Optional[int] = len(new_centroid) if new_centroid else None
        with self._db.conn:
            self._db.execute(
                "UPDATE kg_taxonomy_nodes "
                "SET centroid = ?, centroid_dim = ?, doc_count = ?, "
                "    updated_at = datetime('now') "
                "WHERE node_id = ?",
                (blob, dim, new_doc_count, node_id),
            )
        self._commit()

    def recompute_all_centroids(self, user_id: str) -> None:
        # Bottom-up: deepest nodes first so internal nodes see already-fresh
        # children when they average up.
        rows = self._db.execute(
            "SELECT node_id FROM kg_taxonomy_nodes "
            "WHERE user_id = ? ORDER BY depth DESC",
            (user_id,),
        ).fetchall()
        node_ids = [r["node_id"] for r in rows]
        if not node_ids:
            return

        # Per-project rule: any >10s op must surface progress. Centroid
        # recompute is multi-second on real corpora — show a bar.
        try:
            from rich.progress import (  # noqa: PLC0415
                BarColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
            )
        except ImportError:  # pragma: no cover
            Progress = None  # type: ignore[assignment]

        if Progress is None:
            for nid in node_ids:
                self.recompute_node_centroid(user_id, nid)
            return

        with Progress(
            TextColumn("[bold blue]recompute_all_centroids"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("centroids", total=len(node_ids))
            for nid in node_ids:
                self.recompute_node_centroid(user_id, nid)
                progress.advance(task)

    # ------------------------------------------------------------------
    # Beam descend (hot path)
    # ------------------------------------------------------------------

    def beam_descend(
        self,
        user_id: str,
        query_embedding: list[float],
        beam_width: int,
        max_depth: int,
        min_score: float = 0.05,
        dominance_gap: float = 0.0,
        min_top_score_floor: float = 0.0,
        query_keywords: Optional[list[str]] = None,
        keyword_weight: float = 0.0,
    ) -> DescendResult:
        """Beam-descend the user's taxonomy tree.

        ``dominance_gap`` enables adaptive beam narrowing. After sorting the
        candidates of a level by score, we walk from the top and stop as soon
        as the consecutive gap (s[i-1] - s[i]) exceeds ``dominance_gap``.
        A clear winner (e.g. +0.37 vs +0.13 with gap 0.10) thus narrows the
        beam to 1 even when ``beam_width`` is larger — avoids spuriously
        descending obviously-unrelated branches.
        Set ``dominance_gap`` to 0.0 (default) to disable the heuristic.

        ``min_top_score_floor`` is a confidence floor applied at EVERY level.
        If the BEST candidate at any depth scores below the floor, we narrow
        the beam to 1 at that level — a weak top match means the query has
        no real home in this part of the tree, so dragging siblings along
        only opens unrelated docs. Defaults to 0.0 (disabled).
        """
        root = self.ensure_root(user_id)

        # One SELECT, then in-memory tree walk: avoids N+1 child fetches.
        all_nodes = self.list_nodes(user_id)
        by_id: dict[str, TaxonomyNode] = {n.node_id: n for n in all_nodes}
        children_of: dict[str, list[TaxonomyNode]] = {nid: [] for nid in by_id}
        for n in all_nodes:
            if n.parent_id is not None and n.parent_id in children_of:
                children_of[n.parent_id].append(n)
        for nid in children_of:
            children_of[nid].sort(key=lambda x: x.label)

        if not children_of.get(root.node_id):
            return DescendResult(leaves=[], trace=[])

        # Floor for "missing centroid" — we want such nodes to fall below
        # min_score so they only survive the "keep the best one" rescue.
        missing_score = min_score - 1.0

        frontier: list[tuple[TaxonomyNode, float]] = [(root, 1.0)]
        leaves: list[NodeScore] = []
        trace: list[LevelTrace] = []
        depth = 0

        # Phase 12 — hybrid routing. When keyword_weight > 0 and we have query
        # keywords, blend a normalized keyword-overlap signal into the cosine
        # score: combined = cosine + keyword_weight * overlap. A node with no
        # keywords contributes 0 overlap, so this is a strict no-op on
        # un-keyworded trees (preserves the dense-only behaviour exactly).
        kw_on = bool(keyword_weight) and bool(query_keywords)

        while frontier and depth < max_depth:
            considered: list[NodeScore] = []
            for parent_node, _ in frontier:
                for child in children_of.get(parent_node.node_id, []):
                    if child.centroid is None:
                        cos = missing_score
                    else:
                        cos = self._cosine(query_embedding, child.centroid)
                    kw = 0.0
                    if kw_on and child.keywords:
                        kw = keyword_overlap(query_keywords or [], child.keywords)
                    combined = float(cos) + (keyword_weight * kw if kw_on else 0.0)
                    considered.append(
                        NodeScore(node=child, score=combined, keyword_score=float(kw))
                    )

            if not considered:
                trace.append(LevelTrace(depth=depth, considered=[], kept=[]))
                break

            considered.sort(key=lambda x: x.score, reverse=True)
            above_floor = [c for c in considered if c.score >= min_score]

            # Confidence floor — applied at EVERY level, not just the root.
            # If even the best candidate at this level scores weakly, force
            # beam=1 so we don't drag siblings into a doomed retrieval.
            if (
                min_top_score_floor > 0.0
                and above_floor
                and above_floor[0].score < min_top_score_floor
            ):
                kept = above_floor[:1]
            elif dominance_gap > 0.0 and above_floor:
                # Adaptive beam: walk from the top and stop as soon as the gap
                # between consecutive scores exceeds dominance_gap. A clearly-
                # dominant top score thus narrows the beam without dragging
                # along weakly-related siblings.
                effective_beam = min(beam_width, len(above_floor))
                for i in range(1, effective_beam):
                    if above_floor[i - 1].score - above_floor[i].score >= dominance_gap:
                        effective_beam = i
                        break
                kept = above_floor[:effective_beam]
            else:
                kept = above_floor[:beam_width]

            # Rescue: never empty-out the beam — keep the single best so the
            # GUI sees a (low-confidence) path instead of "nothing matched".
            if not kept:
                kept = [considered[0]]

            trace.append(LevelTrace(depth=depth, considered=considered, kept=kept))

            next_frontier: list[tuple[TaxonomyNode, float]] = []
            for ns in kept:
                if ns.node.is_leaf:
                    leaves.append(ns)
                else:
                    next_frontier.append((ns.node, ns.score))
            frontier = next_frontier
            depth += 1

        return DescendResult(leaves=leaves, trace=trace)


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _mean(vecs: list[list[float]]) -> list[float]:
    dim = len(vecs[0])
    out = [0.0] * dim
    for v in vecs:
        for i in range(dim):
            out[i] += v[i]
    n = float(len(vecs))
    return [x / n for x in out]


def _row_get(row, key: str):
    """Return ``row[key]`` or None when the column is absent.

    sqlite3.Row raises IndexError on an unknown column; on a pre-Phase-12 DB
    the ``keywords`` column may not exist, so we probe ``keys()`` first.
    """
    try:
        if key in row.keys():
            return row[key]
    except (AttributeError, IndexError, TypeError):
        try:
            return row[key]
        except (IndexError, KeyError):
            return None
    return None


def _decode_keywords(raw) -> list[str]:
    """Decode the JSON keywords TEXT column into a clean list of strings."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()]


def _encode_keywords(keywords: Optional[list[str]]) -> Optional[str]:
    """Encode a keyword list to compact JSON TEXT (None when empty)."""
    if not keywords:
        return None
    clean = [str(k).strip() for k in keywords if str(k).strip()]
    if not clean:
        return None
    return json.dumps(clean, ensure_ascii=False)
