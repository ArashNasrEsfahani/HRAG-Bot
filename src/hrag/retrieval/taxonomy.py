"""TaxonomyRetriever: hierarchical-taxonomy retriever (Phase 3).

Retrieval flow
--------------
1. Embed the query.
2. Beam-descend the user's taxonomy tree (``TaxonomyStore.beam_descend``) to
   pick a small set of leaf nodes by cosine on node centroids.
3. Collect the documents living under those leaves.
4. Push the leaf-doc allow-list into Chroma's ``where`` clause via
   ``VectorStore.query(doc_ids=...)`` — the scan only touches chunks belonging
   to those documents. Nothing else gets considered.
5. Hydrate from SQLite (same pattern as :class:`VectorRetriever`) and wrap as
   ``RetrievalResult`` rows tagged ``retriever="taxonomy"``.

The descend trace is always stashed on ``self.last_descend`` (even on the
empty-tree path) and exposed via :meth:`describe_last_descend` for the GUI.
The returned dict carries the full tree topology, per-level beam considered
nodes, picked leaves with their document titles, and a ``stats`` block the
chat UI uses to render the "opened X of Y documents" callout.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from hrag.config import TaxonomyConfig
from hrag.db.connection import Database
from hrag.providers.embeddings import EmbeddingProvider
from hrag.retrieval.base import Retriever
from hrag.retrieval.vector import VectorStore
from hrag.taxonomy.types import DescendResult
from hrag.types import Chunk, RetrievalResult

if TYPE_CHECKING:
    from hrag.intent import Intent

logger = logging.getLogger(__name__)


class TaxonomyRetriever(Retriever):
    """Retriever that scopes vector search to documents under beam-picked leaves."""

    name = "taxonomy"

    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        taxonomy_store: Any,   # hrag.taxonomy.store.TaxonomyStore (forward decl: module in flight)
        cfg: TaxonomyConfig,
        fallback: Retriever,
    ) -> None:
        self._db = db
        self._vector_store = vector_store
        self._embedder = embedder
        self._taxonomy_store = taxonomy_store
        self._cfg = cfg
        self._fallback = fallback
        self.last_descend: DescendResult = DescendResult(leaves=[], trace=[])
        self._tree_empty: bool = False
        self._last_user_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Retriever interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 30,
        source_types: Optional[list[str]] = None,
        intent_hint: Optional["Intent"] = None,
        where: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        self._last_user_id = user_id
        q_emb: list[float] = self._embedder.embed_one(query)

        # The old `short_query_force_top1_words` knob has been removed — its
        # job is now done upstream by the intent classifier (queries like
        # "hey" / "thanks" never reach retrieval at all; they're handled by
        # the GREETING path in orchestrator.chat).
        result: DescendResult = self._taxonomy_store.beam_descend(
            user_id,
            q_emb,
            self._cfg.beam_width,
            self._cfg.max_depth,
            self._cfg.min_node_score,
            dominance_gap=getattr(self._cfg, "beam_dominance_gap", 0.0),
            min_top_score_floor=getattr(self._cfg, "min_top_score_floor", 0.0),
        )
        self.last_descend = result
        self._tree_empty = False

        if not result.leaves:
            # Empty tree (first run) or beam pruned everything — degrade gracefully.
            logger.warning(
                "TaxonomyRetriever: beam_descend produced no leaves for user_id=%s; "
                "delegating to fallback retriever (%s).",
                user_id,
                getattr(self._fallback, "name", "?"),
            )
            self._tree_empty = True
            return self._fallback.retrieve(
                query, user_id, top_k=top_k, source_types=source_types,
                intent_hint=intent_hint, where=where,
            )

        # Per-leaf docs, leaves sorted by score descending so the cap below
        # is order-stable (best leaf gets in first).
        leaves_sorted = sorted(
            result.leaves, key=lambda lf: float(lf.score), reverse=True
        )
        per_leaf_docs: list[tuple[Any, list[str]]] = [
            (leaf, self._taxonomy_store.get_docs_at(leaf.node.node_id))
            for leaf in leaves_sorted
        ]

        # ----- max_docs_pct safety cap --------------------------------------
        # Imbalanced trees can have a single leaf that owns most of the corpus
        # (e.g. "General Scientific Research" with 14 of 24 docs). For
        # low-confidence queries that leaf will still be the only winner —
        # but keeping it opens 58% of the library. Cap the opened-set size
        # by dropping the weakest leaves (always keep at least one).
        max_pct = float(getattr(self._cfg, "max_docs_pct", 1.0))
        if 0.0 < max_pct < 1.0:
            total = self._count_user_docs(user_id)
            cap = max(1, int(total * max_pct)) if total > 0 else 0
            if cap > 0:
                kept_leaves: list[Any] = []
                kept_docs: set[str] = set()
                for leaf, docs in per_leaf_docs:
                    candidate = kept_docs.union(docs)
                    if kept_leaves and len(candidate) > cap:
                        # Adding this leaf would breach the cap — and we
                        # already have at least one. Stop.
                        break
                    kept_leaves.append(leaf)
                    kept_docs = candidate
                if len(kept_leaves) < len(per_leaf_docs):
                    dropped = [
                        lf.node.label for lf, _ in per_leaf_docs[len(kept_leaves):]
                    ]
                    logger.info(
                        "TaxonomyRetriever: max_docs_pct=%.2f → dropped %d leaves "
                        "(%s) so opened-doc count stays within %d of %d.",
                        max_pct, len(dropped), ", ".join(dropped), cap, total,
                    )
                    # Reflect the trim in the descend result so the UI shows
                    # what actually drove retrieval.
                    result.leaves = kept_leaves  # type: ignore[assignment]
                    self.last_descend = result
                    per_leaf_docs = per_leaf_docs[: len(kept_leaves)]

        leaf_doc_ids: list[str] = []
        for _leaf, docs in per_leaf_docs:
            leaf_doc_ids.extend(docs)
        # dedupe while preserving order
        leaf_doc_ids = list(dict.fromkeys(leaf_doc_ids))

        if not leaf_doc_ids:
            logger.warning(
                "TaxonomyRetriever: %d leaves picked but no documents under them "
                "(user_id=%s); delegating to fallback.",
                len(result.leaves),
                user_id,
            )
            return self._fallback.retrieve(
                query, user_id, top_k=top_k, source_types=source_types,
                intent_hint=intent_hint, where=where,
            )

        # Push the allow-list directly into Chroma — the scan only touches
        # chunks belonging to leaf docs. No over-fetch waste.
        pairs = self._vector_store.query(
            user_id=user_id,
            query_embedding=q_emb,
            top_k=top_k,
            source_types=source_types,
            doc_ids=leaf_doc_ids,
            where=where,
        )

        if not pairs:
            return []

        chunk_ids = [cid for cid, _ in pairs[:top_k]]
        score_map: dict[str, float] = {cid: score for cid, score in pairs[:top_k]}
        chunks_by_id = self._hydrate(chunk_ids)

        results: list[RetrievalResult] = []
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score_map[chunk_id],
                    retriever=self.name,
                    rerank_score=None,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Trace exposure (consumed by orchestrator / GUI)
    # ------------------------------------------------------------------

    def describe_last_descend(self, user_id: Optional[str] = None) -> dict[str, Any]:
        """Return a JSON-serializable view of the most recent beam descent.

        Carries everything the GUI needs to render the navigation visual:

        - ``leaves``     — picked leaves with score + doc count + sample titles.
        - ``trace``      — per-level beam, sorted by score, kept-vs-pruned flag.
        - ``tree``       — full tree topology (every node, parent ids) so the
                           chat page can render the WHOLE tree with the chosen
                           path highlighted, not just the descent slice.
        - ``stats``      — corpus-wide counts so the chat can show
                           "opened X of Y documents".
        - ``note``       — set when retrieval fell back (e.g. empty tree).
        """
        result = self.last_descend

        if not result.leaves and not result.trace:
            return {
                "leaves": [],
                "trace": [],
                "tree": {"nodes": [], "root_id": None},
                "stats": {
                    "total_docs": 0,
                    "leaves_picked": 0,
                    "docs_opened": 0,
                    "nodes_considered": 0,
                    "tree_size": 0,
                },
                "note": "tree empty — fell back to vector",
            }

        # ----- leaves (picked) ----------------------------------------------
        leaves_out: list[dict[str, Any]] = []
        opened_doc_ids: set[str] = set()
        for leaf in result.leaves:
            try:
                doc_ids = list(
                    self._taxonomy_store.get_docs_at(leaf.node.node_id)
                )
            except Exception:
                doc_ids = []
            opened_doc_ids.update(doc_ids)
            doc_titles = self._fetch_doc_titles(doc_ids[:6])
            leaves_out.append(
                {
                    "node_id": leaf.node.node_id,
                    "label": leaf.node.label,
                    "score": float(leaf.score),
                    "doc_count": len(doc_ids),
                    "doc_titles": doc_titles,
                }
            )

        # ----- per-level trace (full beam, kept marked) ---------------------
        trace_out: list[dict[str, Any]] = []
        kept_node_ids: set[str] = set()
        considered_total = 0
        for level in result.trace:
            kept_ids = {ns.node.node_id for ns in level.kept}
            kept_node_ids.update(kept_ids)
            considered = [
                {
                    "node_id": ns.node.node_id,
                    "label": ns.node.label,
                    "score": float(ns.score),
                    "kept": ns.node.node_id in kept_ids,
                }
                for ns in level.considered
            ]
            considered.sort(key=lambda row: row["score"], reverse=True)
            considered_total += len(considered)
            trace_out.append({"depth": level.depth, "considered": considered})

        # ----- full tree topology -------------------------------------------
        tree_nodes: list[dict[str, Any]] = []
        root_id: Optional[str] = None
        try:
            if user_id is not None:
                all_nodes = self._taxonomy_store.list_nodes(user_id)
                for n in all_nodes:
                    if n.parent_id is None:
                        root_id = n.node_id
                    tree_nodes.append(
                        {
                            "node_id": n.node_id,
                            "parent_id": n.parent_id,
                            "label": n.label,
                            "depth": n.depth,
                            "is_leaf": bool(n.is_leaf),
                            "doc_count": int(n.doc_count),
                            "on_path": n.node_id in kept_node_ids,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("describe_last_descend: tree topology unavailable: %s", exc)

        # ----- stats --------------------------------------------------------
        total_docs = self._count_user_docs(user_id) if user_id else 0
        stats = {
            "total_docs": total_docs,
            "leaves_picked": len(leaves_out),
            "docs_opened": len(opened_doc_ids),
            "nodes_considered": considered_total,
            "tree_size": len(tree_nodes),
        }

        out: dict[str, Any] = {
            "leaves": leaves_out,
            "trace": trace_out,
            "tree": {"nodes": tree_nodes, "root_id": root_id},
            "stats": stats,
        }
        if self._tree_empty:
            out["note"] = "tree empty — fell back to vector"
        return out

    def _fetch_doc_titles(self, doc_ids: list[str]) -> list[str]:
        """Best-effort lookup of doc titles for the trace payload."""
        if not doc_ids:
            return []
        placeholders = ",".join(["?"] * len(doc_ids))
        try:
            rows = self._db.execute(
                f"SELECT doc_id, title FROM documents WHERE doc_id IN ({placeholders})",
                doc_ids,
            ).fetchall()
        except Exception:  # noqa: BLE001
            return []
        by_id = {r["doc_id"]: (r["title"] or r["doc_id"]) for r in rows}
        return [by_id.get(did, did) for did in doc_ids]

    def _count_user_docs(self, user_id: Optional[str]) -> int:
        if user_id is None:
            return 0
        try:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        except Exception:  # noqa: BLE001
            return 0
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hydrate(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        """SELECT chunks from SQLite; skip tombstoned (excluded=1) rows.

        Mirrors :meth:`VectorRetriever._hydrate` — kept verbatim so the two
        retrievers stay shape-compatible.
        """
        if not chunk_ids:
            return {}

        placeholders = ",".join(["?"] * len(chunk_ids))
        sql = f"""
            SELECT
                chunk_id, doc_id, user_id,
                text, title, section, subsection,
                chunk_index, token_count, source_type,
                excluded, metadata
            FROM chunks
            WHERE chunk_id IN ({placeholders})
        """
        cursor = self._db.execute(sql, chunk_ids)
        rows = cursor.fetchall()

        result: dict[str, Chunk] = {}
        for row in rows:
            if row["excluded"] == 1:
                continue

            raw_meta = row["metadata"]
            meta: dict = {}
            if raw_meta:
                try:
                    meta = json.loads(raw_meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}

            chunk = Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                user_id=row["user_id"],
                text=row["text"],
                embedding_text=row["text"],
                title=row["title"] or "",
                section=row["section"] or "",
                subsection=row["subsection"] or "",
                chunk_index=row["chunk_index"],
                token_count=row["token_count"],
                source_type=row["source_type"],
                metadata=meta,
            )
            result[row["chunk_id"]] = chunk

        return result
