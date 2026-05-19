"""KGPPRRetriever: HippoRAG-faithful knowledge-graph Personalized PageRank retriever.

Pipeline at retrieve time (HippoRAG / HippoRAG 2 alignment):

  1. NER on the query -> surface entity strings.
  2. For EACH entity, find the top-K phrase nodes by cosine similarity over
     the canonicalized embeddings already stored on the phrase nodes.
     Reset probability is distributed proportional to similarity (HippoRAG 2
     "broadened seed set" -- helps "synonymy threshold" spread mass over
     {tau, threshold, synonymy threshold} variants).
  3. Build the FULL sparse adjacency over ALL nodes (phrase + passage), so
     PPR mass naturally flows through phrase->phrase, phrase->passage,
     phrase->phrase->passage paths.
  4. Run Personalized PageRank seeded with the weighted seed set.
  5. HippoRAG 2 "passage-node integration" combined scoring:

         passage_score = alpha * passage_node_ppr_score
                       + (1 - alpha) * sum(phrase_ppr_score
                                            for phrase in passage_phrases)

     - alpha = 0  -> legacy phrase-aggregate behavior
     - alpha = 1  -> read passage scores directly from PPR (HippoRAG 2)
     - alpha = 0.5 -> the paper's recommended mix (default)

     If the KG has no passage nodes, alpha is silently forced to 0 so the
     legacy phrase-aggregate path remains valid.
  6. Optional chunk-position prior (no-op when beta = 0):

         position_weight = 1 - beta * chunk.section_depth   (clamped to >= 0)

     Today no chunk carries `section_depth`; the multiplier is 1.0 in that
     case, so this is a future hook -- see TODO below.
  7. Hydrate Chunks from SQLite, apply source_types filter, drop tombstones.
  8. Return RetrievalResult list with retriever='kg_ppr'.

Design notes
------------
- Seed broadening is done via the embeddings stored on each phrase node
  during canonicalization in KGStore (see `_canonicalize_phrase`).  We do
  NOT call the embedder at query time -- that would couple this module to
  the embedding provider.  We embed the surface form ONCE per query via
  the kg_store-bound NER (which already uses an embedder).  Falling back
  to exact-match-only when no embedder is wired keeps the module
  importable in tests that stub heavy deps.
- Passage-phrase membership for the (1 - alpha) phrase-aggregate term is
  read directly from the in-memory NetworkX graph -- the phrase --contains-->
  passage edges are exactly the membership relation.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from hrag.db.connection import Database
from hrag.retrieval.base import Retriever
from hrag.types import Chunk, RetrievalResult

if TYPE_CHECKING:
    from hrag.intent import Intent
    from hrag.kg.ner import NER
    from hrag.kg.store import KGStore

logger = logging.getLogger(__name__)


def _cosine(a, b) -> float:
    """Cosine similarity over numpy-castable vectors. Returns 0.0 on zero norms."""
    import numpy as np  # noqa: PLC0415

    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


class KGPPRRetriever(Retriever):
    """HippoRAG-faithful KG retriever using Personalized PageRank.

    Given a natural-language query:
    - Extract named entities via NER.
    - Match each entity to its top-K phrase nodes (by cosine similarity of
      embeddings stored on the phrase nodes during canonicalization).
    - Seed PPR with reset probability proportional to similarity.
    - Score passages via the HippoRAG 2 mix:
          alpha * passage_node_ppr + (1 - alpha) * sum(phrase_ppr in passage)
    - Hydrate the top-k passages from SQLite and return them.

    Returns [] when the KG is empty or no seeds match, so the orchestrator
    or router can transparently fall back to vector retrieval.
    """

    name = "kg_ppr"

    def __init__(
        self,
        db: Database,
        kg_store: "KGStore",
        ner: "NER",
        damping: float = 0.5,
        seed_top_k: int = 3,
        passage_node_alpha: float = 0.5,
        section_depth_beta: float = 0.0,
    ) -> None:
        self._db = db
        self._kg_store = kg_store
        self._ner = ner
        self._damping = damping
        self._seed_top_k = max(1, int(seed_top_k))
        self._alpha = max(0.0, min(1.0, float(passage_node_alpha)))
        self._beta = float(section_depth_beta)

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
        """Run PPR over the KG and return up to *top_k* passage chunks.

        Note: this retriever does not honour ``where=`` metadata filters; pass
        to a vector-based retriever instead.
        """
        # `where` is accepted for Retriever Protocol compatibility but ignored
        # — KG-PPR scores chunks via graph mass, not Chroma metadata.
        del where  # explicitly ignored
        # Lazy-import heavy deps so this module is importable without them.
        import numpy as np  # noqa: PLC0415

        # ------------------------------------------------------------------
        # 1. Guard: empty KG
        # ------------------------------------------------------------------
        if self._kg_store.num_phrase_nodes() == 0:
            logger.debug("KGPPRRetriever: KG has no phrase nodes; returning [].")
            return []

        # ------------------------------------------------------------------
        # 2. NER -> surface forms
        # ------------------------------------------------------------------
        surface_forms: list[str] = self._ner.extract(query)
        if not surface_forms:
            logger.debug("KGPPRRetriever: NER returned no entities; returning [].")
            return []

        # ------------------------------------------------------------------
        # 3. Broadened seed set: per-entity top-K phrase nodes weighted by
        #    cosine similarity. Falls back to exact-match canonical node when
        #    no embedder hook is reachable on the kg_store.
        # ------------------------------------------------------------------
        seed_node_ids, seed_weights = self._build_seed_set(surface_forms)
        if not seed_node_ids:
            logger.debug(
                "KGPPRRetriever: no phrase nodes matched %s; returning [].",
                surface_forms,
            )
            return []

        # ------------------------------------------------------------------
        # 4. Sparse adjacency over the FULL graph (phrase + passage nodes).
        #    HippoRAG 2's "passage-node integration" requires PPR to flow
        #    onto passage nodes directly.
        # ------------------------------------------------------------------
        adj, node_ids = self._kg_store.to_sparse_adjacency()
        if adj.shape[0] == 0 or len(node_ids) == 0:
            logger.debug("KGPPRRetriever: empty adjacency matrix; returning [].")
            return []

        # ------------------------------------------------------------------
        # 5. Map seed node_ids to matrix indices, dropping any seed not in
        #    the adjacency (defensive; shouldn't happen).
        # ------------------------------------------------------------------
        node_index: dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}
        seed_indices: list[int] = []
        kept_weights: list[float] = []
        for sid, w in zip(seed_node_ids, seed_weights):
            idx = node_index.get(sid)
            if idx is not None:
                seed_indices.append(idx)
                kept_weights.append(w)

        if not seed_indices:
            logger.debug(
                "KGPPRRetriever: seed nodes not present in adjacency matrix; returning []."
            )
            return []

        # ------------------------------------------------------------------
        # 6. Run PPR with the weighted seed set.
        # ------------------------------------------------------------------
        from hrag.kg.ppr import personalized_pagerank  # noqa: PLC0415

        scores: np.ndarray = personalized_pagerank(
            adj,
            seed_indices,
            damping=self._damping,
            seed_weights=kept_weights if any(w > 0 for w in kept_weights) else None,
        )

        # Guard against degenerate NaN outputs (e.g. disconnected graph)
        if np.any(np.isnan(scores)):
            logger.warning(
                "KGPPRRetriever: PPR returned NaN scores; returning []."
            )
            return []

        # ------------------------------------------------------------------
        # 7. Identify passage-node indices using SQLite as the authoritative
        #    source. If the KG has no passage nodes at all we silently force
        #    alpha = 0 (phrase-aggregate fallback).
        # ------------------------------------------------------------------
        passage_ids: set[str] = self._fetch_passage_node_ids(user_id)
        passage_nodes_present = len(passage_ids) > 0
        effective_alpha = self._alpha if passage_nodes_present else 0.0

        # ------------------------------------------------------------------
        # 8. HippoRAG 2 combined scoring per candidate passage.
        #
        #    passage_score = alpha * passage_node_ppr
        #                  + (1 - alpha) * sum(phrase_ppr in passage)
        #
        #    The phrase-aggregate uses the kg_store's own contains-edge
        #    membership (predecessors of the passage node whose node_type
        #    is "phrase"). When alpha = 1 the phrase term is skipped
        #    entirely (HippoRAG 2 pure passage-node retrieval).
        # ------------------------------------------------------------------
        candidate_pids: list[str]
        if passage_nodes_present:
            # Prefer passage nodes already present in the live graph;
            # the SQLite mirror may be slightly larger.
            candidate_pids = [nid for nid in node_ids if nid in passage_ids]
        else:
            # No passage nodes in the KG -- fall back to projecting via
            # passage_nodes_for(seeded phrase nodes). This is the legacy
            # "co-occurrence" projection. In practice the live graph always
            # has passage nodes (KGStore creates them on every upsert), so
            # this branch is mostly defensive for tests / future graphs.
            candidate_pids = sorted(
                self._kg_store.passage_nodes_for(seed_node_ids)
            )

        if not candidate_pids:
            logger.debug("KGPPRRetriever: no passage candidates; returning [].")
            return []

        # Pre-fetch the in-memory graph once -- cheap because KGStore exposes
        # the live MultiDiGraph as `_graph`. We touch only read-only API.
        graph = getattr(self._kg_store, "_graph", None)

        ranked: list[tuple[str, float]] = []
        for pid in candidate_pids:
            node_idx = node_index.get(pid)
            passage_term = (
                float(scores[node_idx]) if (node_idx is not None and effective_alpha > 0.0) else 0.0
            )

            phrase_term = 0.0
            if effective_alpha < 1.0 and graph is not None and pid in graph:
                # All phrases that "contain" this passage are predecessors
                # in the directed graph. Sum their PPR mass.
                for pred in graph.predecessors(pid):
                    pdata = graph.nodes[pred]
                    if pdata.get("node_type") != "phrase":
                        continue
                    pidx = node_index.get(pred)
                    if pidx is not None:
                        phrase_term += float(scores[pidx])

            score = effective_alpha * passage_term + (1.0 - effective_alpha) * phrase_term

            # ----------------------------------------------------------
            # Chunk-position prior (no-op default).
            #
            # TODO(section_depth): Chunk currently has `section`/`subsection`
            # fields but no numeric `section_depth` we can use here without a
            # wider schema change. When that lands, multiply `score` by
            # max(0, 1 - beta * chunk.section_depth). For now, beta is plumbed
            # through config so the knob exists but the formula collapses
            # to *= 1.0.
            # ----------------------------------------------------------
            position_weight = 1.0
            score = score * position_weight

            ranked.append((pid, score))

        # Sort descending by combined score.
        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[:top_k]

        if not ranked:
            return []

        chunk_ids = [pid for pid, _ in ranked]
        score_map = {pid: sc for pid, sc in ranked}

        # ------------------------------------------------------------------
        # 9. Hydrate from SQLite
        # ------------------------------------------------------------------
        chunks_by_id = self._hydrate(chunk_ids)

        # ------------------------------------------------------------------
        # 10. Build results (respect source_types filter; skip tombstoned)
        # ------------------------------------------------------------------
        results: list[RetrievalResult] = []
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                # Either tombstoned or not yet committed to chunks table.
                continue
            if source_types is not None and chunk.source_type not in source_types:
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
    # Seed-set construction
    # ------------------------------------------------------------------

    def _build_seed_set(
        self, surface_forms: list[str]
    ) -> tuple[list[str], list[float]]:
        """Return (node_ids, weights) for the broadened PPR seed set.

        For each surface form:
        - Always include any exact canonical match (lowercased).
        - When K > 1 AND the kg_store exposes an embedder, additionally
          take the top-(K-1) phrase nodes by cosine similarity over the
          per-node embeddings (set during canonicalization).

        Weights are cosine similarities; exact matches receive weight 1.0
        and synonym matches receive their cosine similarity in [0, 1].
        Duplicate node_ids keep the highest weight.

        Falls back to plain exact-match (the prior behavior) when no
        embedder is reachable -- this preserves test stubs that don't
        wire one through.
        """
        # Step 1: exact canonical matches via the existing API.
        exact_ids = self._kg_store.find_phrase_nodes(surface_forms)

        seeds: dict[str, float] = {nid: 1.0 for nid in exact_ids}

        # Short-circuit if K=1 -- no broadening requested.
        if self._seed_top_k <= 1:
            if not seeds:
                return [], []
            return list(seeds.keys()), [seeds[k] for k in seeds]

        # Step 2: broadening via embeddings. Try to access an embedder;
        # KGStore stores it as `_embedder`. We treat absence as "skip".
        embedder = getattr(self._kg_store, "_embedder", None)
        graph = getattr(self._kg_store, "_graph", None)
        if embedder is None or graph is None:
            if not seeds:
                return [], []
            return list(seeds.keys()), [seeds[k] for k in seeds]

        # Pre-collect all (node_id, embedding) pairs once.
        phrase_nodes: list[tuple[str, list[float]]] = []
        for nid, data in graph.nodes(data=True):
            if data.get("node_type") != "phrase":
                continue
            emb = data.get("embedding")
            if emb is None:
                continue
            phrase_nodes.append((nid, emb))

        if not phrase_nodes:
            if not seeds:
                return [], []
            return list(seeds.keys()), [seeds[k] for k in seeds]

        # For each surface form, embed once and take top-(K) by cosine.
        for sf in surface_forms:
            key = sf.strip()
            if not key:
                continue
            try:
                qvec = embedder.embed_one(key.lower())
            except Exception as exc:  # noqa: BLE001 -- embedders vary
                logger.debug(
                    "KGPPRRetriever: embed_one failed for %r: %s; "
                    "falling back to exact-match for this term.",
                    key, exc,
                )
                continue

            # Score every phrase node against this query embedding.
            sims: list[tuple[str, float]] = []
            for nid, emb in phrase_nodes:
                sim = _cosine(qvec, emb)
                if sim > 0:
                    sims.append((nid, sim))
            sims.sort(key=lambda x: x[1], reverse=True)

            # Keep top-K with strictly positive similarity. The exact-match
            # canonical entry already counts as one of the seeds; we add up
            # to K total per surface form.
            for nid, sim in sims[: self._seed_top_k]:
                prev = seeds.get(nid, 0.0)
                if sim > prev:
                    seeds[nid] = sim

        if not seeds:
            return [], []
        # Stable ordering for reproducibility in tests.
        ordered = sorted(seeds.items(), key=lambda kv: (-kv[1], kv[0]))
        return [nid for nid, _ in ordered], [w for _, w in ordered]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_passage_node_ids(self, user_id: str) -> set[str]:
        """Return all passage node_ids from the SQLite kg_nodes mirror."""
        cursor = self._db.execute(
            "SELECT node_id FROM kg_nodes WHERE user_id = ? AND node_type = 'passage'",
            (user_id,),
        )
        return {row["node_id"] for row in cursor.fetchall()}

    def _hydrate(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        """SELECT chunks matching *chunk_ids* from SQLite; skip excluded rows."""
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
            # Skip tombstoned chunks.
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
                # embedding_text is not stored in SQLite; use text as a fallback.
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
