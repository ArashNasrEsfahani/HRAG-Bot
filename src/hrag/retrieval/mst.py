"""KG2RAG-style minimum-spanning-tree context organizer (Phase 2).

KG2RAG (NAACL 2025) takes a set of retrieved chunks, builds an induced
subgraph from their KG neighbourhood, computes a minimum spanning tree to
drop redundant chunks, and orders the survivors so semantically adjacent
passages end up adjacent in the prompt.

This module owns the post-retrieval re-organisation step. It is OPTIONAL:
when the KG is empty, or when no retrieved chunks have entries in the KG,
``organize`` returns its input unchanged. That makes the organizer safe to
plug in unconditionally — it self-disables when there is nothing to do.

The organizer reads ``KGStore._graph`` directly for read-only traversal.
The alternative (asking ``KGStore`` to expose more API) is deliberately
avoided at this point in the project; access here is read-only and will
not mutate graph state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hrag.types import RetrievalResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hrag.kg.store import KGStore


# Deprecated module-level constant kept for backward compatibility only.
# The live value is now ``MSTOrganizer._min_edge_jaccard`` (default 0.02).
_MIN_JACCARD_FOR_EDGE = 0.02


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets. Empty/empty -> 0.0."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _result_score(result: RetrievalResult) -> float:
    """Pick rerank_score if present, else the raw retrieval score."""
    if result.rerank_score is not None:
        return float(result.rerank_score)
    return float(result.score)


class MSTOrganizer:
    """KG2RAG-style context organizer.

    Takes a list of ``RetrievalResult``, computes an entity-overlap MST over
    the chunks, drops redundancies, and returns an ordered list of survivors.

    The organizer is OPTIONAL. It only fires when the KG is populated. With
    no KG (or with a result set whose chunks aren't in the KG), it returns
    the input unchanged (no-op).
    """

    name = "mst_organizer"

    def __init__(
        self,
        kg_store: "KGStore",
        max_chunks: int = 12,
        redundancy_threshold: float = 0.4,
        min_edge_jaccard: float = 0.02,
    ) -> None:
        """Initialise the organizer.

        Args:
            kg_store: The knowledge-graph store used to look up phrase nodes.
            max_chunks: Hard cap on the number of survivors returned.
            redundancy_threshold: Jaccard threshold above which the lower-scored
                member of a high-overlap MST edge is dropped.
                Empirical: 0.4 drops redundant near-duplicates without losing
                meaningful neighbours.  Higher (0.7) is more conservative;
                lower (0.2) is aggressive.
            min_edge_jaccard: Minimum Jaccard overlap required to add an edge
                between two chunks in the chunk graph.  Chunks below this
                floor are placed in separate singleton components.  Default
                0.02 is permissive enough to connect chunks sharing even one
                entity out of ~50.  Raise to 0.05+ to tighten connectivity.
        """
        self._kg_store = kg_store
        self._max_chunks = int(max_chunks)
        self._redundancy_threshold = float(redundancy_threshold)
        self._min_edge_jaccard = float(min_edge_jaccard)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def organize(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Apply KG2RAG MST organization to a list of retrieval results.

        See module docstring for the algorithm; in brief:
            1. No-op fast paths (empty input / empty KG / no chunks in KG).
            2. Build an undirected weighted graph over chunk_ids; edge weight
               is ``-overlap`` so MST surfaces high-overlap chains.
            3. Compute the MST per connected component.
            4. Prune redundancies: for each MST edge whose Jaccard >=
               ``redundancy_threshold``, mark the lower-scored side for
               removal.
            5. Order survivors by BFS from each component's highest-scored
               chunk.
            6. Cap to ``max_chunks``.
        """
        if not results:
            return results

        # Fast path: empty KG -> nothing we can organise around.
        if self._kg_store.num_phrase_nodes() == 0:
            return results

        # Map every input chunk to its phrase set in the KG. Chunks not in
        # the graph map to an empty set; they'll fall out as singletons.
        chunk_phrases: dict[str, set[str]] = {
            r.chunk.chunk_id: self._chunk_to_phrases(r.chunk.chunk_id)
            for r in results
        }
        if all(not phrases for phrases in chunk_phrases.values()):
            # KG exists but none of these chunks are represented in it —
            # nothing to organise. Return as-is to avoid surprising callers.
            return results

        import networkx as nx  # noqa: PLC0415

        results_by_id: dict[str, RetrievalResult] = {
            r.chunk.chunk_id: r for r in results
        }
        chunk_ids: list[str] = [r.chunk.chunk_id for r in results]

        # Build an undirected weighted graph over chunks.
        graph: "nx.Graph" = nx.Graph()
        for cid in chunk_ids:
            graph.add_node(cid)

        for i, a in enumerate(chunk_ids):
            phrases_a = chunk_phrases[a]
            for b in chunk_ids[i + 1 :]:
                phrases_b = chunk_phrases[b]
                if not phrases_a or not phrases_b:
                    continue
                jaccard = _jaccard(phrases_a, phrases_b)
                if jaccard < self._min_edge_jaccard:
                    continue
                overlap = len(phrases_a & phrases_b)
                # Negative weight so MST favours high-overlap edges first.
                graph.add_edge(
                    a,
                    b,
                    weight=-float(overlap),
                    jaccard=jaccard,
                )

        # MST per connected component.
        components: list[set[str]] = [set(c) for c in nx.connected_components(graph)]

        # Build the global MST; for singleton components MST is trivially
        # the lone node.
        mst: "nx.Graph" = nx.minimum_spanning_tree(graph)

        # Redundancy pruning: walk MST edges, mark the loser of each
        # high-overlap edge.
        to_drop: set[str] = set()
        for u, v, edge_data in mst.edges(data=True):
            jaccard = float(edge_data.get("jaccard", 0.0))
            if jaccard < self._redundancy_threshold:
                continue
            if u in to_drop or v in to_drop:
                # If one side is already gone, leave the other alone.
                continue
            score_u = _result_score(results_by_id[u])
            score_v = _result_score(results_by_id[v])
            loser = u if score_u < score_v else v
            to_drop.add(loser)

        # Order survivors: BFS from highest-scored survivor per component,
        # over the survivor sub-MST. Components ordered by (size desc,
        # max_score desc).
        ordered: list[RetrievalResult] = []
        component_specs: list[tuple[int, float, list[str]]] = []
        for component in components:
            survivors_in_comp = [c for c in component if c not in to_drop]
            if not survivors_in_comp:
                continue
            walk = self._bfs_order(mst, survivors_in_comp, results_by_id, to_drop)
            size = len(walk)
            max_score = max(_result_score(results_by_id[c]) for c in walk)
            component_specs.append((size, max_score, walk))

        # Sort: bigger components first, ties broken by max score (desc).
        component_specs.sort(key=lambda spec: (-spec[0], -spec[1]))
        for _size, _max_score, walk in component_specs:
            for cid in walk:
                ordered.append(results_by_id[cid])

        # Cap.
        return ordered[: self._max_chunks]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _chunk_to_phrases(self, chunk_id: str) -> set[str]:
        """Look up the phrase nodes that ``contains``-edge into this chunk's
        passage node. Returns an empty set if the chunk is not in the graph
        (predates the KG, or KG is empty for that chunk)."""
        graph = self._kg_store._graph
        if chunk_id not in graph:
            return set()
        phrases: set[str] = set()
        for src, _dst, attrs in graph.in_edges(chunk_id, data=True):
            if attrs.get("relation") == "contains":
                phrases.add(src)
        return phrases

    def _bfs_order(
        self,
        mst,
        survivors: list[str],
        results_by_id: dict[str, RetrievalResult],
        dropped: set[str],
    ) -> list[str]:
        """BFS over the survivor sub-MST starting from the highest-scored
        survivor. Falls back to score-desc order for fully-disconnected
        survivors (e.g. singletons or pruned-into-pieces components)."""
        if not survivors:
            return []

        # Sort survivors by score desc — this also defines the fallback order
        # for any nodes the BFS can't reach.
        ranked = sorted(
            survivors,
            key=lambda c: -_result_score(results_by_id[c]),
        )

        # The survivor sub-MST: original MST minus dropped nodes.
        # We BFS over neighbours that are also survivors; that's equivalent
        # to walking ``mst.subgraph(survivors)``.
        survivor_set = set(survivors)
        visited: set[str] = set()
        order: list[str] = []

        for seed in ranked:
            if seed in visited:
                continue
            # BFS from seed.
            queue: list[str] = [seed]
            visited.add(seed)
            while queue:
                # Pop from the front; keep neighbours in score-desc order so
                # the BFS prefers higher-scoring branches when fanning out.
                node = queue.pop(0)
                order.append(node)
                neighbours = [
                    n
                    for n in mst.neighbors(node)
                    if n in survivor_set and n not in visited and n not in dropped
                ]
                neighbours.sort(key=lambda c: -_result_score(results_by_id[c]))
                for nbr in neighbours:
                    visited.add(nbr)
                    queue.append(nbr)
        return order
