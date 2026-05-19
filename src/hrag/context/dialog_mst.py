"""Turn-level conversation history compactor (Phase 4).

The orchestrator may carry a long chat history into every turn's prompt.
Beyond a small horizon that hurts both latency and answer quality — older
turns dilute the prompt and crowd out retrieved passages. ``DialogMSTCompactor``
collapses the older portion of the history into a single synthetic system
message holding LLM-generated summaries of clustered turns, while keeping
the most recent ``keep_recent_turns`` verbatim.

The algorithm is conceptually a sibling of ``hrag.retrieval.mst`` (the
chunk-level KG2RAG organizer) but the graph semantics are different:

  * Nodes are conversation turns, not chunks.
  * Edges are embedding-cosine similarity, not entity overlap.
  * The output is a *new* list of ``Message`` (one summary + the recent
    tail), not a re-ordered list of the input.

Algorithm:
    1. Fast path: ``len(history) <= compact_after_turns`` -> return as-is.
    2. Split into ``old = history[:-keep_recent]`` and ``recent``.
    3. Embed each old turn's content (L2-normalised vectors from the
       project's ``EmbeddingProvider``; cosine == dot on the hot path).
    4. Greedy single-pass clustering: each turn either joins the existing
       cluster whose centroid maximises cosine similarity (if above
       ``cluster_threshold``), or seeds a new cluster.
    5. Order clusters by the index of their oldest member so the synthetic
       summary preserves temporal order.
    6. For each cluster, render ``prompts/dialog_summary.md`` and call the
       LLM once. Concatenate the summaries into a single ``system`` message
       prefixed with ``[Earlier conversation]``.
    7. Return ``[synthetic_message, *recent]``.

Failure mode: if any LLM call raises, the compactor returns the input
history unchanged so the chat path keeps working. The summary is a
nice-to-have, never load-bearing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hrag.types import Message

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hrag.providers.embeddings import EmbeddingProvider
    from hrag.providers.llm import LLMProvider


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "dialog_summary.md"


class DialogMSTCompactor:
    """Compact a long conversation history into a synthetic summary turn
    plus the most-recent N verbatim turns.

    See module docstring for the full algorithm. Designed to be safe to
    plug in unconditionally: short histories pass through untouched and
    LLM failures fall back to the unchanged input.
    """

    name = "dialog_mst"

    def __init__(
        self,
        llm: "LLMProvider",
        embeddings: "EmbeddingProvider",
        *,
        compact_after_turns: int = 12,
        keep_recent_turns: int = 6,
        summary_target_tokens: int = 400,
        cluster_threshold: float = 0.55,
    ) -> None:
        """Initialise the compactor.

        Args:
            llm: LLM provider used to summarise each cluster.
            embeddings: Embedding provider; must return L2-normalised
                vectors so cosine similarity == dot product.
            compact_after_turns: Compaction only fires when the history
                exceeds this length. Below it, ``compact`` is a pure
                pass-through.
            keep_recent_turns: Number of trailing turns to keep verbatim.
                Must be < ``compact_after_turns`` for compaction to do
                anything useful.
            summary_target_tokens: Target length per cluster summary, in
                tokens. The LLM call uses ``max_tokens = int(target * 1.5)``
                to leave a little headroom over the target.
            cluster_threshold: Greedy clustering threshold in cosine
                similarity. Higher -> more, smaller clusters. Lower ->
                fewer, broader clusters.
        """
        self._llm = llm
        self._embeddings = embeddings
        self._compact_after_turns = int(compact_after_turns)
        self._keep_recent_turns = int(keep_recent_turns)
        self._summary_target_tokens = int(summary_target_tokens)
        self._cluster_threshold = float(cluster_threshold)
        # Cached prompt template — read once.
        self._prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compact(self, history: list[Message]) -> list[Message]:
        """Return a (possibly) shorter history with older turns summarised.

        Fast-path returns the input unchanged when:
          * ``len(history) <= compact_after_turns``
          * the old slice is empty (i.e. ``keep_recent_turns >= len(history)``)

        On any LLM exception during summarisation the input is returned
        unchanged.
        """
        if not history or len(history) <= self._compact_after_turns:
            return history

        # Split. keep_recent_turns may be 0 in principle; we still guard
        # against an empty `old` slice below for safety.
        keep_n = max(0, self._keep_recent_turns)
        if keep_n >= len(history):
            return history

        if keep_n == 0:
            old = list(history)
            recent: list[Message] = []
        else:
            old = list(history[:-keep_n])
            recent = list(history[-keep_n:])

        if not old:
            return history

        # Embed old turns. The provider returns L2-normalised vectors.
        try:
            raw_vectors = self._embeddings.embed([t.content for t in old])
        except Exception:
            return history

        # Normalise to a list of list[float] for centroid math. Accept
        # either ndarray or list-of-lists from the provider.
        vectors: list[list[float]] = [list(v) for v in raw_vectors]
        if len(vectors) != len(old):
            return history

        clusters = self._cluster(vectors)

        # Render and summarise each cluster.
        try:
            summaries = [self._summarise_cluster(old, c) for c in clusters]
        except Exception:
            # LLM failure mid-summary -> fall back to unchanged history.
            return history

        joined = "\n\n".join(s.strip() for s in summaries if s and s.strip())
        synthetic = Message(
            role="system",
            content="[Earlier conversation]\n" + joined,
        )
        return [synthetic, *recent]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cluster(self, vectors: list[list[float]]) -> list[list[int]]:
        """Greedy single-pass agglomerative clustering by cosine similarity.

        Vectors are assumed L2-normalised so dot == cosine.

        Returns a list of clusters, each a list of indices into ``vectors``.
        Clusters are returned in the order of their oldest (lowest-index)
        member, preserving the temporal order of the conversation.
        """
        import numpy as np  # noqa: PLC0415 - kept lazy per project convention

        arr = np.asarray(vectors, dtype=float)

        clusters: list[list[int]] = []
        centroids: list[np.ndarray] = []

        for i in range(arr.shape[0]):
            vec_i = arr[i]
            best_cluster = -1
            best_sim = self._cluster_threshold
            for c_idx, centroid in enumerate(centroids):
                sim = float(np.dot(vec_i, centroid))
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = c_idx
            if best_cluster < 0:
                clusters.append([i])
                centroids.append(vec_i.copy())
            else:
                members = clusters[best_cluster]
                # Update centroid: running mean of member vectors. Keep it
                # normalised so subsequent comparisons stay on the cosine
                # scale even after many merges.
                n = len(members)
                new_centroid = (centroids[best_cluster] * n + vec_i) / (n + 1)
                norm = float(np.linalg.norm(new_centroid))
                if norm > 0:
                    new_centroid = new_centroid / norm
                centroids[best_cluster] = new_centroid
                members.append(i)

        # Sort clusters by the index of their oldest member so the
        # synthetic summary reads in temporal order.
        clusters.sort(key=lambda c: c[0])
        return clusters

    def _summarise_cluster(
        self,
        old: list[Message],
        member_indices: list[int],
    ) -> str:
        """Render the dialog-summary prompt for one cluster and call the LLM.

        Member indices are kept in their original (temporal) order inside
        the prompt — sorting here makes the rendered turns easier for the
        LLM to follow.
        """
        ordered = sorted(member_indices)
        lines: list[str] = []
        for idx in ordered:
            turn = old[idx]
            role = (turn.role or "user").capitalize()
            content = (turn.content or "").strip()
            lines.append(f"{role}: {content}")
        turns_text = "\n".join(lines)

        prompt = self._prompt_template.format(
            turns=turns_text,
            summary_target_tokens=self._summary_target_tokens,
        )

        max_tokens = int(self._summary_target_tokens * 1.5)
        text = self._llm.complete(
            prompt,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return (text or "").strip()
