"""Document-scoped retrieval wrapper.

Two regimes:
  A. Explicit-title — query names a paper -> HARD-filter to those docs.
  B. Two-stage — no title match -> coarse doc-ranker picks top-K docs,
     then wrapped retriever filters to those.

Wraps any Retriever (router, hybrid, etc.). Falls through unchanged when
A and B both yield zero docs (graceful degradation).

Implementation note on filtering: existing retrievers (vector, kg_ppr, etc.)
do not accept a ``doc_id_filter`` parameter. We therefore use POST-HOC
filtering: oversample the wrapped retriever (top_k * oversample_factor),
drop any result whose ``chunk.doc_id`` isn't in the allowed set, and
truncate to top_k. This keeps the wrapper retriever-agnostic at the cost
of some wasted retrieval work; oversample_factor=3 is the default.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

from hrag.db.connection import Database
from hrag.providers.embeddings import EmbeddingProvider
from hrag.retrieval.base import Retriever
from hrag.types import RetrievalResult

if TYPE_CHECKING:
    from hrag.intent import Intent

logger = logging.getLogger(__name__)


# Hard-coded alias map for the 3 papers we benchmark on.
# Extending this is a one-line edit. Keys are matched (case-insensitive,
# substring) against the SQLite ``documents.title`` column.
PAPER_ALIASES: dict[str, list[str]] = {
    # HippoRAG (the original 2024 NeurIPS paper). Aliases must NOT include
    # "hipporag 2" — that's a different paper (see below).
    "HIPPORAG": ["hipporag", "hippo-rag", "hippo rag"],
    # HippoRAG 2 (2025 follow-up). Title in SQLite is the arXiv id.
    "2502.14802v2": ["hipporag 2", "hipporag-2", "hipporag2", "hippo-rag 2"],
    # RAGate.
    "2025.findings-naacl.30": ["ragate", "ra-gate", "ra gate"],
}


# Aliases that subsume another base alias must be checked first. When
# "hipporag 2" matches, we must NOT also match the bare "hipporag" key.
# Each tuple is (specific_alias, base_alias_to_suppress).
_ALIAS_PRECEDENCE: list[tuple[str, str]] = [
    ("hipporag 2", "hipporag"),
    ("hipporag-2", "hipporag"),
    ("hipporag2", "hipporag"),
    ("hippo-rag 2", "hippo-rag"),
]


def _alias_regex(alias: str) -> re.Pattern[str]:
    """Compile a word-boundary regex for an alias, escaping special chars.

    We use ``\\b`` so ``hipporag`` doesn't match inside ``hipporag2``.
    """
    return re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)


class DocScopedRetriever(Retriever):
    """Wraps another Retriever, restricting results to a relevant doc subset.

    Regime A (explicit-title): if the query mentions a known paper alias,
    HARD-filter retrieval to documents whose ``title`` contains the
    matching alias-key (case-insensitive substring against SQLite).

    Regime B (two-stage): if no alias matches, run a cheap coarse
    doc-ranker to pick the top-K relevant documents, then post-hoc
    filter the wrapped retriever's oversampled results to those.

    If both regimes yield no docs (e.g. empty corpus), the wrapped
    retriever is invoked unmodified — callers see no behavioural change.
    """

    name = "doc_scoped"

    def __init__(
        self,
        wrapped: Retriever,
        db: Database,
        embedder: EmbeddingProvider,
        coarse_top_k: int = 3,
        oversample_factor: int = 3,
        aliases: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self._wrapped = wrapped
        self._db = db
        self._embedder = embedder
        self._coarse_top_k = max(1, coarse_top_k)
        self._oversample_factor = max(1, oversample_factor)
        self._aliases = aliases if aliases is not None else PAPER_ALIASES

    # ------------------------------------------------------------------
    # Public Retriever interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 20,
        source_types: Optional[list[str]] = None,
        intent_hint: Optional["Intent"] = None,
        where: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        # Regime A: explicit title match against the alias map.
        allowed_doc_ids = self._title_match(query, user_id)
        regime = "title" if allowed_doc_ids else None

        # Regime B: coarse doc ranker fallback.
        if not allowed_doc_ids:
            allowed_doc_ids = self._coarse_doc_rank(
                query, user_id, k=self._coarse_top_k
            )
            regime = "coarse" if allowed_doc_ids else None

        # Graceful fallthrough: nothing matched (e.g. empty corpus).
        if not allowed_doc_ids:
            logger.debug(
                "doc_scope: no docs matched (regime A+B empty); "
                "falling through to wrapped retriever unchanged."
            )
            return self._wrapped.retrieve(
                query, user_id, top_k, source_types,
                intent_hint=intent_hint, where=where,
            )

        logger.debug(
            "doc_scope: regime=%s allowed_docs=%d", regime, len(allowed_doc_ids)
        )

        # Oversample the wrapped retriever and post-hoc filter to allowed docs.
        oversampled_k = top_k * self._oversample_factor
        oversampled = self._wrapped.retrieve(
            query, user_id, oversampled_k, source_types,
            intent_hint=intent_hint, where=where,
        )
        filtered = [r for r in oversampled if r.chunk.doc_id in allowed_doc_ids]
        return filtered[:top_k]

    # ------------------------------------------------------------------
    # Regime A: explicit-title match
    # ------------------------------------------------------------------

    def _title_match(self, query: str, user_id: str) -> set[str]:
        """Match query against alias map; return SQLite doc_ids whose title
        contains the matched alias-key (case-insensitive substring).

        Precedence: more-specific aliases (e.g. "hipporag 2") win over
        their bases (e.g. "hipporag"), so "What about HippoRAG 2?" does
        NOT also match the base HippoRAG paper.
        """
        if not query or not query.strip():
            return set()

        q = query.lower()

        # 1. Find every alias that hits the query at a word boundary.
        matched_aliases: set[str] = set()
        for _key, alias_list in self._aliases.items():
            for alias in alias_list:
                if _alias_regex(alias).search(q):
                    matched_aliases.add(alias.lower())

        if not matched_aliases:
            return set()

        # 2. Apply precedence: drop base aliases when a specific alias
        #    that subsumes them also matched.
        for specific, base in _ALIAS_PRECEDENCE:
            if specific in matched_aliases and base in matched_aliases:
                matched_aliases.discard(base)

        # 3. Map the surviving aliases back to their alias-keys (the
        #    title-substring used to find docs in SQLite).
        title_substrings: set[str] = set()
        for key, alias_list in self._aliases.items():
            lowered = {a.lower() for a in alias_list}
            if lowered & matched_aliases:
                title_substrings.add(key.lower())

        if not title_substrings:
            return set()

        # 4. Look up matching doc_ids in the documents table.
        return self._docs_with_titles(title_substrings, user_id)

    def _docs_with_titles(
        self, title_substrings: set[str], user_id: str
    ) -> set[str]:
        """Return doc_ids whose title contains ANY of the lowercase
        substrings (case-insensitive LIKE)."""
        if not title_substrings:
            return set()

        # OR-combine LIKE clauses; LIKE is case-insensitive in SQLite for ASCII.
        clauses = " OR ".join(["LOWER(title) LIKE ?"] * len(title_substrings))
        params: list[str] = [user_id]
        params.extend(f"%{s}%" for s in title_substrings)
        sql = (
            "SELECT doc_id FROM documents "
            f"WHERE user_id = ? AND ({clauses})"
        )
        try:
            cursor = self._db.execute(sql, params)
            rows = cursor.fetchall()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("doc_scope: title lookup failed: %s", exc)
            return set()
        return {row["doc_id"] for row in rows}

    # ------------------------------------------------------------------
    # Regime B: coarse doc-ranker (vector aggregation by doc_id)
    # ------------------------------------------------------------------

    def _coarse_doc_rank(
        self, query: str, user_id: str, k: int
    ) -> set[str]:
        """Rank docs by aggregating chunk-level vector retrieval scores.

        Pulls top-50 chunks via the wrapped retriever (which for the
        production config is a router whose vector path is always
        active), groups them by ``doc_id`` summing scores, and returns
        the top-k doc_ids.

        Returns empty set when nothing comes back (empty corpus, etc.) —
        the caller falls through to the unfiltered wrapped retriever.
        """
        if not query or not query.strip() or k <= 0:
            return set()

        try:
            results = self._wrapped.retrieve(query, user_id, top_k=50)
        except Exception as exc:
            logger.warning(
                "doc_scope: coarse rank wrapped.retrieve raised: %s", exc
            )
            return set()

        if not results:
            return set()

        score_by_doc: dict[str, float] = {}
        for r in results:
            doc_id = r.chunk.doc_id
            if not doc_id:
                continue
            # Sum scores per doc; missing scores treated as 0.
            score_by_doc[doc_id] = score_by_doc.get(doc_id, 0.0) + (r.score or 0.0)

        if not score_by_doc:
            return set()

        ranked = sorted(score_by_doc.items(), key=lambda kv: kv[1], reverse=True)
        return {doc_id for doc_id, _score in ranked[:k]}
