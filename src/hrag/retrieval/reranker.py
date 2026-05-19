"""ACP-RAG style LLM relevance reranker (0-3 integer scoring)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from hrag.providers.llm import LLMProvider
from hrag.types import RetrievalResult

ProgressCallback = Callable[[int, int, int], None]
"""Progress callback signature: (index, total, score) -> None."""

# Default prompt used when the file is missing and no template was supplied.
_FALLBACK_TEMPLATE = """Score how relevant the passage is to answering the query.

Output ONLY a single integer: 0, 1, 2, or 3. No explanation. No punctuation.

Scoring rubric:
0 = Irrelevant — unrelated topic, wrong domain, or pure noise
1 = Tangentially related — shares vocabulary but contains no useful information
2 = Related — provides useful context that partially supports an answer
3 = Directly answers — contains the specific fact or data the query is asking for

Query: {query}

Passage: {passage}

Score:"""

# Regex that finds the first digit 0-3 in a response string.
_SCORE_RE = re.compile(r"[0-3]")


def _load_prompt_template() -> str:
    """Load prompt from the canonical prompts/rerank.md; fall back to inline default."""
    prompt_path = Path(__file__).parents[1] / "prompts" / "rerank.md"
    if prompt_path.exists():
        try:
            return prompt_path.read_text(encoding="utf-8")
        except OSError:
            pass
    return _FALLBACK_TEMPLATE


class LLMReranker:
    """Scores each RetrievalResult with an LLM (0-3) and filters/sorts the list.

    Scoring protocol (ACP-RAG)
    --------------------------
    For every result the LLM is asked to return an integer 0-3:
        0 = irrelevant
        1 = barely related
        2 = relevant
        3 = directly answers

    Results below *threshold* are dropped; survivors are sorted by
    (rerank_score DESC, original score DESC) then optionally truncated to top_k.

    Sequential execution is intentional: Phase 1 keeps this simple and avoids
    threading complexity with shared LLM clients.
    """

    def __init__(
        self,
        llm: LLMProvider,
        prompt_template: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self._template: str = (
            prompt_template if prompt_template is not None else _load_prompt_template()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        threshold: int = 2,
        top_k: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> list[RetrievalResult]:
        """Score, filter, and sort *results* using the LLM scorer.

        Parameters
        ----------
        query:      The user's original query string.
        results:    Candidate retrieval results to be scored.
        threshold:  Minimum rerank_score (inclusive) to keep a result.
        top_k:      If given, truncate the final ranked list to this length.

        Returns
        -------
        A new list of RetrievalResult with .rerank_score populated,
        filtered and sorted.
        """
        if not results:
            return []

        scored: list[RetrievalResult] = []
        total = len(results)
        for idx, result in enumerate(results, start=1):
            score = self._score_one(query=query, passage=result.chunk.text)
            result.rerank_score = score
            if progress is not None:
                progress(idx, total, score)
            if score >= threshold:
                scored.append(result)

        # Sort: primary = rerank_score desc, secondary = vector score desc.
        scored.sort(key=lambda r: (r.rerank_score, r.score), reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_one(self, query: str, passage: str) -> int:
        """Ask the LLM for a 0-3 relevance score; default to 0 on any failure."""
        prompt = self._template.format(query=query, passage=passage)
        try:
            response = self._llm.complete(prompt)
        except Exception:
            return 0

        return _parse_score(response)


def _parse_score(text: str) -> int:
    """Extract the first 0-3 integer from *text*; return 0 if none found."""
    match = _SCORE_RE.search(text.strip())
    if match is None:
        return 0
    return int(match.group())
