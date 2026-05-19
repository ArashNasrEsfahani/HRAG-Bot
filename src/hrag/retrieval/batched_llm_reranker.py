"""LLM reranker that scores all candidates in ONE batched call."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

from hrag.providers.llm import LLMProvider
from hrag.types import RetrievalResult

ProgressCallback = Callable[[int, int, float], None]
"""Progress callback: (chunk_index_1based, total_chunks, score) -> None."""

# Regex to strip Markdown code-fences from LLM output.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.MULTILINE)

# Default prompt used when the prompt file is missing and no template was supplied.
_FALLBACK_TEMPLATE = """Score how relevant EACH passage is to answering the query.

Output ONLY a JSON array of integers — one integer per passage, same order.
No prose. No markdown fences. No keys. Just the array, e.g. [2, 0, 3].
Length MUST equal the number of passages. If unsure, return 0.

Scoring rubric:
0 = Irrelevant — unrelated topic, wrong domain, or pure noise
1 = Tangentially related — shares vocabulary but no useful information
2 = Related — useful context that partially supports an answer
3 = Directly answers — contains the specific fact the query asks for

Query: {query}

{numbered_passages}

Output:"""


def _load_prompt_template() -> str:
    """Load prompt from prompts/rerank_batched.md; fall back to inline default."""
    prompt_path = Path(__file__).parents[1] / "prompts" / "rerank_batched.md"
    if prompt_path.exists():
        try:
            return prompt_path.read_text(encoding="utf-8")
        except OSError:
            pass
    return _FALLBACK_TEMPLATE


def _parse_scores(response: str, expected_count: int) -> list[int] | None:
    """Parse a JSON array of ints from *response*.

    Returns a list of ints on success, or None if the response is malformed
    or the list length does not match *expected_count*.
    """
    cleaned = _FENCE_RE.sub("", response).strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, list):
        return None
    if len(parsed) != expected_count:
        return None

    scores: list[int] = []
    for item in parsed:
        if isinstance(item, int) and 0 <= item <= 3:
            scores.append(item)
        elif isinstance(item, float) and item == int(item) and 0 <= int(item) <= 3:
            scores.append(int(item))
        else:
            return None  # any malformed element → reject whole batch

    return scores


class BatchedLLMReranker:
    """LLM reranker that scores all candidates in ONE call (vs. N calls).

    Trade-off: one long-context LLM call instead of N short ones. Often
    faster on local models because forward-pass setup cost is amortized,
    and KV-cache warmup happens once. Quality depends on the LLM's ability
    to produce well-formed JSON arrays.

    Scoring: 0-3 integer per ACP-RAG rubric, parsed from a JSON array.
    On parse failure the entire batch is scored 0, keeping cost predictable.
    """

    name = "batched_llm"

    def __init__(
        self,
        llm: LLMProvider,
        prompt_template: str | None = None,
        max_passages_per_batch: int = 12,
    ) -> None:
        self._llm = llm
        self._template: str = (
            prompt_template if prompt_template is not None else _load_prompt_template()
        )
        self._max_batch = max(1, max_passages_per_batch)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        threshold: int = 2,
        top_k: int | None = None,
        progress: Optional[ProgressCallback] = None,
    ) -> list[RetrievalResult]:
        """Score, filter, and sort *results* using a single batched LLM call.

        Parameters
        ----------
        query:      The user's original query string.
        results:    Candidate retrieval results to be scored.
        threshold:  Minimum rerank_score (inclusive) to keep a result.
        top_k:      If given, truncate the final ranked list to this length.
        progress:   Optional callback emitted per chunk: (index, total, score).

        Returns
        -------
        A new list of RetrievalResult with .rerank_score populated,
        filtered and sorted.
        """
        if not results:
            return []

        total = len(results)
        chunk_cursor = 0  # global index across batches (0-based)

        # Split into batches and score each.
        for batch_start in range(0, total, self._max_batch):
            batch = results[batch_start : batch_start + self._max_batch]
            scores = self._score_batch(query, batch)

            for local_idx, (result, score) in enumerate(zip(batch, scores)):
                result.rerank_score = float(score)
                chunk_cursor += 1
                if progress is not None:
                    progress(chunk_cursor, total, float(score))

        # Filter, sort, truncate.
        passed = [r for r in results if (r.rerank_score or 0.0) >= threshold]
        passed.sort(key=lambda r: (r.rerank_score or 0.0, r.score), reverse=True)

        if top_k is not None:
            passed = passed[:top_k]

        return passed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_batch(self, query: str, batch: list[RetrievalResult]) -> list[int]:
        """Call the LLM once for *batch*; return a list of 0-3 scores.

        Falls back to all-zeros on any error or malformed response.
        """
        numbered_passages = "\n\n".join(
            f"[{i + 1}] {result.chunk.text}" for i, result in enumerate(batch)
        )
        prompt = self._template.format(
            query=query,
            numbered_passages=numbered_passages,
        )

        try:
            response = self._llm.complete(prompt)
        except Exception:
            return [0] * len(batch)

        scores = _parse_scores(response, expected_count=len(batch))
        if scores is None:
            return [0] * len(batch)

        return scores
