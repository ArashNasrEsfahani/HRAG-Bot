"""Document-title topic detection — a pre-classifier that overrides the
intent classifier when the user's query mentions a token that exists in their
document corpus.

Why this exists
---------------
The regex / LLM intent classifier is good but not infallible. Edge cases
have repeatedly slipped through ("what do you know about hipporag" once
matched PERSONAL because of the "what do you know about" template; "so
what is hipporag" once slipped past the FACTUAL-opener regex because of
the discourse prefix). Each failure routed a substantive corpus question
into the wrong path.

When the query contains a token that appears in a document title, the
answer is unambiguous: the user is asking about a known topic in their
own library. Force FACTUAL — bypass the classifier — emit a verdict with
``source="named_topic"`` so the UI surfaces *why*.

Algorithm
---------
At construction, for every non-episodic document, tokenize the title:
  1. Split on non-alphanumeric.
  2. For each chunk, further split letter↔digit boundaries (so "Priority1"
     becomes ``["Priority", "1"]``).
  3. Lowercase.
  4. Drop tokens that are pure-numeric, shorter than ``min_token_len`` (3),
     or in ``_STOPWORDS`` (function words, pronouns, file extensions, …).

At query time, scan the query (also lowercased) for any topic term, using a
``\\b…\\b`` word-boundary regex so "rag" doesn't match inside "fragmentation".
If any term hits, return the set of matched terms.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hrag.db.connection import Database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token extraction regex
# ---------------------------------------------------------------------------

_RE_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
_RE_LETTER_OR_DIGIT_RUN = re.compile(r"[A-Za-z]+|[0-9]+")


# ---------------------------------------------------------------------------
# Stopwords — never used as topic terms even when they appear in a title
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        # Function words / determiners
        "the", "and", "or", "of", "in", "on", "at", "to", "for", "by", "with",
        "from", "into", "onto", "upon", "than", "but", "not", "no", "yes",
        "a", "an", "this", "that", "these", "those", "some", "any", "all",
        "every", "each", "both", "few", "more", "most", "other", "such",
        # Pronouns — CRITICAL: never include "me" / "my" / "i" as topic terms
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself",
        "we", "us", "our", "ours", "ourselves",
        "he", "him", "his", "she", "her", "hers", "herself",
        "they", "them", "their", "theirs", "themselves",
        "it", "its", "itself",
        # Auxiliaries / common verbs
        "is", "are", "was", "were", "be", "been", "being", "am",
        "has", "have", "had", "having",
        "do", "does", "did", "doing", "done",
        "will", "would", "should", "could", "can", "may", "might", "must",
        "shall", "ought",
        # Question words and discourse markers
        "what", "who", "when", "where", "why", "how", "which",
        "so", "well", "now", "then", "also", "very", "just", "only", "really",
        "ok", "okay", "yeah", "sure", "hey", "hi", "hello",
        # File extensions / format suffixes that pollute titles
        "pdf", "md", "txt", "doc", "docx", "html", "htm", "xml", "ppt", "pptx",
        # Document boilerplate
        "paper", "draft", "version", "final", "rev", "appendix",
        # Numerals as words (just in case)
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "zero",
    }
)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class KnownTopicDetector:
    """Pre-classifier that flags queries which mention a known document topic.

    The detector loads document titles lazily on first ``detect()`` call per
    user and re-reads them every time — SQLite reads are microseconds and
    cheap-on-every-turn is preferable to a stale cache.
    """

    name: str = "named_topic"

    def __init__(self, db: "Database", *, min_token_len: int = 3) -> None:
        self._db = db
        self._min_token_len = int(min_token_len)
        # Per-user topic-term cache. Invalidated by ``refresh()``; the
        # orchestrator calls refresh on ingest hooks where it has the user
        # id; ad-hoc calls re-read directly when the user_id is unknown.
        self._terms_by_user: dict[str, frozenset[str]] = {}

    # ------------------------------------------------------------------
    # Token extraction
    # ------------------------------------------------------------------

    def _extract_tokens(self, title: str) -> set[str]:
        """Return the set of topic tokens contributed by one document title."""
        tokens: set[str] = set()
        for chunk in _RE_NON_ALNUM.split(title):
            for sub in _RE_LETTER_OR_DIGIT_RUN.findall(chunk):
                low = sub.lower()
                if low.isdigit():
                    continue
                if len(low) < self._min_token_len:
                    continue
                if low in _STOPWORDS:
                    continue
                tokens.add(low)
        return tokens

    # ------------------------------------------------------------------
    # Load + refresh
    # ------------------------------------------------------------------

    def refresh(self, user_id: str) -> frozenset[str]:
        """Re-read document titles for *user_id* and rebuild the topic set."""
        try:
            rows = self._db.execute(
                "SELECT title FROM documents "
                "WHERE user_id = ? AND source_type != 'episodic'",
                (user_id,),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("KnownTopicDetector.refresh failed (%s)", exc)
            self._terms_by_user[user_id] = frozenset()
            return frozenset()

        terms: set[str] = set()
        for row in rows:
            title = (row["title"] or "").strip() if hasattr(row, "keys") else (row[0] or "").strip()
            if title:
                terms.update(self._extract_tokens(title))
        result = frozenset(terms)
        self._terms_by_user[user_id] = result
        logger.info(
            "KnownTopicDetector: user_id=%s topic_terms=%d (sample=%s)",
            user_id, len(result),
            ", ".join(sorted(result)[:8]) + ("…" if len(result) > 8 else ""),
        )
        return result

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, query: str, user_id: str) -> set[str]:
        """Return the set of topic terms that appear in *query* as whole words.

        Empty set ⇒ no known topic; the caller should fall back to the
        regular intent classifier.
        """
        if not query or not query.strip():
            return set()

        terms = self._terms_by_user.get(user_id)
        if terms is None:
            terms = self.refresh(user_id)
        if not terms:
            return set()

        query_lower = query.lower()
        matched: set[str] = set()
        for term in terms:
            # Word-boundary match prevents "rag" → "fragmentation".
            if re.search(rf"\b{re.escape(term)}\b", query_lower):
                matched.add(term)
        return matched
