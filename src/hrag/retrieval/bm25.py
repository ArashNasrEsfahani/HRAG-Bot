"""BM25Retriever: sparse keyword retriever backed by Okapi BM25.

Index lifecycle
---------------
The BM25 index is built in-memory from SQLite on instantiation (or on
explicit `.refresh()` calls). Filtering by user_id and source_types happens
at query time so the same index can serve multiple users without rebuilding.
For corpora under ~100k chunks the build takes well under one second.

After new chunks are ingested, callers should either construct a fresh
BM25Retriever or call `.refresh()` to pick up the new documents.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Optional

from rank_bm25 import BM25Okapi

from hrag.db.connection import Database
from hrag.retrieval.base import Retriever
from hrag.types import Chunk, RetrievalResult

if TYPE_CHECKING:
    from hrag.intent import Intent


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase word-boundary split; drop single-character tokens.

    Uses ``re.findall(r'[a-zA-Z0-9]+', text.lower())`` then filters tokens
    whose length is 1.  No explicit stop-word list — BM25 IDF handles high-
    frequency words naturally.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1]


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever(Retriever):
    """Sparse keyword retriever using Okapi BM25.

    Index lifecycle: BM25 needs an in-memory index of all chunks. We rebuild
    it from SQLite on instantiation. For typical corpora (<100k chunks) this
    takes <1 second. After ingestion of new chunks, callers should construct
    a new BM25Retriever (or call `.refresh()`).

    Tokenizer: lowercase split on word boundaries; remove single-char tokens
    and punctuation. No stop-word removal (BM25 IDF handles common words).
    """

    name = "bm25"

    def __init__(self, db: Database, user_id_filter: Optional[str] = None) -> None:
        self._db = db
        self._user_id_filter = user_id_filter

        # Populated by refresh()
        self._chunk_ids: list[str] = []
        self._chunks: dict[str, Chunk] = {}
        self._user_index: dict[str, list[int]] = {}
        self._source_type_index: dict[str, list[int]] = {}
        self._bm25: Optional[BM25Okapi] = None

        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the BM25 index from SQLite.

        Loads ALL non-excluded chunks (for all users unless *user_id_filter*
        was supplied at construction time). Filtering by user_id and
        source_types happens at query time.
        """
        if self._user_id_filter is not None:
            sql = """
                SELECT
                    chunk_id, doc_id, user_id,
                    text, title, section, subsection,
                    chunk_index, token_count, source_type,
                    metadata
                FROM chunks
                WHERE excluded = 0
                  AND user_id = ?
                ORDER BY chunk_id
            """
            cursor = self._db.execute(sql, (self._user_id_filter,))
        else:
            sql = """
                SELECT
                    chunk_id, doc_id, user_id,
                    text, title, section, subsection,
                    chunk_index, token_count, source_type,
                    metadata
                FROM chunks
                WHERE excluded = 0
                ORDER BY chunk_id
            """
            cursor = self._db.execute(sql, ())

        rows = cursor.fetchall()

        chunk_ids: list[str] = []
        chunks: dict[str, Chunk] = {}
        user_index: dict[str, list[int]] = {}
        source_type_index: dict[str, list[int]] = {}
        corpus: list[list[str]] = []

        for idx, row in enumerate(rows):
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
                # embedding_text is not stored in SQLite; fall back to text
                # (matches VectorRetriever behaviour).
                embedding_text=row["text"],
                title=row["title"] or "",
                section=row["section"] or "",
                subsection=row["subsection"] or "",
                chunk_index=row["chunk_index"],
                token_count=row["token_count"],
                source_type=row["source_type"],
                metadata=meta,
            )

            chunk_ids.append(chunk.chunk_id)
            chunks[chunk.chunk_id] = chunk
            corpus.append(_tokenize(chunk.text))

            user_index.setdefault(chunk.user_id, []).append(idx)
            source_type_index.setdefault(chunk.source_type, []).append(idx)

        self._chunk_ids = chunk_ids
        self._chunks = chunks
        self._user_index = user_index
        self._source_type_index = source_type_index

        if corpus:
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 30,
        source_types: Optional[list[str]] = None,
        intent_hint: Optional["Intent"] = None,
        where: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Score chunks belonging to *user_id*; filter by *source_types* and excluded=0.

        Returns a list of RetrievalResult sorted by BM25 score descending,
        with ``retriever="bm25"``.

        Note: this retriever does not honour ``where=`` metadata filters; pass
        to a vector-based retriever instead.
        """
        # `where` is accepted for Retriever Protocol compatibility but ignored
        # — BM25 has no metadata index. Phase 7-A math-meta filtering happens
        # in the vector path; callers wanting BM25 with metadata should fuse
        # via the router/hybrid retrievers.
        del where  # explicitly ignored
        if self._bm25 is None:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        # Full-corpus BM25 scores (numpy array, length == len(corpus))
        scores = self._bm25.get_scores(tokens)

        # Candidate indices: intersection of user filter and source_type filter
        user_indices = set(self._user_index.get(user_id, []))
        if not user_indices:
            return []

        if source_types is not None:
            type_indices: set[int] = set()
            for st in source_types:
                type_indices.update(self._source_type_index.get(st, []))
            candidate_indices = user_indices & type_indices
        else:
            candidate_indices = user_indices

        if not candidate_indices:
            return []

        # Sort candidates by score descending; take top_k
        ranked = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)
        ranked = ranked[:top_k]

        results: list[RetrievalResult] = []
        for idx in ranked:
            chunk_id = self._chunk_ids[idx]
            chunk = self._chunks[chunk_id]
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(scores[idx]),
                    retriever=self.name,
                    rerank_score=None,
                )
            )

        return results
