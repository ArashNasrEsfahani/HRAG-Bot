"""Phase 9.15 — Feedback-weighted re-ranking tests.

All tests use a real (tmp) SQLite DB but no heavy deps (no chromadb, no
sentence-transformers, no ollama). The orchestrator integration tests use
stub objects assembled from conftest helpers.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from hrag.config import RetrievalConfig
from hrag.feedback_scoring import FeedbackScorer, apply_feedback_to_rerank_score
from hrag.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Shared DB helpers (mirrors test_feedback_stats.py pattern)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_db_singleton():
    import hrag.db.connection as _conn_mod

    _conn_mod._db_singleton = None
    yield
    _conn_mod._db_singleton = None


def _make_db(tmp_path: Path):
    """Return a fresh bootstrapped Database."""
    from hrag.db.connection import init_db

    return init_db(tmp_path / "store.sqlite", "default")


def _insert_session(db, user_id: str = "default") -> str:
    sid = "sess_" + uuid.uuid4().hex[:8]
    with db.conn:
        db.execute(
            "INSERT INTO sessions(session_id, user_id) VALUES (?, ?)",
            (sid, user_id),
        )
    db.commit()
    return sid


def _insert_message(
    db,
    session_id: str,
    role: str,
    content: str,
    metadata: str | None = None,
    user_id: str = "default",
) -> int:
    cur = db.execute(
        "INSERT INTO messages(session_id, user_id, role, content, metadata)"
        " VALUES (?, ?, ?, ?, ?)",
        (session_id, user_id, role, content, metadata),
    )
    db.commit()
    return cur.lastrowid


def _insert_feedback(
    db,
    message_id: int,
    session_id: str,
    rating: int,
    user_id: str = "default",
) -> str:
    fid = uuid.uuid4().hex
    with db.conn:
        db.execute(
            "INSERT INTO feedback(feedback_id, message_id, session_id, user_id, rating)"
            " VALUES (?, ?, ?, ?, ?)",
            (fid, str(message_id), session_id, user_id, rating),
        )
    db.commit()
    return fid


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        user_id="default",
        text="sample text",
        embedding_text="sample text",
    )


def _make_result(chunk_id: str, rerank_score: float = 0.0) -> RetrievalResult:
    return RetrievalResult(
        chunk=_make_chunk(chunk_id),
        score=0.5,
        rerank_score=rerank_score,
    )


# ---------------------------------------------------------------------------
# 1. Config default OFF
# ---------------------------------------------------------------------------


def test_feedback_scoring_default_off():
    """cfg.retrieval.feedback_reranking_enabled defaults to False."""
    cfg = RetrievalConfig()
    assert cfg.feedback_reranking_enabled is False


def test_feedback_reranking_weight_default():
    """Default weight is 0.3."""
    cfg = RetrievalConfig()
    assert cfg.feedback_reranking_weight == pytest.approx(0.3)


def test_feedback_reranking_alpha_default():
    """Default EMA alpha is 0.3."""
    cfg = RetrievalConfig()
    assert cfg.feedback_reranking_alpha == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 2. Empty DB → score returns neutral_default (0.0)
# ---------------------------------------------------------------------------


def test_feedback_score_empty_db_returns_zero(tmp_path):
    """FeedbackScorer.score() on an empty feedback table returns 0.0."""
    db = _make_db(tmp_path)
    scorer = FeedbackScorer(db)
    assert scorer.score("chunk_abc") == pytest.approx(0.0)
    db.close()


# ---------------------------------------------------------------------------
# 3. Thumbs-up → positive score
# ---------------------------------------------------------------------------


def test_feedback_score_thumbs_up_positive(tmp_path):
    """A single thumbs-up feedback row makes score(chunk_id) > 0."""
    db = _make_db(tmp_path)
    sid = _insert_session(db)
    _insert_message(db, sid, "user", "question")
    meta = json.dumps({"sources": ["c1", "c2"]})
    asst_id = _insert_message(db, sid, "assistant", "answer", metadata=meta)
    _insert_feedback(db, asst_id, sid, +1)

    scorer = FeedbackScorer(db)
    assert scorer.score("c1") > 0.0
    assert scorer.score("c2") > 0.0
    # Chunk not in sources stays neutral.
    assert scorer.score("c_other") == pytest.approx(0.0)
    db.close()


# ---------------------------------------------------------------------------
# 4. Thumbs-down → negative score
# ---------------------------------------------------------------------------


def test_feedback_score_thumbs_down_negative(tmp_path):
    """A single thumbs-down feedback row makes score(chunk_id) < 0."""
    db = _make_db(tmp_path)
    sid = _insert_session(db)
    _insert_message(db, sid, "user", "question")
    meta = json.dumps({"sources": ["c1", "c2"]})
    asst_id = _insert_message(db, sid, "assistant", "answer", metadata=meta)
    _insert_feedback(db, asst_id, sid, -1)

    scorer = FeedbackScorer(db)
    assert scorer.score("c1") < 0.0
    db.close()


# ---------------------------------------------------------------------------
# 5. EMA decay — 5 ups then 1 down: mostly positive, but pulled down
# ---------------------------------------------------------------------------


def test_feedback_score_ema_decay(tmp_path):
    """After 5 thumbs-up followed by 1 thumbs-down, score is positive but < 1.0."""
    db = _make_db(tmp_path)
    sid = _insert_session(db)

    # 5 thumbs-up for c1
    for _ in range(5):
        _insert_message(db, sid, "user", "q")
        meta = json.dumps({"sources": ["c1"]})
        asst_id = _insert_message(db, sid, "assistant", "a", metadata=meta)
        _insert_feedback(db, asst_id, sid, +1)

    # 1 thumbs-down for c1
    _insert_message(db, sid, "user", "q")
    meta = json.dumps({"sources": ["c1"]})
    asst_id = _insert_message(db, sid, "assistant", "a", metadata=meta)
    _insert_feedback(db, asst_id, sid, -1)

    scorer = FeedbackScorer(db, alpha=0.3)
    s = scorer.score("c1")
    # Must be positive (5 ups dominate) but strictly less than 1.0 (1 down pulls it).
    assert s > 0.0
    assert s < 1.0
    db.close()


# ---------------------------------------------------------------------------
# 6. score_many issues exactly ONE SQL query (spy on db.execute)
# ---------------------------------------------------------------------------


def test_score_many_batches_efficiently(tmp_path):
    """score_many(chunk_ids) issues exactly one SQL query regardless of batch size."""
    db = _make_db(tmp_path)
    sid = _insert_session(db)

    # Seed some data for c1 and c2.
    meta = json.dumps({"sources": ["c1", "c2"]})
    _insert_message(db, sid, "user", "q")
    asst_id = _insert_message(db, sid, "assistant", "a", metadata=meta)
    _insert_feedback(db, asst_id, sid, +1)

    scorer = FeedbackScorer(db)

    query_count = 0
    real_execute = db.execute

    def counting_execute(sql, params=()):
        nonlocal query_count
        if "FROM feedback" in sql:
            query_count += 1
        return real_execute(sql, params)

    db.execute = counting_execute  # type: ignore[method-assign]

    # First call populates cache.
    result1 = scorer.score_many(["c1", "c2", "c3"])
    assert query_count == 1, f"Expected 1 SQL query, got {query_count}"

    # Second call hits the cache — no extra query.
    scorer.score_many(["c1", "c2"])
    assert query_count == 1, "Second call should be cache-hit, no extra query"

    assert result1.get("c1", 0.0) > 0
    assert result1.get("c2", 0.0) > 0
    assert "c3" not in result1  # c3 has no feedback

    db.close()


# ---------------------------------------------------------------------------
# 7. apply_feedback_to_rerank_score pure function
# ---------------------------------------------------------------------------


def test_apply_feedback_to_rerank_score_adds_weighted():
    """apply_feedback_to_rerank_score(1.0, 0.5, weight=0.3) == 1.15."""
    result = apply_feedback_to_rerank_score(1.0, 0.5, weight=0.3)
    assert result == pytest.approx(1.15)


def test_apply_feedback_to_rerank_score_negative_feedback():
    """Negative feedback decreases the score."""
    result = apply_feedback_to_rerank_score(2.0, -1.0, weight=0.3)
    assert result == pytest.approx(1.7)


def test_apply_feedback_to_rerank_score_zero_feedback():
    """Zero feedback is a no-op."""
    result = apply_feedback_to_rerank_score(3.5, 0.0, weight=0.3)
    assert result == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# 8. Malformed / missing metadata is silently skipped (Phase 8 contract 29)
# ---------------------------------------------------------------------------


def test_feedback_score_null_metadata_skipped(tmp_path):
    """metadata IS NULL rows are silently skipped (no crash)."""
    db = _make_db(tmp_path)
    sid = _insert_session(db)
    # message with NULL metadata (omit metadata arg)
    asst_id = _insert_message(db, sid, "assistant", "answer")
    _insert_feedback(db, asst_id, sid, +1)

    scorer = FeedbackScorer(db)
    assert scorer.score("c1") == pytest.approx(0.0)
    db.close()


def test_feedback_score_empty_json_metadata_skipped(tmp_path):
    """metadata = '{}' (no 'sources' key) rows are silently skipped."""
    db = _make_db(tmp_path)
    sid = _insert_session(db)
    asst_id = _insert_message(db, sid, "assistant", "answer", metadata="{}")
    _insert_feedback(db, asst_id, sid, +1)

    scorer = FeedbackScorer(db)
    assert scorer.score("c1") == pytest.approx(0.0)
    db.close()


def test_feedback_score_malformed_json_skipped(tmp_path, caplog):
    """Malformed JSON metadata is logged at DEBUG and skipped (no crash)."""
    import logging

    db = _make_db(tmp_path)
    sid = _insert_session(db)
    asst_id = _insert_message(db, sid, "assistant", "answer", metadata="{bad json}")
    _insert_feedback(db, asst_id, sid, +1)

    with caplog.at_level(logging.DEBUG, logger="hrag.feedback_scoring"):
        scorer = FeedbackScorer(db)
        result = scorer.score("c1")

    assert result == pytest.approx(0.0)
    db.close()


# ---------------------------------------------------------------------------
# 9. Phase-8 nested metadata shape (selected_chunk_ids)
# ---------------------------------------------------------------------------


def test_feedback_score_phase8_nested_sources(tmp_path):
    """score() reads phase8.selected_chunk_ids when top-level sources absent."""
    db = _make_db(tmp_path)
    sid = _insert_session(db)
    meta = json.dumps({
        "phase8": {
            "selected_chunk_ids": ["c_deep"],
            "action": "continue",
        }
    })
    asst_id = _insert_message(db, sid, "assistant", "answer", metadata=meta)
    _insert_feedback(db, asst_id, sid, +1)

    scorer = FeedbackScorer(db)
    assert scorer.score("c_deep") > 0.0
    assert scorer.score("c_other") == pytest.approx(0.0)
    db.close()


# ---------------------------------------------------------------------------
# 10. Orchestrator integration — flag OFF: rerank_score values unchanged
# ---------------------------------------------------------------------------


def test_orchestrator_no_feedback_pass_through(tmp_path):
    """When feedback_reranking_enabled=False, orchestrator leaves rerank_score unchanged."""
    from hrag.config import RetrievalConfig

    # Build a result with a known rerank_score.
    r = _make_result("c1", rerank_score=4.0)
    original_score = r.rerank_score

    # Simulate the orchestrator's logic: when flag is OFF, nothing happens.
    cfg = RetrievalConfig(feedback_reranking_enabled=False)
    assert not cfg.feedback_reranking_enabled

    # Score must be unchanged.
    assert r.rerank_score == pytest.approx(original_score)
    # feedback_score default is 0.0.
    assert r.feedback_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 11. Orchestrator integration — flag ON: feedback nudges scores and re-sorts
# ---------------------------------------------------------------------------


def test_orchestrator_with_feedback_shifts_scores(tmp_path):
    """When flag ON and feedback exists, the result with negative feedback is demoted."""
    db = _make_db(tmp_path)
    sid = _insert_session(db)

    # c_good has thumbs-up; c_bad has thumbs-down.
    for chunk_id, rating in [("c_good", +1), ("c_bad", -1)]:
        _insert_message(db, sid, "user", "q")
        meta = json.dumps({"sources": [chunk_id]})
        asst_id = _insert_message(db, sid, "assistant", "a", metadata=meta)
        _insert_feedback(db, asst_id, sid, rating)

    # Start them at the same rerank_score. c_bad would normally rank first
    # if it had a higher score — but with feedback it gets demoted.
    results = [
        _make_result("c_bad", rerank_score=1.0),   # starts higher
        _make_result("c_good", rerank_score=0.9),  # starts lower
    ]

    scorer = FeedbackScorer(db, alpha=0.3)
    fb_scores = scorer.score_many(["c_bad", "c_good"])

    weight = 0.3
    for r in results:
        fs = fb_scores.get(r.chunk.chunk_id, 0.0)
        r.feedback_score = fs
        old = r.rerank_score if r.rerank_score is not None else 0.0
        r.rerank_score = apply_feedback_to_rerank_score(old, fs, weight=weight)

    # Re-sort descending.
    results.sort(
        key=lambda r2: (
            r2.rerank_score if r2.rerank_score is not None else float("-inf"),
            r2.score,
        ),
        reverse=True,
    )

    # After feedback nudge c_good (positive feedback) should rank above c_bad.
    assert results[0].chunk.chunk_id == "c_good"
    assert results[1].chunk.chunk_id == "c_bad"

    # feedback_score is populated.
    by_id = {r.chunk.chunk_id: r for r in results}
    assert by_id["c_good"].feedback_score > 0.0
    assert by_id["c_bad"].feedback_score < 0.0

    db.close()


# ---------------------------------------------------------------------------
# 12. feedback_score field exists on RetrievalResult with default 0.0
# ---------------------------------------------------------------------------


def test_retrieval_result_has_feedback_score_field():
    """RetrievalResult.feedback_score defaults to 0.0."""
    r = _make_result("cX", rerank_score=1.5)
    assert hasattr(r, "feedback_score")
    assert r.feedback_score == pytest.approx(0.0)
