"""Phase 9 wave 3 — focused coverage for the remaining tickets.

Covers (pure functions + config defaults):
  9.2  async_preflight_enabled config flag
  9.9  rerank_fallback_telemetry config flag + table existence
  9.10 first_token_latency_enabled config flag
  9.13 context_compression_* config flags
  9.16 crag_enabled / crag_score_floor / multiplier flags
  9.17 self_rag_enabled flag + extract_uncertain_spans pure helper
"""
from __future__ import annotations

from hrag.config import CompactionConfig, RetrievalConfig
from hrag.gating.uncertain import extract_uncertain_spans


# ---------------------------------------------------------------------------
# 9.2 — Async pre-retrieval
# ---------------------------------------------------------------------------


def test_async_preflight_default_off():
    cfg = RetrievalConfig()
    assert cfg.async_preflight_enabled is False


# ---------------------------------------------------------------------------
# 9.9 — Rerank-fallback telemetry
# ---------------------------------------------------------------------------


def test_rerank_fallback_telemetry_default_off():
    cfg = RetrievalConfig()
    assert cfg.rerank_fallback_telemetry_enabled is False


def test_rerank_fallback_telemetry_table_created(tmp_path):
    from hrag.db.connection import init_db

    db = init_db(tmp_path / "x.sqlite", "u")
    try:
        row = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='rerank_fallback_events'"
        ).fetchone()
        assert row is not None
    finally:
        db.close()


def test_feedback_summary_includes_rerank_fallback(tmp_path):
    from hrag.db.connection import init_db
    from hrag.feedback_stats import feedback_summary

    db = init_db(tmp_path / "x.sqlite", "u")
    try:
        out = feedback_summary(db)
        assert "rerank_fallback_count" in out
        assert out["rerank_fallback_count"] == 0
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 9.10 — First-token latency tracker
# ---------------------------------------------------------------------------


def test_first_token_latency_default_off():
    cfg = RetrievalConfig()
    assert cfg.first_token_latency_enabled is False


# ---------------------------------------------------------------------------
# 9.13 — Context compression
# ---------------------------------------------------------------------------


def test_context_compression_default_off():
    cfg = CompactionConfig()
    assert cfg.context_compression_enabled is False
    assert cfg.context_budget_chars == 12_000


# ---------------------------------------------------------------------------
# 9.16 — CRAG re-routing
# ---------------------------------------------------------------------------


def test_crag_default_off():
    cfg = RetrievalConfig()
    assert cfg.crag_enabled is False
    assert cfg.crag_score_floor == 0.0
    assert cfg.crag_retry_top_k_multiplier == 2.0


# ---------------------------------------------------------------------------
# 9.17 — Self-RAG span re-retrieval
# ---------------------------------------------------------------------------


def test_self_rag_default_off():
    cfg = CompactionConfig()
    assert cfg.self_rag_enabled is False
    assert cfg.self_rag_max_spans == 2


def test_extract_uncertain_spans_empty():
    assert extract_uncertain_spans("") == []
    assert extract_uncertain_spans("plain text with no markers") == []


def test_extract_uncertain_spans_single():
    text = "HippoRAG uses PPR for retrieval [UNCERTAIN]. It is hierarchical."
    spans = extract_uncertain_spans(text)
    assert len(spans) == 1
    assert "PPR" in spans[0]
    assert "[UNCERTAIN]" not in spans[0]


def test_extract_uncertain_spans_multiple():
    text = (
        "Claim one happens here. The first detail X [UNCERTAIN]. "
        "Then claim two. The second detail Y [UNCERTAIN]."
    )
    spans = extract_uncertain_spans(text)
    assert len(spans) == 2
    assert "X" in spans[0]
    assert "Y" in spans[1]


def test_extract_uncertain_spans_respects_max_chars():
    long_prefix = "x" * 600
    text = long_prefix + " [UNCERTAIN]"
    spans = extract_uncertain_spans(text, max_chars_per_span=100)
    assert len(spans) == 1
    assert len(spans[0]) <= 100
