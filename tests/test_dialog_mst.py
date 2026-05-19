"""Tests for hrag.context.dialog_mst.DialogMSTCompactor.

These tests stub both the LLM and the embedding provider so they run
offline and deterministically. The compactor's contract is:

  - Short histories pass through unchanged.
  - Long histories are split into (old, recent); old turns are
    embedding-clustered and each cluster is summarised by one LLM call.
  - Output is ``[synthetic_system_message, *recent]`` where the synthetic
    message holds ``[Earlier conversation]`` followed by joined summaries.
  - On any LLM exception during summarisation, the input is returned
    unchanged (the chat path must never break).
"""

from __future__ import annotations

from typing import Optional

import pytest

from hrag.context.dialog_mst import DialogMSTCompactor
from hrag.types import Message


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubLLM:
    name = "stub"

    def __init__(self, reply: str = "Earlier the user discussed X.", *, raises: bool = False):
        self.reply = reply
        self.raises = raises
        self.calls: list[dict] = []

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        self.calls.append(
            {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
        )
        if self.raises:
            raise RuntimeError("simulated LLM failure")
        return self.reply


class _StubEmbeddings:
    """Returns pre-baked vectors; assumes caller treats them as L2-normalised."""

    name = "stub"

    def __init__(self, vectors):
        self.vectors = vectors
        self.last_inputs: Optional[list[str]] = None

    def embed(self, texts):
        self.last_inputs = list(texts)
        return self.vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _orthogonal_unit_vectors(n: int, dim: int = 32) -> list[list[float]]:
    """One-hot unit vectors so each turn forms its own cluster by default."""
    out: list[list[float]] = []
    for i in range(n):
        v = [0.0] * max(dim, n)
        v[i % len(v)] = 1.0
        out.append(v)
    return out


def _make_history(n: int) -> list[Message]:
    """Alternating user/assistant turns with unique content."""
    history: list[Message] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        history.append(Message(role=role, content=f"turn-{i}: content body {i}"))
    return history


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_short_history_passthrough():
    """Histories at or below compact_after_turns should be returned unchanged."""
    history = _make_history(8)
    llm = _StubLLM()
    emb = _StubEmbeddings(_orthogonal_unit_vectors(8))
    compactor = DialogMSTCompactor(
        llm=llm,
        embeddings=emb,
        compact_after_turns=12,
        keep_recent_turns=6,
    )
    out = compactor.compact(history)
    assert out == history
    assert llm.calls == []
    assert emb.last_inputs is None  # no embedding calls either


def test_long_history_compacted():
    """20 turns, compact_after=12, keep_recent=6 -> output is shorter and the
    first element is a synthetic system message."""
    history = _make_history(20)
    llm = _StubLLM(reply="Cluster summary line.")
    emb = _StubEmbeddings(_orthogonal_unit_vectors(14, dim=32))  # 14 = old slice
    compactor = DialogMSTCompactor(
        llm=llm,
        embeddings=emb,
        compact_after_turns=12,
        keep_recent_turns=6,
    )
    out = compactor.compact(history)
    assert len(out) < len(history)
    assert out[0].role == "system"
    # The last 6 turns must be present verbatim at the end.
    assert out[-6:] == history[-6:]


def test_recent_turns_preserved_verbatim():
    history = _make_history(20)
    llm = _StubLLM(reply="summary")
    emb = _StubEmbeddings(_orthogonal_unit_vectors(14))
    compactor = DialogMSTCompactor(
        llm=llm,
        embeddings=emb,
        compact_after_turns=12,
        keep_recent_turns=6,
    )
    out = compactor.compact(history)
    tail = out[-6:]
    assert len(tail) == 6
    for got, want in zip(tail, history[-6:]):
        assert got.role == want.role
        assert got.content == want.content


def test_summary_uses_dialog_summary_prompt():
    """The prompt sent to the LLM must come from prompts/dialog_summary.md
    — assert it carries that template's distinctive instruction phrase."""
    history = _make_history(20)
    llm = _StubLLM(reply="summary")
    emb = _StubEmbeddings(_orthogonal_unit_vectors(14))
    compactor = DialogMSTCompactor(
        llm=llm,
        embeddings=emb,
        compact_after_turns=12,
        keep_recent_turns=6,
    )
    compactor.compact(history)
    assert len(llm.calls) >= 1
    seen_prompt = llm.calls[0]["prompt"]
    # Distinctive phrase from dialog_summary.md.
    assert "neutral third-person voice" in seen_prompt
    # The turns block must include role-prefixed lines from the old slice.
    assert "User:" in seen_prompt or "Assistant:" in seen_prompt


def test_clustering_groups_similar_turns():
    """If two old turns share an identical vector and are orthogonal to the
    rest, they must be summarised in a single LLM call (one cluster).

    Setup: 13 old turns (compact_after=12, keep_recent=0 forces full
    history-as-old). Indices 0 and 1 share the same one-hot vector;
    the rest get unique one-hots so they each form their own cluster.

    Expected number of LLM calls = 12 (one merged cluster + 11 singletons).
    """
    # 13 old turns, keep_recent=0 so all 13 are 'old'.
    n_old = 13
    # Build unique one-hots; then force index 1 == index 0 so they cluster.
    dim = max(32, n_old)
    vectors = [[0.0] * dim for _ in range(n_old)]
    for i in range(n_old):
        vectors[i][i] = 1.0
    vectors[1] = list(vectors[0])  # identical to index 0 -> cosine 1.0

    history = _make_history(n_old)
    llm = _StubLLM(reply="cluster summary")
    emb = _StubEmbeddings(vectors)
    compactor = DialogMSTCompactor(
        llm=llm,
        embeddings=emb,
        compact_after_turns=12,
        keep_recent_turns=0,
        cluster_threshold=0.55,
    )
    out = compactor.compact(history)
    # One LLM call per cluster: 1 merged + 11 singletons = 12.
    assert len(llm.calls) == 12
    # Output should be just the synthetic system message (recent is empty).
    assert len(out) == 1
    assert out[0].role == "system"


def test_llm_exception_falls_back_to_unchanged():
    history = _make_history(20)
    llm = _StubLLM(raises=True)
    emb = _StubEmbeddings(_orthogonal_unit_vectors(14))
    compactor = DialogMSTCompactor(
        llm=llm,
        embeddings=emb,
        compact_after_turns=12,
        keep_recent_turns=6,
    )
    out = compactor.compact(history)
    assert out == history


def test_summary_target_tokens_propagates_to_max_tokens():
    """max_tokens passed to the LLM must be 1.5x the configured target."""
    history = _make_history(20)
    llm = _StubLLM(reply="summary")
    emb = _StubEmbeddings(_orthogonal_unit_vectors(14))
    compactor = DialogMSTCompactor(
        llm=llm,
        embeddings=emb,
        compact_after_turns=12,
        keep_recent_turns=6,
        summary_target_tokens=300,
    )
    compactor.compact(history)
    assert llm.calls, "expected at least one LLM call"
    for call in llm.calls:
        assert call["max_tokens"] == 450  # 300 * 1.5


def test_synthetic_message_role_is_system():
    history = _make_history(20)
    llm = _StubLLM(reply="summary")
    emb = _StubEmbeddings(_orthogonal_unit_vectors(14))
    compactor = DialogMSTCompactor(
        llm=llm,
        embeddings=emb,
        compact_after_turns=12,
        keep_recent_turns=6,
    )
    out = compactor.compact(history)
    assert out[0].role == "system"


def test_synthetic_message_marker_in_content():
    history = _make_history(20)
    llm = _StubLLM(reply="cluster summary line")
    emb = _StubEmbeddings(_orthogonal_unit_vectors(14))
    compactor = DialogMSTCompactor(
        llm=llm,
        embeddings=emb,
        compact_after_turns=12,
        keep_recent_turns=6,
    )
    out = compactor.compact(history)
    assert out[0].content.startswith("[Earlier conversation]")
    # And it should carry the summary text we stubbed back from the LLM.
    assert "cluster summary line" in out[0].content


def test_empty_history_passthrough():
    """Empty input should return empty output without touching LLM/embeddings."""
    llm = _StubLLM()
    emb = _StubEmbeddings([])
    compactor = DialogMSTCompactor(llm=llm, embeddings=emb)
    assert compactor.compact([]) == []
    assert llm.calls == []
    assert emb.last_inputs is None


def test_embedding_exception_falls_back_to_unchanged():
    """If the embedding provider raises, return the history unchanged."""

    class _BoomEmbeddings:
        name = "boom"

        def embed(self, texts):
            raise RuntimeError("embedding offline")

    history = _make_history(20)
    llm = _StubLLM()
    compactor = DialogMSTCompactor(
        llm=llm,
        embeddings=_BoomEmbeddings(),
        compact_after_turns=12,
        keep_recent_turns=6,
    )
    out = compactor.compact(history)
    assert out == history
    assert llm.calls == []


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
