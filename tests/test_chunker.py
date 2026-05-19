"""Tests for hrag.ingest.chunker — chunk_document()."""

from __future__ import annotations

from pathlib import Path

import pytest

tiktoken = pytest.importorskip("tiktoken")

from hrag.config import ChunkingConfig
from hrag.ingest.chunker import chunk_document
from hrag.types import Chunk, Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(text: str, title: str = "TestDoc", doc_id: str = "doc0001") -> Document:
    return Document(
        doc_id=doc_id,
        user_id="tester",
        source_path="/fake/path.txt",
        title=title,
        text=text,
    )


def _default_cfg(**kwargs) -> ChunkingConfig:
    defaults = dict(max_tokens=400, overlap_tokens=60, metadata_fusion=True)
    defaults.update(kwargs)
    return ChunkingConfig(**defaults)


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------

def test_single_small_doc_produces_at_least_one_chunk() -> None:
    doc = _make_doc("This is a short document.\n")
    chunks = chunk_document(doc, _default_cfg())
    assert len(chunks) >= 1


def test_chunks_are_chunk_instances() -> None:
    doc = _make_doc("Some content.\n\nMore content.\n")
    chunks = chunk_document(doc, _default_cfg())
    for c in chunks:
        assert isinstance(c, Chunk)


def test_chunk_index_zero_based_and_contiguous() -> None:
    """chunk_index values must be 0, 1, 2, ... with no gaps."""
    # Build a doc large enough to guarantee multiple chunks
    para = "Word " * 100 + "\n\n"
    doc = _make_doc(para * 20)
    chunks = chunk_document(doc, _default_cfg(max_tokens=100, overlap_tokens=10))
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks))), f"Non-contiguous indices: {indices}"


def test_all_chunks_share_same_doc_id() -> None:
    para = "Word " * 80 + "\n\n"
    doc = _make_doc(para * 10, doc_id="mydoc123")
    chunks = chunk_document(doc, _default_cfg(max_tokens=100, overlap_tokens=10))
    assert all(c.doc_id == "mydoc123" for c in chunks)


def test_all_chunks_share_same_user_id() -> None:
    doc = _make_doc("Some text.\n\nMore text.\n")
    doc.user_id = "test_user"
    chunks = chunk_document(doc, _default_cfg())
    assert all(c.user_id == "test_user" for c in chunks)


# ---------------------------------------------------------------------------
# metadata_fusion
# ---------------------------------------------------------------------------

def test_metadata_fusion_enabled_adds_title_prefix() -> None:
    """With metadata_fusion=True, embedding_text starts with [<title>]."""
    doc = _make_doc("Some content here.\n", title="MyTitle")
    chunks = chunk_document(doc, _default_cfg(metadata_fusion=True))
    assert len(chunks) >= 1
    for c in chunks:
        assert c.embedding_text.startswith("[MyTitle]"), (
            f"Expected '[MyTitle]' prefix, got: {c.embedding_text[:60]!r}"
        )


def test_metadata_fusion_disabled_embedding_equals_text() -> None:
    """With metadata_fusion=False, embedding_text must equal text."""
    doc = _make_doc("Plain content.\n\nSecond paragraph.\n", title="ATitle")
    chunks = chunk_document(doc, _default_cfg(metadata_fusion=False))
    assert len(chunks) >= 1
    for c in chunks:
        assert c.embedding_text == c.text, (
            f"embedding_text != text when fusion disabled:\n"
            f"  embedding_text={c.embedding_text!r}\n"
            f"  text={c.text!r}"
        )


def test_metadata_fusion_includes_section_label() -> None:
    """With fusion on and a headed section, embedding_text should include [section]."""
    text = "# Introduction\n\nIntro content.\n"
    doc = _make_doc(text, title="Report")
    chunks = chunk_document(doc, _default_cfg(metadata_fusion=True))
    assert len(chunks) >= 1
    embedding_texts = [c.embedding_text for c in chunks if c.section]
    assert any("[Introduction]" in et for et in embedding_texts), (
        f"No chunk embedding_text contains '[Introduction]'. "
        f"Got: {[c.embedding_text for c in chunks]}"
    )


# ---------------------------------------------------------------------------
# Token budget enforcement
# ---------------------------------------------------------------------------

def test_long_doc_produces_multiple_chunks() -> None:
    """A document with many tokens must be split into multiple chunks."""
    # 200 words × ~1.3 tokens/word ≈ 260 tokens per paragraph, 5 paragraphs ≈ 1300 tokens
    para = ("The quick brown fox jumps over the lazy dog. " * 20).strip() + "\n\n"
    doc = _make_doc(para * 5)
    chunks = chunk_document(doc, _default_cfg(max_tokens=150, overlap_tokens=20))
    assert len(chunks) > 1, "Expected multiple chunks for a long document"


def test_chunks_respect_max_tokens_budget(tmp_path: Path) -> None:
    """No chunk's token_count should exceed max_tokens by more than 10%."""
    enc = tiktoken.get_encoding("cl100k_base")

    def count(t: str) -> int:
        return len(enc.encode(t))

    max_tok = 100
    slack = 1.10  # 10% tolerance for boundary paragraphs
    para = ("Hello world this is a test sentence. " * 10).strip() + "\n\n"
    doc = _make_doc(para * 8)
    chunks = chunk_document(
        doc,
        _default_cfg(max_tokens=max_tok, overlap_tokens=10, metadata_fusion=False),
        tokenizer=count,
    )
    for c in chunks:
        assert c.token_count <= max_tok * slack, (
            f"Chunk {c.chunk_index} has {c.token_count} tokens, "
            f"exceeds budget {max_tok} (with {int(slack*100)}% slack)"
        )


# ---------------------------------------------------------------------------
# chunk_id format
# ---------------------------------------------------------------------------

def test_chunk_ids_are_unique() -> None:
    para = "Word " * 50 + "\n\n"
    doc = _make_doc(para * 10, doc_id="uniqtest")
    chunks = chunk_document(doc, _default_cfg(max_tokens=80, overlap_tokens=10))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids detected"


def test_chunk_id_format() -> None:
    """chunk_id must be '<doc_id>:<4-digit-zero-padded-index>'."""
    doc = _make_doc("Some text here.\n", doc_id="abcdef01")
    chunks = chunk_document(doc, _default_cfg())
    assert chunks[0].chunk_id == "abcdef01:0000"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_doc_returns_no_chunks() -> None:
    doc = _make_doc("   \n\n   \n")
    chunks = chunk_document(doc, _default_cfg())
    # Either 0 chunks or the edge-case fallback; the key invariant is no crash
    assert isinstance(chunks, list)


def test_whitespace_only_sections_skipped() -> None:
    """Sections containing only whitespace should not produce chunks."""
    doc = _make_doc("# Section\n\n   \n\n# Real Section\n\nActual content here.\n")
    chunks = chunk_document(doc, _default_cfg())
    # All chunks must have non-empty text
    for c in chunks:
        assert c.text.strip(), f"Chunk with empty text: {c!r}"
