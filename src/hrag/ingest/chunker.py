"""Document chunker: split Document into Chunks for embedding and retrieval.

Strategy:
  1. Use detect_sections to get section/subsection spans.
  2. Within each section, split on paragraph boundaries (double newline).
  3. Greedily pack paragraphs into chunks up to chunking_cfg.max_tokens.
  4. Apply chunking_cfg.overlap_tokens overlap between adjacent same-section chunks.
  5. Apply HeteRAG metadata fusion to embedding_text when configured.
"""

from __future__ import annotations

from typing import Any, Callable

from hrag.config import ChunkingConfig
from hrag.types import Chunk, Document
from hrag.ingest.metadata import detect_sections, find_references_offset
from hrag.ingest.math_detect import (
    find_display_math_spans,
    has_math,
    position_in_span,
)


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def _build_tokenizer() -> Callable[[str], int]:
    """Return a count_tokens function using tiktoken cl100k_base."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        def count_tokens(text: str) -> int:
            return len(enc.encode(text))

        return count_tokens
    except ImportError:
        # Fallback: rough approximation (1 token ~= 4 chars)
        def count_tokens_approx(text: str) -> int:
            return max(1, len(text) // 4)

        return count_tokens_approx


# Module-level lazy singleton so we don't re-load the vocab on each call
_default_count_tokens: Callable[[str], int] | None = None


def count_tokens(text: str) -> int:
    """Count tokens using the module-level cl100k_base tokenizer."""
    global _default_count_tokens
    if _default_count_tokens is None:
        _default_count_tokens = _build_tokenizer()
    return _default_count_tokens(text)


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    """Split text on one or more blank lines; preserve non-empty paragraphs.

    Math-aware: a blank line that falls *inside* a display-math span
    ($$...$$ or \\begin{X}...\\end{X}) is NOT treated as a paragraph
    boundary, so multi-line equations stay in a single paragraph and
    therefore in a single chunk.
    """
    import re

    math_spans = find_display_math_spans(text)
    if not math_spans:
        parts = re.split(r"\n{2,}", text)
        return [p.strip() for p in parts if p.strip()]

    # Manual split that honours math spans.
    parts: list[str] = []
    cursor = 0
    for m in re.finditer(r"\n{2,}", text):
        # If the blank line is inside a math span, skip it.
        if position_in_span(m.start(), math_spans) is not None:
            continue
        parts.append(text[cursor:m.start()])
        cursor = m.end()
    parts.append(text[cursor:])
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Core greedy packer
# ---------------------------------------------------------------------------

def _pack_paragraphs(
    paragraphs: list[str],
    max_tokens: int,
    overlap_tokens: int,
    count_fn: Callable[[str], int],
) -> list[str]:
    """Greedily pack paragraphs into chunks; add token overlap between chunks."""
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_paras: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_fn(para)

        # If a single paragraph is too large, split it by sentences / lines.
        # Trade-off: when the paragraph contains a display-math block we
        # refuse to line-split it — tearing an equation apart corrupts its
        # meaning far more than an over-budget chunk does. The over-budget
        # math chunk is emitted as-is and the embedder/LLM truncate gracefully.
        if para_tokens > max_tokens:
            # Flush current buffer first
            if current_paras:
                chunks.append("\n\n".join(current_paras))
                current_paras = []
                current_tokens = 0
            if find_display_math_spans(para):
                # Keep the equation-bearing paragraph whole, even if over budget.
                chunks.append(para)
                continue
            # Hard-split the oversized paragraph by lines
            for line in para.splitlines():
                line = line.strip()
                if not line:
                    continue
                line_tokens = count_fn(line)
                if current_tokens + line_tokens > max_tokens and current_paras:
                    chunks.append("\n\n".join(current_paras))
                    current_paras = []
                    current_tokens = 0
                current_paras.append(line)
                current_tokens += line_tokens
            continue

        if current_tokens + para_tokens > max_tokens and current_paras:
            chunks.append("\n\n".join(current_paras))

            # Build overlap: take paragraphs from the tail of the current chunk
            overlap_paras: list[str] = []
            overlap_toks = 0
            for op in reversed(current_paras):
                ot = count_fn(op)
                if overlap_toks + ot > overlap_tokens:
                    break
                overlap_paras.insert(0, op)
                overlap_toks += ot

            current_paras = overlap_paras + [para]
            current_tokens = overlap_toks + para_tokens
        else:
            current_paras.append(para)
            current_tokens += para_tokens

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    doc: Document,
    chunking_cfg: ChunkingConfig,
    tokenizer: Any = None,
) -> list[Chunk]:
    """Split a Document into Chunks ready for embedding.

    Parameters
    ----------
    doc:
        The loaded Document.
    chunking_cfg:
        ChunkingConfig with max_tokens, overlap_tokens, metadata_fusion.
    tokenizer:
        Optional callable(str) -> int for token counting.
        If None, uses tiktoken cl100k_base.
    """
    if tokenizer is not None:
        count_fn: Callable[[str], int] = tokenizer
    else:
        global _default_count_tokens
        if _default_count_tokens is None:
            _default_count_tokens = _build_tokenizer()
        count_fn = _default_count_tokens

    # Optionally truncate at the References/Bibliography heading. PDFs of
    # academic papers otherwise contribute many bibliography-only chunks that
    # pollute retrieval. The flag lets users keep them when needed.
    body_text = doc.text
    if chunking_cfg.drop_after_references:
        ref_off = find_references_offset(body_text)
        if ref_off is not None and ref_off > 0:
            body_text = body_text[:ref_off]

    sections = detect_sections(body_text)
    chunks: list[Chunk] = []
    chunk_index = 0

    for start_off, end_off, section_title, subsection_title in sections:
        section_text = body_text[start_off:end_off]
        if not section_text.strip():
            continue

        paragraphs = _split_paragraphs(section_text)
        if not paragraphs:
            continue

        raw_chunks = _pack_paragraphs(
            paragraphs,
            chunking_cfg.max_tokens,
            chunking_cfg.overlap_tokens,
            count_fn,
        )

        for raw_text in raw_chunks:
            if not raw_text.strip():
                continue

            token_count = count_fn(raw_text)

            if chunking_cfg.metadata_fusion:
                parts = [f"[{doc.title}]"]
                if section_title:
                    parts.append(f"[{section_title}]")
                if subsection_title:
                    parts.append(f"[{subsection_title}]")
                prefix = " ".join(parts) + "\n\n"
                embedding_text = prefix + raw_text
            else:
                embedding_text = raw_text

            chunk = Chunk(
                chunk_id=f"{doc.doc_id}:{chunk_index:04d}",
                doc_id=doc.doc_id,
                user_id=doc.user_id,
                text=raw_text,
                embedding_text=embedding_text,
                title=doc.title,
                section=section_title,
                subsection=subsection_title,
                chunk_index=chunk_index,
                token_count=token_count,
                source_type=doc.source_type,
                metadata={"has_math": has_math(raw_text)},
            )
            chunks.append(chunk)
            chunk_index += 1

    # Edge case: doc has text but no chunks produced (e.g. all whitespace sections)
    if not chunks and body_text.strip():
        raw_text = body_text.strip()
        embedding_text = f"[{doc.title}]\n\n{raw_text}" if chunking_cfg.metadata_fusion else raw_text
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:0000",
                doc_id=doc.doc_id,
                user_id=doc.user_id,
                text=raw_text,
                embedding_text=embedding_text,
                title=doc.title,
                section="",
                subsection="",
                chunk_index=0,
                token_count=count_fn(raw_text),
                source_type=doc.source_type,
                metadata={"has_math": has_math(raw_text)},
            )
        )

    return chunks
