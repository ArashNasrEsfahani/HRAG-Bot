"""Tests for hrag.ingest.loaders — load_document()."""

from __future__ import annotations

from pathlib import Path

import pytest

from hrag.ingest.loaders import load_document
from hrag.types import Document


# ---------------------------------------------------------------------------
# Plain-text loader
# ---------------------------------------------------------------------------

def test_txt_loader_returns_document(sample_txt_path: Path) -> None:
    doc = load_document(sample_txt_path, "user1")
    assert isinstance(doc, Document)


def test_txt_loader_title_is_stem(sample_txt_path: Path) -> None:
    doc = load_document(sample_txt_path, "user1")
    assert doc.title == sample_txt_path.stem   # "sample"


def test_txt_loader_text_nonempty(sample_txt_path: Path) -> None:
    doc = load_document(sample_txt_path, "user1")
    assert len(doc.text) > 0


def test_txt_loader_full_text_content(tmp_path: Path) -> None:
    """The entire file content must be present in doc.text."""
    content = "Hello world.\nSecond line.\n"
    p = tmp_path / "hello.txt"
    p.write_text(content, encoding="utf-8")
    doc = load_document(p, "user1")
    assert doc.text == content


def test_txt_loader_metadata_format(sample_txt_path: Path) -> None:
    doc = load_document(sample_txt_path, "user1")
    assert doc.metadata.get("format") == "txt"


def test_txt_loader_user_id_preserved(sample_txt_path: Path) -> None:
    doc = load_document(sample_txt_path, "alice")
    assert doc.user_id == "alice"


def test_txt_loader_source_path_is_absolute(sample_txt_path: Path) -> None:
    doc = load_document(sample_txt_path, "user1")
    assert Path(doc.source_path).is_absolute()


# ---------------------------------------------------------------------------
# Markdown loader
# ---------------------------------------------------------------------------

def test_md_loader_returns_document(sample_md_path: Path) -> None:
    doc = load_document(sample_md_path, "user1")
    assert isinstance(doc, Document)


def test_md_loader_title_is_stem(sample_md_path: Path) -> None:
    doc = load_document(sample_md_path, "user1")
    assert doc.title == sample_md_path.stem   # "sample"


def test_md_loader_metadata_format(sample_md_path: Path) -> None:
    doc = load_document(sample_md_path, "user1")
    assert doc.metadata.get("format") == "md"


def test_md_loader_text_contains_headings(sample_md_path: Path) -> None:
    doc = load_document(sample_md_path, "user1")
    assert "Introduction" in doc.text
    assert "Methods" in doc.text


def test_markdown_extension_accepted(tmp_path: Path) -> None:
    """The .markdown extension (not just .md) must be accepted."""
    p = tmp_path / "notes.markdown"
    p.write_text("# Heading\n\nBody text.\n", encoding="utf-8")
    doc = load_document(p, "user1")
    assert doc.title == "notes"
    assert doc.metadata.get("format") == "md"


# ---------------------------------------------------------------------------
# doc_id stability and user-scoping
# ---------------------------------------------------------------------------

def test_doc_id_stable_across_calls(sample_txt_path: Path) -> None:
    """Same path + same user_id must always produce the same doc_id."""
    doc1 = load_document(sample_txt_path, "user1")
    doc2 = load_document(sample_txt_path, "user1")
    assert doc1.doc_id == doc2.doc_id


def test_doc_id_differs_for_different_users(sample_txt_path: Path) -> None:
    """Different user_ids must produce different doc_ids for the same file."""
    doc_alice = load_document(sample_txt_path, "alice")
    doc_bob = load_document(sample_txt_path, "bob")
    assert doc_alice.doc_id != doc_bob.doc_id


def test_doc_id_is_16_hex_chars(sample_txt_path: Path) -> None:
    doc = load_document(sample_txt_path, "user1")
    assert len(doc.doc_id) == 16
    assert all(c in "0123456789abcdef" for c in doc.doc_id)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_unsupported_extension_raises_value_error(tmp_path: Path) -> None:
    p = tmp_path / "file.xyz"
    p.write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_document(p, "user1")


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    p = tmp_path / "nonexistent.txt"
    with pytest.raises(FileNotFoundError):
        load_document(p, "user1")


# ---------------------------------------------------------------------------
# PDF roundtrip (optional — requires pymupdf)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    pytest.importorskip("fitz", reason="pymupdf not installed") is None,
    reason="pymupdf not installed",
)
def test_pdf_loader_roundtrip(tmp_path: Path) -> None:
    """Generate a minimal PDF with pymupdf and verify load_document reads it back."""
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "tiny.pdf"
    pdf_doc = fitz.open()
    page = pdf_doc.new_page()
    page.insert_text((72, 72), "Hello from PDF.")
    pdf_doc.save(str(pdf_path))
    pdf_doc.close()

    doc = load_document(pdf_path, "user1")
    assert isinstance(doc, Document)
    assert "Hello from PDF" in doc.text
    assert doc.metadata.get("format") == "pdf"
    assert doc.title == "tiny"


# ---------------------------------------------------------------------------
# PDF page-artifact stripping (unit-level, no pymupdf needed)
# ---------------------------------------------------------------------------

def test_strip_page_artifacts_removes_bare_page_numbers() -> None:
    from hrag.ingest.loaders import _strip_page_artifacts

    page_text = "Some opening sentence.\n12\nNext sentence after a page number."
    cleaned = _strip_page_artifacts(page_text)
    assert "12" not in cleaned.splitlines()
    assert "Some opening sentence." in cleaned
    assert "Next sentence after a page number." in cleaned


def test_strip_page_artifacts_removes_page_x_of_y() -> None:
    from hrag.ingest.loaders import _strip_page_artifacts

    page_text = "Body content here.\nPage 3 of 50\nMore body."
    cleaned = _strip_page_artifacts(page_text)
    assert "Page 3 of 50" not in cleaned
    assert "Body content here." in cleaned
    assert "More body." in cleaned


def test_strip_page_artifacts_keeps_inline_numbers() -> None:
    """Numbers embedded in real prose must be preserved."""
    from hrag.ingest.loaders import _strip_page_artifacts

    page_text = "We trained on 250 examples.\nThe loss decreased by 12.4%."
    cleaned = _strip_page_artifacts(page_text)
    assert "We trained on 250 examples." in cleaned
    assert "The loss decreased by 12.4%." in cleaned


@pytest.mark.skipif(
    pytest.importorskip("fitz", reason="pymupdf not installed") is None,
    reason="pymupdf not installed",
)
def test_pdf_loader_inserts_blank_line_between_pages(tmp_path: Path) -> None:
    """Joining pages with \\n\\n lets paragraph splitters break at page boundaries."""
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "two_page.pdf"
    pdf_doc = fitz.open()
    p1 = pdf_doc.new_page()
    p1.insert_text((72, 72), "End of page one paragraph.")
    p2 = pdf_doc.new_page()
    p2.insert_text((72, 72), "Start of page two paragraph.")
    pdf_doc.save(str(pdf_path))
    pdf_doc.close()

    doc = load_document(pdf_path, "user1")
    # The two page contents should be separated by at least one blank line
    # (i.e. \n\n) so downstream paragraph splitting respects the boundary.
    assert "\n\n" in doc.text


@pytest.mark.skipif(
    pytest.importorskip("fitz", reason="pymupdf not installed") is None,
    reason="pymupdf not installed",
)
def test_pdf_loader_emits_page_spans(tmp_path: Path) -> None:
    """Phase 13.1: _load_pdf_pymupdf must emit page_spans with monotonic,
    non-overlapping entries; page_for_offset(0, spans) must equal 1.
    """
    fitz = pytest.importorskip("fitz")
    from hrag.ingest.metadata import page_for_offset

    pdf_path = tmp_path / "spans_test.pdf"
    pdf_doc = fitz.open()
    p1 = pdf_doc.new_page()
    p1.insert_text((72, 72), "Content on page one.")
    p2 = pdf_doc.new_page()
    p2.insert_text((72, 72), "Content on page two.")
    pdf_doc.save(str(pdf_path))
    pdf_doc.close()

    doc = load_document(pdf_path, "user1")
    spans = doc.metadata.get("page_spans")
    assert spans is not None, "page_spans must be present in PDF metadata"
    assert len(spans) == 2, f"Expected 2 spans for 2-page PDF, got {len(spans)}"

    # Spans must be non-overlapping and monotonically increasing
    for i, (start, end, pno) in enumerate(spans):
        assert start < end, f"Span {i} has start >= end: {start} >= {end}"
        if i > 0:
            prev_end = spans[i - 1][1]
            assert start >= prev_end, (
                f"Span {i} start {start} overlaps previous span end {prev_end}"
            )

    # Page numbers must be 1, 2 in order
    page_nos = [pno for _, _, pno in spans]
    assert page_nos == [1, 2], f"Expected page numbers [1, 2], got {page_nos}"

    # page_for_offset(0, spans) must return page 1
    assert page_for_offset(0, spans) == 1, (
        "page_for_offset(0, spans) must be 1 (first page)"
    )
