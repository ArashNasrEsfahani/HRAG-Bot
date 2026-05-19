"""Tests for the Nougat-OCR PDF loader scaffold (Phase 7-C).

These tests verify the wiring and fallback behaviour WITHOUT requiring
nougat-ocr to be installed.  All tests pass in the stock dev environment.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. is_nougat_available() returns a bool (True or False; both are valid here)
# ---------------------------------------------------------------------------

def test_is_nougat_available_returns_bool():
    # Reset the module-level cache so we get a fresh probe.
    import hrag.ingest.nougat_loader as nl
    nl._NOUGAT_AVAILABLE = None  # reset cache

    from hrag.ingest.nougat_loader import is_nougat_available
    result = is_nougat_available()
    assert isinstance(result, bool), "is_nougat_available() must return a bool"


# ---------------------------------------------------------------------------
# 2. load_pdf_nougat() raises ImportError with actionable message when
#    nougat is not installed (mock is_nougat_available to return False).
# ---------------------------------------------------------------------------

def test_load_pdf_nougat_raises_import_error_when_not_installed(tmp_path):
    # Create a dummy PDF path (doesn't need to be a real PDF for this test).
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    import hrag.ingest.nougat_loader as nl

    with patch.object(nl, "is_nougat_available", return_value=False):
        with pytest.raises(ImportError) as exc_info:
            nl.load_pdf_nougat(fake_pdf)

    msg = str(exc_info.value)
    assert "nougat-ocr" in msg, "Error message should name the package"
    assert "pip install" in msg, "Error message should give installation command"


# ---------------------------------------------------------------------------
# 3. _load_pdf with use_nougat=False uses PyMuPDF (never calls nougat).
# ---------------------------------------------------------------------------

def test_load_pdf_uses_pymupdf_when_nougat_disabled(tmp_path):
    from hrag.config import Config, IngestConfig

    cfg = Config()
    cfg.ingest = IngestConfig(use_nougat=False)

    # Build a mock fitz document that returns one page of text.
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Hello world"
    mock_page.number = 0

    mock_fitz_doc = MagicMock()
    mock_fitz_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_fitz_doc.close = MagicMock()

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    import hrag.ingest.loaders as loaders

    with patch.dict(sys.modules, {"fitz": MagicMock(open=MagicMock(return_value=mock_fitz_doc))}):
        # Patch fitz inside the loaders module directly
        with patch("hrag.ingest.loaders._load_pdf_pymupdf") as mock_pymupdf:
            mock_pymupdf.return_value = ("Hello world", {"format": "pdf", "page_count": 1, "pages": [1]})
            text, metadata = loaders._load_pdf(fake_pdf, cfg=cfg)

    mock_pymupdf.assert_called_once_with(fake_pdf)
    assert text == "Hello world"
    assert metadata["format"] == "pdf"


# ---------------------------------------------------------------------------
# 4. _load_pdf with use_nougat=True but nougat not installed falls back
#    to PyMuPDF and logs an info message.
# ---------------------------------------------------------------------------

def test_load_pdf_falls_back_to_pymupdf_when_nougat_not_installed(tmp_path, caplog):
    from hrag.config import Config, IngestConfig

    cfg = Config()
    cfg.ingest = IngestConfig(use_nougat=True)

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    import hrag.ingest.loaders as loaders
    import hrag.ingest.nougat_loader as nl

    expected_text = "PyMuPDF fallback text"
    expected_meta = {"format": "pdf", "page_count": 0, "pages": []}

    with patch.object(nl, "is_nougat_available", return_value=False):
        with patch("hrag.ingest.loaders._load_pdf_pymupdf") as mock_pymupdf:
            mock_pymupdf.return_value = (expected_text, expected_meta)

            import logging
            with caplog.at_level(logging.INFO, logger="hrag.ingest.loaders"):
                text, metadata = loaders._load_pdf(fake_pdf, cfg=cfg)

    mock_pymupdf.assert_called_once_with(fake_pdf)
    assert text == expected_text
    # Should have logged that nougat is not installed
    assert any("not installed" in r.message.lower() or "nougat" in r.message.lower()
                for r in caplog.records)


# ---------------------------------------------------------------------------
# 5. Importing hrag.ingest.nougat_loader does NOT import nougat at module load.
# ---------------------------------------------------------------------------

def test_nougat_loader_import_has_no_nougat_side_effect():
    # Remove from sys.modules if it was already imported in this session.
    for key in list(sys.modules.keys()):
        if key == "hrag.ingest.nougat_loader":
            del sys.modules[key]

    # Confirm nougat is not in sys.modules before our import.
    nougat_keys_before = [k for k in sys.modules if k == "nougat" or k.startswith("nougat.")]

    import hrag.ingest.nougat_loader  # noqa: F401

    nougat_keys_after = [k for k in sys.modules if k == "nougat" or k.startswith("nougat.")]

    # Any nougat keys that appeared are new ones introduced by the import.
    new_keys = set(nougat_keys_after) - set(nougat_keys_before)
    assert not new_keys, (
        f"Importing nougat_loader pulled in nougat modules: {new_keys}. "
        "All nougat imports must be deferred inside functions."
    )
