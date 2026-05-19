"""Nougat-OCR loader for academic PDFs.

This module provides an optional Nougat-OCR loader that reconstructs LaTeX-rich
text from academic PDFs. It is an opt-in alternative to the default PyMuPDF
loader and is only active when ``ingest.use_nougat: true`` is set in config
AND the ``nougat-ocr`` package is installed.

All nougat imports are deferred inside functions so that importing this module
has zero side effects — ``hrag`` starts normally even without nougat installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache: None = not yet checked, True/False = result of last check.
_NOUGAT_AVAILABLE: bool | None = None


def is_nougat_available() -> bool:
    """Return True iff the nougat-ocr package can be imported.

    The result is cached after the first call so repeated calls are cheap.
    """
    global _NOUGAT_AVAILABLE
    if _NOUGAT_AVAILABLE is not None:
        return _NOUGAT_AVAILABLE

    try:
        import importlib
        # The PyPI package is nougat-ocr; the importable name is 'nougat'.
        importlib.import_module("nougat")
        _NOUGAT_AVAILABLE = True
    except ImportError:
        _NOUGAT_AVAILABLE = False

    return _NOUGAT_AVAILABLE


def load_pdf_nougat(
    path: Path,
    *,
    model: str = "facebook/nougat-base",
) -> str:
    """Load a PDF with Nougat-OCR and return LaTeX-rich text.

    Parameters
    ----------
    path:
        Absolute path to the PDF file.
    model:
        HuggingFace checkpoint ID, e.g. ``"facebook/nougat-base"`` (default)
        or ``"facebook/nougat-small"``.  If the model weights are not on disk,
        Nougat's own download path runs automatically (~800 MB on first use).

    Raises
    ------
    ImportError
        If ``nougat-ocr`` is not installed.
    FileNotFoundError
        If *path* does not exist.
    """
    if not is_nougat_available():
        raise ImportError(
            "nougat-ocr not installed. "
            "Run: pip install nougat-ocr  (~800MB model download on first use)"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    # All heavy imports are deferred here so module-level import stays free of
    # side effects.
    import torch  # nougat depends on torch
    from nougat import NougatModel
    from nougat.utils.dataset import LazyDataset
    from nougat.utils.checkpoint import get_checkpoint
    import fitz  # pymupdf — needed to count pages

    logger.info("Nougat OCR: loading model %s for %s", model, path.name)

    checkpoint = get_checkpoint(model)
    nougat_model = NougatModel.from_pretrained(checkpoint)
    nougat_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nougat_model = nougat_model.to(device)

    dataset = LazyDataset(
        str(path),
        nougat_model.encoder.prepare_input,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )

    pages: list[str] = []
    for sample, _ in dataloader:
        if sample is None:
            continue
        model_output = nougat_model.inference(image_tensors=sample)
        page_text = model_output["predictions"][0]
        if page_text.strip():
            pages.append(page_text)

    full_text = "\n\n".join(pages)
    logger.info(
        "Nougat OCR: extracted %d pages / %d chars from %s",
        len(pages),
        len(full_text),
        path.name,
    )
    return full_text
