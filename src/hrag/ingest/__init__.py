"""hrag.ingest — document ingestion subsystem.

Public API:
    load_document   - load a file from disk into a Document
    chunk_document  - split a Document into Chunks
    IngestPipeline  - orchestrator: load -> chunk -> embed -> store
"""

from __future__ import annotations

from hrag.ingest.loaders import load_document
from hrag.ingest.chunker import chunk_document
from hrag.ingest.pipeline import IngestPipeline

__all__ = [
    "load_document",
    "chunk_document",
    "IngestPipeline",
]
