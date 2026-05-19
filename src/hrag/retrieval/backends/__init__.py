"""Pluggable vector-store backends for :class:`hrag.retrieval.vector.VectorStore`.

The :class:`VectorBackend` protocol describes the minimal contract the
:class:`VectorStore` facade depends on. ChromaDB is the default (and currently
only) implementation; :class:`SqliteVecBackend` is a stub kept here so the
backend-selection plumbing is in place when sqlite-vec lands.
"""

from __future__ import annotations

from .base import VectorBackend
from .chroma import ChromaBackend
from .sqlite_vec import SqliteVecBackend

__all__ = ["VectorBackend", "ChromaBackend", "SqliteVecBackend"]
