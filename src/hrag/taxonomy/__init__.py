"""Hierarchical document-taxonomy layer (Phase 2b).

A user-editable tree where each leaf node holds a small set of documents.
Retrieval beam-searches from the root, opens only a few leaves, and runs
chunk retrieval scoped to the documents found there.

Modules
-------
- ``types``     — TaxonomyNode dataclass + DescendTrace dataclass.
- ``store``     — TaxonomyStore: SQLite CRUD, centroid math, beam descend.
- ``builder``   — TaxonomyBuilder: LLM-proposes a tree from the existing corpus.
- ``assigner``  — DocAssigner: files a single new doc into the tree on ingest.
"""

from hrag.taxonomy.types import (
    TaxonomyNode,
    NodeScore,
    LevelTrace,
    DescendResult,
)

__all__ = [
    "TaxonomyNode",
    "NodeScore",
    "LevelTrace",
    "DescendResult",
]
