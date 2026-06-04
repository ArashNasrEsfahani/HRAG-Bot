"""Shared dataclasses for the hierarchical taxonomy layer.

Kept here (not in ``hrag.types``) because they are internal to the taxonomy
subsystem; only ``RetrievalResult`` crosses the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaxonomyNode:
    """A node in the user's document taxonomy.

    Internal nodes group child nodes; leaf nodes hold a list of assigned
    document IDs (queried lazily via :class:`TaxonomyStore`).
    """

    node_id: str
    user_id: str
    parent_id: Optional[str]
    label: str
    description: str = ""
    depth: int = 0
    is_leaf: bool = False
    centroid: Optional[list[float]] = None
    doc_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    # Phase 12 — per-node keywords for hybrid (dense + sparse) routing. Empty
    # until a tree is built with the keyword-aware propose prompt or backfilled
    # via `hrag taxonomy keywords`. Stored as a JSON list on kg_taxonomy_nodes.
    keywords: list[str] = field(default_factory=list)


@dataclass
class NodeScore:
    """A node with its score against the current query.

    ``score`` is the combined routing score. When hybrid keyword routing is
    active it equals ``cosine + keyword_weight * keyword_score``; otherwise it
    is the plain cosine. ``keyword_score`` (0..1) is kept separately so the
    GUI trace can show how much the keyword signal contributed.
    """

    node: TaxonomyNode
    score: float          # combined routing score
    keyword_score: float = 0.0   # normalized keyword overlap contribution (0..1)


@dataclass
class LevelTrace:
    """Diagnostic record of one beam-descent level.

    The GUI renders these to show *all* considered branches at each level,
    with the ones the beam kept highlighted.
    """

    depth: int
    considered: list[NodeScore] = field(default_factory=list)   # everything scored
    kept: list[NodeScore] = field(default_factory=list)         # survived top-B prune


@dataclass
class DescendResult:
    """Outcome of one beam descent of the tree.

    ``leaves`` are the terminal nodes the beam reached, paired with their
    cosine score. ``trace`` is the full level-by-level record used for UI
    visualization.
    """

    leaves: list[NodeScore] = field(default_factory=list)
    trace: list[LevelTrace] = field(default_factory=list)
