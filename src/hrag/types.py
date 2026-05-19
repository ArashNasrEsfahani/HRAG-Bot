"""Shared dataclasses used across modules.

Keep these stable: every subsystem (ingest, retrieval, orchestrator) imports from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Document:
    """A raw source document loaded from disk before chunking."""

    doc_id: str
    user_id: str
    source_path: str
    title: str
    text: str
    source_type: str = "document"   # "document" | "episodic"
    metadata: dict[str, Any] = field(default_factory=dict)
    ingested_at: Optional[datetime] = None


@dataclass
class Chunk:
    """A chunk of a document, ready for embedding/retrieval.

    `embedding_text` is what the embedding model sees (HeteRAG metadata fusion):
    it concatenates title/section onto `text`. `text` is what the LLM sees at
    generation time (no padding).
    """

    chunk_id: str
    doc_id: str
    user_id: str
    text: str
    embedding_text: str
    title: str = ""
    section: str = ""
    subsection: str = ""
    chunk_index: int = 0
    token_count: int = 0
    source_type: str = "document"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A retrieved chunk with its retrieval score and provenance."""

    chunk: Chunk
    score: float
    retriever: str = "vector"        # "vector" | "kg_ppr" | "community"
    rerank_score: Optional[float] = None
    """Reranker score. Scale depends on the reranker:
       LLMReranker (ACP-RAG)   = int 0-3
       CrossEncoderReranker    = float logit (typically -12..+12 for ms-marco models)
       BatchedLLMReranker      = int 0-3
    """


@dataclass
class Message:
    role: str        # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationRequest:
    messages: list[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[list[str]] = None


@dataclass
class GenerationResponse:
    text: str
    raw: Any = None                  # provider-specific raw response
