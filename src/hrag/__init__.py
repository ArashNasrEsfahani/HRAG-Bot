"""HRAG-Bot: hierarchical RAG chatbot with growing memory."""

from __future__ import annotations

__version__ = "0.1.0"

from hrag.orchestrator import ChatResult, Orchestrator

__all__ = ["Orchestrator", "ChatResult"]
