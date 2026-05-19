"""Phase 3 context-building layer.

ContextBuilder is the single hot-path call site that replaces the literal
``user_profile=""`` in Orchestrator.chat with the rendered profile string.
"""

from hrag.context.builder import ContextBuilder

__all__ = ["ContextBuilder"]
