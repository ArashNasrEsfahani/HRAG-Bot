"""Phase 2 smoke test — runs four queries (one per routing path) against a fully populated KG.

Captures the router classification, retriever name, and source diversity so
we can confirm:
  - entity queries   -> kg_ppr fires
  - global queries   -> community fires
  - cross_document   -> RRF fuses kg_ppr + community + vector with multi-doc sources
  - ambiguous        -> RRF fuses kg_ppr + vector

Run:  py tools/smoke_phase2.py
"""

from __future__ import annotations

import sys
from collections import Counter

from hrag.config import load_config
from hrag.orchestrator import Orchestrator


SMOKES: list[tuple[str, str]] = [
    ("entity", "What is the PPR damping factor in HippoRAG?"),
    ("global", "Summarize the main themes across all three papers."),
    ("cross_document", "How do HippoRAG and RAGate differ in their approach?"),
    ("ambiguous", "Tell me more about retrieval."),
]


def _events_collector():
    events: list[tuple[str, dict]] = []

    def cb(name: str, payload: dict) -> None:
        events.append((name, payload))

    return events, cb


def _doc_count(results) -> int:
    """Distinct doc_ids across the source chunks (ignores synthetic 'community::' rows)."""
    return len({r.chunk.doc_id for r in results if not r.chunk.chunk_id.startswith("community::")})


def main() -> int:
    cfg = load_config()
    print(f"config: retriever={cfg.retrieval.retriever}  kg.enabled={cfg.kg.enabled}  llm.model={cfg.llm.model}")
    if cfg.retrieval.retriever != "router":
        print("WARNING: retriever is not 'router'; smoke test less meaningful.")
    if not cfg.kg.enabled:
        print("WARNING: kg.enabled=False; KG retrievers will be inactive.")

    orch = Orchestrator(cfg)

    rc = 0
    for label, q in SMOKES:
        print()
        print("=" * 78)
        print(f"[{label}] {q}")
        print("=" * 78)

        events, cb = _events_collector()
        try:
            result = orch.chat(question=q, user_id=cfg.user.default_user_id, progress=cb, stream=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  CHAT ERROR: {exc}")
            rc = 1
            continue

        # Pull what we care about from the events
        retrieve_events = [p for n, p in events if n == "retrieve"]
        organize = next((p for n, p in events if n == "organize_done"), None)

        retr_names = Counter(r.retriever for r in result.sources)
        sources_doc_count = _doc_count(result.sources)
        community_count = sum(1 for r in result.sources if r.chunk.chunk_id.startswith("community::"))

        print(f"  retrievers fired : {dict(retr_names)}")
        print(f"  community rows   : {community_count}")
        print(f"  distinct docs    : {sources_doc_count}")
        if organize:
            print(f"  MST              : input={organize.get('input')}  output={organize.get('output')}  dropped={organize.get('dropped')}")
        print(f"  answer (first 240 chars):")
        print(f"    {result.answer[:240].strip()!r}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
