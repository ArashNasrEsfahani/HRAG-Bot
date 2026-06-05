"""Phase 13.1 — orchestrator-level tests for the agentic multi-stage deep read.

These reuse the stub pattern from ``test_orchestrator_review.py``: the
orchestrator is built against ``sample_config`` with the LLM swapped for a
scripted planner stub and the retriever swapped for a spy that feeds a
deterministic seed list. A small document (100 chunks → 10 parts) is seeded
directly into the temp SQLite DB so the deterministic ``read_part`` SQL path
has real rows to open.

What's covered:
  * a STRUCTURAL question ("how many chapters") routes into the agentic read;
  * ``read_part`` opens the EXACT chunk_index range of a chapter (no vector
    search) — the "go read chapter 7" proof;
  * the action-repeat guard redirects a re-read of an already-read part;
  * a WEAK one-pass FACTUAL turn escalates into the agentic read (and does NOT
    when the flag is off — the FACTUAL→GENERAL swap fires instead);
  * an episodic-only seed (no document) falls through to the one-pass path
    (Contract 58).
"""

from __future__ import annotations

import json


def _reset_db_singleton() -> None:
    import hrag.db.connection as _conn_mod

    _conn_mod._db_singleton = None


class _PlanLLM:
    """Scripted stub: pops a queued action dict for each deep-read plan call,
    returns a canned synthesis / followups / intent for the others."""

    name = "plan-stub"

    def __init__(self, plan_queue=None) -> None:
        self.plan_queue = list(plan_queue or [])
        self.calls: list[str] = []

    def _dispatch(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "Document map" in prompt:  # deep_read_pass.md (the action planner)
            if self.plan_queue:
                return json.dumps(self.plan_queue.pop(0))
            return json.dumps({"action": "answer"})
        if "Write a clear, cohesive answer" in prompt:  # deep_read_synthesize.md
            return "The document is organised into chapters. [synth]"
        if "follow-up" in prompt.lower():
            return "What is chapter 7 about?\nWho is Toni Wolff?\nWhat is in Liber Primus?"
        if "Intent Classification" in prompt or "Output (one word only)" in prompt:
            return "factual"
        return "one-pass answer body"

    def complete(self, prompt, system=None, temperature=None, max_tokens=None):
        return self._dispatch(prompt)

    def generate(self, request):
        from hrag.types import GenerationResponse

        prompt = " ".join(m.content for m in request.messages)
        return GenerationResponse(text=self._dispatch(prompt), raw=None)

    def generate_stream(self, request):
        yield self.generate(request).text


class _SpyRetriever:
    name = "spy"

    def __init__(self, results=None) -> None:
        self.calls: list[dict] = []
        self._results = list(results or [])

    def retrieve(self, query, user_id, top_k=10, source_types=None,
                 intent_hint=None, where=None):
        self.calls.append({"query": query, "top_k": top_k, "where": where})
        return list(self._results)


def _seed_result(i, score=0.5, doc_id="deepdoc", title="The Red Book"):
    from hrag.types import Chunk, RetrievalResult

    ch = Chunk(
        chunk_id=f"ch{i:04d}", doc_id=doc_id, user_id="default",
        text=f"Body text of chunk {i}.", embedding_text=f"Body {i}",
        title=title, section=f"Chapter {i // 10 + 1}", chunk_index=i,
        page=(i // 10) + 1,
    )
    return RetrievalResult(chunk=ch, score=score)


def _seed_doc(db, *, doc_id="deepdoc", user_id="default", n=100, title="The Red Book"):
    """Insert a user + document + n document chunks (10 parts × 10 chunks)."""
    with db.conn:
        db.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
        db.execute(
            "INSERT OR REPLACE INTO documents "
            "(doc_id, user_id, source_path, title, source_type) "
            "VALUES (?, ?, ?, ?, 'document')",
            (doc_id, user_id, f"/tmp/{doc_id}.pdf", title),
        )
        for i in range(n):
            chapter = f"Chapter {i // 10 + 1}"
            db.execute(
                "INSERT OR REPLACE INTO chunks "
                "(chunk_id, doc_id, user_id, text, title, section, subsection, "
                " chunk_index, token_count, source_type, page, chapter, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, '', ?, 20, 'document', ?, ?, '{}')",
                (f"ch{i:04d}", doc_id, user_id, f"Body text of chunk {i}. " * 4,
                 title, chapter, i, (i // 10) + 1, chapter),
            )


def _build_orch(sample_config, *, seed_results, plan_queue, floor=0.0):
    sample_config.retrieval.rerank_enabled = False
    sample_config.retrieval.reflection_mode = "off"   # no reflective LLM calls
    sample_config.intent.enabled = False              # force FACTUAL
    sample_config.intent.corpus_relevance_floor = floor
    _reset_db_singleton()
    from hrag.orchestrator import Orchestrator

    orch = Orchestrator(sample_config)
    llm = _PlanLLM(plan_queue)
    orch.llm = llm
    if orch.gate is not None:
        orch.gate.llm = llm
    if orch.clue is not None:
        orch.clue.llm = llm
    orch.retriever = _SpyRetriever(seed_results)
    _seed_doc(orch.db)
    return orch, orch.retriever, llm


def _run(orch, question, *, stream=True):
    events: list[tuple[str, dict]] = []
    try:
        out = orch.chat(
            question, user_id="default",
            progress=lambda n, p: events.append((n, p)), stream=stream,
        )
    finally:
        orch.close()
        _reset_db_singleton()
    return out, events


# ---------------------------------------------------------------------------
# 1. Structural routing
# ---------------------------------------------------------------------------

def test_structural_question_routes_to_deep_read(sample_config):
    orch, _spy, _llm = _build_orch(
        sample_config,
        seed_results=[_seed_result(0), _seed_result(1), _seed_result(2)],
        plan_queue=[{"action": "answer"}, {"action": "answer"}],
    )
    out, events = _run(orch, "how many chapters does this book have?")

    fired = {n for n, _ in events}
    assert "deep_read_start" in fired
    start = next(p for n, p in events if n == "deep_read_start")
    assert start["structural"] is True
    assert start["pages_available"] is True   # seeded chunks carry page numbers
    assert out.answer


# ---------------------------------------------------------------------------
# 2. read_part determinism — "go read chapter 7"
# ---------------------------------------------------------------------------

def test_read_part_opens_exact_chapter_range(sample_config):
    orch, _spy, _llm = _build_orch(
        sample_config,
        seed_results=[_seed_result(0), _seed_result(1)],
        plan_queue=[
            {"action": "read_part", "part_idx": 6, "note": "moving to chapter 7"},
            {"action": "answer"},
        ],
    )
    out, events = _run(orch, "tell me everything about the red book")

    src_ids = {r.chunk.chunk_id for r in out.sources}
    # Part 6 == chunk_index 60..69; chunks_per_pass=6 caps the read at 60..65.
    assert {f"ch{i:04d}" for i in range(60, 66)} <= src_ids
    # The read was deterministic (by chunk_index), NOT the spy's seed/search:
    assert "ch0050" not in src_ids   # part 5 untouched
    assert "ch0070" not in src_ids   # part 7 untouched
    # The narration carried the action + the opened chapter.
    passes = [p for n, p in events if n == "deep_read_pass"]
    assert passes[0]["action"] == "read_part"
    assert passes[0]["arg"]["idx"] == 6


# ---------------------------------------------------------------------------
# 3. Action-repeat guard
# ---------------------------------------------------------------------------

def test_read_part_repeat_guard_redirects(sample_config):
    orch, _spy, _llm = _build_orch(
        sample_config,
        seed_results=[_seed_result(0)],
        plan_queue=[
            {"action": "read_part", "part_idx": 3},
            {"action": "read_part", "part_idx": 3},   # already read → redirect
            {"action": "answer"},
        ],
    )
    _out, events = _run(orch, "walk me through the red book")

    passes = [p for n, p in events if n == "deep_read_pass"]
    assert passes[0]["action"] == "read_part" and passes[0]["arg"]["idx"] == 3
    # Second request for part 3 is redirected to a different (unread) part.
    assert passes[1]["action"] == "read_part" and passes[1]["arg"]["idx"] != 3


# ---------------------------------------------------------------------------
# 4. Weak-one-pass escalation
# ---------------------------------------------------------------------------

def test_weak_one_pass_escalates_to_deep_read(sample_config):
    orch, _spy, _llm = _build_orch(
        sample_config,
        seed_results=[_seed_result(0, score=0.05), _seed_result(1, score=0.04)],
        plan_queue=[{"action": "answer"}, {"action": "answer"}],
        floor=0.5,   # top_score 0.05 < 0.5 → weak
    )
    # Non-broad, non-structural FACTUAL question → reaches the one-pass swap hook.
    out, events = _run(orch, "what did jung write regarding the folio")

    fired = {n for n, _ in events}
    assert "deep_read_start" in fired
    start = next(p for n, p in events if n == "deep_read_start")
    assert start["escalated"] is True
    assert out.answer


def test_weak_escalation_disabled_falls_to_general_swap(sample_config):
    orch, _spy, _llm = _build_orch(
        sample_config,
        seed_results=[_seed_result(0, score=0.05)],
        plan_queue=[],
        floor=0.5,
    )
    sample_config.deep_read.escalate_on_weak_answer = False
    _out, events = _run(orch, "what did jung write regarding the folio")

    fired = {n for n, _ in events}
    assert "deep_read_start" not in fired
    routes = [p for n, p in events if n == "intent_route"]
    assert any(p.get("swapped_from") == "factual" for p in routes)


# ---------------------------------------------------------------------------
# 5. Fall-through when no document matches (Contract 58)
# ---------------------------------------------------------------------------

def test_no_document_falls_through_to_one_pass(sample_config):
    from hrag.types import Chunk, RetrievalResult

    epi = Chunk(
        chunk_id="e1", doc_id="epidoc", user_id="default",
        text="user enjoys hiking", embedding_text="user enjoys hiking",
        title="memory", section="", chunk_index=0, source_type="episodic",
    )
    orch, _spy, _llm = _build_orch(
        sample_config,
        seed_results=[RetrievalResult(chunk=epi, score=0.9)],
        plan_queue=[],
    )
    out, events = _run(orch, "how many chapters does this book have?")

    fired = {n for n, _ in events}
    assert "deep_read_start" not in fired   # episodic-only → no document → fall through
    assert out.answer                       # one-pass still produced an answer
