"""Phase 8.3 — "be a friend, not a search engine" tests.

Three bugs from a real user session, fixed together:

1. The intent classifier ignored prior turns. A follow-up like
   "what about Mahmoud?" right after a PERSONAL turn that surfaced Mahmoud
   should classify as PERSONAL, not FACTUAL. Fix: pass the last 2 exchanges
   into the LLM prompt, AND add a pre-LLM fast path keyed on the previous
   turn's intent + entities the prior memories surfaced.

2. The review pause fired on score_floor even when the bot already had a
   strong episodic memory hit. Fix: in ``should_pause``, when intent is
   PERSONAL and at least one episodic result has ``score >= 0.10``, return
   ``[]`` immediately — the modal would have nothing useful to add.

3. When the bot had a memory but no doc match, the answer was a flat
   "I know X" reply. Fix: a new ``answer_personal_known.md`` template that
   leads with the fact, admits the limit, and offers to dig further. Routed
   to when intent is PERSONAL AND at least one episodic memory was returned
   AND no document chunk earned a positive rerank score.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from hrag.intent import (
    ENTITY_TRACKER_CAP,
    Intent,
    IntentClassifier,
    IntentVerdict,
    extract_entities,
)
from hrag.interaction.review import should_pause
from hrag.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# LLM stub that records every prompt it sees
# ---------------------------------------------------------------------------


class _PromptRecorderLLM:
    """LLM stub that records every prompt and returns a fixed reply.

    The intent classifier is the only LLM consumer the tests in this file
    care about. It calls ``complete(prompt, temperature, max_tokens)``.
    """

    name = "prompt-recorder"

    def __init__(self, reply: str = "factual") -> None:
        self.prompts: list[str] = []
        self.reply = reply

    def complete(
        self,
        prompt: str,
        system=None,
        temperature=None,
        max_tokens=None,
    ) -> str:
        self.prompts.append(prompt)
        return self.reply

    def generate(self, request):
        from hrag.types import GenerationResponse

        text = " ".join(m.content for m in request.messages)
        self.prompts.append(text)
        return GenerationResponse(text=self.reply, raw=None)

    def generate_stream(self, request):
        yield self.reply


# ---------------------------------------------------------------------------
# Test 1 — follow-up entity fast path on the classifier
# ---------------------------------------------------------------------------


def test_personal_followup_with_entity_stays_personal() -> None:
    """Previous turn surfaced "Mahmoud" from memory; next user message
    mentions Mahmoud → classifier short-circuits to PERSONAL via the
    fast path with NO LLM call."""
    llm = _PromptRecorderLLM(reply="factual")
    clf = IntentClassifier(llm=llm)

    # The fast-path branch must run BEFORE the LLM cache + LLM call. We
    # confirm by asserting the LLM was never invoked.
    verdict = clf.classify(
        "what about Mahmoud's work?",
        history=[
            ("user", "who am I?"),
            ("assistant", "You are Arash, and Mahmoud is your friend."),
        ],
        prev_intent=Intent.PERSONAL,
        prev_memory_entities={"Mahmoud"},
    )

    assert verdict.intent == Intent.PERSONAL
    assert verdict.source == "follow_up_entity"
    assert verdict.confidence == pytest.approx(0.9)
    # The LLM must NOT have been called — this is the cheap fast path.
    assert llm.prompts == []


def test_follow_up_does_not_fire_when_prev_intent_is_factual() -> None:
    """The follow-up fast path requires the previous turn to be PERSONAL.
    If the previous turn was FACTUAL, mentioning a tracked token does
    NOT short-circuit to PERSONAL — the LLM path is taken instead."""
    llm = _PromptRecorderLLM(reply="factual")
    clf = IntentClassifier(llm=llm)

    verdict = clf.classify(
        "show me more about HippoRAG and its citations",  # not a fast-path shape
        history=[("user", "what is HippoRAG?"), ("assistant", "HippoRAG is …")],
        prev_intent=Intent.FACTUAL,
        prev_memory_entities={"HippoRAG"},
    )

    # The follow-up fast path is NOT engaged; the LLM was consulted.
    assert verdict.source != "follow_up_entity"
    assert len(llm.prompts) == 1


# ---------------------------------------------------------------------------
# Test 2 — LLM prompt carries the conversation_history slot
# ---------------------------------------------------------------------------


def test_intent_classifier_uses_history() -> None:
    """When the fast paths are inconclusive, the LLM prompt must contain the
    rendered history block so the prompt-side "follow-up about a person
    already framed as personal" rule has something to chew on."""
    llm = _PromptRecorderLLM(reply="personal")
    clf = IntentClassifier(llm=llm)

    history = [
        ("user", "who am I?"),
        ("assistant", "You are Arash, and Mahmoud is your friend."),
    ]
    # Use a query that does NOT match the fast-path personal phrases,
    # the factual openers, or the greeting vocab — so the LLM path runs.
    # AND make sure no tracked entity appears in the query, so the
    # follow-up fast path stays off.
    verdict = clf.classify(
        "could you elaborate on the relationship",
        history=history,
        prev_intent=Intent.PERSONAL,
        prev_memory_entities={"SomeoneElse"},
    )

    assert verdict.source == "llm"
    assert llm.prompts, "the LLM should have been called"
    sent = llm.prompts[0]
    # The history block must be in the prompt — sentinel lines and the
    # rendered slot header.
    assert "Recent conversation" in sent
    assert "You are Arash" in sent
    assert "Mahmoud is your friend" in sent


# ---------------------------------------------------------------------------
# Test 3 / 4 — review pause behaviour
# ---------------------------------------------------------------------------


def _review_cfg(**overrides) -> SimpleNamespace:
    defaults = dict(
        review_enabled=True,
        review_mode="smart_auto",
        review_score_floor=-3.0,
        review_ambiguity_delta=0.4,
        review_branch_threshold=2,
        review_timeout_s=90.0,
        rephrasings_enabled=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _chunk(chunk_id: str, source_type: str = "episodic", text: str = "memory") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        user_id="default",
        text=text,
        embedding_text=text,
        title=f"Title-{chunk_id}",
        section="Section",
        source_type=source_type,
    )


def _result(
    chunk_id: str,
    score: float,
    rerank_score: Optional[float] = None,
    source_type: str = "episodic",
    text: str = "memory",
) -> RetrievalResult:
    return RetrievalResult(
        chunk=_chunk(chunk_id, source_type=source_type, text=text),
        score=score,
        rerank_score=rerank_score,
    )


def _verdict(intent_value: str) -> SimpleNamespace:
    return SimpleNamespace(intent=SimpleNamespace(value=intent_value), confidence=0.9)


def test_review_skips_when_personal_with_strong_episodic() -> None:
    """A PERSONAL turn with at least one episodic result at score>=0.10
    must NOT trigger the modal, even when other triggers (score_floor)
    would otherwise fire."""
    # rerank_score -9.0 ≪ score_floor -3.0 → score_floor would fire …
    results = [_result("c1", score=0.42, rerank_score=-9.0, source_type="episodic")]

    fired = should_pause(
        cfg=_review_cfg(),
        results=results,
        descend=None,
        intent_verdict=_verdict("personal"),
        router_label=None,
        factual_general_swap_imminent=False,
    )
    # …but the PERSONAL+episodic early-return kills the pause.
    assert fired == []


def test_review_still_pauses_when_personal_no_memory() -> None:
    """Same low-score setup but the only result is a document chunk
    (no episodic). The early-return must NOT fire — score_floor still
    triggers the modal."""
    results = [_result("c1", score=0.42, rerank_score=-9.0, source_type="document")]

    fired = should_pause(
        cfg=_review_cfg(),
        results=results,
        descend=None,
        intent_verdict=_verdict("personal"),
        router_label=None,
        factual_general_swap_imminent=False,
    )
    # score_floor must still appear in the reasons list.
    from hrag.interaction.review import PauseReason

    assert PauseReason.SCORE_FLOOR in fired


def test_review_skip_threshold_excludes_weak_episodic() -> None:
    """An episodic memory with score < 0.10 should NOT be treated as
    'strong' — the review pause must still fire when other triggers do."""
    results = [_result("c1", score=0.05, rerank_score=-9.0, source_type="episodic")]

    fired = should_pause(
        cfg=_review_cfg(),
        results=results,
        descend=None,
        intent_verdict=_verdict("personal"),
        router_label=None,
        factual_general_swap_imminent=False,
    )
    from hrag.interaction.review import PauseReason

    assert PauseReason.SCORE_FLOOR in fired


# ---------------------------------------------------------------------------
# Test 5 / 6 — orchestrator routes to the right template
# ---------------------------------------------------------------------------


class _ScriptedClassifier:
    def __init__(self, intent: Intent) -> None:
        self._intent = intent

    def classify(self, text: str, **kwargs) -> IntentVerdict:
        return IntentVerdict(
            intent=self._intent,
            confidence=1.0,
            source="test",
            raw_label=self._intent.value,
        )


class _RecordingRetriever:
    name = "recording"

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    def retrieve(
        self,
        query,
        user_id,
        top_k=10,
        source_types=None,
        intent_hint=None,
        where=None,
    ):
        return list(self._results)


class _PromptCapturingLLM:
    name = "prompt-capture"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, request):
        from hrag.types import GenerationResponse

        text = " ".join(m.content for m in request.messages)
        self.prompts.append(text)
        return GenerationResponse(text="ok", raw=None)

    def complete(self, prompt, system=None, temperature=None, max_tokens=None):
        self.prompts.append(prompt)
        if "Intent Classification" in prompt or "Output (one word only)" in prompt:
            return "personal"
        if "Score:" in prompt or "0-3" in prompt or "0, 1, 2, or 3" in prompt:
            return "2"
        return "ok, got it."

    def generate_stream(self, request):
        text = " ".join(m.content for m in request.messages)
        self.prompts.append(text)
        yield "ok"


def _make_orch(sample_config, results: list[RetrievalResult]):
    """Build an Orchestrator with PERSONAL classifier + recording retriever."""
    import hrag.db.connection as _conn_mod

    _conn_mod._db_singleton = None
    sample_config.retrieval.rerank_enabled = False

    from hrag.orchestrator import Orchestrator

    orch = Orchestrator(sample_config)
    llm = _PromptCapturingLLM()
    orch.llm = llm  # type: ignore[assignment]
    if getattr(orch, "gate", None) is not None:
        orch.gate.llm = llm
    if getattr(orch, "clue", None) is not None:
        orch.clue.llm = llm
    orch.intent_classifier = _ScriptedClassifier(Intent.PERSONAL)  # type: ignore[assignment]
    orch.retriever = _RecordingRetriever(results)  # type: ignore[assignment]
    return orch, llm


def _answer_prompt(llm: _PromptCapturingLLM) -> str:
    """Return the last generation prompt (filter out intent + rerank)."""
    gen_prompts = [
        p
        for p in llm.prompts
        if "Intent Classification" not in p
        and "Output (one word only)" not in p
        and "Score:" not in p
        and "0-3" not in p
    ]
    assert gen_prompts, "no generation prompt was captured"
    return gen_prompts[-1]


def test_orchestrator_routes_to_known_template(sample_config) -> None:
    """PERSONAL + 1 episodic + 0 strong doc → the rendered prompt comes
    from ``answer_personal_known.md`` (memory-led friendly template)."""
    ep = _result("c1", score=0.9, source_type="episodic", text="Mahmoud is the user's friend.")
    orch, llm = _make_orch(sample_config, results=[ep])
    try:
        orch.chat("what about Mahmoud?", user_id="default")
    finally:
        orch.close()
        import hrag.db.connection as _conn_mod

        _conn_mod._db_singleton = None

    prompt = _answer_prompt(llm)
    # Sentinels from the new memory-led template.
    assert "memory-led" in prompt
    assert "honestly admit the limit" in prompt
    # The main personal template's distinctive heading must NOT be here.
    assert "What you've remembered about the user" not in prompt


def test_orchestrator_keeps_main_template_when_strong_doc(sample_config) -> None:
    """PERSONAL + 1 episodic + 1 doc with rerank>0 → the orchestrator stays
    on the main ``answer_personal.md`` template (memory-led path requires
    NO strong doc)."""
    ep = _result("c1", score=0.9, source_type="episodic")
    doc = _result("c2", score=0.85, rerank_score=2.5, source_type="document")
    orch, llm = _make_orch(sample_config, results=[ep, doc])
    try:
        orch.chat("what about Mahmoud?", user_id="default")
    finally:
        orch.close()
        import hrag.db.connection as _conn_mod

        _conn_mod._db_singleton = None

    prompt = _answer_prompt(llm)
    # Main template sentinel — present.
    assert "What you've remembered about the user" in prompt
    # Memory-led template sentinel — absent.
    assert "memory-led" not in prompt


# ---------------------------------------------------------------------------
# Test 7 — entity extraction helper basics
# ---------------------------------------------------------------------------


def test_extract_entities_ascii_proper_nouns() -> None:
    """Capitalised ASCII proper nouns are picked up; lowercase/stopwords
    are not."""
    ents = extract_entities("The user's friend is Mahmoud and Arash is the user.")
    # The proper-noun tokens land.
    assert "Mahmoud" in ents
    assert "Arash" in ents
    # The lowercase tokens / stopwords don't.
    assert "user" not in ents
    assert "friend" not in ents
    # "The" is a stopword.
    assert "The" not in ents


def test_extract_entities_includes_unicode_names() -> None:
    """Persian-script names (no casing) are still extracted because
    non-ASCII multi-letter tokens pass the loose unicode word filter."""
    ents = extract_entities("محمود is the user's friend.")
    # "محمود" is non-ASCII and length >= 3 → kept.
    assert "محمود" in ents


def test_entity_tracker_cap_is_documented() -> None:
    """The cap constant is exposed so the orchestrator can import it
    without re-declaring; assert the publicly-documented value."""
    assert ENTITY_TRACKER_CAP == 50


# ---------------------------------------------------------------------------
# Test 8 — the new template file exists and renders
# ---------------------------------------------------------------------------


def test_personal_known_template_renders() -> None:
    """The new prompt file is present on disk AND the registry exposes
    ``render_personal_known`` that interpolates the four required slots."""
    from hrag.prompts_registry import PromptRegistry

    prompts_dir = (
        Path(__file__).resolve().parents[1] / "src" / "hrag" / "prompts"
    )
    # The file is on disk.
    assert (prompts_dir / "answer_personal_known.md").exists()

    registry = PromptRegistry(prompts_dir)
    rendered = registry.render_personal_known(
        retrieved_memories="Mahmoud is Arash's friend.",
        retrieved_docs_summary="(no relevant documents found)",
        conversation_history="User: who am I?\nAssistant: …",
        question="what about Mahmoud?",
    )

    # The four slots interpolate correctly.
    assert "Mahmoud is Arash's friend." in rendered
    assert "(no relevant documents found)" in rendered
    assert "what about Mahmoud?" in rendered
    # And the distinctive sentinel survives.
    assert "memory-led" in rendered
