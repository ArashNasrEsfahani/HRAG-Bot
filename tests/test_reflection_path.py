"""Phase 11 / 11.1 — reflective / opinion personal-question path.

Real-user bug: asked "what do you think about me?", the bot recited one saved
fact ("you work at KareOne") and offered to search the documents — the
``answer_personal_known.md`` script — instead of forming an impression. The
question is an opinion request, not a fact lookup.

Phase 11.1 makes detection switchable (``retrieval.reflection_mode`` =
off|regex|hybrid) and two-tier with asymmetric precision:
- ``is_reflective_strict`` — high precision, gates *coercion* of a non-PERSONAL
  verdict (protects "describe me a function").
- ``is_reflective_query`` — recall tier (strict OR two-factor self-ref × cue),
  used *within* an already-PERSONAL turn.
- hybrid mode adds an LLM yes/no fallback (``ReflectiveClassifier`` /
  combined-preflight ``reflective`` field) when the regex misses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from hrag.intent import (
    Intent,
    IntentVerdict,
    has_reflective_anchor,
    has_self_reference,
    is_reflective_query,
    is_reflective_strict,
)
from hrag.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# 1. Pure detector — strict (coercion-grade) tier
# ---------------------------------------------------------------------------


def test_strict_matches_unambiguous_opinion_shapes() -> None:
    positives = [
        "what do you think about me?",
        "so what do you think about me",
        "what do you think of me",
        "what's your opinion of me",
        "what is your honest impression of me",
        "your thoughts on me?",
        "how would you describe me",
        "how do you see me",
        "how do I come across",
        "describe me",
        "describe me.",
        "sum me up",
        "what kind of person am I",
    ]
    for q in positives:
        assert is_reflective_strict(q), q


def test_strict_rejects_describe_me_plus_object() -> None:
    """The bare 'describe me' branch must NOT fire on 'describe me a/the X' —
    that is a factual request and would be wrongly coerced to PERSONAL."""
    negatives = [
        "describe me a function that sorts a list",
        "describe me a sunset",
        "describe me the binary search algorithm",
        "describe me how transformers work",
        "what do you think about transformers",
        "what is RAG",
        "do you know me",
        "what do you know about me",
        "",
    ]
    for q in negatives:
        assert not is_reflective_strict(q), q


def test_strict_matches_farsi_and_finglish() -> None:
    assert is_reflective_strict("درباره من چی فکر می‌کنی؟")
    assert is_reflective_strict("نظرت در مورد من چیه")
    assert is_reflective_strict("nazaret darbare man chie")


def test_zwnj_normalization() -> None:
    """The ZWNJ and non-ZWNJ spellings of 'how do you see me' both match,
    even though only the joined form is listed."""
    assert is_reflective_strict("منو چطور می‌بینی")  # with ZWNJ
    assert is_reflective_strict("منو چطور میبینی")    # without ZWNJ


# ---------------------------------------------------------------------------
# 2. Pure detector — recall tier (two-factor) + self-reference helper
# ---------------------------------------------------------------------------


def test_recall_tier_adds_two_factor_matches() -> None:
    # strict shapes still pass …
    assert is_reflective_query("what do you think about me")
    # … plus looser two-factor (self-ref AND opinion-cue).
    assert is_reflective_query("sum me up")
    assert is_reflective_query("honestly, how do I come across to you")
    # No opinion cue OR no self-ref → recall tier stays off.
    assert not is_reflective_query("what do you think about transformers")
    assert not is_reflective_query("what is my name")


def test_has_self_reference() -> None:
    assert has_self_reference("describe me")
    assert has_self_reference("what is my GPA")
    assert has_self_reference("نظرت درباره من")
    assert not has_self_reference("what is RAG")
    assert not has_self_reference("describe a binary tree")


# ---------------------------------------------------------------------------
# 3. Prompt registry renders the synthesis template
# ---------------------------------------------------------------------------


def test_personal_reflect_template_renders() -> None:
    from hrag.prompts_registry import PromptRegistry

    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "hrag" / "prompts"
    assert (prompts_dir / "answer_personal_reflect.md").exists()

    registry = PromptRegistry(prompts_dir)
    rendered = registry.render_personal_reflect(
        user_profile="Facts: employer: KareOne",
        retrieved_memories="The user is named Arash.",
        retrieved_docs="Arash researches HCI and ML.",
        conversation_history="(no prior conversation)",
        question="what do you think about me?",
    )

    assert "Facts: employer: KareOne" in rendered
    assert "Arash researches HCI and ML." in rendered
    assert "what do you think about me?" in rendered
    assert "reflective impression" in rendered
    assert "offer a concrete next step" not in rendered


# ---------------------------------------------------------------------------
# Orchestrator wiring helpers (self-contained stubs)
# ---------------------------------------------------------------------------


class _ScriptedClassifier:
    def __init__(self, intent: Intent) -> None:
        self._intent = intent

    def classify(self, text: str, **kwargs) -> IntentVerdict:
        return IntentVerdict(
            intent=self._intent, confidence=1.0, source="test",
            raw_label=self._intent.value,
        )


class _RecordingRetriever:
    name = "recording"

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.queries: list[str] = []

    def retrieve(self, query, user_id, top_k=10, source_types=None,
                 intent_hint=None, where=None):
        self.queries.append(query)
        return list(self._results)


class _PromptCapturingLLM:
    """Records prompts; answers 'yes' to the reflective-check prompt."""

    name = "prompt-capture"

    def __init__(self, reflective_reply: str = "yes") -> None:
        self.prompts: list[str] = []
        self._reflective_reply = reflective_reply
        self.reflective_calls = 0

    def generate(self, request):
        from hrag.types import GenerationResponse
        text = " ".join(m.content for m in request.messages)
        self.prompts.append(text)
        return GenerationResponse(text="ok", raw=None)

    def complete(self, prompt, system=None, temperature=None, max_tokens=None):
        self.prompts.append(prompt)
        if "reflective / opinion request" in prompt or "reflective/opinion" in prompt:
            self.reflective_calls += 1
            return self._reflective_reply
        if "Intent Classification" in prompt or "Output (one word only)" in prompt:
            return "personal"
        if "Score:" in prompt or "0-3" in prompt:
            return "2"
        return "ok"

    def generate_stream(self, request):
        text = " ".join(m.content for m in request.messages)
        self.prompts.append(text)
        yield "ok"


def _chunk(chunk_id: str, source_type: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=f"doc_{chunk_id}", user_id="default",
        text=text, embedding_text=text, title=f"Title-{chunk_id}",
        section="Section", source_type=source_type,
    )


def _result(chunk_id: str, score: float, source_type: str, text: str,
            rerank_score: Optional[float] = None) -> RetrievalResult:
    return RetrievalResult(
        chunk=_chunk(chunk_id, source_type, text), score=score,
        rerank_score=rerank_score,
    )


def _make_orch(sample_config, results, classifier_intent: Intent,
               reflection_mode: str = "hybrid", reflective_reply: str = "yes"):
    import hrag.db.connection as _conn_mod

    _conn_mod._db_singleton = None
    sample_config.retrieval.rerank_enabled = False
    sample_config.retrieval.reflection_mode = reflection_mode

    from hrag.orchestrator import Orchestrator

    orch = Orchestrator(sample_config)
    llm = _PromptCapturingLLM(reflective_reply=reflective_reply)
    orch.llm = llm  # type: ignore[assignment]
    if getattr(orch, "gate", None) is not None:
        orch.gate.llm = llm
    if getattr(orch, "clue", None) is not None:
        orch.clue.llm = llm
    orch.intent_classifier = _ScriptedClassifier(classifier_intent)  # type: ignore[assignment]
    retr = _RecordingRetriever(results)
    orch.retriever = retr  # type: ignore[assignment]
    return orch, llm, retr


def _answer_prompt(llm: _PromptCapturingLLM) -> str:
    gen = [
        p for p in llm.prompts
        if "Intent Classification" not in p
        and "Output (one word only)" not in p
        and "Score:" not in p
        and "0-3" not in p
        and "reflective / opinion request" not in p
    ]
    assert gen, "no generation prompt captured"
    return gen[-1]


def _teardown(orch) -> None:
    orch.close()
    import hrag.db.connection as _conn_mod
    _conn_mod._db_singleton = None


# ---------------------------------------------------------------------------
# 4. Orchestrator routes reflective PERSONAL turns to the synthesis prompt
# ---------------------------------------------------------------------------


def test_reflective_turn_uses_synthesis_prompt(sample_config) -> None:
    """PERSONAL + reflective question → answer_personal_reflect.md."""
    ep = _result("c1", 0.9, "episodic", "The user works at KareOne.")
    orch, llm, _ = _make_orch(sample_config, [ep], Intent.PERSONAL,
                              reflection_mode="regex")
    try:
        orch.chat("what do you think about me?", user_id="default")
    finally:
        _teardown(orch)

    prompt = _answer_prompt(llm)
    assert "reflective impression" in prompt
    assert "honestly admit the limit" not in prompt


def test_strict_coerces_misclassified_factual(sample_config) -> None:
    """Classifier returns FACTUAL but the question is unambiguously reflective
    → strict regex coerces to PERSONAL → synthesis prompt. (regex mode, no LLM)"""
    ep = _result("c1", 0.9, "episodic", "The user works at KareOne.")
    orch, llm, _ = _make_orch(sample_config, [ep], Intent.FACTUAL,
                              reflection_mode="regex")
    try:
        orch.chat("how would you describe me?", user_id="default")
    finally:
        _teardown(orch)

    prompt = _answer_prompt(llm)
    assert "reflective impression" in prompt


def test_factual_with_describe_me_object_not_coerced(sample_config) -> None:
    """ASYMMETRIC PRECISION: 'describe me a function' is FACTUAL and must NOT
    be coerced — the strict tier rejects it, so no synthesis prompt."""
    doc = _result("c1", 0.9, "document", "def quicksort(): ...")
    orch, llm, _ = _make_orch(sample_config, [doc], Intent.FACTUAL,
                              reflection_mode="regex")
    try:
        orch.chat("describe me a function that sorts a list", user_id="default")
    finally:
        _teardown(orch)

    prompt = _answer_prompt(llm)
    assert "reflective impression" not in prompt


def test_reflection_off_is_noop(sample_config) -> None:
    """mode='off' → no synthesis prompt, no reflective LLM call."""
    ep = _result("c1", 0.9, "episodic", "The user works at KareOne.")
    orch, llm, _ = _make_orch(sample_config, [ep], Intent.PERSONAL,
                              reflection_mode="off")
    try:
        orch.chat("what do you think about me?", user_id="default")
    finally:
        _teardown(orch)

    prompt = _answer_prompt(llm)
    assert "reflective impression" not in prompt
    assert llm.reflective_calls == 0


def test_regex_mode_never_calls_llm_judge(sample_config) -> None:
    """mode='regex' must never consult the ReflectiveClassifier."""
    ep = _result("c1", 0.9, "episodic", "The user works at KareOne.")
    orch, llm, _ = _make_orch(sample_config, [ep], Intent.PERSONAL,
                              reflection_mode="regex")
    try:
        # A question the regex tiers MISS (no opinion cue token), so hybrid
        # would fall to the LLM — but regex mode must not.
        orch.chat("tell me your overall vibe and read on me as a human",
                  user_id="default")
    finally:
        _teardown(orch)
    assert llm.reflective_calls == 0


def test_hybrid_uses_llm_when_regex_misses(sample_config) -> None:
    """mode='hybrid': a reflective-but-unusual phrasing the regex misses is
    rescued by the LLM yes/no judge → synthesis prompt."""
    ep = _result("c1", 0.9, "episodic", "The user works at KareOne.")
    # "what's your gut feeling on who I am" — has self-ref ('i', 'your') but
    # no opinion-cue token in the recall regex, and not a strict phrase.
    q = "what's your gut feeling on who I am as a person"
    assert not is_reflective_query(q)  # regex genuinely misses
    orch, llm, _ = _make_orch(sample_config, [ep], Intent.PERSONAL,
                              reflection_mode="hybrid", reflective_reply="yes")
    try:
        orch.chat(q, user_id="default")
    finally:
        _teardown(orch)

    prompt = _answer_prompt(llm)
    assert llm.reflective_calls >= 1
    assert "reflective impression" in prompt


# ---------------------------------------------------------------------------
# 5. Corroboration anchor — a lone weak LLM "yes" can't hijack a neutral message
#    Regression for: "Ok i want to test you" → a fabricated self-portrait built
#    from misattributed document chunks.
# ---------------------------------------------------------------------------


def test_anchor_rejects_neutral_and_bare_subject_i() -> None:
    """Bare subject 'i' and neutral meta-statements carry no reflective anchor."""
    negatives = [
        "Ok i want to test you",      # the reported failure
        "let's start the test",
        "can you help me with this",  # 'help me' is imperative-object, stripped
        "i want to ask you something",
        "what is RAG",
        "summarise this paper",
        "",
    ]
    for q in negatives:
        assert not has_reflective_anchor(q), q


def test_anchor_accepts_genuine_reflective_phrasings() -> None:
    """Opinion-cue, object/possessive self-ref, or user-as-subject phrasings
    all anchor — including ones the recall regex misses."""
    positives = [
        "what do you think about me",          # opinion cue + me
        "describe me",                          # opinion cue
        "how would you describe me",
        "what's your honest read on who I am as a person",  # read + who i am
        "what's your gut feeling on who I am as a person",  # subject-only anchor
        "tell me about myself",                 # myself survives 'tell me' strip
        "نظرت درباره من",                       # FA strong self-ref
    ]
    for q in positives:
        assert has_reflective_anchor(q), q


def test_neutral_message_not_coerced_despite_llm_yes(sample_config) -> None:
    """The reported bug: 'Ok i want to test you' with a stray LLM 'yes' must NOT
    become a reflective self-portrait. No anchor → the judge is never even
    consulted → no coercion, no synthesis prompt — for BOTH a FACTUAL base
    verdict (coercion path) and a PERSONAL one (within-PERSONAL path)."""
    doc = _result("c1", 0.9, "document",
                  "And the king sought to control the child from its birth.")
    for base in (Intent.FACTUAL, Intent.PERSONAL):
        orch, llm, _ = _make_orch(sample_config, [doc], base,
                                  reflection_mode="hybrid", reflective_reply="yes")
        try:
            orch.chat("Ok i want to test you", user_id="default")
        finally:
            _teardown(orch)
        prompt = _answer_prompt(llm)
        assert "reflective impression" not in prompt, base
        assert llm.reflective_calls == 0, base


def test_anchored_reflective_still_coerces_via_llm_in_hybrid(sample_config) -> None:
    """A genuine reflective question the strict regex misses, but which carries
    an anchor, is still rescued by the LLM judge and coerces FACTUAL → reflective
    synthesis — the corroboration gate must not regress real recall."""
    ep = _result("c1", 0.9, "episodic", "The user works at KareOne.")
    q = "what's your honest read on who I am as a person"
    assert not is_reflective_strict(q)   # strict genuinely misses ("on who I am")
    assert has_reflective_anchor(q)      # but it anchors
    orch, llm, _ = _make_orch(sample_config, [ep], Intent.FACTUAL,
                              reflection_mode="hybrid", reflective_reply="yes")
    try:
        orch.chat(q, user_id="default")
    finally:
        _teardown(orch)
    prompt = _answer_prompt(llm)
    assert llm.reflective_calls >= 1
    assert "reflective impression" in prompt


def test_reflect_prompt_warns_against_document_misattribution() -> None:
    """Layer A: the synthesis template must explicitly frame document excerpts
    as NOT facts about the user, so the model can't recast a document's story
    (a king, a birth) as the user's biography."""
    from hrag.prompts_registry import PromptRegistry

    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "hrag" / "prompts"
    registry = PromptRegistry(prompts_dir)
    rendered = registry.render_personal_reflect(
        user_profile="(no profile yet)",
        retrieved_memories="(nothing saved yet)",
        retrieved_docs="And the king sought to control the child from its birth.",
        conversation_history="(no prior conversation)",
        question="Ok i want to test you",
    )
    low = rendered.lower()
    assert "not facts about the user" in low or "not the user" in low
    assert "biography" in low


# ---------------------------------------------------------------------------
# 6. Layer A hardening — document content can't be recited as the user's bio.
#    Even if a reflective turn DOES fire, an unrelated document chunk (a novel's
#    plot) never reaches the synthesis prompt, so a weak model can't misattribute
#    it. Model-independent guard.
# ---------------------------------------------------------------------------


def test_user_identifying_terms_and_about_user() -> None:
    from hrag.orchestrator import _chunk_is_about_user, _user_identifying_terms

    assert _user_identifying_terms("(no profile yet)") == set()
    assert _user_identifying_terms("") == set()
    terms = _user_identifying_terms("Facts about you: employer KareOne; researches HRAG")
    assert "kareone" in terms and "hrag" in terms
    assert "about" not in terms and "your" not in terms  # stopworded
    # A Red Book chunk mentions none of the user's distinctive terms.
    assert not _chunk_is_about_user("And the king sought to control the child.", terms)
    # A chunk mentioning the employer IS about the user.
    assert _chunk_is_about_user("Arash works at KareOne on retrieval.", terms)
    # Empty terms (thin/no profile) ⇒ never about the user.
    assert not _chunk_is_about_user("anything at all", set())


def test_reflect_excludes_documents_not_about_user(sample_config) -> None:
    """A genuine reflective turn whose only retrieved doc is unrelated narrative
    (no user terms, thin profile) must keep that text OUT of the synthesis
    prompt — the actual fix for the Red-Book-as-biography failure."""
    red_book = _result("c1", 0.9, "document",
                       "And the king sought to control the child from its birth.")
    orch, llm, _ = _make_orch(sample_config, [red_book], Intent.PERSONAL,
                              reflection_mode="regex")
    try:
        orch.chat("what do you think about me?", user_id="default")
    finally:
        _teardown(orch)

    prompt = _answer_prompt(llm)
    assert "reflective impression" in prompt            # reflect template used
    assert "the king sought to control" not in prompt   # doc content excluded
    assert "no personal documents on file" in prompt
