"""Unit tests for hrag.intent — IntentClassifier + IntentVerdict."""

from __future__ import annotations

import pytest

from hrag.intent import Intent, IntentClassifier


# ---------------------------------------------------------------------------
# Local stubs — kept in this file to avoid touching tests/conftest.py
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Configurable stub that records every prompt it receives."""

    name = "fake_intent"

    def __init__(self, output: str = "factual") -> None:
        self.output = output
        self.calls: list[str] = []

    def complete(self, prompt: str, *, temperature=None, max_tokens=None) -> str:
        self.calls.append(prompt)
        return self.output


class _RaisingLLM:
    """Stub whose complete() always raises."""

    name = "raising"

    def complete(self, prompt: str, *, temperature=None, max_tokens=None) -> str:
        raise RuntimeError("ollama unavailable")


# ---------------------------------------------------------------------------
# Parametrized fast-path cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, expected_intent",
    [
        # --- GREETING (single-word, repeated chars, punctuation) ---
        ("hey",           Intent.GREETING),
        ("Hey",           Intent.GREETING),
        ("HEY!",          Intent.GREETING),
        ("heeey",         Intent.GREETING),   # char-collapse: heeey → hey
        ("HEEEEY!!",      Intent.GREETING),   # char-collapse + outer punct strip
        ("yo",            Intent.GREETING),
        ("yooo",          Intent.GREETING),
        ("yo what's up",  Intent.GREETING),   # 3 tokens, all in vocab
        ("thanks",        Intent.GREETING),
        ("thaaanks",      Intent.GREETING),   # char-collapse: thaaanks → thanks
        ("good morning",  Intent.GREETING),
        ("ok",            Intent.GREETING),
        ("ok cool",       Intent.GREETING),
        # --- PERSONAL (identity phrases beat greeting vocab) ---
        ("what's my name",             Intent.PERSONAL),
        ("What's my name?",            Intent.PERSONAL),
        ("who am I",                   Intent.PERSONAL),
        ("do you know me",             Intent.PERSONAL),
        ("do you remember me",         Intent.PERSONAL),
        ("tell me about myself",       Intent.PERSONAL),
        ("what do you know about me",  Intent.PERSONAL),
        ("hey, do you remember me",    Intent.PERSONAL),  # personal beats greeting
        # --- UNCLEAR (immediate, no LLM call) ---
        ("",    Intent.UNCLEAR),
        ("   ", Intent.UNCLEAR),
        ("?",   Intent.UNCLEAR),
        (".",   Intent.UNCLEAR),
    ],
)
def test_fast_path_classification(query: str, expected_intent: Intent) -> None:
    """Fast-path cases must resolve without an LLM call."""
    # llm=None is deliberate — if the fast path ever tries to call it,
    # we'll get an AttributeError and the test fails.
    clf = IntentClassifier(llm=None, fast_only=True)
    verdict = clf.classify(query)
    assert verdict.intent == expected_intent, (
        f"query={query!r}: expected {expected_intent}, got {verdict.intent}"
    )


# ---------------------------------------------------------------------------
# Source field checks
# ---------------------------------------------------------------------------


def test_fast_path_returns_correct_source() -> None:
    """When the fast path resolves, verdict.source must be 'fast_path'."""
    clf = IntentClassifier(llm=None, fast_only=True)
    for query in ("hey", "heeey", "what's my name", "", "?"):
        v = clf.classify(query)
        assert v.source == "fast_path", (
            f"query={query!r}: expected source='fast_path', got {v.source!r}"
        )


# ---------------------------------------------------------------------------
# Empty query never touches the LLM
# ---------------------------------------------------------------------------


def test_empty_query_skips_llm() -> None:
    """An empty string must be resolved by the fast path; LLM is never called."""
    llm = _FakeLLM("factual")
    clf = IntentClassifier(llm=llm)
    v = clf.classify("")

    assert llm.calls == [], "LLM must not be called for an empty query"
    assert v.intent == Intent.UNCLEAR
    assert v.source == "fast_path"


# ---------------------------------------------------------------------------
# fast_only=True with a non-greeting query
# ---------------------------------------------------------------------------


def test_fast_only_mode_returns_unclear_when_no_match() -> None:
    """fast_only=True: fast path returns None → UNCLEAR from fast_path, no LLM."""
    clf = IntentClassifier(llm=None, fast_only=True)
    # Declarative form (no interrogative opener) so fast-path FACTUAL regex
    # doesn't grab it. Genuinely ambiguous; only the LLM could resolve it.
    v = clf.classify("the moon is interesting i guess")

    assert v.intent == Intent.UNCLEAR
    assert v.confidence == pytest.approx(0.3)
    assert v.source == "fast_path"


# ---------------------------------------------------------------------------
# LLM fallback invoked for substantive queries
# ---------------------------------------------------------------------------


def test_llm_fallback_invoked_when_fast_path_returns_none() -> None:
    """A substantive query without a fast-path opener must reach the LLM."""
    llm = _FakeLLM("factual")
    clf = IntentClassifier(llm=llm)
    # No interrogative opener, not in greeting vocab — fast path returns None.
    v = clf.classify("the moon is interesting i guess")

    assert len(llm.calls) == 1, "LLM should have been called exactly once"
    assert v.intent == Intent.FACTUAL
    assert v.confidence > 0
    assert v.source == "llm"


# ---------------------------------------------------------------------------
# LLM output parsing — each of the four valid labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "llm_output, expected_intent",
    [
        # Exact lowercase
        ("greeting",  Intent.GREETING),
        ("personal",  Intent.PERSONAL),
        ("factual",   Intent.FACTUAL),
        ("unclear",   Intent.UNCLEAR),
        # With surrounding whitespace
        (" greeting\n", Intent.GREETING),
        ("\nfactual ",  Intent.FACTUAL),
        # Mixed case
        ("PERSONAL",   Intent.PERSONAL),
        ("Factual",    Intent.FACTUAL),
        # Extra trailing newline (common in some LLMs)
        ("factual\n",  Intent.FACTUAL),
        ("unclear\n",  Intent.UNCLEAR),
    ],
)
def test_llm_fallback_parses_each_label(llm_output: str, expected_intent: Intent) -> None:
    """The LLM output parser handles whitespace and casing variations."""
    llm = _FakeLLM(llm_output)
    clf = IntentClassifier(llm=llm)
    # "asdfgh" is not in any vocab so fast path returns None
    v = clf.classify("asdfgh")
    assert v.intent == expected_intent, (
        f"llm_output={llm_output!r}: expected {expected_intent}, got {v.intent}"
    )


# ---------------------------------------------------------------------------
# Unparseable LLM output → UNCLEAR/fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "garbage",
    [
        "???",
        "definitely not a label",
        "I am an LLM and I cannot classify",
        "42",
        "---",
    ],
)
def test_llm_fallback_returns_unclear_on_unparseable_output(garbage: str) -> None:
    """Garbage LLM output must produce UNCLEAR with source 'fallback'."""
    llm = _FakeLLM(garbage)
    clf = IntentClassifier(llm=llm)
    v = clf.classify("asdfgh")

    assert v.intent == Intent.UNCLEAR
    assert v.source == "fallback"


# ---------------------------------------------------------------------------
# LLM exception → UNCLEAR/fallback (no propagation)
# ---------------------------------------------------------------------------


def test_llm_failure_returns_unclear() -> None:
    """A crashing LLM must not propagate the exception; result is UNCLEAR/fallback."""
    clf = IntentClassifier(llm=_RaisingLLM())
    # Use a query that genuinely falls past the fast path so the LLM is
    # invoked and gets a chance to crash.
    v = clf.classify("the moon is interesting i guess")

    assert v.intent == Intent.UNCLEAR
    assert v.source == "fallback"


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


def test_cache_hit_skips_llm() -> None:
    """The second call with the same (normalized) query must not invoke the LLM."""
    llm = _FakeLLM("factual")
    clf = IntentClassifier(llm=llm)

    query = "asdfgh"
    v1 = clf.classify(query)
    v2 = clf.classify(query)

    assert len(llm.calls) == 1, "LLM should be called exactly once (cache hit on 2nd)"
    assert v1 == v2


def test_cache_keyed_by_normalized_form() -> None:
    """Two queries that normalize to the same string share one cache entry."""
    llm = _FakeLLM("factual")
    clf = IntentClassifier(llm=llm)

    # "asdfgh" and "ASDFGH" both normalize to "asdfgh" (lowercase)
    clf.classify("asdfgh")
    clf.classify("ASDFGH")

    # Only one LLM call because the second lookup is a cache hit
    assert len(llm.calls) == 1, (
        "Normalized queries should share a cache entry — LLM called more than once"
    )


def test_cache_hit_on_char_collapsed_repeat() -> None:
    """Queries that differ only in char-run length share a cache entry after collapse."""
    # "heeey" and "heeeeeey" both collapse to "hey"; "hey" hits the fast path
    # (GREETING), so neither calls the LLM.  The important check is that they
    # produce equal verdicts.
    clf = IntentClassifier(llm=None, fast_only=True)
    v1 = clf.classify("heeey")
    v2 = clf.classify("heeeeeey")

    assert v1.intent == v2.intent == Intent.GREETING
    assert v1 == v2


# ---------------------------------------------------------------------------
# GENERAL is never emitted by the classifier
# ---------------------------------------------------------------------------


def test_intent_general_is_never_emitted_by_classifier() -> None:
    """GENERAL is an orchestrator-side rewrite — the classifier must never emit it."""
    all_llm_outputs = ["greeting", "personal", "factual", "unclear"]
    fast_path_inputs = [
        "hey", "heeey", "HEEEEY!!", "yo", "yooo",
        "thanks", "thaaanks", "good morning", "ok cool",
        "what's my name", "who am I", "do you remember me",
        "", "   ", "?", ".",
    ]

    # Fast-path inputs (llm never reached)
    clf_fast = IntentClassifier(llm=None, fast_only=True)
    for query in fast_path_inputs:
        v = clf_fast.classify(query)
        assert v.intent != Intent.GENERAL, (
            f"fast-path query={query!r} produced GENERAL — must not happen"
        )

    # LLM-path outputs — one classifier per label to avoid cache collisions
    for label in all_llm_outputs:
        clf = IntentClassifier(llm=_FakeLLM(label))
        v = clf.classify("explain personalized pagerank")
        assert v.intent != Intent.GENERAL, (
            f"LLM output {label!r} produced GENERAL — must not happen"
        )


# ---------------------------------------------------------------------------
# hiii-there edge case (spec note: depends on whether "there" is in vocab)
# ---------------------------------------------------------------------------


def test_hiii_there_does_not_raise() -> None:
    """'hiii there' must not raise regardless of which intent it resolves to."""
    llm = _FakeLLM("greeting")
    clf = IntentClassifier(llm=llm)
    v = clf.classify("hiii there")
    # Accept GREETING (if "there" is in vocab) or whatever the LLM returns —
    # the key assertion is no exception and a valid Intent.
    assert v.intent in set(Intent)


# ---------------------------------------------------------------------------
# Stale-state hardening: fast-path FACTUAL openers, no-cache, versioned cache
# ---------------------------------------------------------------------------


def test_factual_opener_fast_path_what_is() -> None:
    """The exact bug that prompted this round: 'what is hipporag' must
    resolve FACTUAL on the fast path without any LLM call."""
    clf = IntentClassifier(llm=None, fast_only=True)
    v = clf.classify("what is hipporag?")
    assert v.intent == Intent.FACTUAL
    assert v.source == "fast_path"


def test_factual_opener_does_not_cannibalize_personal() -> None:
    """'what is my name' contains 'my name' (a personal phrase) — must stay
    PERSONAL because the personal-phrase check runs BEFORE factual openers."""
    clf = IntentClassifier(llm=None, fast_only=True)
    assert clf.classify("what is my name?").intent == Intent.PERSONAL
    assert clf.classify("tell me about myself").intent == Intent.PERSONAL
    assert clf.classify("who am I").intent == Intent.PERSONAL


def test_factual_opener_handles_various_shapes() -> None:
    """Sweep the major factual-opener shapes — all must be FACTUAL fast-path."""
    clf = IntentClassifier(llm=None, fast_only=True)
    for q in [
        "what is RAG",
        "what's HippoRAG",
        "explain personalized pagerank",
        "describe knowledge graphs",
        "tell me about transformers",
        "summarize the attention paper",
        "define embedding",
        "how does PPR work",
        "why are graph neural networks useful",
    ]:
        v = clf.classify(q)
        assert v.intent == Intent.FACTUAL, f"{q!r} → {v.intent.value}"
        assert v.source == "fast_path"


def test_short_interrogative_greeting_stays_greeting() -> None:
    """'what's up' / 'how are you' must stay GREETING, not get grabbed by
    the FACTUAL-opener regex."""
    clf = IntentClassifier(llm=None, fast_only=True)
    assert clf.classify("what's up").intent == Intent.GREETING
    assert clf.classify("how are you").intent == Intent.GREETING


def test_fast_path_verdicts_not_cached() -> None:
    """Fast-path verdicts must NOT be stored in the cache — only LLM verdicts.
    Without this, a source-level fix can't take effect in a long-lived
    server process."""
    clf = IntentClassifier(llm=None, fast_only=True)
    clf.classify("hey")
    clf.classify("what is RAG?")
    clf.classify("what's my name?")
    assert clf._cache == {}, f"Expected empty cache, got {clf._cache!r}"


def test_versioned_cache_rejects_stale_entries() -> None:
    """LLM verdicts stored under one cache version must be ignored after
    the version stamp changes — simulates a prompt-template edit."""
    llm = _FakeLLM("factual")
    clf = IntentClassifier(llm=llm)
    # First call: hits LLM, stores in cache.
    clf.classify("is the moon real")
    assert len(clf._cache) == 1
    assert len(llm.calls) == 1
    # Simulate a prompt edit that changes the cache version.
    clf._cache_version = "deadbeef0000"
    # Same query — must NOT return the stale entry; LLM is re-invoked.
    v = clf.classify("is the moon real")
    assert v.source == "llm"
    assert len(llm.calls) == 2, "LLM should have been re-invoked after version change"


def test_cache_version_is_short_hex_string() -> None:
    """Sanity: the cache version is a short hex slug usable in log lines."""
    clf = IntentClassifier(llm=None, fast_only=True)
    assert isinstance(clf._cache_version, str)
    assert len(clf._cache_version) == 12
    assert all(c in "0123456789abcdef" for c in clf._cache_version)


def test_boot_self_test_clean_on_default_classifier() -> None:
    """Default classifier's fast-path must pass every entry in the boot
    golden set."""
    clf = IntentClassifier(llm=None, fast_only=True)
    failures = clf.self_test()
    assert failures == [], (
        f"Default classifier failed self-test on {len(failures)} queries: "
        + "; ".join(f"{q!r} expected={e.value} got={a.value}" for q, e, a in failures)
    )


def test_boot_self_test_returns_failures_when_personal_phrases_break(monkeypatch) -> None:
    """If someone breaks _PERSONAL_PHRASES (e.g. by adding a too-broad
    'what is' entry), the self-test must surface the regression."""
    import hrag.intent as intent_mod
    # Add a too-broad phrase that would mis-match FACTUAL openers.
    bad_phrases = intent_mod._PERSONAL_PHRASES + ("what is",)
    monkeypatch.setattr(intent_mod, "_PERSONAL_PHRASES", bad_phrases)
    clf = IntentClassifier(llm=None, fast_only=True)
    failures = clf.self_test()
    # At least the "what is RAG" / "what is hipporag" entries should fail now.
    failed_queries = {q for q, _, _ in failures}
    assert "what is RAG" in failed_queries
    assert "what is hipporag" in failed_queries


# ---------------------------------------------------------------------------
# Discourse-marker prefix tolerance — "so what is X" must be FACTUAL
# ---------------------------------------------------------------------------


def test_discourse_prefix_factual() -> None:
    """The exact phrasing the user reported: 'so what is hipporag?' must
    resolve FACTUAL on the fast path."""
    clf = IntentClassifier(llm=None, fast_only=True)
    for q in [
        "so what is hipporag",
        "so what is hipporag?",
        "well, what is RAG",
        "and tell me about transformers",
        "hmm, what is pagerank",
        "btw, explain attention",
        "actually, summarize the paper",
        "alright, define embedding",
        "now describe knowledge graphs",
        "wait, how does PPR work",
    ]:
        v = clf.classify(q)
        assert v.intent == Intent.FACTUAL, f"{q!r} → {v.intent.value}"
        assert v.source == "fast_path"


def test_discourse_prefix_preserves_greeting() -> None:
    """Discourse prefixes must not convert greetings into factual queries."""
    clf = IntentClassifier(llm=None, fast_only=True)
    assert clf.classify("so what's up").intent == Intent.GREETING
    assert clf.classify("well, hey there").intent == Intent.GREETING
    assert clf.classify("so hey").intent == Intent.GREETING


def test_discourse_prefix_preserves_personal() -> None:
    """Discourse prefixes must not convert personal queries into factual."""
    clf = IntentClassifier(llm=None, fast_only=True)
    assert clf.classify("so what is my name").intent == Intent.PERSONAL
    assert clf.classify("well, tell me about myself").intent == Intent.PERSONAL
