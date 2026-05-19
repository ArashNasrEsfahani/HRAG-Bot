"""Hybrid intent classifier for HRAG-Bot.

Fast-path: normalized vocabulary lookup (deterministic, <1 ms). The fast path
covers four shapes: personal-phrase substring, short-interrogative greeting
("what's up"), factual openers ("what is X", "tell me about Y"), and pure
greeting vocab. **Fast-path verdicts are NOT cached** — they run on every
call (~20 µs) so source-level fixes (a new personal phrase, a tweak to the
factual-opener regex) take effect on the next message in the same long-lived
Streamlit process; no orchestrator rebuild needed.

LLM fallback: structured-output prompt when fast path is inconclusive. LLM
verdicts ARE cached, but the cache is **version-stamped** with a SHA-1 of
the prompt template + vocab sets; entries from a stale version are dropped
silently on lookup. This stops fixed verdicts from haunting long-lived
sessions.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from hrag.providers.llm import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-scope compiled regexes — never re-compiled per call
# ---------------------------------------------------------------------------

# Strip surrounding punctuation (leading and trailing)
_RE_STRIP_PUNCT = re.compile(r"^[.!?,:;\"'`~\-]+|[.!?,:;\"'`~\-]+$")

# Collapse 3+ consecutive identical characters to 1: "heeey" → "hey"
_RE_COLLAPSE = re.compile(r"(.)\1{2,}")

# Strip trailing punctuation embedded in tokens
_RE_TOKEN_TRAIL = re.compile(r"[.!?,:;\"'`~\-]+$")

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    GREETING = "greeting"
    PERSONAL = "personal"
    FACTUAL = "factual"
    UNCLEAR = "unclear"
    # GENERAL is never emitted by the classifier itself — the LLM only knows
    # the four labels above. The orchestrator's post-retrieval check rewrites
    # FACTUAL → GENERAL when the level-0 taxonomy max score is below the
    # corpus-relevance floor, meaning the question is substantive but the
    # local corpus has nothing about it (e.g. "where is Tehran?" against an
    # ML/AI library). GENERAL routes to answer_general.md, where the LLM
    # answers from world knowledge with a short caveat.
    GENERAL = "general"


@dataclass(frozen=True)
class IntentVerdict:
    intent: Intent
    confidence: float
    source: Literal["fast_path", "llm", "fallback", "named_topic"]
    raw_label: Optional[str] = None  # LLM literal output, for telemetry


# ---------------------------------------------------------------------------
# Vocabulary sets — defined once at module scope
# ---------------------------------------------------------------------------

_PERSONAL_PHRASES: tuple[str, ...] = (
    "my name",
    "who am i",
    "do you know me",
    "do you remember me",
    "remember my name",
    "tell me about myself",
    "what do you know about me",
)

_GREETING_VOCAB: frozenset[str] = frozenset(
    {
        "hi", "hey", "hello", "yo", "sup", "howdy",
        "thanks", "thank", "thx", "ty",
        "bye", "goodbye",
        "ok", "okay", "cool", "nice", "alright", "sure",
        "yeah", "yep", "yes", "nope", "nah",
        "good", "morning", "afternoon", "evening", "night", "day",
        "see", "you", "ya", "later", "farewell",
        "greetings", "you?", "up", "what's", "there",
    }
)

# Map LLM output words to Intent members
_LABEL_TO_INTENT: dict[str, Intent] = {
    "greeting": Intent.GREETING,
    "personal": Intent.PERSONAL,
    "factual":  Intent.FACTUAL,
    "unclear":  Intent.UNCLEAR,
}

# Discourse markers that can prefix a question without changing its intent.
# "so what is hipporag", "well, what's RAG", "and tell me about transformers"
# should all parse the same as the bare interrogative. Stripping these at the
# start lets the FACTUAL-opener regex below work uniformly.
_DISCOURSE_PREFIX_RE = re.compile(
    r"^(?:"
    r"so|well|and|now|but|then|hey|ok|okay|um|uh|hmm|"
    r"btw|actually|anyway|alright|wait|please|hmm+|"
    r"i\s+mean|like|y'?know"
    r")[\s,.!\-]+",
    re.IGNORECASE,
)

# Factual-opener fast-path. Matches the common interrogative + definitional
# shape ("what is X", "explain Y", "summarize Z", …) where X is NOT the user.
# The negative-lookahead blocks me/myself/my-name/I/I'm — those go through
# the earlier personal-phrase check, never reaching here.
_FACTUAL_OPENERS_RE = re.compile(
    r"^(?:"
    r"what(?:'?s| is| are| was| were)|"
    r"who(?:'?s| is| are| was| were)|"
    r"when(?:'?s| is| are| was| were)|"
    r"where(?:'?s| is| are| was| were)|"
    r"why(?:'?s| is| are| was| were| does| do| did)|"
    r"how(?:'?s| is| are| do(?:es)?| did| can| could| should| would)|"
    r"which|"
    r"define|defines|definition of|"
    r"explain|explains|explanation of|"
    r"describe|describes|description of|"
    r"summari[sz]e|summary of|"
    r"tell me about|"
    r"give me (?:a )?(?:summary|overview|definition|explanation) of"
    r")\s+(?!(?:me\b|myself\b|my\s+(?:name|self)\b|i\b|i'?m\b))"
)

# Interrogatives that look greeting-shaped only when paired with greeting
# vocab. Together with `_GREETING_VOCAB` they let "what's up", "how are you",
# "how's it going" pass as GREETING before the FACTUAL regex grabs them.
_GREETING_INTERROGATIVES: frozenset[str] = frozenset(
    {"what", "what's", "whats", "is", "are", "how", "how's", "hows", "it", "going"}
)

# Boot self-test golden set — exercises every fast-path branch with realistic
# queries. Verified on every Orchestrator boot; mismatches → one logger.warning
# line so silent fast-path regressions show up in the streamlit log.
_BOOT_GOLDEN: tuple[tuple[str, Intent], ...] = (
    ("hey",                          Intent.GREETING),
    ("hello there",                  Intent.GREETING),
    ("what's up",                    Intent.GREETING),
    ("so what's up",                 Intent.GREETING),  # discourse prefix + greeting
    ("what is RAG",                  Intent.FACTUAL),
    ("what is hipporag",             Intent.FACTUAL),
    ("so what is hipporag",          Intent.FACTUAL),   # discourse prefix + factual
    ("well, explain pagerank",       Intent.FACTUAL),   # discourse prefix
    ("and tell me about transformers", Intent.FACTUAL), # discourse prefix
    ("explain knowledge graphs",     Intent.FACTUAL),
    ("tell me about transformers",   Intent.FACTUAL),
    ("summarize the attention paper",Intent.FACTUAL),
    ("what's my name",               Intent.PERSONAL),
    ("who am I",                     Intent.PERSONAL),
    ("what do you know about me",    Intent.PERSONAL),
    ("",                             Intent.UNCLEAR),
    ("?",                            Intent.UNCLEAR),
)


# ---------------------------------------------------------------------------
# Normalizer (pure function, no side-effects)
# ---------------------------------------------------------------------------


def _normalize(query: str) -> str:
    """Return the normalized form of *query* used for all fast-path checks."""
    # 1. Lowercase and strip outer whitespace
    text = query.lower().strip()
    # 2. Strip surrounding punctuation
    text = _RE_STRIP_PUNCT.sub("", text).strip()
    # 3. Collapse 3+ consecutive identical chars to 1
    text = _RE_COLLAPSE.sub(r"\1", text)
    return text


def _tokenize(normalized: str) -> list[str]:
    """Split *normalized* text into tokens with trailing punct stripped."""
    raw_tokens = normalized.split()
    return [_RE_TOKEN_TRAIL.sub("", tok) for tok in raw_tokens]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class IntentClassifier:
    """Hybrid intent classifier: fast vocabulary lookup + LLM fallback."""

    name: str = "hybrid"

    def __init__(
        self,
        llm: Optional["LLMProvider"],
        *,
        fast_only: bool = False,
        max_tokens: int = 30,
    ) -> None:
        self._llm = llm
        self._fast_only = fast_only
        self._max_tokens = max_tokens

        prompt_path = Path(__file__).parent / "prompts" / "intent_classify.md"
        self._template = prompt_path.read_text(encoding="utf-8")

        # SHA-1 of the LLM template + vocab signature. Cache entries carry
        # this version; mismatched entries are dropped on lookup. Short
        # 12-hex form keeps logs tidy.
        vocab_sig = (
            "|".join(_PERSONAL_PHRASES)
            + "||"
            + "|".join(sorted(_GREETING_VOCAB))
        )
        self._cache_version: str = hashlib.sha1(
            (self._template + "||" + vocab_sig).encode("utf-8")
        ).hexdigest()[:12]

        # In-process cache for LLM/fallback verdicts only. Fast-path verdicts
        # are NOT cached (see classify() — they run microseconds and caching
        # them would trap stale verdicts past a source-level fix).
        self._cache: dict[str, tuple[IntentVerdict, str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> IntentVerdict:
        """Classify *query* and return an :class:`IntentVerdict`.

        Fast path runs FIRST and is NEVER cached — that guarantees a source
        edit (new personal phrase, tweaked factual-opener regex) takes effect
        on the next message. Only the LLM path's result is cached, and the
        cache is version-stamped so it self-invalidates when the prompt
        template or vocab sets change between releases.
        """
        normalized = _normalize(query)

        # 1. Fast path — always re-evaluated, never cached.
        verdict = self._fast_path(query, normalized)
        if verdict is not None:
            logger.info(
                "intent verdict: query=%r intent=%s confidence=%.2f source=%s "
                "cached=False",
                query[:80], verdict.intent.value, verdict.confidence, verdict.source,
            )
            return verdict

        # 2. Versioned LLM cache lookup (drop stale-version entries).
        cached_entry = self._cache.get(normalized)
        if cached_entry is not None and cached_entry[1] == self._cache_version:
            cached_verdict = cached_entry[0]
            logger.info(
                "intent verdict: query=%r intent=%s confidence=%.2f source=%s "
                "cached=True cache_version=%s",
                query[:80], cached_verdict.intent.value,
                cached_verdict.confidence, cached_verdict.source,
                self._cache_version,
            )
            return cached_verdict

        # 3. LLM (or fast-only fallback)
        if self._fast_only:
            verdict = IntentVerdict(Intent.UNCLEAR, 0.3, "fast_path", None)
        else:
            verdict = self._llm_classify(query, normalized)

        # 4. Cache the slow-path verdict with the current version stamp.
        self._cache[normalized] = (verdict, self._cache_version)
        logger.info(
            "intent verdict: query=%r intent=%s confidence=%.2f source=%s "
            "cached=False cache_version=%s",
            query[:80], verdict.intent.value, verdict.confidence, verdict.source,
            self._cache_version,
        )
        return verdict

    # ------------------------------------------------------------------
    # Fast path
    # ------------------------------------------------------------------

    def _fast_path(self, query: str, normalized: str) -> Optional[IntentVerdict]:
        """Return a verdict without an LLM call, or ``None`` if inconclusive."""

        # Step 4: Empty / whitespace-only / single non-alphanumeric → UNCLEAR
        if not normalized or not any(ch.isalnum() for ch in normalized):
            return IntentVerdict(Intent.UNCLEAR, 0.95, "fast_path", None)

        tokens = _tokenize(normalized)

        # Single non-alphanumeric token (after strip) also → UNCLEAR
        if len(tokens) == 1 and not any(ch.isalnum() for ch in tokens[0]):
            return IntentVerdict(Intent.UNCLEAR, 0.95, "fast_path", None)

        # Step 5: Identity / personal phrases (takes precedence over greeting
        # and factual openers, so "what is my name" stays PERSONAL).
        for phrase in _PERSONAL_PHRASES:
            if phrase in normalized:
                return IntentVerdict(Intent.PERSONAL, 0.95, "fast_path", None)

        # Strip a leading discourse marker once — "so what is X" / "well,
        # what's RAG" / "and tell me about transformers" parse the same as
        # the bare interrogative. The stripped form is used for the
        # short-greeting check AND the FACTUAL-opener match below.
        stripped = _DISCOURSE_PREFIX_RE.sub("", normalized, count=1).strip()
        stripped_tokens = _tokenize(stripped) if stripped != normalized else tokens

        # Step 5.5: Short interrogative greetings — "what's up", "how are
        # you", "how's it going", "so what's up". The discourse-strip lets
        # variants like "so what's up" hit this branch.
        if (
            1 < len(stripped_tokens) <= 4
            and all(tok in (_GREETING_VOCAB | _GREETING_INTERROGATIVES) for tok in stripped_tokens)
        ):
            return IntentVerdict(Intent.GREETING, 0.9, "fast_path", None)

        # Step 6: Factual openers — "what is X", "explain Y", "tell me about
        # Z" where Z is NOT me/myself. Negative-lookahead in the regex.
        if _FACTUAL_OPENERS_RE.match(stripped):
            return IntentVerdict(Intent.FACTUAL, 0.9, "fast_path", None)

        # Step 7: Greeting vocab — short query, all tokens in greeting vocab.
        # Uses the discourse-stripped form so "so hey" / "well, ok" work.
        check_tokens = stripped_tokens or tokens
        if check_tokens and len(check_tokens) <= 4 and all(tok in _GREETING_VOCAB for tok in check_tokens):
            return IntentVerdict(Intent.GREETING, 0.95, "fast_path", None)

        # Step 8: Inconclusive — let the LLM decide.
        return None

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    def _llm_classify(self, query: str, normalized: str) -> IntentVerdict:
        """Call the LLM and parse a single intent label from its output."""
        try:
            # The query may carry embedded newlines if upstream stitched a
            # follow-up onto its prior turn ("what is X\nsearch for it"). Those
            # newlines break the "User message: {query}\n\nOutput:" template
            # — the LLM thinks the user message ended at the linebreak and
            # returns empty. Flatten to single-line before formatting.
            flat_query = " | ".join(
                line.strip() for line in query.split("\n") if line.strip()
            )
            prompt = self._template.format(query=flat_query)
            raw: str = self._llm.complete(prompt, temperature=0.0, max_tokens=self._max_tokens)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "IntentClassifier LLM call failed (%s); defaulting to UNCLEAR.", exc
            )
            return IntentVerdict(Intent.UNCLEAR, 0.3, "fallback", None)

        raw_label = (raw or "").strip()
        cleaned = raw_label.lower()

        # Be permissive: find the first matching intent word anywhere in output
        intent: Optional[Intent] = None
        for word in cleaned.split():
            word_clean = _RE_TOKEN_TRAIL.sub("", word)
            if word_clean in _LABEL_TO_INTENT:
                intent = _LABEL_TO_INTENT[word_clean]
                break

        if intent is None:
            # Substring fallback — handle cases like "greeting." or "factual:"
            for label, mapped in _LABEL_TO_INTENT.items():
                if label in cleaned:
                    intent = mapped
                    break

        if intent is None:
            return IntentVerdict(Intent.UNCLEAR, 0.3, "fallback", raw_label=raw_label)

        return IntentVerdict(intent, 0.85, "llm", raw_label=raw_label)

    # ------------------------------------------------------------------
    # Boot-time self-test
    # ------------------------------------------------------------------

    def self_test(self) -> list[tuple[str, "Intent", "Intent"]]:
        """Exercise every fast-path branch against a golden set.

        Returns a list of ``(query, expected, actual)`` triples for every
        query whose fast-path verdict disagrees with the golden answer.
        Empty list = healthy. Does NOT call the LLM (uses ``_fast_path``
        directly), so it's free to run at boot.
        """
        failures: list[tuple[str, Intent, Intent]] = []
        for q, expected in _BOOT_GOLDEN:
            verdict = self._fast_path(q, _normalize(q))
            actual = verdict.intent if verdict is not None else Intent.UNCLEAR
            if actual != expected:
                failures.append((q, expected, actual))
        return failures
