"""Hybrid intent classifier for HRAG-Bot.

Fast-path: normalized vocabulary lookup (deterministic, <1 ms). The fast path
covers four shapes: personal-phrase substring, short-interrogative greeting
("what's up"), factual openers ("what is X", "tell me about Y"), and pure
greeting vocab. **Fast-path verdicts are NOT cached** — they run on every
call (~20 µs) so source-level fixes (a new personal phrase, a tweak to the
factual-opener regex) take effect on the next message in the same long-lived
server process; no orchestrator rebuild needed.

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
    r"give me (?:a )?(?:summary|overview|definition|explanation) of|"
    r"check|show|list|find|get|compare|look (?:at|up)|search (?:for)?|"
    r"retrieve|fetch|walk me through|calculate|compute"
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
# line so silent fast-path regressions show up in the server log.
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
    ("check the ml tree",            Intent.FACTUAL),   # imperative fast-path
    ("list all papers about RAG",    Intent.FACTUAL),   # imperative fast-path
    ("show the results",             Intent.FACTUAL),   # imperative fast-path
    ("what's my name",               Intent.PERSONAL),
    ("who am I",                     Intent.PERSONAL),
    ("what do you know about me",    Intent.PERSONAL),
    ("",                             Intent.UNCLEAR),
    ("?",                            Intent.UNCLEAR),
)


# ---------------------------------------------------------------------------
# Phase 8.3 — entity extraction + history rendering helpers
# ---------------------------------------------------------------------------

# Loose proper-noun token detector. Matches:
#   - ASCII capitalised tokens (Mahmoud, NAACL, Arash) of 2+ chars
#   - All-Unicode-letter "words" of length >= 3 that are NOT in the common
#     English stop-list below. This captures Farsi/Persian names like
#     "محمود" that have no case distinction.
# We're intentionally permissive: false positives in the tracker are
# inert (the follow-up fast path only fires when the user message ALSO
# contains the same token).
_RE_PROPER_NOUN_ASCII = re.compile(r"\b[A-Z][A-Za-z'\-]{1,}\b")
_RE_WORD_UNICODE = re.compile(r"[^\W\d_]{3,}", flags=re.UNICODE)

_ENTITY_STOPWORDS: frozenset[str] = frozenset(
    {
        # Pronouns, articles, common verbs that frequently appear capitalised
        # at sentence start.
        "the", "a", "an", "and", "or", "but", "if", "then", "than", "that",
        "this", "these", "those", "is", "are", "was", "were", "be", "been",
        "being", "do", "does", "did", "have", "has", "had", "i", "you",
        "he", "she", "we", "they", "it", "me", "my", "your", "his", "her",
        "our", "their", "its", "yes", "no", "ok", "okay", "yeah", "yep",
        "to", "of", "in", "on", "at", "by", "for", "with", "from", "as",
        "user", "assistant", "source", "untitled", "section",
        # Source-formatter words from _format_passages.
        "n/a",
    }
)

# Per-session entity-tracker cap. Keeps the in-memory dict tiny under heavy
# multi-turn use; evicts oldest entries first when exceeded.
ENTITY_TRACKER_CAP: int = 50


def extract_entities(text: str) -> set[str]:
    """Extract candidate proper-noun entities from *text*.

    Pure function — no LLM, no I/O. Used by the orchestrator to build the
    follow-up entity tracker from the text of retrieved episodic memories.
    """
    if not text:
        return set()
    out: set[str] = set()
    # ASCII proper nouns.
    for tok in _RE_PROPER_NOUN_ASCII.findall(text):
        low = tok.lower()
        if low in _ENTITY_STOPWORDS:
            continue
        out.add(tok)
    # Unicode words (Persian, Arabic, etc. — no case marker available).
    for tok in _RE_WORD_UNICODE.findall(text):
        # Skip if it's already a pure-ASCII English stopword captured by
        # the first pass; lowercased compare.
        if tok.lower() in _ENTITY_STOPWORDS:
            continue
        # Skip pure-ASCII tokens that lack a leading uppercase letter — they
        # are common words like "remember", "memory" that aren't entities.
        # Tokens with a non-ASCII codepoint pass (Persian / CJK / etc.).
        if tok.isascii() and not tok[:1].isupper():
            continue
        out.add(tok)
    return out


# ---------------------------------------------------------------------------
# Phase 11.1 — reflective / opinion personal-question detector (two-tier)
# ---------------------------------------------------------------------------
#
# Two precision tiers serve the asymmetric cost of getting this wrong:
#   * STRICT (high precision) gates *coercion* — overriding a FACTUAL/UNCLEAR
#     verdict to PERSONAL. A false positive here hijacks a real question
#     ("describe me a function"), so only unambiguous opinion-about-me phrases
#     qualify.
#   * RECALL (high recall) fires only *within* an already-PERSONAL turn, where
#     a false positive merely picks the synthesis prompt over the recital one
#     (cheap). It adds a two-factor self-reference × opinion-cue match on top
#     of the strict phrases.
# Both tiers are pure functions — no LLM, no I/O (contract-19 property).

# Zero-width non-joiner — Persian text sprinkles these between morphemes
# ("می‌بینی"); stripping them collapses the ZWNJ / non-ZWNJ spelling variants
# so we don't have to enumerate both.
_ZWNJ = "‌"


def _normalize_reflect(query: str) -> str:
    """Lowercase, strip ZWNJ, and collapse whitespace for reflective matching."""
    text = (query or "").lower().replace(_ZWNJ, "")
    return " ".join(text.split())


# STRICT phrase regex — unambiguous "your opinion/impression OF ME" shapes.
# The bare `describe me` branch is end-anchored AND guarded by a negative
# lookahead so "describe me a function / how X / the Y" do NOT match.
_REFLECTIVE_STRICT_RE = re.compile(
    r"(?:"
    r"what\s+do\s+you\s+(?:think|reckon|feel|make)\s+(?:about|of)\s+me|"
    r"what(?:'?s| is)\s+your\s+(?:opinion|impression|take|view|thought|read)"
    r"\s+(?:on|of|about)\s+me|"
    r"how\s+(?:do|would)\s+you\s+(?:see|describe|view|perceive|picture)\s+me|"
    r"how\s+do\s+i\s+come\s+across|"
    r"describe\s+me(?!\s+(?:a|an|the|how|what|why|to|some|like\s+a))\s*[?.!]*$|"
    r"sum\s+me\s+up|"
    r"what\s+(?:kind|sort|type)\s+of\s+(?:person|guy|human|man|woman)\s+am\s+i|"
    r"what\s+(?:am\s+i|do\s+you\s+think\s+i'?m|do\s+you\s+think\s+i\s+am)\s+like|"
    r"your\s+(?:honest\s+)?(?:opinion|impression|thoughts?|take|read|assessment)"
    r"\s+(?:on|of|about)\s+me"
    r")",
    re.IGNORECASE,
)

# Farsi/Persian strict phrases ("what do you think about me", "your opinion
# about me", "how do you see me"). Matched against the ZWNJ-normalized text,
# so only the joined spelling is listed.
_REFLECTIVE_FARSI: tuple[str, ...] = (
    "درباره من چی فکر",
    "در مورد من چی فکر",
    "نظرت درباره من",
    "نظرت در مورد من",
    "منو چطور میبینی",
    "من چطور آدمی",
)

# Finglish (Romanized Persian) strict phrases.
_REFLECTIVE_FINGLISH: tuple[str, ...] = (
    "nazaret darbare man",
    "nazaret dar morede man",
    "man chetori",
    "darbare man chi fekr",
)

# Imperative ditransitives where "me" is a grammatical object, NOT a
# self-reference subject — "tell me about X", "show me Y". Stripped before the
# self-reference test so a factual "tell me about hipporag" is not mistaken
# for a question about the user.
_IMPERATIVE_ME_RE = re.compile(
    r"\b(?:tell|give|show|let|help|teach|find|get|bring|send|email|remind|ask|"
    r"walk|point|hand|offer|fetch)\s+me\b",
    re.IGNORECASE,
)

# RECALL tier — two-factor. A self-reference token AND an opinion-cue token
# anywhere in the (normalized) query. Word-boundaried so "image" does not
# trip "i" and "metaphor" does not trip "me".
_SELF_REF_RE = re.compile(
    r"\b(?:me|myself|i|i'?m|my)\b|من|منو|خودم",
    re.IGNORECASE,
)


def _self_ref_text(query: str) -> str:
    """Normalized query with imperative-'me' phrases removed, for self-ref tests."""
    return _IMPERATIVE_ME_RE.sub(" ", _normalize_reflect(query))
_OPINION_CUE_RE = re.compile(
    r"\b(?:think|thoughts?|opinion|impression|view|take|perceive|describe|"
    r"personality|vibe|reckon|assessment|judge|judgement|read)\b",
    re.IGNORECASE,
)

# A self-reference that the user is the SUBJECT of — object/possessive pronouns
# (me/my/myself; FA من/منو/خودم). Deliberately excludes the bare subject "i",
# which appears in countless non-personal sentences ("i want to test you").
_STRONG_SELF_REF_RE = re.compile(r"\b(?:me|myself|my)\b|من|منو|خودم", re.IGNORECASE)

# Phrasings that frame the USER as the thing being characterised — "who I am",
# "what I'm like", "as a person". These carry reflective intent without an
# opinion-cue or object-pronoun, so they anchor on their own.
_REFLECT_SUBJECT_RE = re.compile(
    r"\bwho\s+(?:i\s+am|am\s+i)\b|"
    r"\bwhat\s+(?:i\s+am|i'?m|am\s+i)\b|"
    r"\bas\s+a\s+(?:person|human|individual|man|woman|guy)\b|"
    r"\babout\s+me\b",
    re.IGNORECASE,
)


def is_reflective_strict(query: str) -> bool:
    """High-precision reflective test — gates verdict *coercion*.

    Pure function, no LLM/IO. Only unambiguous "opinion/impression of me"
    phrasings (EN + FA + Finglish) return True, so a misclassified FACTUAL
    question like "describe me a function" is never coerced to PERSONAL.
    """
    if not query:
        return False
    norm = _normalize_reflect(query)
    if _REFLECTIVE_STRICT_RE.search(norm):
        return True
    if any(p in norm for p in _REFLECTIVE_FINGLISH):
        return True
    # Farsi phrases: match the normalized (lowercased is a no-op for Persian).
    return any(p in norm for p in _REFLECTIVE_FARSI)


def has_self_reference(query: str) -> bool:
    """True when *query* refers to the user (me/I/my/myself; FA من/منو/خودم).

    Pure function. Used to gate the hybrid LLM reflective call so it only
    fires on plausibly-personal turns.
    """
    if not query:
        return False
    return bool(_SELF_REF_RE.search(_self_ref_text(query)))


def is_reflective_query(query: str) -> bool:
    """Recall-tier reflective test — used *within* an already-PERSONAL turn.

    Returns True for the strict phrases OR a two-factor match (a self-reference
    token AND an opinion-cue token). The looser two-factor branch boosts recall
    where false positives are cheap. Pure function, no LLM/IO.
    """
    if not query:
        return False
    if is_reflective_strict(query):
        return True
    norm = _normalize_reflect(query)
    return bool(_SELF_REF_RE.search(_self_ref_text(query)) and _OPINION_CUE_RE.search(norm))


def has_reflective_anchor(query: str) -> bool:
    """Minimal textual signal that a message could be an opinion/impression
    request *about the user* — corroborates the hybrid LLM "yes".

    The hybrid LLM judge is a small, error-prone model; a lone false "yes" on a
    neutral message ("Ok i want to test you") must not be enough to coerce the
    turn into a reflective self-portrait. We only consult the judge — and only
    let it coerce — when the text itself carries a reflective anchor:

    * an opinion cue (think / opinion / impression / describe …), OR
    * an object/possessive self-reference (me / my / myself; FA من/منو/خودم), OR
    * a user-as-subject phrasing ("who I am", "what I'm like", "as a person").

    The bare subject pronoun "i" alone is NOT an anchor. Pure function, no I/O.
    """
    if not query:
        return False
    norm = _normalize_reflect(query)
    return bool(
        _OPINION_CUE_RE.search(norm)
        or _STRONG_SELF_REF_RE.search(_self_ref_text(query))
        or _REFLECT_SUBJECT_RE.search(norm)
    )


def _mentions_entity(query: str, entities: set[str]) -> bool:
    """Case-insensitive substring check: does *query* mention any *entity*?"""
    if not query or not entities:
        return False
    q_lower = query.lower()
    for ent in entities:
        if not ent:
            continue
        if ent.lower() in q_lower:
            return True
    return False


def _render_history_block(history: Optional[list[tuple[str, str]]]) -> str:
    """Render the last 2 exchanges into a compact prompt block.

    Returns ``"(no prior conversation)"`` when *history* is empty. Each
    message is truncated to 200 chars so the prompt stays small.
    """
    if not history:
        return "(no prior conversation)"
    recent = history[-4:]  # 2 user / assistant exchanges
    lines: list[str] = []
    for role, content in recent:
        label = "User" if role == "user" else "Assistant"
        text = (content or "").replace("\n", " ").strip()
        if len(text) > 200:
            text = text[:200] + "…"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


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

    def classify(
        self,
        query: str,
        *,
        history: Optional[list[tuple[str, str]]] = None,
        prev_intent: Optional[Intent] = None,
        prev_memory_entities: Optional[set[str]] = None,
    ) -> IntentVerdict:
        """Classify *query* and return an :class:`IntentVerdict`.

        Fast path runs FIRST and is NEVER cached — that guarantees a source
        edit (new personal phrase, tweaked factual-opener regex) takes effect
        on the next message. Only the LLM path's result is cached, and the
        cache is version-stamped so it self-invalidates when the prompt
        template or vocab sets change between releases.

        Phase 8.3 — when *prev_intent* is PERSONAL and *prev_memory_entities*
        contains a token that the current message mentions, short-circuit to
        PERSONAL via a pre-LLM fast path. This handles the common follow-up
        ("what about Mahmoud?" right after a PERSONAL turn that surfaced
        Mahmoud) without paying for an LLM call.

        Phase 8.3 — the LLM path is also fed up to the last 2 user/assistant
        exchanges via *history* so the prompt-side rules ("follow-up about a
        person already framed as personal") can apply.
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

        # 1b. Phase 8.3 — follow-up entity fast path. When the previous turn
        # was PERSONAL and the user's current message mentions any entity
        # the previous PERSONAL turn surfaced from memory, this is a
        # continuation of the personal thread. No LLM call.
        if (
            prev_intent == Intent.PERSONAL
            and prev_memory_entities
            and _mentions_entity(query, prev_memory_entities)
        ):
            verdict = IntentVerdict(
                intent=Intent.PERSONAL,
                confidence=0.9,
                source="follow_up_entity",
                raw_label=None,
            )
            logger.info(
                "intent verdict: query=%r intent=%s confidence=%.2f source=%s "
                "cached=False (entity follow-up)",
                query[:80], verdict.intent.value, verdict.confidence, verdict.source,
            )
            return verdict

        # 2. Versioned LLM cache lookup (drop stale-version entries).
        # Cache key includes a compact history fingerprint so the same query
        # at different points in the conversation does not collide.
        cache_key = normalized
        if history:
            # Last 2 exchanges = up to 4 messages. Cheap content hash.
            recent = history[-4:]
            hist_sig = "||".join(f"{r}:{c[:40]}" for r, c in recent)
            cache_key = normalized + "@@" + hashlib.sha1(hist_sig.encode("utf-8")).hexdigest()[:8]
        cached_entry = self._cache.get(cache_key)
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
            verdict = self._llm_classify(query, normalized, history=history)

        # 4. Cache the slow-path verdict with the current version stamp.
        self._cache[cache_key] = (verdict, self._cache_version)
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

    def _llm_classify(
        self,
        query: str,
        normalized: str,
        *,
        history: Optional[list[tuple[str, str]]] = None,
    ) -> IntentVerdict:
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
            # Render the last 2 exchanges (up to 4 messages) into the prompt
            # so a follow-up like "what about Mahmoud?" inherits the prior
            # personal framing.
            history_block = _render_history_block(history)
            try:
                prompt = self._template.format(
                    query=flat_query, conversation_history=history_block,
                )
            except (KeyError, IndexError):
                # Backwards-compat path for any custom template without the
                # new {conversation_history} slot.
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


# ---------------------------------------------------------------------------
# Phase 11.1 — reflective LLM fallback classifier (hybrid mode)
# ---------------------------------------------------------------------------


class ReflectiveClassifier:
    """Tiny yes/no LLM judge: is *query* a reflective/opinion request?

    Mirrors :class:`IntentClassifier`'s caching philosophy — a version-stamped
    in-process cache keyed on the normalized query (+ a short history
    fingerprint), so a fixed verdict never haunts a long-lived session past a
    prompt edit. Consulted ONLY in hybrid mode, only when the pure-regex tiers
    miss AND the turn is plausibly personal — so the LLM cost lands on rare
    ambiguous turns.
    """

    name: str = "reflective"

    def __init__(self, llm: Optional["LLMProvider"], *, max_tokens: int = 4) -> None:
        self._llm = llm
        self._max_tokens = max_tokens
        prompt_path = Path(__file__).parent / "prompts" / "reflective_check.md"
        self._template = prompt_path.read_text(encoding="utf-8")
        self._cache_version: str = hashlib.sha1(
            self._template.encode("utf-8")
        ).hexdigest()[:12]
        self._cache: dict[str, tuple[bool, str]] = {}

    def classify(
        self,
        query: str,
        *,
        history: Optional[list[tuple[str, str]]] = None,
    ) -> bool:
        """Return True when the LLM judges *query* a reflective request.

        Defensive: any LLM/parse failure returns False (the safe default —
        the regex tiers already had their say, and a missed reflective turn
        merely falls back to the prior personal behaviour).
        """
        if not query or self._llm is None:
            return False

        norm = _normalize_reflect(query)
        cache_key = norm
        if history:
            recent = history[-4:]
            hist_sig = "||".join(f"{r}:{c[:40]}" for r, c in recent)
            cache_key = norm + "@@" + hashlib.sha1(
                hist_sig.encode("utf-8")
            ).hexdigest()[:8]

        cached = self._cache.get(cache_key)
        if cached is not None and cached[1] == self._cache_version:
            return cached[0]

        verdict = False
        try:
            history_block = _render_history_block(history)
            flat_query = " | ".join(
                line.strip() for line in query.split("\n") if line.strip()
            )
            try:
                prompt = self._template.format(
                    query=flat_query, conversation_history=history_block,
                )
            except (KeyError, IndexError):
                prompt = self._template.format(query=flat_query)
            raw: str = self._llm.complete(
                prompt, temperature=0.0, max_tokens=self._max_tokens,
            )
            verdict = "yes" in (raw or "").strip().lower()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ReflectiveClassifier LLM call failed (%s); defaulting to False.",
                exc,
            )
            verdict = False

        self._cache[cache_key] = (verdict, self._cache_version)
        logger.info(
            "reflective verdict: query=%r reflective=%s", query[:80], verdict,
        )
        return verdict
