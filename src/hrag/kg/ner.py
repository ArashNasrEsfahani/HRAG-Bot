"""Named Entity Recognition (NER) for the HRAG knowledge-graph layer.

Two implementations are provided:

- SpacyNER  — lightweight default; uses spaCy en_core_web_sm with a regex
              fallback when spaCy or the model is unavailable.
- LLMNER    — LLM-based; renders prompts/ner.md and parses the JSON response.

Use build_ner() to construct the right one from config.
"""

from __future__ import annotations

import json
import re
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

from hrag.providers.llm import LLMProvider

# ---------------------------------------------------------------------------
# Entity type allowlist for spaCy NER
# ---------------------------------------------------------------------------

_SPACY_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "PRODUCT",
        "WORK_OF_ART",
        "EVENT",
        "LAW",
        "FAC",
        "NORP",
    }
)

# ---------------------------------------------------------------------------
# Regex patterns for the fallback proper-noun detector
# ---------------------------------------------------------------------------

# A "token" for the fallback: a word that starts with a capital letter OR is an
# all-caps acronym OR is a CamelCase/mixed-caps word (e.g. "HippoRAG", "PageRank").
# We define a "capitalised token" broadly as any word where the first character
# is uppercase.
_RE_CAP_TOKEN = re.compile(r"\b[A-Z][A-Za-z]*\b")

# All-caps acronyms (two or more uppercase letters)
_RE_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")


def _regex_extract(query: str) -> list[str]:
    """Regex-based fallback: capitalized/mixed-caps runs + acronyms.

    Groups consecutive capitalised tokens into multi-word phrases, then adds
    standalone all-caps acronyms that were not already captured.
    """
    seen: dict[str, None] = {}

    # Walk through the query character by character to group consecutive
    # capitalised tokens (separated only by spaces) into multi-word phrases.
    tokens = query.split()
    phrase_tokens: list[str] = []
    for tok in tokens:
        # Strip trailing punctuation for matching purposes
        clean = tok.rstrip(".,!?;:")
        if _RE_CAP_TOKEN.fullmatch(clean):
            phrase_tokens.append(clean)
        else:
            if phrase_tokens:
                key = " ".join(phrase_tokens).lower().strip()
                if key and key not in seen:
                    seen[key] = None
                phrase_tokens = []
    # Flush any remaining phrase
    if phrase_tokens:
        key = " ".join(phrase_tokens).lower().strip()
        if key and key not in seen:
            seen[key] = None

    # Also catch standalone all-caps acronyms not captured above
    for match in _RE_ACRONYM.finditer(query):
        key = match.group().lower().strip()
        if key and key not in seen:
            seen[key] = None

    return list(seen)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class NER(ABC):
    """Extract named entities (noun phrases) from a query string."""

    name: str = "abstract"

    @abstractmethod
    def extract(self, query: str) -> list[str]:
        """Return a list of normalized entity strings (lowercased, stripped)."""


# ---------------------------------------------------------------------------
# SpacyNER
# ---------------------------------------------------------------------------


class SpacyNER(NER):
    """Lightweight default. Uses spaCy en_core_web_sm.

    On first call, lazily loads the model. If spaCy or the model is not
    installed, falls back to a regex-based proper-noun detector (capitalized
    multi-word sequences) and emits a single warning so the pipeline never
    breaks.
    """

    name = "spacy"

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self._model_name = model_name
        self._nlp = None  # lazy-loaded
        self._fallback: bool = False
        self._warned: bool = False

    def _load(self) -> None:
        """Attempt to load the spaCy model; set fallback on failure."""
        if self._nlp is not None or self._fallback:
            return
        try:
            import spacy  # noqa: PLC0415

            self._nlp = spacy.load(self._model_name)
        except (ImportError, OSError):
            self._fallback = True
            if not self._warned:
                warnings.warn(
                    f"SpacyNER: spaCy or model {self._model_name!r} not available; "
                    "falling back to regex-based proper-noun detection.",
                    UserWarning,
                    stacklevel=3,
                )
                self._warned = True

    def extract(self, query: str) -> list[str]:
        if not query or not query.strip():
            return []

        self._load()

        if self._fallback:
            return _regex_extract(query)

        # spaCy path
        doc = self._nlp(query)  # type: ignore[misc]
        seen: dict[str, None] = {}

        # Named entities in the allowlist
        for ent in doc.ents:
            if ent.label_ in _SPACY_ENTITY_TYPES:
                key = ent.text.lower().strip()
                if key and key not in seen:
                    seen[key] = None

        # Noun chunks whose root is a proper noun (catches technical terms)
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ == "PROPN":
                key = chunk.text.lower().strip()
                if key and key not in seen:
                    seen[key] = None

        return list(seen)


# ---------------------------------------------------------------------------
# LLMNER
# ---------------------------------------------------------------------------

# Strip markdown code fences that some LLMs wrap around JSON
_RE_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_llm_json(raw: str) -> list[str]:
    """Parse a JSON list of strings from raw LLM output.

    Tolerates:
    - Code fences (```json ... ```)
    - Leading/trailing prose before/after the JSON list
    - Completely malformed output → returns []
    """
    text = raw.strip()

    # Strip code fences
    fence_match = _RE_CODE_FENCE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try to find the first JSON array in the text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(parsed, list):
        return []

    result: dict[str, None] = {}
    for item in parsed:
        if not isinstance(item, str):
            continue
        key = item.lower().strip()
        if key and key not in result:
            result[key] = None
    return list(result)


class LLMNER(NER):
    """LLM-based NER. Renders prompts/ner.md with the query and parses a JSON
    list of strings. On any error returns [].
    """

    name = "llm"

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        prompt_path = Path(__file__).parent.parent / "prompts" / "ner.md"
        self._template = prompt_path.read_text(encoding="utf-8")

    def extract(self, query: str) -> list[str]:
        if not query or not query.strip():
            return []
        prompt = self._template.format(query=query)
        try:
            raw = self._llm.complete(prompt, temperature=0.0)
        except Exception:
            return []
        return _parse_llm_json(raw)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_ner(cfg, llm: LLMProvider) -> NER:  # cfg: KGConfig (forward-ref safe)
    """Factory. Modes: 'spacy' (default), 'llm'. Anything else raises ValueError."""
    # Defensive attribute access so we don't depend on the exact KGConfig
    # field being finalised yet (Wave 1D may still be evolving it).
    mode = (getattr(cfg, "ner", "spacy") or "spacy").lower().strip()

    if mode == "spacy":
        return SpacyNER()
    if mode == "llm":
        return LLMNER(llm)

    raise ValueError(
        f"Unknown NER mode: {mode!r}. Expected one of: spacy, llm."
    )
