"""Pure, dependency-free keyword extraction + overlap for taxonomy routing.

Two jobs, both stdlib-only and bilingual (English + Persian/Farsi):

* :func:`extract_keywords` — a lightweight YAKE-style scorer over a node's
  member texts (doc summaries + label/description). Used to *backfill* nodes
  the LLM left without keywords and by the ``hrag taxonomy keywords`` command.
* :func:`tokenize` / :func:`keyword_overlap` — the hot-path query side. The
  retriever turns a raw query into tokens with no LLM, and scores it against a
  node's stored keywords.

Everything here is a pure function — no LLM, no I/O, no global state — so it is
cheap enough to run on every turn (mirrors the contract-19 ``_is_math_meta_query``
style) and trivially testable.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Zero-width non-joiner — Persian sprinkles these between morphemes
# ("می‌بینی"); stripping them collapses spelling variants so tokenisation is
# stable. Mirrors the normalisation used in ``hrag.intent``.
_ZWNJ = "‌"

# Token = a run of word characters (Unicode letters/digits, incl. Persian) of
# length >= 2. Apostrophes inside a token are kept ("don't" -> "don't").
_TOKEN_RE = re.compile(r"[^\W\d_][\w'’]*", flags=re.UNICODE)

# Common English stopwords (compact list — enough to keep keyword noise down).
_STOP_EN: frozenset[str] = frozenset(
    """
    a an the this that these those and or but if then else for of to in on at by
    with without from as is are was were be been being do does did has have had
    it its it's they them their we us our you your i me my he she his her not no
    yes can could should would may might must will shall about into over under
    between within across per via using use used uses also more most such than
    very much many some any all each both few other another which who whom whose
    what when where why how here there out up down off again only own same so
    """.split()
)

# Common Persian stopwords (compact, high-frequency function words).
_STOP_FA: frozenset[str] = frozenset(
    """
    و در به از که این آن را با برای تا یا اما اگر چون های ها هم یک دو می نمی
    است بود شد باشد کند کرد شده خود ما شما او آنها ایشان من تو آنان نیز هر چه
    چی کجا کی چرا چگونه چند بر بی روی زیر بین درباره مورد طور همه بعضی دیگر
    """.split()
)

_STOPWORDS: frozenset[str] = _STOP_EN | _STOP_FA


def _normalize(text: str) -> str:
    """Lowercase, strip ZWNJ, collapse whitespace."""
    return " ".join((text or "").replace(_ZWNJ, "").lower().split())


def tokenize(text: str) -> list[str]:
    """Split *text* into normalized content tokens (stopwords removed)."""
    if not text:
        return []
    norm = _normalize(text)
    out: list[str] = []
    for tok in _TOKEN_RE.findall(norm):
        if len(tok) < 2:
            continue
        if tok in _STOPWORDS:
            continue
        # Drop pure-ASCII length-2 tokens (mostly noise like "ab"); keep
        # non-ASCII short tokens (Persian words can be short and meaningful).
        if len(tok) == 2 and tok.isascii():
            continue
        out.append(tok)
    return out


def extract_keywords(
    texts: list[str],
    *,
    top_k: int = 8,
    max_ngram: int = 2,
) -> list[str]:
    """Extract up to *top_k* keywords/keyphrases from *texts*.

    A compact YAKE-flavoured scorer: candidates are unigrams and bigrams of
    content tokens; each is ranked by ``frequency * distinctness`` where
    distinctness rewards terms that are NOT spread across every text (a term
    appearing in one focused summary is more characteristic than one appearing
    everywhere). Bigrams get a mild boost so phrases like "reinforcement
    learning" outrank their parts. Pure function — no I/O, no LLM.
    """
    if not texts:
        return []

    per_text_tokens: list[list[str]] = [tokenize(t) for t in texts if t]
    per_text_tokens = [toks for toks in per_text_tokens if toks]
    if not per_text_tokens:
        return []

    n_texts = len(per_text_tokens)
    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()  # in how many texts the candidate appears

    for toks in per_text_tokens:
        seen: set[str] = set()
        # Unigrams.
        for t in toks:
            tf[t] += 1
            seen.add(t)
        # n-grams up to max_ngram.
        for n in range(2, max_ngram + 1):
            for i in range(len(toks) - n + 1):
                gram = " ".join(toks[i : i + n])
                tf[gram] += 1
                seen.add(gram)
        for cand in seen:
            df[cand] += 1

    scores: dict[str, float] = {}
    for cand, freq in tf.items():
        # Distinctness: characteristic terms appear in a focused subset of
        # texts, not all of them. df==n_texts (everywhere) is penalised.
        spread = df[cand] / n_texts
        distinctness = 1.0 - 0.5 * spread
        ngram_boost = 1.0 + 0.4 * (cand.count(" "))
        scores[cand] = float(freq) * distinctness * ngram_boost * (1.0 + math.log1p(freq))

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

    # Greedy de-duplication: skip a unigram that is fully contained in an
    # already-selected phrase (keep "reinforcement learning", drop "learning").
    chosen: list[str] = []
    chosen_tokens: set[str] = set()
    for cand, _ in ranked:
        parts = cand.split(" ")
        if len(parts) == 1 and cand in chosen_tokens:
            continue
        chosen.append(cand)
        chosen_tokens.update(parts)
        if len(chosen) >= top_k:
            break
    return chosen


def keyword_overlap(query_keywords: list[str], node_keywords: list[str]) -> float:
    """Normalized overlap (0..1) between a query's and a node's keywords.

    Token-level Jaccard-ish recall over the node's keywords: how many of the
    node's keyword *tokens* are present among the query tokens, normalised by
    the node's keyword token count. A node keyword phrase contributes each of
    its tokens. Pure, fast (set intersection). Returns 0.0 when either side is
    empty, so hybrid routing is a true no-op on un-keyworded trees.
    """
    if not query_keywords or not node_keywords:
        return 0.0
    q_tokens: set[str] = set()
    for kw in query_keywords:
        q_tokens.update(kw.split(" "))
    n_tokens: list[str] = []
    for kw in node_keywords:
        n_tokens.extend(kw.split(" "))
    if not n_tokens:
        return 0.0
    hits = sum(1 for t in n_tokens if t in q_tokens)
    return hits / len(n_tokens)
