"""Tests for hrag.topics.KnownTopicDetector — document-title topic extraction
and word-boundary query matching."""

from __future__ import annotations

import pytest

from hrag.db.connection import Database
from hrag.topics import KnownTopicDetector, _STOPWORDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_with_docs(tmp_path):
    """Build a fresh SQLite with a handful of doc + episodic titles."""
    db = Database(tmp_path / "topics.sqlite")
    db.init_schema()
    db.ensure_user("u1")
    titles = [
        # doc titles
        ("HIPPORAG",                       "document"),
        ("121_Memory_Never_Fades_Boostin", "document"),
        ("2025.findings-naacl.30",         "document"),
        ("4043_ReMindRAG_Low_Cost_LLM_Gu", "document"),
        ("hiwa pitch deck-pdf",            "document"),
        # episodic — MUST be excluded from topic terms
        ("I am arash nasr esfahani.",      "episodic"),
        ("I am designing a hierarchical rag system.", "episodic"),
    ]
    for i, (title, src) in enumerate(titles):
        db.execute(
            "INSERT INTO documents(doc_id, user_id, source_path, title, source_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"d{i}", "u1", f"/tmp/{i}", title, src),
        )
    db.commit()
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


def test_extract_tokens_simple_title(db_with_docs):
    det = KnownTopicDetector(db_with_docs)
    assert det._extract_tokens("HIPPORAG") == {"hipporag"}


def test_extract_tokens_letter_digit_split(db_with_docs):
    det = KnownTopicDetector(db_with_docs)
    # "Priority1" → ["Priority", "1"] → only "priority" kept (numeric filter)
    assert det._extract_tokens("Priority1") == {"priority"}


def test_extract_tokens_numeric_filter(db_with_docs):
    det = KnownTopicDetector(db_with_docs)
    # arxiv-style IDs are mostly numeric — nothing useful.
    assert det._extract_tokens("2025.findings-naacl.30") == {"findings", "naacl"}


def test_extract_tokens_underscore_split(db_with_docs):
    det = KnownTopicDetector(db_with_docs)
    # "Boostin" survives (truncated word from arxiv title); "121" filtered.
    assert det._extract_tokens("121_Memory_Never_Fades_Boostin") == {
        "memory", "never", "fades", "boostin",
    }


def test_extract_tokens_stopwords_filtered(db_with_docs):
    det = KnownTopicDetector(db_with_docs)
    # "me" / "my" / "the" / "of" must never appear as topic terms.
    title = "the_paper_on_me_and_my_work"
    tokens = det._extract_tokens(title)
    assert "me" not in tokens
    assert "my" not in tokens
    assert "the" not in tokens
    assert "of" not in tokens
    assert "paper" not in tokens  # filtered (boilerplate)
    # "work" should survive though
    assert "work" in tokens


# ---------------------------------------------------------------------------
# refresh() — loads from SQLite, excludes episodic
# ---------------------------------------------------------------------------


def test_refresh_excludes_episodic(db_with_docs):
    """Memory titles must NOT contribute topic terms — otherwise the user's
    name leaks into the topic set and 'what is arash' would force FACTUAL."""
    det = KnownTopicDetector(db_with_docs)
    terms = det.refresh("u1")
    assert "hipporag" in terms
    assert "naacl" in terms
    # From episodic titles — must be excluded
    assert "arash" not in terms
    assert "esfahani" not in terms
    # From episodic project title — must be excluded
    assert "hierarchical" not in terms


def test_refresh_returns_frozenset(db_with_docs):
    det = KnownTopicDetector(db_with_docs)
    terms = det.refresh("u1")
    assert isinstance(terms, frozenset)


# ---------------------------------------------------------------------------
# detect() — word-boundary substring match
# ---------------------------------------------------------------------------


def test_detect_matches_topic_in_query(db_with_docs):
    det = KnownTopicDetector(db_with_docs)
    assert det.detect("so what is hipporag?", "u1") == {"hipporag"}
    assert det.detect("what is hipporag", "u1") == {"hipporag"}
    assert det.detect("tell me about hipporag and naacl", "u1") == {
        "hipporag", "naacl",
    }


def test_detect_word_boundary(db_with_docs):
    """A topic term must match as a whole word — 'hipporag' must NOT match
    inside an unrelated word like 'hipporagism' (hypothetical) and 'naacl'
    must not match 'naaclitis'. Also 'memory' must match its plural cautiously."""
    det = KnownTopicDetector(db_with_docs)
    # "fragmentation" doesn't contain "naacl" so easy negative.
    assert det.detect("fragmentation is fun", "u1") == set()
    # Substring inside a bigger word — should NOT match.
    assert det.detect("naaclitis is not a thing", "u1") == set()


def test_detect_empty_query(db_with_docs):
    det = KnownTopicDetector(db_with_docs)
    assert det.detect("", "u1") == set()
    assert det.detect("   ", "u1") == set()


def test_detect_query_with_no_topics(db_with_docs):
    """Greetings, off-corpus questions, and pure personal queries must
    return no topic matches."""
    det = KnownTopicDetector(db_with_docs)
    assert det.detect("hey", "u1") == set()
    assert det.detect("what is my name", "u1") == set()
    assert det.detect("where is Tehran", "u1") == set()


def test_detect_lazy_loads_topics(db_with_docs):
    """The first call to detect() loads topics for that user automatically."""
    det = KnownTopicDetector(db_with_docs)
    assert "u1" not in det._terms_by_user
    det.detect("what is hipporag", "u1")
    assert "u1" in det._terms_by_user


def test_detect_empty_user_returns_empty(db_with_docs):
    """A user with no documents must return an empty term set + no matches."""
    det = KnownTopicDetector(db_with_docs)
    db_with_docs.ensure_user("u2")
    assert det.detect("what is hipporag", "u2") == set()


# ---------------------------------------------------------------------------
# Stopword sanity
# ---------------------------------------------------------------------------


def test_stopwords_include_personal_pronouns():
    """Critical: pronouns must be stopwords. If 'me' or 'my' became topic
    terms, the system would force FACTUAL on every personal query."""
    for w in ("i", "me", "my", "myself", "yourself", "we", "us", "our"):
        assert w in _STOPWORDS, f"{w!r} must be a stopword"
