"""SessionAutoExtractor: end-to-end with mocked LLM JSON output."""

from __future__ import annotations

import pytest


class _CannedExtractor:
    """Stands in for PreferenceExtractor.extract — returns a fixed candidate list."""

    def __init__(self, candidates):
        self._candidates = candidates
        self.calls = 0

    def extract(self, conversation):
        self.calls += 1
        return self._candidates


def _seed_session(db, user_id, session_id):
    db.execute(
        "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
        (session_id, user_id),
    )
    db.execute(
        "INSERT INTO messages (session_id, user_id, role, content) VALUES (?, ?, ?, ?)",
        (session_id, user_id, "user", "I'm an engineer in Berlin."),
    )
    db.execute(
        "INSERT INTO messages (session_id, user_id, role, content) VALUES (?, ?, ?, ?)",
        (session_id, user_id, "assistant", "Good to know."),
    )
    db.commit()


@pytest.fixture()
def stores(tmp_db):
    from hrag.db.migrations import run_migrations
    from hrag.memory.profile import ProfileStore

    run_migrations(tmp_db)
    _seed_session(tmp_db, "default", "sess-1")
    return tmp_db, ProfileStore(tmp_db)


def test_only_upserts_above_min_confidence(stores):
    from hrag.memory.auto_extract import SessionAutoExtractor
    from hrag.memory.extractor import PreferenceCandidate

    db, profile_store = stores
    candidates = [
        PreferenceCandidate("fact", "occupation", "engineer", 0.95),
        PreferenceCandidate("fact", "location", "Berlin", 0.6),  # below 0.7
    ]
    extractor = _CannedExtractor(candidates)
    auto = SessionAutoExtractor(db, extractor, profile_store, min_confidence=0.7)
    auto.on_session_close("default", "sess-1", block=True)

    prefs = profile_store.list_all("default")
    topics = {p.topic for p in prefs}
    assert "occupation" in topics
    assert "location" not in topics


def test_swallows_extractor_exceptions(stores):
    from hrag.memory.auto_extract import SessionAutoExtractor

    db, profile_store = stores

    class _BoomExtractor:
        def extract(self, conv):
            raise RuntimeError("extraction died")

    auto = SessionAutoExtractor(db, _BoomExtractor(), profile_store)
    # Must not raise; thread joins cleanly.
    auto.on_session_close("default", "sess-1", block=True)
    assert profile_store.list_all("default") == []


def test_records_session_id_on_upsert(stores):
    from hrag.memory.auto_extract import SessionAutoExtractor
    from hrag.memory.extractor import PreferenceCandidate

    db, profile_store = stores
    candidates = [PreferenceCandidate("fact", "language", "Python", 0.9)]
    auto = SessionAutoExtractor(
        db, _CannedExtractor(candidates), profile_store, min_confidence=0.7
    )
    auto.on_session_close("default", "sess-1", block=True)

    prefs = profile_store.list_all("default")
    assert len(prefs) == 1
    assert prefs[0].source_session_id == "sess-1"


def test_no_messages_means_no_upserts(tmp_db):
    from hrag.db.migrations import run_migrations
    from hrag.memory.auto_extract import SessionAutoExtractor
    from hrag.memory.profile import ProfileStore

    run_migrations(tmp_db)
    auto = SessionAutoExtractor(
        tmp_db, _CannedExtractor(["never-called"]), ProfileStore(tmp_db)
    )
    auto.on_session_close("default", "absent-session", block=True)
    assert ProfileStore(tmp_db).list_all("default") == []
