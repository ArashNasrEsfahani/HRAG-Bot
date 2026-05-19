"""ProfileStore: upsert idempotency, render output, delete."""

from __future__ import annotations

import pytest


@pytest.fixture()
def store(tmp_db):
    from hrag.db.migrations import run_migrations
    from hrag.memory.profile import ProfileStore

    run_migrations(tmp_db)
    return ProfileStore(tmp_db)


def test_upsert_inserts_new_row(store):
    pid = store.upsert("default", "fact", "occupation", "engineer", confidence=0.9)
    assert isinstance(pid, int) and pid > 0
    prefs = store.list_all("default")
    assert len(prefs) == 1
    assert prefs[0].polarity == "fact"
    assert prefs[0].topic == "occupation"
    assert prefs[0].value == "engineer"


def test_upsert_overwrites_on_conflict(store):
    pid1 = store.upsert("default", "fact", "occupation", "engineer", confidence=0.9)
    pid2 = store.upsert("default", "fact", "occupation", "scientist", confidence=0.95)
    assert pid1 == pid2, "ON CONFLICT should update the same row"
    prefs = store.list_all("default")
    assert len(prefs) == 1
    assert prefs[0].value == "scientist"
    assert prefs[0].confidence == pytest.approx(0.95)


def test_upsert_distinct_polarities_coexist(store):
    store.upsert("default", "fact", "occupation", "engineer")
    store.upsert("default", "like", "occupation", "love it")
    prefs = store.list_all("default")
    polarities = sorted(p.polarity for p in prefs)
    assert polarities == ["fact", "like"]


def test_upsert_rejects_bad_polarity(store):
    with pytest.raises(ValueError):
        store.upsert("default", "neutral", "topic", "value")


def test_upsert_rejects_empty_topic(store):
    with pytest.raises(ValueError):
        store.upsert("default", "fact", "   ", "value")


def test_upsert_clamps_confidence(store):
    store.upsert("default", "fact", "x", "y", confidence=2.5)
    prefs = store.list_all("default")
    assert prefs[0].confidence == 1.0


def test_delete_removes_row(store):
    pid = store.upsert("default", "fact", "occupation", "engineer")
    assert store.delete("default", pid) is True
    assert store.list_all("default") == []


def test_render_empty(store):
    assert store.render("default") == "(no profile yet)"


def test_render_groups_by_polarity(store):
    store.upsert("default", "fact", "occupation", "data engineer")
    store.upsert("default", "fact", "location", "Singapore")
    store.upsert("default", "style", "response length", "shorter answers")
    store.upsert("default", "like", "language", "Python over R")
    store.upsert("default", "dislike", "basic SQL", "skip explanation")

    rendered = store.render("default")
    assert "Facts:" in rendered
    assert "Style preferences:" in rendered
    assert "Likes:" in rendered
    assert "Dislikes:" in rendered
    # Order check — facts should come first
    lines = rendered.split("\n")
    assert lines[0].startswith("Facts:")


def test_render_drops_low_confidence(store):
    store.upsert("default", "fact", "high", "x", confidence=0.9)
    store.upsert("default", "fact", "low", "y", confidence=0.3)
    rendered = store.render("default", min_confidence=0.5)
    assert "high" in rendered
    assert "low" not in rendered


def test_render_caps_max_items(store):
    for i in range(20):
        store.upsert("default", "fact", f"topic{i}", f"value{i}", confidence=0.9)
    rendered = store.render("default", max_items=5)
    # Each entry contains "topicN: valueN"; count by colon separator.
    entries = rendered.count(": ") - 1  # subtract the "Facts:" header
    assert entries == 5
