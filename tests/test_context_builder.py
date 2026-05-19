"""ContextBuilder: profile string flows into the prompt kwargs."""

from __future__ import annotations


def test_build_returns_no_profile_placeholder_when_empty(tmp_db):
    from hrag.context import ContextBuilder
    from hrag.db.migrations import run_migrations
    from hrag.memory.profile import ProfileStore

    run_migrations(tmp_db)
    builder = ContextBuilder(ProfileStore(tmp_db))
    out = builder.build("default")
    assert out == {"user_profile": "(no profile yet)"}


def test_build_renders_profile_when_present(tmp_db):
    from hrag.context import ContextBuilder
    from hrag.db.migrations import run_migrations
    from hrag.memory.profile import ProfileStore

    run_migrations(tmp_db)
    store = ProfileStore(tmp_db)
    store.upsert("default", "fact", "occupation", "engineer", confidence=0.9)
    builder = ContextBuilder(store)
    out = builder.build("default")
    assert "Facts:" in out["user_profile"]
    assert "engineer" in out["user_profile"]


def test_build_respects_min_confidence(tmp_db):
    from hrag.context import ContextBuilder
    from hrag.db.migrations import run_migrations
    from hrag.memory.profile import ProfileStore

    run_migrations(tmp_db)
    store = ProfileStore(tmp_db)
    store.upsert("default", "fact", "low_conf", "x", confidence=0.3)
    builder = ContextBuilder(store, min_confidence=0.6)
    assert builder.build("default") == {"user_profile": "(no profile yet)"}


def test_build_respects_max_items(tmp_db):
    from hrag.context import ContextBuilder
    from hrag.db.migrations import run_migrations
    from hrag.memory.profile import ProfileStore

    run_migrations(tmp_db)
    store = ProfileStore(tmp_db)
    for i in range(10):
        store.upsert("default", "fact", f"t{i}", f"v{i}", confidence=0.9)
    builder = ContextBuilder(store, max_items=3)
    out = builder.build("default")
    # 3 items -> 3 colon-value pairs on the Facts line, plus the "Facts:" header.
    assert out["user_profile"].count(": ") == 4
