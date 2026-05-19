"""Unit tests for hrag.interaction.store.InteractionStore."""

from __future__ import annotations

import threading
import time

from hrag.interaction.store import InteractionStore, PendingTurn


def test_create_and_submit_unblocks():
    """A thread waiting on wait_for_decision must unblock when submit fires."""
    store = InteractionStore(ttl_s=10.0, reap_interval_s=10.0)
    try:
        turn_id = "t1"
        store.create_turn(turn_id)

        got: list[dict] = []

        def _waiter():
            d = store.wait_for_decision(turn_id, timeout_s=2.0)
            if d is not None:
                got.append(d)

        thread = threading.Thread(target=_waiter)
        thread.start()
        # Tiny sleep to let the waiter reach event.wait().
        time.sleep(0.05)
        ok = store.submit_decision(turn_id, {"action": "filter"})
        thread.join(timeout=2.0)

        assert ok is True
        assert not thread.is_alive()
        assert got == [{"action": "filter"}]
    finally:
        store.shutdown()


def test_submit_decision_idempotent():
    """Submitting twice should return True then False."""
    store = InteractionStore(ttl_s=10.0, reap_interval_s=10.0)
    try:
        turn_id = "t2"
        store.create_turn(turn_id)
        assert store.submit_decision(turn_id, {"action": "continue"}) is True
        assert store.submit_decision(turn_id, {"action": "filter"}) is False
        # First decision wins.
        assert store.get_decision(turn_id) == {"action": "continue"}
    finally:
        store.shutdown()


def test_submit_decision_unknown_turn():
    """Submitting against an unknown turn_id is a no-op returning False."""
    store = InteractionStore(ttl_s=10.0, reap_interval_s=10.0)
    try:
        assert store.submit_decision("does-not-exist", {"action": "x"}) is False
    finally:
        store.shutdown()


def test_wait_for_decision_timeout():
    """No submission → wait returns None after the timeout elapses."""
    store = InteractionStore(ttl_s=10.0, reap_interval_s=10.0)
    try:
        store.create_turn("t3")
        t0 = time.time()
        result = store.wait_for_decision("t3", timeout_s=0.05)
        elapsed = time.time() - t0
        assert result is None
        # Sanity check: we actually waited roughly the requested timeout
        # (not zero, not a full second). Generous bounds for slow CI.
        assert 0.03 <= elapsed < 1.0
    finally:
        store.shutdown()


def test_wait_for_decision_unknown_turn_returns_none_fast():
    """wait_for_decision on an unknown turn must return None immediately."""
    store = InteractionStore(ttl_s=10.0, reap_interval_s=10.0)
    try:
        t0 = time.time()
        result = store.wait_for_decision("nope", timeout_s=5.0)
        elapsed = time.time() - t0
        assert result is None
        assert elapsed < 0.5  # didn't wait the full 5s
    finally:
        store.shutdown()


def test_get_decision_before_submit_returns_none():
    store = InteractionStore(ttl_s=10.0, reap_interval_s=10.0)
    try:
        store.create_turn("t4")
        assert store.get_decision("t4") is None
        store.submit_decision("t4", {"a": 1})
        assert store.get_decision("t4") == {"a": 1}
    finally:
        store.shutdown()


def test_remove_drops_turn():
    store = InteractionStore(ttl_s=10.0, reap_interval_s=10.0)
    try:
        store.create_turn("t5")
        assert store.get("t5") is not None
        store.remove("t5")
        assert store.get("t5") is None
        # Removing again is safe.
        store.remove("t5")
    finally:
        store.shutdown()


def test_reap_drops_old_turns():
    """Stale turns should be reaped within the next reap tick."""
    store = InteractionStore(ttl_s=0.05, reap_interval_s=0.02)
    try:
        store.create_turn("stale")
        assert store.get("stale") is not None
        # Wait long enough for at least one reap pass to catch it.
        time.sleep(0.25)
        assert store.get("stale") is None
    finally:
        store.shutdown()


def test_shutdown_joins_reaper():
    """shutdown() must terminate the reaper thread; second call is a no-op."""
    store = InteractionStore(ttl_s=10.0, reap_interval_s=0.05)
    reaper = store._reaper  # noqa: SLF001 — test inspecting internals
    assert reaper.is_alive()
    store.shutdown()
    # The reaper thread should have exited.
    assert not reaper.is_alive()
    # Calling shutdown a second time must not raise.
    store.shutdown()


def test_pending_turn_dataclass_defaults():
    """PendingTurn should be constructable with minimal args."""
    t = PendingTurn(turn_id="x", created_at=time.time())
    assert t.turn_id == "x"
    assert t.decision is None
    assert t.sources_snapshot == []
    assert isinstance(t.event, threading.Event)
    assert not t.event.is_set()


def test_create_turn_with_sources_snapshot():
    store = InteractionStore(ttl_s=10.0, reap_interval_s=10.0)
    try:
        snap = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        turn = store.create_turn("t6", sources_snapshot=snap)
        assert turn.sources_snapshot == snap
        # Snapshot is a copy, not the same list reference.
        snap.append({"chunk_id": "c3"})
        assert len(store.get("t6").sources_snapshot) == 2
    finally:
        store.shutdown()
