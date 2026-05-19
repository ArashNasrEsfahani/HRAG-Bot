"""Thread-safe in-memory store of pending review turns.

Each pending turn carries an ``threading.Event``: the orchestrator thread
blocks on ``wait_for_decision``; the HTTP resume handler thread unblocks it
via ``submit_decision``. A daemon reaper drops entries older than ``ttl_s``
so a crashed frontend never leaks memory. Restarting the server invalidates
every pending turn by design — there is no persistence.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PendingTurn:
    """A turn waiting for the user's review decision.

    Attributes
    ----------
    turn_id:
        Stable identifier for the turn (typically ``uuid4().hex``). The
        orchestrator allocates this and the frontend echoes it back when the
        user submits a decision.
    created_at:
        ``time.time()`` snapshot at registration. Used by the reaper.
    event:
        Unblocks the orchestrator thread when the decision arrives. Owned
        per-turn so concurrent turns don't fight over a shared signal.
    decision:
        ``None`` until set by :py:meth:`InteractionStore.submit_decision`,
        then the raw dict payload the frontend POSTed.
    sources_snapshot:
        Optional copy of the source list captured at pause time. Useful for
        post-hoc analytics / debugging; not used by the wait path.
    """

    turn_id: str
    created_at: float
    event: threading.Event = field(default_factory=threading.Event)
    decision: Optional[dict[str, Any]] = None
    sources_snapshot: list[dict[str, Any]] = field(default_factory=list)


class InteractionStore:
    """In-memory store of pending review turns keyed by ``turn_id``.

    Thread-safe. A daemon thread reaps entries older than ``ttl_s`` every
    ``reap_interval_s`` seconds. Restarting the server invalidates pending
    turns by design.
    """

    def __init__(self, ttl_s: float = 300.0, reap_interval_s: float = 30.0) -> None:
        self._turns: dict[str, PendingTurn] = {}
        self._lock = threading.Lock()
        self._ttl_s = ttl_s
        self._reap_interval_s = reap_interval_s
        self._stop = threading.Event()
        self._reaper = threading.Thread(
            target=self._reap_loop,
            name="InteractionStoreReaper",
            daemon=True,
        )
        self._reaper.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_turn(
        self,
        turn_id: str,
        sources_snapshot: Optional[list[dict[str, Any]]] = None,
    ) -> PendingTurn:
        """Register a new pending turn and return the :class:`PendingTurn`.

        Re-registering the same ``turn_id`` overwrites the previous entry —
        the old event will never be set, but no callers should be waiting on
        it (the orchestrator never reuses a turn_id).
        """
        turn = PendingTurn(
            turn_id=turn_id,
            created_at=time.time(),
            sources_snapshot=list(sources_snapshot) if sources_snapshot else [],
        )
        with self._lock:
            self._turns[turn_id] = turn
        return turn

    def submit_decision(self, turn_id: str, decision: dict[str, Any]) -> bool:
        """Attach *decision* to the turn and unblock the orchestrator thread.

        Returns ``True`` if this is the first submission for *turn_id*,
        ``False`` if the turn is unknown or already has a decision (the
        second call is a no-op). Idempotent.
        """
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None:
                return False
            if turn.decision is not None or turn.event.is_set():
                return False
            turn.decision = dict(decision)
            event = turn.event
        # Release the lock before signalling so a waiter doesn't wake up
        # only to immediately fight us for the same lock.
        event.set()
        return True

    def get_decision(self, turn_id: str) -> Optional[dict[str, Any]]:
        """Return the decision dict if one has been submitted, else ``None``."""
        with self._lock:
            turn = self._turns.get(turn_id)
            return None if turn is None else turn.decision

    def wait_for_decision(
        self,
        turn_id: str,
        timeout_s: float,
    ) -> Optional[dict[str, Any]]:
        """Block until the decision arrives or *timeout_s* elapses.

        Returns the decision dict on success, ``None`` on timeout or when
        the turn is unknown. The ``event.wait`` happens OUTSIDE the lock so
        a concurrent ``submit_decision`` is not blocked.
        """
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None:
                return None
            event = turn.event
            # If the decision already arrived (race window between create
            # and wait), short-circuit.
            if turn.decision is not None:
                return turn.decision

        # Wait outside the lock.
        signalled = event.wait(timeout=timeout_s)
        if not signalled:
            return None

        with self._lock:
            turn = self._turns.get(turn_id)
            return None if turn is None else turn.decision

    def get(self, turn_id: str) -> Optional[PendingTurn]:
        """Return the :class:`PendingTurn` if known, else ``None``."""
        with self._lock:
            return self._turns.get(turn_id)

    def remove(self, turn_id: str) -> None:
        """Drop *turn_id* from the store. Safe to call when unknown."""
        with self._lock:
            self._turns.pop(turn_id, None)

    def shutdown(self) -> None:
        """Stop the reaper thread. Safe to call exactly once.

        Subsequent calls are no-ops (the stop event is already set).
        """
        self._stop.set()
        if self._reaper.is_alive():
            self._reaper.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reap_loop(self) -> None:
        """Background loop: every ``reap_interval_s`` drop stale entries.

        Stale = ``time.time() - created_at > ttl_s``. We iterate over a
        copied key list (under the lock) so the dict isn't mutated during
        iteration; the actual deletions happen under the same lock.
        """
        # Use ``_stop.wait`` instead of ``time.sleep`` so ``shutdown()``
        # returns within ``reap_interval_s`` worst-case rather than blocking
        # for the full interval.
        while not self._stop.wait(timeout=self._reap_interval_s):
            now = time.time()
            with self._lock:
                stale = [
                    tid for tid, turn in list(self._turns.items())
                    if now - turn.created_at > self._ttl_s
                ]
                for tid in stale:
                    # Wake any (legitimate) waiter so it sees the missing
                    # turn and returns None instead of hanging until its
                    # own timeout. The decision stays None so the
                    # orchestrator's downstream code treats this as a
                    # timeout, which is the right semantic.
                    turn = self._turns.pop(tid, None)
                    if turn is not None:
                        turn.event.set()
