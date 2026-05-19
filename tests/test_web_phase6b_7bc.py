"""Phase 6-B / 7-B / 7-C — new config knobs + three new GET endpoints.

Eight tests:
1.  GET /api/config returns the five new keys with defaults.
2.  POST /api/config with num_keep round-trips.
3.  POST /api/config with use_nougat round-trips.
4.  POST /api/config with adaptive_retriever_per_intent merges; existing keys preserved.
5.  POST /api/config with bogus retriever value rejects (HTTP 400).
6.  GET /api/embeddings/suggested returns 4 suggestions + current model.
7.  GET /api/ingest/nougat_status returns {available: false, ...} (Nougat not installed).
8.  GET /api/feedback/stats returns zeros on a fresh DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from hrag.web.app import _State, app  # noqa: E402


def _reset_state() -> None:
    """Tear down the singleton so each test starts with a fresh in-memory config."""
    with _State.lock:
        _State.cfg = None
        _State.orch = None


@pytest.fixture(autouse=True)
def reset_state():
    _reset_state()
    yield
    _reset_state()


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Test 1 — GET /api/config exposes all five new keys
# ---------------------------------------------------------------------------


def test_get_config_new_keys_present(client: TestClient) -> None:
    """The five Phase-6B/7B/7C knobs must appear in GET /api/config."""
    r = client.get("/api/config")
    assert r.status_code == 200, r.text
    body = r.json()

    # llm.num_keep
    assert "llm" in body
    assert "num_keep" in body["llm"], "llm.num_keep missing"
    # The default is None
    assert body["llm"]["num_keep"] is None, f"Expected None got {body['llm']['num_keep']!r}"

    # embeddings section
    assert "embeddings" in body, "embeddings section missing"
    assert "model" in body["embeddings"]
    assert "dim" in body["embeddings"]

    # retrieval.adaptive_retriever_per_intent
    assert "retrieval" in body
    assert "adaptive_retriever_per_intent" in body["retrieval"], (
        "retrieval.adaptive_retriever_per_intent missing"
    )
    arint = body["retrieval"]["adaptive_retriever_per_intent"]
    assert isinstance(arint, dict)
    assert "factual" in arint  # at least one default intent present

    # ingest section
    assert "ingest" in body, "ingest section missing"
    assert "use_nougat" in body["ingest"], "ingest.use_nougat missing"
    assert "nougat_model" in body["ingest"], "ingest.nougat_model missing"
    assert body["ingest"]["use_nougat"] is False, "Default use_nougat should be False"


# ---------------------------------------------------------------------------
# Test 2 — POST num_keep round-trips
# ---------------------------------------------------------------------------


def test_patch_num_keep_roundtrip(client: TestClient) -> None:
    r = client.post("/api/config", json={"num_keep": 256})
    assert r.status_code == 200, r.text
    assert r.json()["llm"]["num_keep"] == 256

    r2 = client.get("/api/config")
    assert r2.status_code == 200
    assert r2.json()["llm"]["num_keep"] == 256


# ---------------------------------------------------------------------------
# Test 3 — POST use_nougat round-trips
# ---------------------------------------------------------------------------


def test_patch_use_nougat_roundtrip(client: TestClient) -> None:
    r = client.post("/api/config", json={"use_nougat": True})
    assert r.status_code == 200, r.text
    assert r.json()["ingest"]["use_nougat"] is True

    r2 = client.get("/api/config")
    assert r2.status_code == 200
    assert r2.json()["ingest"]["use_nougat"] is True


# ---------------------------------------------------------------------------
# Test 4 — POST adaptive_retriever_per_intent merges; existing keys preserved
# ---------------------------------------------------------------------------


def test_patch_adaptive_retriever_per_intent_merges(client: TestClient) -> None:
    # First set factual to taxonomy
    r = client.post("/api/config", json={
        "adaptive_retriever_per_intent": {"factual": "taxonomy"},
    })
    assert r.status_code == 200, r.text
    arint = r.json()["retrieval"]["adaptive_retriever_per_intent"]
    assert arint["factual"] == "taxonomy", f"factual should be taxonomy, got {arint['factual']!r}"

    # All other default intents must still be present (they started as "default")
    for intent in ("greeting", "personal", "general", "unclear"):
        assert intent in arint, f"intent {intent!r} was dropped after merge"
        assert arint[intent] == "default", (
            f"intent {intent!r} was unexpectedly mutated to {arint[intent]!r}"
        )

    # Now set general to vector without touching factual
    r2 = client.post("/api/config", json={
        "adaptive_retriever_per_intent": {"general": "vector"},
    })
    assert r2.status_code == 200, r2.text
    arint2 = r2.json()["retrieval"]["adaptive_retriever_per_intent"]
    assert arint2["factual"] == "taxonomy", "factual must still be taxonomy after second patch"
    assert arint2["general"] == "vector", "general should now be vector"


# ---------------------------------------------------------------------------
# Test 5 — POST adaptive_retriever_per_intent with bogus value → HTTP 400
# ---------------------------------------------------------------------------


def test_patch_adaptive_retriever_per_intent_rejects_bogus(client: TestClient) -> None:
    r = client.post("/api/config", json={
        "adaptive_retriever_per_intent": {"factual": "bogus_retriever"},
    })
    # Must fail — either 400 (HTTPException) or 422 (validation error)
    assert r.status_code in (400, 422), (
        f"Expected 400 or 422, got {r.status_code}: {r.text}"
    )
    # Config must not have changed
    r2 = client.get("/api/config")
    assert r2.status_code == 200
    assert r2.json()["retrieval"]["adaptive_retriever_per_intent"]["factual"] == "default", (
        "factual should still be 'default' after a rejected patch"
    )


# ---------------------------------------------------------------------------
# Test 6 — GET /api/embeddings/suggested
# ---------------------------------------------------------------------------


def test_get_embeddings_suggested(client: TestClient) -> None:
    r = client.get("/api/embeddings/suggested")
    assert r.status_code == 200, r.text
    body = r.json()

    assert "current" in body, "current model missing"
    assert "current_dim" in body, "current_dim missing"
    assert "suggestions" in body, "suggestions list missing"

    suggestions = body["suggestions"]
    # Config ships 4 curated suggestions
    assert len(suggestions) >= 4, f"Expected >= 4 suggestions, got {len(suggestions)}"

    # Each suggestion must have the three required fields
    for s in suggestions:
        assert "label" in s, f"suggestion missing 'label': {s}"
        assert "model" in s, f"suggestion missing 'model': {s}"
        assert "dim" in s, f"suggestion missing 'dim': {s}"

    # The all-mpnet entry must be present
    mpnet_models = [s for s in suggestions if "all-mpnet" in s["model"]]
    assert mpnet_models, "all-mpnet suggestion not found in suggestions list"


# ---------------------------------------------------------------------------
# Test 7 — GET /api/ingest/nougat_status (nougat not installed on this box)
# ---------------------------------------------------------------------------


def test_get_nougat_status(client: TestClient) -> None:
    r = client.get("/api/ingest/nougat_status")
    assert r.status_code == 200, r.text
    body = r.json()

    assert "available" in body
    assert "model" in body
    assert "use_nougat" in body

    # On a stock dev box without nougat-ocr installed, available must be False
    assert body["available"] is False, (
        "Expected available=False because nougat-ocr is not installed in this environment"
    )
    # Default model string must be non-empty
    assert body["model"], "nougat_model must be a non-empty string"


# ---------------------------------------------------------------------------
# Test 8 — GET /api/feedback/stats on a fresh DB returns all zeros
# ---------------------------------------------------------------------------


def test_get_feedback_stats_empty(client: TestClient) -> None:
    r = client.get("/api/feedback/stats")
    assert r.status_code == 200, r.text
    body = r.json()

    assert "thumbs_up" in body
    assert "thumbs_down" in body
    assert "total" in body
    assert "sessions" in body
    assert "top_negative" in body

    assert body["thumbs_up"] == 0, f"Expected 0 got {body['thumbs_up']}"
    assert body["thumbs_down"] == 0, f"Expected 0 got {body['thumbs_down']}"
    assert body["total"] == 0, f"Expected 0 got {body['total']}"
    assert body["sessions"] == 0, f"Expected 0 got {body['sessions']}"
    assert body["top_negative"] == [], f"Expected [] got {body['top_negative']}"
