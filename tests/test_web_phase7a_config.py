"""Phase 7-A config-knob exposure tests for the FastAPI web layer.

Four tests:
1. GET /api/config returns Phase 7-A keys with their default values.
2. POST /api/config with math_meta_filter_enabled round-trips correctly.
3. POST /api/config with formula_extraction_enabled + formula_extraction_max_tokens
   updates the nested config and GET reflects it.
4. POST /api/config with math_meta_rerank_threshold round-trips.
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
    """Tear down the singleton so each test starts with a fresh config."""
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
# Test 1 — GET returns Phase 7-A defaults
# ---------------------------------------------------------------------------

def test_get_config_phase7a_defaults(client: TestClient) -> None:
    r = client.get("/api/config")
    assert r.status_code == 200, r.text
    body = r.json()

    # retrieval section — math meta filter knobs
    ret = body["retrieval"]
    assert "math_meta_filter_enabled" in ret, "retrieval.math_meta_filter_enabled missing from GET /api/config"
    assert ret["math_meta_filter_enabled"] is False, (
        f"Expected False got {ret['math_meta_filter_enabled']!r}"
    )

    assert "math_meta_rerank_threshold" in ret, "retrieval.math_meta_rerank_threshold missing"
    assert ret["math_meta_rerank_threshold"] == -10.0, (
        f"Expected -10.0 got {ret['math_meta_rerank_threshold']!r}"
    )

    # formula_extraction section
    assert "formula_extraction" in body, "formula_extraction section missing from GET /api/config"
    fe = body["formula_extraction"]
    assert fe["enabled"] is False, f"Expected False got {fe['enabled']!r}"
    assert fe["max_tokens"] == 400, f"Expected 400 got {fe['max_tokens']!r}"


# ---------------------------------------------------------------------------
# Test 2 — POST math_meta_filter_enabled round-trips
# ---------------------------------------------------------------------------

def test_patch_math_meta_filter_enabled_roundtrip(client: TestClient) -> None:
    r = client.post("/api/config", json={"math_meta_filter_enabled": True})
    assert r.status_code == 200, r.text

    r2 = client.get("/api/config")
    assert r2.status_code == 200
    assert r2.json()["retrieval"]["math_meta_filter_enabled"] is True


# ---------------------------------------------------------------------------
# Test 3 — POST formula_extraction_enabled + formula_extraction_max_tokens
# ---------------------------------------------------------------------------

def test_patch_formula_extraction_roundtrip(client: TestClient) -> None:
    payload = {
        "formula_extraction_enabled": True,
        "formula_extraction_max_tokens": 600,
    }
    r = client.post("/api/config", json=payload)
    assert r.status_code == 200, r.text

    r2 = client.get("/api/config")
    assert r2.status_code == 200
    fe = r2.json()["formula_extraction"]
    assert fe["enabled"] is True, f"Expected True got {fe['enabled']!r}"
    assert fe["max_tokens"] == 600, f"Expected 600 got {fe['max_tokens']!r}"


# ---------------------------------------------------------------------------
# Test 4 — POST math_meta_rerank_threshold round-trips
# ---------------------------------------------------------------------------

def test_patch_math_meta_rerank_threshold_roundtrip(client: TestClient) -> None:
    r = client.post("/api/config", json={"math_meta_rerank_threshold": -6.0})
    assert r.status_code == 200, r.text

    r2 = client.get("/api/config")
    assert r2.status_code == 200
    assert r2.json()["retrieval"]["math_meta_rerank_threshold"] == -6.0
