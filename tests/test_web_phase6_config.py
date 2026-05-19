"""Phase 6 config-knob exposure tests for the FastAPI web layer.

Four tests:
1. GET /api/config returns Phase 6 keys with their default values.
2. POST /api/config with keep_alive round-trips correctly.
3. POST /api/config with adaptive_enabled + adaptive_top_k round-trips;
   unknown keys are silently dropped.
4. POST /api/config with vector_backend does NOT change vector_backend
   (backend swaps are intentionally not patchable).
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
# Test 1 — GET returns Phase 6 defaults
# ---------------------------------------------------------------------------

def test_get_config_phase6_defaults(client: TestClient) -> None:
    r = client.get("/api/config")
    assert r.status_code == 200, r.text
    body = r.json()

    # llm.keep_alive
    assert "keep_alive" in body["llm"], "llm.keep_alive missing from GET /api/config"
    assert body["llm"]["keep_alive"] == "30m", (
        f"Expected '30m' got {body['llm']['keep_alive']!r}"
    )

    # retrieval adaptive knobs
    ret = body["retrieval"]
    assert "adaptive_enabled" in ret, "retrieval.adaptive_enabled missing"
    assert ret["adaptive_enabled"] is False

    assert "adaptive_personal_episodic_bias" in ret, "retrieval.adaptive_personal_episodic_bias missing"

    assert "adaptive_top_k" in ret, "retrieval.adaptive_top_k missing"
    assert isinstance(ret["adaptive_top_k"], dict)

    # vector_backend (display-only)
    assert "vector_backend" in ret, "retrieval.vector_backend missing"
    assert ret["vector_backend"] == "chroma"

    # kg section
    assert "kg" in body, "kg section missing from GET /api/config"
    assert body["kg"]["backend"] == "networkx"


# ---------------------------------------------------------------------------
# Test 2 — POST keep_alive round-trips
# ---------------------------------------------------------------------------

def test_patch_keep_alive_roundtrip(client: TestClient) -> None:
    r = client.post("/api/config", json={"keep_alive": "1h"})
    assert r.status_code == 200, r.text

    r2 = client.get("/api/config")
    assert r2.status_code == 200
    assert r2.json()["llm"]["keep_alive"] == "1h"


# ---------------------------------------------------------------------------
# Test 3 — POST adaptive_enabled + adaptive_top_k; unknown keys dropped
# ---------------------------------------------------------------------------

def test_patch_adaptive_top_k_unknown_keys_dropped(client: TestClient) -> None:
    payload = {
        "adaptive_enabled": True,
        "adaptive_top_k": {
            "greeting": 0,
            "factual": 4,
            "bogus": 99,   # must be silently dropped
        },
    }
    r = client.post("/api/config", json=payload)
    assert r.status_code == 200, r.text

    r2 = client.get("/api/config")
    assert r2.status_code == 200
    body = r2.json()

    assert body["retrieval"]["adaptive_enabled"] is True

    top_k = body["retrieval"]["adaptive_top_k"]
    assert top_k["greeting"] == 0
    assert top_k["factual"] == 4
    assert "bogus" not in top_k, "Unknown key 'bogus' should have been dropped"


# ---------------------------------------------------------------------------
# Test 4 — POST vector_backend does NOT change vector_backend
# ---------------------------------------------------------------------------

def test_patch_vector_backend_not_patchable(client: TestClient) -> None:
    # Read the current value first.
    r0 = client.get("/api/config")
    original = r0.json()["retrieval"]["vector_backend"]

    # Attempt to change it — should be silently ignored (Pydantic extra=ignore).
    r = client.post("/api/config", json={"vector_backend": "sqlite_vec"})
    assert r.status_code == 200, r.text

    r2 = client.get("/api/config")
    assert r2.json()["retrieval"]["vector_backend"] == original, (
        "vector_backend must not change via POST /api/config"
    )
