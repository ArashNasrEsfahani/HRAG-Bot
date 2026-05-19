"""Tests for src/hrag/kg/ppr.py — personalized PageRank."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.sparse")

from hrag.kg.ppr import personalized_pagerank  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csr(data, rows, cols, n):
    """Construct a CSR sparse matrix of shape (n, n)."""
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)


# ---------------------------------------------------------------------------
# Test 1: 3-node line graph  0 → 1 → 2, seed [0]
# PPR scores should be monotonically decreasing: score[0] > score[1] > score[2]
# ---------------------------------------------------------------------------

def test_line_graph_monotone():
    # 0 → 1, 1 → 2
    A = _csr([1.0, 1.0], [0, 1], [1, 2], 3)
    scores = personalized_pagerank(A, seed_indices=[0], damping=0.85)
    assert scores.shape == (3,)
    assert scores[0] > scores[1] > scores[2], (
        f"Expected monotone decrease; got {scores}"
    )


# ---------------------------------------------------------------------------
# Test 2: damping == 0 → result equals personalization vector exactly
# ---------------------------------------------------------------------------

def test_damping_zero_returns_personalization():
    A = _csr([1.0, 1.0, 1.0], [0, 1, 2], [1, 2, 3], 4)
    scores = personalized_pagerank(A, seed_indices=[2], damping=0.0)
    expected = np.array([0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_almost_equal(scores, expected)


# ---------------------------------------------------------------------------
# Test 3: symmetric 3-node triangle, single seed
# Seed node should have highest score; the other two are equal.
# ---------------------------------------------------------------------------

def test_triangle_single_seed():
    # Fully connected symmetric triangle (undirected)
    rows = [0, 0, 1, 1, 2, 2]
    cols = [1, 2, 0, 2, 0, 1]
    data = [1.0] * 6
    A = _csr(data, rows, cols, 3)
    scores = personalized_pagerank(A, seed_indices=[0], damping=0.85)
    assert scores[0] > scores[1], f"seed node not highest: {scores}"
    assert scores[0] > scores[2], f"seed node not highest: {scores}"
    np.testing.assert_almost_equal(scores[1], scores[2], decimal=6)


# ---------------------------------------------------------------------------
# Test 4: empty graph (n == 0) → empty array
# ---------------------------------------------------------------------------

def test_empty_graph():
    A = sp.csr_matrix((0, 0), dtype=float)
    scores = personalized_pagerank(A, seed_indices=[], damping=0.85)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (0,)


# ---------------------------------------------------------------------------
# Test 5: invalid inputs raise ValueError
# ---------------------------------------------------------------------------

def test_invalid_non_square():
    A = sp.csr_matrix((3, 4), dtype=float)
    with pytest.raises(ValueError, match="square"):
        personalized_pagerank(A, seed_indices=[0])


def test_invalid_empty_seeds():
    A = _csr([1.0], [0], [1], 3)
    with pytest.raises(ValueError, match="non-empty"):
        personalized_pagerank(A, seed_indices=[])


def test_invalid_seed_out_of_range():
    A = _csr([1.0], [0], [1], 3)
    with pytest.raises(ValueError, match="out of range"):
        personalized_pagerank(A, seed_indices=[5])


def test_invalid_damping_too_large():
    A = _csr([1.0], [0], [1], 3)
    with pytest.raises(ValueError, match="damping"):
        personalized_pagerank(A, seed_indices=[0], damping=1.5)


def test_invalid_seed_weights_wrong_length():
    A = _csr([1.0], [0], [1], 3)
    with pytest.raises(ValueError, match="same length"):
        personalized_pagerank(A, seed_indices=[0, 1], seed_weights=[0.5])


# ---------------------------------------------------------------------------
# Test 6: seed weights are honored on a disconnected graph
# seeds [0, 1] with weights [3, 1]; no edges → scores purely from teleport
# score[0] / score[1] should be ≈ 3
# ---------------------------------------------------------------------------

def test_seed_weights_honored():
    # 4 nodes, no edges at all
    A = sp.csr_matrix((4, 4), dtype=float)
    scores = personalized_pagerank(
        A,
        seed_indices=[0, 1],
        seed_weights=[3.0, 1.0],
        damping=0.85,
    )
    ratio = scores[0] / scores[1]
    np.testing.assert_almost_equal(ratio, 3.0, decimal=6)


# ---------------------------------------------------------------------------
# Test 7: stationary distribution sums to ≈ 1.0
# ---------------------------------------------------------------------------

def test_scores_sum_to_one():
    rng = np.random.default_rng(42)
    n = 5
    # Random sparse symmetric graph
    dense = rng.random((n, n))
    dense = (dense + dense.T) / 2
    dense[dense < 0.6] = 0.0  # ~40% fill
    A = sp.csr_matrix(dense)
    scores = personalized_pagerank(A, seed_indices=[0, 2], damping=0.85)
    assert abs(scores.sum() - 1.0) < 1e-6, f"scores.sum() = {scores.sum()}"


# ---------------------------------------------------------------------------
# Test 8: convergence within max_iter=50 on 10-node random graph
# max_iter=200 and max_iter=50 should give same result within 1e-5
# ---------------------------------------------------------------------------

def test_convergence():
    rng = np.random.default_rng(99)
    n = 10
    dense = rng.random((n, n))
    dense[dense < 0.3] = 0.0
    A = sp.csr_matrix(dense)
    scores50 = personalized_pagerank(A, seed_indices=[0], damping=0.85, max_iter=50)
    scores200 = personalized_pagerank(A, seed_indices=[0], damping=0.85, max_iter=200)
    np.testing.assert_allclose(scores50, scores200, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 9: dangling node handling — no NaN / Inf
# 3 nodes: 0→1, 0→2; nodes 1 and 2 have no outgoing edges (dangling)
# ---------------------------------------------------------------------------

def test_dangling_nodes_no_nan():
    # 0 → 1, 0 → 2; nodes 1 and 2 are sinks
    A = _csr([1.0, 1.0], [0, 0], [1, 2], 3)
    scores = personalized_pagerank(A, seed_indices=[0], damping=0.85)
    assert scores.shape == (3,)
    assert not np.any(np.isnan(scores)), f"NaN in scores: {scores}"
    assert not np.any(np.isinf(scores)), f"Inf in scores: {scores}"
    np.testing.assert_almost_equal(scores.sum(), 1.0, decimal=6)
