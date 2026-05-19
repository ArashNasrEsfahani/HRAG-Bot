"""Personalized PageRank (PPR) for the HRAG knowledge graph layer.

Pure-math module — no I/O, no LLM, no DB.
scipy and numpy are lazy-imported so this module can be imported even
when those heavy deps are absent (they will only be required when the
function is actually called).
"""
from __future__ import annotations

from typing import Sequence


def personalized_pagerank(
    adjacency,  # scipy.sparse matrix, shape (n, n), nonneg weights
    seed_indices: Sequence[int],
    damping: float = 0.5,  # alpha; HippoRAG default
    max_iter: int = 50,
    tol: float = 1e-6,
    seed_weights=None,  # Optional[Sequence[float]] — uniform if None
) -> "np.ndarray":
    """Run personalized PageRank on a sparse adjacency matrix.

    Parameters
    ----------
    adjacency:
        A scipy.sparse matrix of shape (n, n) with non-negative edge weights.
    seed_indices:
        Node indices from which the random walk is seeded (teleportation targets).
    damping:
        Probability of following an edge (vs. teleporting to seed). Range [0, 1].
        Default 0.85 is common in literature; HippoRAG uses 0.5.
    max_iter:
        Maximum number of power-iteration steps.
    tol:
        L1 convergence tolerance.
    seed_weights:
        Optional non-negative weights for each seed node. Normalised internally.
        When None, each seed node receives equal weight.

    Returns
    -------
    np.ndarray
        1-D float array of length n with PPR scores summing to ≈ 1.
        Returns an empty array when n == 0.
    """
    try:
        import numpy as np  # noqa: PLC0415
        import scipy.sparse as sp  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "scipy is required for PPR; install with: "
            "pip install 'hrag[kg]'"
        ) from exc

    # ------------------------------------------------------------------
    # 1. Basic validation
    # ------------------------------------------------------------------
    shape = adjacency.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            f"adjacency must be a square matrix; got shape {shape}"
        )
    n = shape[0]

    # Edge case: empty graph
    if n == 0:
        return np.array([], dtype=float)

    seed_indices = list(seed_indices)
    if len(seed_indices) == 0:
        raise ValueError("seed_indices must be non-empty")

    for idx in seed_indices:
        if not (0 <= idx < n):
            raise ValueError(
                f"seed index {idx} is out of range for graph with {n} nodes"
            )

    if not (0.0 <= damping <= 1.0):
        raise ValueError(
            f"damping must be in [0, 1]; got {damping}"
        )

    if seed_weights is not None:
        sw = list(seed_weights)
        if len(sw) != len(seed_indices):
            raise ValueError(
                "seed_weights must have the same length as seed_indices; "
                f"got {len(sw)} vs {len(seed_indices)}"
            )
        if sum(sw) <= 0:
            raise ValueError("seed_weights must sum to a positive value")

    # ------------------------------------------------------------------
    # 2. Build row-stochastic transition matrix
    # ------------------------------------------------------------------
    A = adjacency.tocsr().astype(float)

    # Row sums; rows with sum == 0 are dangling nodes.
    row_sums = np.asarray(A.sum(axis=1)).flatten()
    dangling_mask = row_sums == 0.0

    # Normalise rows with nonzero sum.
    # Divide each row i by row_sums[i] where row_sums[i] > 0.
    # We do this by scaling with a diagonal matrix.
    inv_row_sums = np.where(dangling_mask, 0.0, 1.0 / np.where(dangling_mask, 1.0, row_sums))
    D_inv = sp.diags(inv_row_sums, format="csr")
    M = D_inv @ A  # row-stochastic (dangling rows are all-zero)

    # We will use M.T to multiply column-stochastically: M.T @ x
    MT = M.T.tocsr()

    # ------------------------------------------------------------------
    # 3. Personalization vector
    # ------------------------------------------------------------------
    p = np.zeros(n, dtype=float)
    if seed_weights is None:
        p[seed_indices] = 1.0 / len(seed_indices)
    else:
        sw_arr = np.array(seed_weights, dtype=float)
        sw_arr = sw_arr / sw_arr.sum()
        for i, idx in enumerate(seed_indices):
            p[idx] = sw_arr[i]

    # ------------------------------------------------------------------
    # 4. Power iteration
    # ------------------------------------------------------------------
    x = p.copy()

    # Pre-compute which node indices are dangling for fast mass calculation
    dangling_indices = np.where(dangling_mask)[0]

    for _ in range(max_iter):
        # Dangling node mass: probability mass that would "leak"
        dangling_mass = x[dangling_indices].sum() if len(dangling_indices) > 0 else 0.0

        x_new = (
            damping * (np.asarray(MT @ x).flatten())
            + damping * dangling_mass * p
            + (1.0 - damping) * p
        )

        if np.linalg.norm(x_new - x, 1) < tol:
            x = x_new
            break
        x = x_new

    return x
