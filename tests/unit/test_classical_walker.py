"""Tests de src/graph/classical_walk.py y src/quantum/classical_walker.py."""

from __future__ import annotations

import numpy as np
import pytest
from src.graph.classical_walk import random_walk_distribution, transition_matrix
from src.graph.subgraph_selector import Subgraph
from src.quantum.classical_walker import ClassicalWalker


def test_transition_matrix_row_stochastic() -> None:
    W = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 2.0], [0.5, 2.0, 0.0]])
    P = transition_matrix(W)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)


def test_transition_matrix_isolated_node_uniform() -> None:
    W = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    P = transition_matrix(W)
    # Nodo 2 está aislado: fila uniforme
    np.testing.assert_allclose(P[2], 1.0 / 3, atol=1e-12)


@pytest.mark.parametrize("k", [0, 1, 2, 5, 10])
def test_random_walk_sums_to_one(k: int) -> None:
    rng = np.random.default_rng(0)
    W = rng.random((5, 5))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    p = random_walk_distribution(W, seed_idx=0, k=k)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-10)


def test_classical_walker_returns_global_indices() -> None:
    rng = np.random.default_rng(0)
    W_local = rng.random((5, 5))
    W_local = 0.5 * (W_local + W_local.T)
    np.fill_diagonal(W_local, 0.0)
    sub = Subgraph(
        W_local=W_local,
        global_node_ids=np.array([10, 11, 12, 13, 14]),
        seed_idx_local=0,
    )
    walker = ClassicalWalker()
    out = walker.candidate_set(sub, k=3, m=2)
    assert out.shape == (2,)
    assert set(out.tolist()) <= {10, 11, 12, 13, 14}
