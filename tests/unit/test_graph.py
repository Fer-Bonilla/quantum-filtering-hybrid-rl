"""Tests del módulo src/graph/ (affinity, sparsify, builder, subgraph)."""

from __future__ import annotations

import numpy as np
import pytest
from src.graph.affinity import (
    affinity_matrix,
    normalize_affinity,
    positive_correlation,
    sector_affinity,
)
from src.graph.graph_builder import GraphSpec, build_graph
from src.graph.sparsify import degree_vector, knn_sparsify
from src.graph.subgraph_selector import seed_score, select_subgraph

# ---------------------------------------------------------------------------
# Affinity
# ---------------------------------------------------------------------------


def test_positive_correlation_symmetric_and_nonneg() -> None:
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((100, 5))
    C = positive_correlation(returns)
    assert C.shape == (5, 5)
    np.testing.assert_allclose(C, C.T, atol=1e-12)
    assert (C >= 0).all()
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-12)


def test_sector_affinity_binary() -> None:
    S = sector_affinity(["IT", "IT", "FIN", "IT", "FIN"])
    np.testing.assert_array_equal(S, S.T)
    assert S[0, 1] == 1.0
    assert S[0, 2] == 0.0
    assert S[2, 4] == 1.0


def test_affinity_alpha_beta_validation() -> None:
    rng = np.random.default_rng(0)
    R = rng.standard_normal((50, 3))
    with pytest.raises(ValueError):
        affinity_matrix(R, ["A", "B", "C"], alpha=0.5, beta=0.3)
    # alpha=1, beta=0 sin sectores debe funcionar
    A = affinity_matrix(R, None, alpha=1.0, beta=0.0)
    assert A.shape == (3, 3)


def test_normalize_affinity_within_unit_interval() -> None:
    rng = np.random.default_rng(0)
    M = rng.random((6, 6)) * 10
    M = 0.5 * (M + M.T)
    A = normalize_affinity(M, eps=1e-8)
    assert (A >= 0).all()
    # Tolerancia laxa por el eps aditivo
    assert A.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Sparsify
# ---------------------------------------------------------------------------


def test_knn_sparsify_symmetric_avg() -> None:
    rng = np.random.default_rng(0)
    A = rng.random((10, 10))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    W = knn_sparsify(A, k=3, sym_mode="avg")
    assert W.shape == (10, 10)
    np.testing.assert_allclose(W, W.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(W), 0.0, atol=1e-12)
    assert (W >= 0).all()


def test_knn_sparsify_mutual_subset_of_avg() -> None:
    rng = np.random.default_rng(0)
    A = rng.random((8, 8))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    W_avg = knn_sparsify(A, k=2, sym_mode="avg")
    W_mut = knn_sparsify(A, k=2, sym_mode="mutual")
    # Las aristas mutuas son subconjunto de las avg
    edges_avg = W_avg > 0
    edges_mut = W_mut > 0
    assert ((edges_mut & ~edges_avg).sum()) == 0


def test_knn_sparsify_each_node_has_at_most_k_outgoing() -> None:
    """En modo 'mutual' cada nodo tiene <= k vecinos; en 'avg' puede tener más."""
    rng = np.random.default_rng(0)
    A = rng.random((20, 20))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    W = knn_sparsify(A, k=3, sym_mode="mutual")
    nonzero_per_row = (W > 0).sum(axis=1)
    assert (nonzero_per_row <= 3).all()


def test_knn_sparsify_invalid_k() -> None:
    A = np.eye(5)
    with pytest.raises(ValueError):
        knn_sparsify(A, k=0)
    with pytest.raises(ValueError):
        knn_sparsify(A, k=5)


def test_degree_vector_matches_row_sums() -> None:
    rng = np.random.default_rng(0)
    W = rng.random((4, 4))
    np.testing.assert_allclose(degree_vector(W), W.sum(axis=1))


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def test_build_graph_smoke() -> None:
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((60, 6))
    names = ["A", "B", "C", "D", "E", "F"]
    spec = GraphSpec(alpha=1.0, beta=0.0, k_neighbors=2, sym_mode="avg")
    g = build_graph(returns, names, sectors=None, spec=spec)
    assert g.n_nodes == 6
    assert g.node_names == tuple(names)
    np.testing.assert_allclose(g.W, g.W.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(g.W), 0.0, atol=1e-12)


def test_build_graph_with_sectors() -> None:
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((60, 4))
    sectors = ["IT", "FIN", "IT", "FIN"]
    spec = GraphSpec(alpha=0.6, beta=0.4, k_neighbors=2)
    g = build_graph(returns, ["A", "B", "C", "D"], sectors=sectors, spec=spec)
    assert g.W.shape == (4, 4)


# ---------------------------------------------------------------------------
# Subgraph selector
# ---------------------------------------------------------------------------


def test_seed_score_higher_for_better_sharpe() -> None:
    # Activo 0 con retorno alto y vol baja; activo 1 con retorno bajo y vol alta
    R = np.column_stack(
        [
            np.full(50, 0.01) + 1e-3 * np.random.default_rng(0).standard_normal(50),
            np.zeros(50) + 0.05 * np.random.default_rng(1).standard_normal(50),
        ]
    )
    scores = seed_score(R)
    assert scores[0] > scores[1]


def test_select_subgraph_includes_seed_and_size() -> None:
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((60, 10))
    spec = GraphSpec(alpha=1.0, beta=0.0, k_neighbors=3, sym_mode="avg")
    g = build_graph(returns, [f"T{i}" for i in range(10)], sectors=None, spec=spec)
    scores = seed_score(returns[-20:])
    sub = select_subgraph(g, scores, max_size=5)
    assert sub.size <= 5
    assert sub.size >= 2
    assert int(scores.argmax()) == sub.seed_global
    # seed_idx_local debe corresponder al primer elemento
    assert sub.seed_idx_local == 0


def test_select_subgraph_uses_eligible_mask() -> None:
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((60, 6))
    spec = GraphSpec(alpha=1.0, beta=0.0, k_neighbors=3)
    g = build_graph(returns, [f"T{i}" for i in range(6)], sectors=None, spec=spec)
    scores = seed_score(returns[-20:])
    elig = np.array([True, False, False, True, True, True])
    sub = select_subgraph(g, scores, max_size=4, eligible_mask=elig)
    # El seed elegido debe estar en {0, 3, 4, 5}
    assert sub.seed_global in {0, 3, 4, 5}


def test_subgraph_W_local_symmetric_no_diag() -> None:
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((60, 8))
    spec = GraphSpec(alpha=1.0, beta=0.0, k_neighbors=3)
    g = build_graph(returns, [f"T{i}" for i in range(8)], sectors=None, spec=spec)
    scores = seed_score(returns[-20:])
    sub = select_subgraph(g, scores, max_size=5)
    np.testing.assert_allclose(sub.W_local, sub.W_local.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(sub.W_local), 0.0, atol=1e-12)
