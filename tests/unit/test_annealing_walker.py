"""Tests del AnnealingWalker y del QUBO de selección (v6 — Punto 3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.graph.subgraph_selector import Subgraph
from src.quantum.annealing_walker import (
    AnnealingWalker,
    anneal_marginals,
    build_qubo_energies,
)
from src.quantum.random_walker import RandomWalker


def _star_subgraph(m_nodes: int = 6, strong: int = 1) -> Subgraph:
    """Estrella: seed=0 conectado a todos; el nodo ``strong`` con peso alto."""
    W = np.zeros((m_nodes, m_nodes))
    for j in range(1, m_nodes):
        W[0, j] = W[j, 0] = 0.2
    W[0, strong] = W[strong, 0] = 0.9
    return Subgraph(
        W_local=W,
        global_node_ids=np.arange(100, 100 + m_nodes, dtype=np.int64),
        seed_idx_local=0,
    )


# ---------------------------------------------------------------------------
# QUBO
# ---------------------------------------------------------------------------


def test_qubo_ground_state_has_cardinality_m() -> None:
    """El estado fundamental del QUBO tiene exactamente m unos."""
    sub = _star_subgraph(6)
    for m in (2, 3, 4):
        e = build_qubo_energies(sub.W_local, sub.seed_idx_local, m)
        ground = int(np.argmin(e))
        n_ones = bin(ground).count("1")
        assert n_ones == m, f"ground state con {n_ones} unos; esperado m={m}"


def test_qubo_ground_state_prefers_strong_affinity() -> None:
    """Con m=2, el fundamental contiene seed + el vecino de mayor peso."""
    sub = _star_subgraph(6, strong=3)
    e = build_qubo_energies(sub.W_local, sub.seed_idx_local, 2)
    ground = int(np.argmin(e))
    selected = {i for i in range(6) if (ground >> i) & 1}
    assert 0 in selected, "el seed debe estar en el conjunto óptimo"
    assert 3 in selected, "el vecino fuerte debe estar en el conjunto óptimo"


def test_qubo_rejects_oversized_subgraph() -> None:
    W = np.zeros((17, 17))
    with pytest.raises(ValueError, match="excede"):
        build_qubo_energies(W, 0, 3)


# ---------------------------------------------------------------------------
# Evolución adiabática
# ---------------------------------------------------------------------------


def test_anneal_marginals_in_unit_interval() -> None:
    sub = _star_subgraph(5)
    e = build_qubo_energies(sub.W_local, sub.seed_idx_local, 3)
    p = anneal_marginals(e, 5, n_steps=60)
    assert p.shape == (5,)
    assert np.all(p >= -1e-12) and np.all(p <= 1.0 + 1e-12)


def test_anneal_concentrates_on_ground_state_members() -> None:
    """Con annealing suficientemente lento, las marginales de los nodos del
    conjunto óptimo superan a las del resto."""
    sub = _star_subgraph(6, strong=2)
    m = 2
    e = build_qubo_energies(sub.W_local, sub.seed_idx_local, m)
    p = anneal_marginals(e, 6, n_steps=200, total_time=40.0)
    ground = int(np.argmin(e))
    members = [i for i in range(6) if (ground >> i) & 1]
    non_members = [i for i in range(6) if i not in members]
    assert min(p[members]) > max(p[non_members]), (
        f"marginales miembros {p[members]} deben superar a no-miembros "
        f"{p[non_members]}"
    )


def test_anneal_is_deterministic() -> None:
    """Sin muestreo: la evolución es determinista (mismas marginales)."""
    sub = _star_subgraph(5)
    e = build_qubo_energies(sub.W_local, sub.seed_idx_local, 2)
    p1 = anneal_marginals(e, 5, n_steps=50)
    p2 = anneal_marginals(e, 5, n_steps=50)
    np.testing.assert_allclose(p1, p2, atol=1e-14)


# ---------------------------------------------------------------------------
# Walker (protocolo LocalModule)
# ---------------------------------------------------------------------------


def test_annealing_walker_returns_m_global_ids() -> None:
    sub = _star_subgraph(6)
    walker = AnnealingWalker()
    out = walker.candidate_set(sub, k=3, m=3)
    assert out.shape == (3,)
    assert set(out).issubset(set(sub.global_node_ids.tolist()))


def test_annealing_walker_includes_strong_neighbor() -> None:
    sub = _star_subgraph(6, strong=4)
    walker = AnnealingWalker()
    out = walker.candidate_set(sub, k=5, m=2)
    assert 104 in out, f"el vecino fuerte (global 104) debe estar en {out}"


def test_annealing_walker_m_clipped_to_subgraph_size() -> None:
    sub = _star_subgraph(3)
    walker = AnnealingWalker()
    out = walker.candidate_set(sub, k=3, m=10)
    assert out.shape == (3,)


def test_annealing_walker_single_node() -> None:
    sub = Subgraph(
        W_local=np.zeros((1, 1)),
        global_node_ids=np.array([7], dtype=np.int64),
        seed_idx_local=0,
    )
    out = AnnealingWalker().candidate_set(sub, k=3, m=5)
    assert out.tolist() == [7]


# ---------------------------------------------------------------------------
# RandomWalker (Modelo R)
# ---------------------------------------------------------------------------


def test_random_walker_returns_m_unique_global_ids() -> None:
    sub = _star_subgraph(6)
    walker = RandomWalker(seed=42)
    out = walker.candidate_set(sub, k=3, m=4)
    assert out.shape == (4,)
    assert len(set(out.tolist())) == 4
    assert set(out).issubset(set(sub.global_node_ids.tolist()))


def test_random_walker_reproducible_by_seed() -> None:
    sub = _star_subgraph(8)
    seq1 = [RandomWalker(seed=7).candidate_set(sub, 3, 3).tolist() for _ in range(1)]
    seq2 = [RandomWalker(seed=7).candidate_set(sub, 3, 3).tolist() for _ in range(1)]
    assert seq1 == seq2


def test_random_walker_varies_across_calls() -> None:
    """Llamadas sucesivas del MISMO walker producen conjuntos distintos
    (el generador avanza): es una máscara aleatoria, no fija."""
    sub = _star_subgraph(10)
    walker = RandomWalker(seed=0)
    outs = {tuple(sorted(walker.candidate_set(sub, 3, 3).tolist())) for _ in range(20)}
    assert len(outs) > 1
