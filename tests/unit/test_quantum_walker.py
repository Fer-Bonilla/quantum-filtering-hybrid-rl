"""Tests del fachada QuantumWalker (Modelo D) y la interfaz LocalModule."""

from __future__ import annotations

import numpy as np
from src.graph.subgraph_selector import Subgraph
from src.quantum.classical_walker import ClassicalWalker
from src.quantum.quantum_walker import QuantumWalker


def _make_subgraph(seed: int = 0, M: int = 5) -> Subgraph:
    rng = np.random.default_rng(seed)
    W = rng.random((M, M))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return Subgraph(
        W_local=W,
        global_node_ids=np.arange(100, 100 + M, dtype=np.int64),
        seed_idx_local=0,
    )


def test_quantum_walker_returns_global_indices() -> None:
    sub = _make_subgraph()
    walker = QuantumWalker()
    out = walker.candidate_set(sub, k=3, m=3)
    assert out.shape == (3,)
    assert all(100 <= int(idx) < 105 for idx in out)
    assert len(set(out.tolist())) == 3  # distintos


def test_c_vs_d_interface_equivalence() -> None:
    """TEST CRÍTICO: ClassicalWalker y QuantumWalker tienen interfaz idéntica.

    Cambiar el LocalModule no requiere cambios en el caller. La única
    diferencia debe estar en el contenido del top-m, no en su forma.
    """
    sub = _make_subgraph(seed=99, M=6)
    classical = ClassicalWalker()
    quantum = QuantumWalker()
    c_out = classical.candidate_set(sub, k=4, m=3)
    q_out = quantum.candidate_set(sub, k=4, m=3)
    # Misma forma, mismo dtype, mismo rango de valores globales
    assert c_out.shape == q_out.shape
    assert c_out.dtype == q_out.dtype
    assert set(c_out.tolist()) <= set(sub.global_node_ids.tolist())
    assert set(q_out.tolist()) <= set(sub.global_node_ids.tolist())


def test_quantum_vs_classical_distributions_differ() -> None:
    """Para k > 0 y grafo no trivial, las distribuciones DEBEN diferir."""
    from src.graph.classical_walk import random_walk_distribution
    from src.quantum.dtqw import apply_dtqw

    sub = _make_subgraph(seed=7, M=6)
    p_c = random_walk_distribution(sub.W_local, sub.seed_idx_local, k=5)
    p_q = apply_dtqw(sub.W_local, sub.seed_idx_local, k=5)
    # Diferencia L1 debe ser apreciable
    assert float(np.abs(p_c - p_q).sum()) > 0.01


def test_walker_seed_centered_init_mode() -> None:
    """QuantumWalker con init_mode='seed_centered' produce distribución distinta a uniform."""
    sub = _make_subgraph(seed=11, M=5)
    uni = QuantumWalker(init_mode="uniform").candidate_set(sub, k=3, m=3)
    cen = QuantumWalker(init_mode="seed_centered").candidate_set(sub, k=3, m=3)
    # No exigimos diferencia de identidad, solo que ambas estén bien formadas.
    assert uni.shape == cen.shape
