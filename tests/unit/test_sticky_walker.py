"""Tests de src/quantum/sticky_walker.py (Modelo S — dosis-respuesta v7)."""

from __future__ import annotations

import numpy as np
import pytest
from src.graph.subgraph_selector import Subgraph
from src.quantum.sticky_walker import StickyWalker


def _subgraph(global_ids: list[int]) -> Subgraph:
    M = len(global_ids)
    rng = np.random.default_rng(0)
    W = rng.random((M, M))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return Subgraph(
        W_local=W,
        global_node_ids=np.array(global_ids, dtype=np.int64),
        seed_idx_local=0,
    )


def test_returns_global_indices() -> None:
    sub = _subgraph([10, 11, 12, 13, 14])
    walker = StickyWalker(seed=0, rotation_p=0.5)
    out = walker.candidate_set(sub, k=3, m=2)
    assert out.shape == (2,)
    assert set(out.tolist()) <= {10, 11, 12, 13, 14}


def test_unique_indices_each_call() -> None:
    sub = _subgraph(list(range(8)))
    walker = StickyWalker(seed=1, rotation_p=0.5)
    for _ in range(50):
        out = walker.candidate_set(sub, k=3, m=3)
        assert out.size == 3
        assert len(set(out.tolist())) == 3  # sin duplicados


def test_m_greater_than_M_is_clamped() -> None:
    sub = _subgraph([5, 6, 7])
    walker = StickyWalker(seed=0, rotation_p=0.7)
    out = walker.candidate_set(sub, k=3, m=10)
    assert out.size == 3
    assert set(out.tolist()) == {5, 6, 7}


def test_p0_frozen_on_stable_subgraph() -> None:
    """rotation_p=0 → la selección no cambia si H_t es estable."""
    sub = _subgraph(list(range(8)))
    walker = StickyWalker(seed=2, rotation_p=0.0)
    first = walker.candidate_set(sub, k=3, m=3)
    for _ in range(20):
        nxt = walker.candidate_set(sub, k=3, m=3)
        np.testing.assert_array_equal(np.sort(nxt), np.sort(first))
    assert walker.mean_turnover == pytest.approx(0.0)


def test_p1_high_turnover() -> None:
    """rotation_p=1 → re-sorteo total: rotación realizada alta y > 0."""
    sub = _subgraph(list(range(8)))
    walker = StickyWalker(seed=3, rotation_p=1.0)
    for _ in range(200):
        walker.candidate_set(sub, k=3, m=3)
    # m=3 de M=8: el solapamiento esperado por azar es bajo → turnover alto.
    assert walker.mean_turnover > 0.5


def test_turnover_monotonic_in_p() -> None:
    """La rotación realizada crece con rotation_p (la perilla funciona)."""
    sub = _subgraph(list(range(8)))
    turnovers = []
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        w = StickyWalker(seed=7, rotation_p=p)
        for _ in range(400):
            w.candidate_set(sub, k=3, m=3)
        turnovers.append(w.mean_turnover)
    # Monótonamente no decreciente (con margen numérico).
    for a, b in zip(turnovers, turnovers[1:]):
        assert b >= a - 1e-9
    assert turnovers[-1] > turnovers[0] + 0.3  # separación clara extremo-extremo


def test_reproducible_given_seed() -> None:
    sub = _subgraph(list(range(8)))
    w1 = StickyWalker(seed=42, rotation_p=0.5)
    w2 = StickyWalker(seed=42, rotation_p=0.5)
    for _ in range(30):
        np.testing.assert_array_equal(
            w1.candidate_set(sub, k=3, m=3),
            w2.candidate_set(sub, k=3, m=3),
        )


def test_forced_resample_when_node_leaves_subgraph() -> None:
    """rotation_p=0 pero un nodo retenido desaparece → se reemplaza por uno válido."""
    walker = StickyWalker(seed=5, rotation_p=0.0)
    sub_a = _subgraph([0, 1, 2, 3])
    held = walker.candidate_set(sub_a, k=3, m=3)
    # Subgrafo nuevo SIN algunos de los nodos retenidos.
    sub_b = _subgraph([2, 100, 101, 102])
    out = walker.candidate_set(sub_b, k=3, m=3)
    assert out.size == 3
    assert set(out.tolist()) <= {2, 100, 101, 102}
    # Cualquier nodo retenido que ya no existe NO puede aparecer.
    assert {0, 1, 3}.isdisjoint(set(out.tolist()))


def test_invalid_rotation_p_raises() -> None:
    with pytest.raises(ValueError):
        StickyWalker(seed=0, rotation_p=1.5)
    with pytest.raises(ValueError):
        StickyWalker(seed=0, rotation_p=-0.1)


def test_name_encodes_p() -> None:
    assert StickyWalker(seed=0, rotation_p=0.0).name == "sticky_p0.00"
    assert StickyWalker(seed=0, rotation_p=1.0).name == "sticky_p1.00"
