"""Tests de los constructores de topologías regulares (v8 — EXP-5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.graph.regular_subgraphs import TopologyWrapper, bipartite, cycle, d_regular
from src.graph.subgraph_selector import Subgraph
from src.quantum.dtqw import build_coin, build_shift
from src.quantum.encoding import build_port_map


def _sub(M: int = 8, seed: int = 0) -> Subgraph:
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.05, 1.0, size=(M, M))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0.0)
    return Subgraph(W_local=A, global_node_ids=np.arange(M, dtype=np.int64),
                    seed_idx_local=0)


def _degrees(W: np.ndarray) -> np.ndarray:
    return (W > 0).sum(axis=1)


@pytest.mark.parametrize("M,d", [(8, 3), (16, 4)])
def test_d_regular_exact(M: int, d: int) -> None:
    sub = _sub(M)
    for uniform in (False, True):
        reg = d_regular(sub, d, np.random.default_rng(1), uniform=uniform)
        assert (_degrees(reg.W_local) == d).all()
        assert np.allclose(reg.W_local, reg.W_local.T)
        assert np.allclose(np.diag(reg.W_local), 0.0)
        assert (reg.global_node_ids == sub.global_node_ids).all()


def test_d_regular_uniform_weights_are_one() -> None:
    reg = d_regular(_sub(8), 3, np.random.default_rng(2), uniform=True)
    vals = reg.W_local[reg.W_local > 0]
    assert np.allclose(vals, 1.0)


def test_cycle_structure() -> None:
    c = cycle(_sub(8))
    deg = _degrees(c.W_local)
    assert (deg == 2).all()
    # Conexo: el ciclo debe visitar los 8 nodos (una sola componente).
    visitados = {0}
    frontera = [0]
    while frontera:
        v = frontera.pop()
        for u in np.flatnonzero(c.W_local[v] > 0):
            if int(u) not in visitados:
                visitados.add(int(u))
                frontera.append(int(u))
    assert len(visitados) == 8


def test_bipartite_structure_fallback_scores() -> None:
    sub = _sub(8)
    scores = np.linspace(0, 1, 8)
    b = bipartite(sub, sectors=None, seed_scores=scores)
    grupo_a = b.W_local[0] == 0  # no-vecinos de 0 (más el propio 0)
    grupo_a[0] = True
    dentro = np.ix_(np.flatnonzero(grupo_a), np.flatnonzero(grupo_a))
    assert (b.W_local[dentro] == 0).all()  # sin aristas intra-grupo
    a = int(grupo_a.sum())
    assert (_degrees(b.W_local)[grupo_a] == 8 - a).all()


def test_bipartite_by_sector() -> None:
    sub = _sub(8)
    sectors = ["T"] * 20  # global ids 0..7 -> todos 'T' salvo dos
    sectors[2] = sectors[5] = "F"
    b = bipartite(sub, sectors=sectors)
    assert b.W_local[2, 5] == 0.0      # mismo grupo (resto)
    assert b.W_local[2, 0] == 1.0      # grupos distintos


@pytest.mark.parametrize("kind", ["dreg_aff", "dreg_uni", "cycle"])
def test_unitarity_on_rewired(kind: str) -> None:
    """U_t = S·C debe ser unitario sobre cada topología recableada."""
    wrapper = TopologyWrapper(inner=None, kind=kind, d=3, rng_seed=3)
    reg = wrapper.rewire(_sub(8))
    pmap = build_port_map(reg.W_local)
    C = build_coin(pmap)
    S = build_shift(pmap)
    U = S @ C
    assert np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=1e-10)


def test_wrapper_delegates_and_falls_back() -> None:
    class Dummy:
        name = "dummy"

        def candidate_set(self, subgraph, k, m):
            self.last = subgraph
            return subgraph.global_node_ids[:m]

    inner = Dummy()
    w = TopologyWrapper(inner=inner, kind="cycle", rng_seed=0)
    out = w.candidate_set(_sub(8), k=3, m=3)
    assert len(out) == 3
    assert (_degrees(inner.last.W_local) == 2).all()  # recibió el ciclo
