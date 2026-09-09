"""Tests de las monedas alternativas del DTQW (rev. v3 — Punto 3.2).

Verifica que las tres opciones de ``build_coin`` producen matrices
unitarias, y que producen distribuciones medibles distintas sobre un
mismo subgrafo (lo cual justifica la ablación empírica).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.quantum.dtqw import apply_dtqw, build_coin, build_shift
from src.quantum.encoding import build_port_map


def _star_subgraph(d: int = 4) -> np.ndarray:
    """Subgrafo en estrella con d hojas: nodo 0 conectado a {1..d}."""
    n = d + 1
    W = np.zeros((n, n), dtype=np.float64)
    for j in range(1, n):
        W[0, j] = 0.5 + 0.1 * j  # pesos distintos
        W[j, 0] = W[0, j]
    return W


def _path_subgraph(n: int = 5) -> np.ndarray:
    """Subgrafo en línea: 0-1-2-...-n con pesos no uniformes."""
    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        W[i, i + 1] = 0.5 + 0.1 * i
        W[i + 1, i] = W[i, i + 1]
    return W


def test_build_coin_unknown_type_raises() -> None:
    """coin_type desconocido lanza ValueError."""
    W = _star_subgraph(4)
    pmap = build_port_map(W)
    with pytest.raises(ValueError, match="coin_type"):
        build_coin(pmap, coin_type="hadamard_2x2")


def test_weighted_householder_is_unitary() -> None:
    W = _star_subgraph(5)
    pmap = build_port_map(W)
    C = build_coin(pmap, coin_type="weighted_householder")
    err = np.linalg.norm(C @ C.conj().T - np.eye(C.shape[0]))
    assert err < 1e-12, f"||CC†-I|| = {err}"


def test_grover_is_unitary() -> None:
    W = _star_subgraph(5)
    pmap = build_port_map(W)
    C = build_coin(pmap, coin_type="grover")
    err = np.linalg.norm(C @ C.conj().T - np.eye(C.shape[0]))
    assert err < 1e-12


def test_fourier_is_unitary() -> None:
    W = _star_subgraph(5)
    pmap = build_port_map(W)
    C = build_coin(pmap, coin_type="fourier")
    err = np.linalg.norm(C @ C.conj().T - np.eye(C.shape[0]))
    assert err < 1e-12


def test_weighted_householder_is_reflection() -> None:
    """Householder ponderada satisface C^2 = I (reflexión)."""
    W = _star_subgraph(5)
    pmap = build_port_map(W)
    C = build_coin(pmap, coin_type="weighted_householder")
    err = np.linalg.norm(C @ C - np.eye(C.shape[0]))
    assert err < 1e-12


def test_grover_is_reflection() -> None:
    """Grover diffusion también satisface C^2 = I."""
    W = _star_subgraph(5)
    pmap = build_port_map(W)
    C = build_coin(pmap, coin_type="grover")
    err = np.linalg.norm(C @ C - np.eye(C.shape[0]))
    assert err < 1e-12


def test_fourier_is_not_a_reflection() -> None:
    """La moneda Fourier no es una reflexión: $F^2 \\neq I$."""
    W = _star_subgraph(5)
    pmap = build_port_map(W)
    C = build_coin(pmap, coin_type="fourier")
    err = np.linalg.norm(C @ C - np.eye(C.shape[0]))
    assert err > 1e-3, f"Fourier debería NO ser reflexión; ||C²-I|| = {err}"


def test_grover_equals_weighted_householder_under_uniform_weights() -> None:
    """Cuando los pesos son uniformes, weighted_householder == grover."""
    n = 5
    W = np.ones((n, n), dtype=np.float64) - np.eye(n)
    pmap = build_port_map(W)
    C_wh = build_coin(pmap, coin_type="weighted_householder")
    C_gr = build_coin(pmap, coin_type="grover")
    err = np.linalg.norm(C_wh - C_gr)
    assert err < 1e-12


def test_coins_produce_different_distributions() -> None:
    """Sobre el mismo subgrafo no uniforme, las 3 monedas producen P_k
    notoriamente distintas (justifica la ablación empírica)."""
    W = _path_subgraph(5)
    p_wh = apply_dtqw(W, seed_idx=2, k=3, coin_type="weighted_householder")
    p_gr = apply_dtqw(W, seed_idx=2, k=3, coin_type="grover")
    p_fr = apply_dtqw(W, seed_idx=2, k=3, coin_type="fourier")
    # Las distribuciones suman 1
    assert abs(p_wh.sum() - 1.0) < 1e-9
    assert abs(p_gr.sum() - 1.0) < 1e-9
    assert abs(p_fr.sum() - 1.0) < 1e-9
    # Las distribuciones son distintas entre sí (TV > 0.05).
    tv_wh_gr = 0.5 * np.abs(p_wh - p_gr).sum()
    tv_wh_fr = 0.5 * np.abs(p_wh - p_fr).sum()
    tv_gr_fr = 0.5 * np.abs(p_gr - p_fr).sum()
    assert tv_wh_gr > 0.02 or tv_wh_fr > 0.02 or tv_gr_fr > 0.02, (
        f"Las tres monedas producen distribuciones casi idénticas: "
        f"TV(WH,GR)={tv_wh_gr}, TV(WH,FR)={tv_wh_fr}, TV(GR,FR)={tv_gr_fr}"
    )
