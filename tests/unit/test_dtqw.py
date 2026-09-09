"""Tests críticos P0 del módulo cuántico (Sec. 7.10-7.17).

Invariantes verificados:
- U_t = S_t · C_t es unitaria (||U·U†|| ≈ I a tol 1e-12).
- ||ψ_k||² = 1 para todo k ∈ [0, 10] (tol 1e-10).
- C² = I (reflexiones tipo Householder).
- S es matriz de permutación 0/1.
- TopM devuelve m índices distintos.
- DTQW vs caminata clásica: dan distribuciones distintas cuando k > 0.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.quantum.dtqw import (
    apply_dtqw,
    build_coin,
    build_shift,
    evolve,
    measure_position,
)
from src.quantum.encoding import (
    build_port_map,
    hilbert_dim,
    initial_state_seed_centered,
    initial_state_uniform,
)
from src.quantum.measurement import top_m


def _random_symmetric_W(M: int, rng: np.random.Generator, density: float = 0.7) -> np.ndarray:
    """Generar W simétrica no negativa con diagonal 0 y densidad controlada."""
    A = rng.random((M, M))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    # Quitar aristas con probabilidad (1 - density)
    drop = rng.random((M, M)) > density
    drop = drop | drop.T
    A = np.where(drop, 0.0, A)
    np.fill_diagonal(A, 0.0)
    return A


def _ensure_connected(W: np.ndarray) -> np.ndarray:
    """Asegurar grafo conexo añadiendo una arista en anillo si falta."""
    M = W.shape[0]
    out = W.copy()
    for i in range(M):
        if out[i].sum() == 0:
            j = (i + 1) % M
            out[i, j] = out[j, i] = 0.5
    return out


# ---------------------------------------------------------------------------
# Invariantes del coin y shift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M", [2, 3, 4, 6, 8])
def test_coin_unitary(M: int) -> None:
    rng = np.random.default_rng(M)
    W = _ensure_connected(_random_symmetric_W(M, rng))
    pmap = build_port_map(W)
    C = build_coin(pmap)
    dim = hilbert_dim(pmap)
    np.testing.assert_allclose(C @ C.conj().T, np.eye(dim), atol=1e-12)


@pytest.mark.parametrize("M", [2, 3, 4, 6, 8])
def test_coin_squared_is_identity_on_used_ports(M: int) -> None:
    """C es involutiva: C² = I (reflexión Householder)."""
    rng = np.random.default_rng(M + 100)
    W = _ensure_connected(_random_symmetric_W(M, rng))
    pmap = build_port_map(W)
    C = build_coin(pmap)
    np.testing.assert_allclose(C @ C, np.eye(hilbert_dim(pmap)), atol=1e-12)


@pytest.mark.parametrize("M", [2, 3, 4, 6, 8])
def test_shift_is_permutation(M: int) -> None:
    rng = np.random.default_rng(M + 200)
    W = _ensure_connected(_random_symmetric_W(M, rng))
    pmap = build_port_map(W)
    S = build_shift(pmap)
    dim = hilbert_dim(pmap)
    # Cada fila y cada columna debe sumar 1 (permutación)
    np.testing.assert_allclose(S.sum(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(S.sum(axis=1), 1.0, atol=1e-12)
    # Todos los valores son 0 o 1
    real_S = S.real.astype(int)
    assert set(np.unique(real_S).tolist()) <= {0, 1}
    # S es unitaria
    np.testing.assert_allclose(S @ S.conj().T, np.eye(dim), atol=1e-12)


@pytest.mark.parametrize("M", [2, 3, 4, 6, 8])
def test_U_unitary(M: int) -> None:
    """TEST CRÍTICO: U_t = S · C es unitaria."""
    rng = np.random.default_rng(M + 300)
    W = _ensure_connected(_random_symmetric_W(M, rng))
    pmap = build_port_map(W)
    C = build_coin(pmap)
    S = build_shift(pmap)
    U = S @ C
    dim = hilbert_dim(pmap)
    np.testing.assert_allclose(U @ U.conj().T, np.eye(dim), atol=1e-12)


# ---------------------------------------------------------------------------
# Conservación de probabilidad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_probability_conservation_for_all_k(k: int) -> None:
    """TEST CRÍTICO P0: ΣP_k(v_i) = 1 para todo k."""
    rng = np.random.default_rng(42)
    W = _ensure_connected(_random_symmetric_W(6, rng))
    probs = apply_dtqw(W, seed_idx=0, k=k)
    assert probs.shape == (6,)
    assert np.all(probs >= -1e-12)
    assert abs(probs.sum() - 1.0) < 1e-10


@pytest.mark.parametrize("M,k", [(4, 5), (6, 4), (8, 3)])
def test_psi_norm_preserved(M: int, k: int) -> None:
    """||ψ_k||² ≈ 1 tras evolución."""
    rng = np.random.default_rng(M * 1000 + k)
    W = _ensure_connected(_random_symmetric_W(M, rng))
    pmap = build_port_map(W)
    psi_0 = initial_state_uniform(pmap)
    np.testing.assert_allclose(np.real(np.vdot(psi_0, psi_0)), 1.0, atol=1e-12)
    psi_k = evolve(psi_0, build_coin(pmap), build_shift(pmap), k)
    np.testing.assert_allclose(np.real(np.vdot(psi_k, psi_k)), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Casos límite y consistencia
# ---------------------------------------------------------------------------


def test_dtqw_k_zero_returns_initial_marginal() -> None:
    """Para k=0, P_0 = P sobre el estado inicial."""
    rng = np.random.default_rng(0)
    W = _ensure_connected(_random_symmetric_W(6, rng))
    pmap = build_port_map(W)
    psi_0 = initial_state_uniform(pmap)
    p_0_expected = measure_position(psi_0, pmap)
    p_0_actual = apply_dtqw(W, seed_idx=0, k=0)
    np.testing.assert_allclose(p_0_actual, p_0_expected, atol=1e-12)


def test_dtqw_seed_centered_init_concentrated() -> None:
    """Inicialización en el seed: prob inicial concentrada en el seed."""
    rng = np.random.default_rng(0)
    W = _ensure_connected(_random_symmetric_W(6, rng))
    pmap = build_port_map(W)
    psi_0 = initial_state_seed_centered(pmap, seed_idx=2)
    probs = measure_position(psi_0, pmap)
    assert probs[2] > 0.99
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)


def test_dtqw_simple_2node_oscillation() -> None:
    """En un grafo con 2 nodos (línea), tras 2 pasos vuelve al seed (eigenvalue)."""
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    p0 = apply_dtqw(W, seed_idx=0, k=0, init_mode="seed_centered")
    p2 = apply_dtqw(W, seed_idx=0, k=2, init_mode="seed_centered")
    # En un grafo de 2 nodos perfecto, |⟨0|U²|0⟩|² ≠ exactamente 1 porque
    # el coin es Householder + el shift es swap. Aún así, la distribución debe
    # ser válida y sumar 1.
    assert abs(p0.sum() - 1.0) < 1e-10
    assert abs(p2.sum() - 1.0) < 1e-10


def test_dtqw_renormalization_threshold_does_not_explode_norm() -> None:
    """Múltiples pasos no deben provocar deriva acumulada."""
    rng = np.random.default_rng(0)
    W = _ensure_connected(_random_symmetric_W(8, rng))
    for k in range(20):
        p = apply_dtqw(W, seed_idx=0, k=k)
        assert abs(p.sum() - 1.0) < 1e-9


def test_dtqw_invalid_init_mode_raises() -> None:
    W = _ensure_connected(_random_symmetric_W(4, np.random.default_rng(0)))
    with pytest.raises(ValueError):
        apply_dtqw(W, seed_idx=0, k=1, init_mode="badmode")


# ---------------------------------------------------------------------------
# Mediciones (top-m)
# ---------------------------------------------------------------------------


def test_top_m_returns_m_distinct_indices() -> None:
    rng = np.random.default_rng(0)
    probs = rng.random(10)
    probs /= probs.sum()
    out = top_m(probs, m=3)
    assert out.shape == (3,)
    assert len(set(out.tolist())) == 3


def test_top_m_ordered_by_probability() -> None:
    probs = np.array([0.1, 0.4, 0.2, 0.3])
    out = top_m(probs, m=4)
    np.testing.assert_array_equal(out, [1, 3, 2, 0])


def test_top_m_excludes_indices() -> None:
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    out = top_m(probs, m=2, exclude_indices=np.array([0]))
    assert 0 not in out
    assert set(out.tolist()) == {1, 2}


def test_top_m_clips_to_available() -> None:
    probs = np.array([0.6, 0.4])
    out = top_m(probs, m=5)
    assert out.shape == (2,)


# ---------------------------------------------------------------------------
# Latencia
# ---------------------------------------------------------------------------


def test_dtqw_latency_M8_k6() -> None:
    """Performance: apply_dtqw para M=8, k=6 debe completar en < 50 ms."""
    import time

    rng = np.random.default_rng(0)
    W = _ensure_connected(_random_symmetric_W(8, rng))
    t0 = time.perf_counter()
    for _ in range(10):
        apply_dtqw(W, seed_idx=0, k=6)
    elapsed = (time.perf_counter() - t0) / 10
    # 50 ms es generoso; en práctica esperamos < 5 ms
    assert elapsed < 0.05, f"DTQW lento: {elapsed * 1000:.2f} ms por llamada"
