"""Tests del backend PennyLane (Sec. 7.10 / 8.20.5).

Invariantes:
- Sin ruido: ``apply_dtqw_pennylane`` ≈ ``apply_dtqw`` (matricial) a tol 1e-6.
- Σ P = 1 para todos los k.
- Bajo depolarizing ruidoso, la distribución tiende hacia uniforme (cuanto
  mayor el ruido, menor la varianza de las probs).
"""

from __future__ import annotations

import numpy as np
import pytest
from src.quantum.dtqw import apply_dtqw
from src.quantum.noise import NoiseSpec, apply_noise_layer
from src.quantum.pennylane_backend import (
    _next_power_of_two,
    _pad_state,
    _pad_unitary,
    apply_dtqw_pennylane,
)


def _ensure_connected(W: np.ndarray) -> np.ndarray:
    """Asegurar grafo conexo añadiendo aristas anillo si falta."""
    out = W.copy()
    for i in range(W.shape[0]):
        if out[i].sum() == 0:
            j = (i + 1) % W.shape[0]
            out[i, j] = out[j, i] = 0.5
    return out


def _random_W(M: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.random((M, M))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return _ensure_connected(W)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def test_next_power_of_two() -> None:
    assert _next_power_of_two(1) == 1
    assert _next_power_of_two(2) == 2
    assert _next_power_of_two(3) == 4
    assert _next_power_of_two(8) == 8
    assert _next_power_of_two(9) == 16
    assert _next_power_of_two(64) == 64


def test_pad_unitary_preserves_block() -> None:
    U = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)  # swap 2x2
    Up = _pad_unitary(U, 4)
    np.testing.assert_array_equal(Up[:2, :2], U)
    np.testing.assert_array_equal(Up[2:, 2:], np.eye(2))
    # Unitariedad
    np.testing.assert_allclose(Up @ Up.conj().T, np.eye(4), atol=1e-12)


def test_pad_state_zeros_in_padding() -> None:
    psi = np.array([1.0, 0.0], dtype=np.complex128)
    psi_p = _pad_state(psi, 4)
    np.testing.assert_array_equal(psi_p, [1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Consistencia matricial ↔ PennyLane (sin ruido)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,k", [(2, 1), (3, 2), (4, 3), (4, 5), (6, 4)])
@pytest.mark.quantum
def test_pennylane_matches_matrix(M: int, k: int) -> None:
    """TEST CRÍTICO: matrix backend ≈ PennyLane backend sin ruido."""
    W = _random_W(M, seed=M * 100 + k)
    p_matrix = apply_dtqw(W, seed_idx=0, k=k)
    p_pl = apply_dtqw_pennylane(W, seed_idx=0, k=k)
    np.testing.assert_allclose(p_pl, p_matrix, atol=1e-6)


@pytest.mark.quantum
def test_pennylane_seed_centered_consistency() -> None:
    W = _random_W(4, seed=42)
    p_m = apply_dtqw(W, seed_idx=1, k=3, init_mode="seed_centered")
    p_p = apply_dtqw_pennylane(W, seed_idx=1, k=3, init_mode="seed_centered")
    np.testing.assert_allclose(p_p, p_m, atol=1e-6)


# ---------------------------------------------------------------------------
# Conservación de probabilidad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 2, 5, 8])
@pytest.mark.quantum
def test_pennylane_probability_conservation(k: int) -> None:
    W = _random_W(4, seed=k + 1)
    probs = apply_dtqw_pennylane(W, seed_idx=0, k=k)
    assert probs.shape == (4,)
    assert (probs >= -1e-10).all()
    assert abs(probs.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Ruido
# ---------------------------------------------------------------------------


@pytest.mark.quantum
def test_noise_spec_inactive_by_default() -> None:
    assert not NoiseSpec().is_active()
    assert not NoiseSpec(depolarizing_prob=0.0, dephasing_prob=0.0).is_active()
    assert NoiseSpec(depolarizing_prob=0.1).is_active()
    assert NoiseSpec(dephasing_prob=0.05).is_active()


@pytest.mark.quantum
def test_pennylane_with_noise_returns_valid_distribution() -> None:
    """Con ruido el resultado debe seguir siendo una distribución válida."""
    W = _random_W(4, seed=7)
    noise = NoiseSpec(depolarizing_prob=0.1, dephasing_prob=0.05)
    probs = apply_dtqw_pennylane(W, seed_idx=0, k=3, noise=noise)
    assert probs.shape == (4,)
    assert (probs >= -1e-10).all()
    assert abs(probs.sum() - 1.0) < 1e-9


@pytest.mark.quantum
def test_high_depolarizing_flattens_distribution() -> None:
    """Con ruido alto, la distribución debe ser MÁS uniforme que sin ruido.

    Métrica: la varianza de P_k debe ser menor con ruido alto que sin él
    (saturación hacia distribución máximamente mezclada).
    """
    W = _random_W(4, seed=11)
    p_clean = apply_dtqw_pennylane(W, seed_idx=0, k=5, init_mode="seed_centered")
    p_noisy = apply_dtqw_pennylane(
        W,
        seed_idx=0,
        k=5,
        init_mode="seed_centered",
        noise=NoiseSpec(depolarizing_prob=0.3),
    )
    # Tras ruido moderado, la varianza es menor (más cerca de uniforme)
    assert np.var(p_noisy) < np.var(p_clean)


@pytest.mark.quantum
def test_apply_noise_layer_no_op_when_inactive() -> None:
    """apply_noise_layer es no-op si NoiseSpec().is_active() = False.

    Verificable por consistencia matrix vs pennylane con noise inactivo.
    """
    W = _random_W(3, seed=33)
    p1 = apply_dtqw_pennylane(W, seed_idx=0, k=3, noise=NoiseSpec())
    p2 = apply_dtqw_pennylane(W, seed_idx=0, k=3, noise=None)
    np.testing.assert_allclose(p1, p2, atol=1e-12)
    # apply_noise_layer no requiere QNode aquí (solo verificamos su existencia)
    _ = apply_noise_layer  # smoke


# ---------------------------------------------------------------------------
# QuantumWalker integración
# ---------------------------------------------------------------------------


@pytest.mark.quantum
def test_quantum_walker_matrix_vs_pennylane() -> None:
    """QuantumWalker(backend='pennylane') sin ruido coincide con backend='matrix'."""
    from src.graph.subgraph_selector import Subgraph
    from src.quantum.quantum_walker import QuantumWalker

    W = _random_W(4, seed=55)
    sub = Subgraph(
        W_local=W,
        global_node_ids=np.array([10, 20, 30, 40]),
        seed_idx_local=0,
    )
    walker_m = QuantumWalker(backend="matrix")
    walker_p = QuantumWalker(backend="pennylane")
    out_m = walker_m.candidate_set(sub, k=3, m=2)
    out_p = walker_p.candidate_set(sub, k=3, m=2)
    np.testing.assert_array_equal(out_m, out_p)


@pytest.mark.quantum
def test_quantum_walker_with_noise_name() -> None:
    from src.quantum.quantum_walker import QuantumWalker

    clean = QuantumWalker(backend="matrix")
    noisy = QuantumWalker(noise=NoiseSpec(depolarizing_prob=0.1))
    assert clean.name == "quantum"
    assert noisy.name == "quantum_noisy"
    assert noisy._effective_backend() == "pennylane"
