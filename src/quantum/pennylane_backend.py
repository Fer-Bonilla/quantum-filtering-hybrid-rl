"""Backend cuántico PennyLane para la DTQW (Sec. 7.10, ablación 8.20.5).

Wrapper opcional sobre la implementación matricial. Útil para:

1. **Validación cruzada** del backend matricial: ``apply_dtqw_pennylane`` debe
   coincidir con ``apply_dtqw`` (matrix) sobre M ≤ 6 cuando ``noise=None``.
2. **Experimentos con ruido**: cuando ``NoiseSpec.is_active()``, se aplican
   canales depolarizing/dephasing entre pasos sobre ``default.mixed``.

Detalle técnico — Padding:
    PennyLane requiere matrices unitarias de tamaño 2^n por 2^n. Como el
    Hilbert space del DTQW es M*d_max (no necesariamente potencia de 2),
    se padea con bloque identidad. El estado inicial es 0 en las
    posiciones extra; el padding ``I_pad`` no produce probabilidad
    espuria porque ``psi_pad * I_pad * psi_pad = 0`` cuando ``psi_pad``
    está en el bloque útil.
"""

from __future__ import annotations

import math

import numpy as np
import pennylane as qml

from src.quantum.dtqw import build_coin, build_shift
from src.quantum.encoding import (
    build_port_map,
    hilbert_dim,
    initial_state_seed_centered,
    initial_state_uniform,
)
from src.quantum.noise import NoiseSpec, apply_noise_layer


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pad_unitary(U: np.ndarray, pad_dim: int) -> np.ndarray:
    """Extender U a un bloque de tamaño ``pad_dim`` con identidad fuera del bloque."""
    if pad_dim == U.shape[0]:
        return U
    out = np.eye(pad_dim, dtype=U.dtype)
    out[: U.shape[0], : U.shape[0]] = U
    return out


def _pad_state(psi: np.ndarray, pad_dim: int) -> np.ndarray:
    """Extender ψ con ceros hasta tamaño ``pad_dim``."""
    if pad_dim == psi.shape[0]:
        return psi
    out = np.zeros(pad_dim, dtype=psi.dtype)
    out[: psi.shape[0]] = psi
    return out


def apply_dtqw_pennylane(
    W_local: np.ndarray,
    seed_idx: int,
    k: int,
    *,
    init_mode: str = "uniform",
    noise: NoiseSpec | None = None,
) -> np.ndarray:
    """DTQW vía PennyLane (validación o experimentos con ruido).

    Args:
        W_local: matriz simétrica ``(M, M)`` con diagonal 0.
        seed_idx: índice local del nodo semilla.
        k: pasos.
        init_mode: ``"uniform"`` o ``"seed_centered"``.
        noise: ``NoiseSpec`` con probabilidades depolarizing/dephasing por
            qubit por paso. Si None o `is_active()` es False, usa
            ``default.qubit`` (puro).

    Returns:
        Vector ``(M,)`` con P_k(v_i), suma 1, no negativo.
    """
    pmap = build_port_map(W_local)
    if init_mode == "uniform":
        psi_0 = initial_state_uniform(pmap)
    elif init_mode == "seed_centered":
        psi_0 = initial_state_seed_centered(pmap, seed_idx)
    else:
        raise ValueError(f"init_mode desconocido: {init_mode!r}")

    coin = build_coin(pmap)
    shift = build_shift(pmap)
    U = shift @ coin

    dim = hilbert_dim(pmap)
    pad_dim = _next_power_of_two(dim)
    n_qubits = max(int(math.log2(pad_dim)), 1)
    U_pad = _pad_unitary(U, pad_dim)
    psi_pad = _pad_state(psi_0, pad_dim)

    noise_active = noise is not None and noise.is_active()
    device_name = "default.mixed" if noise_active else "default.qubit"
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="numpy")
    def circuit() -> np.ndarray:
        qml.StatePrep(psi_pad, wires=range(n_qubits))
        for _ in range(k):
            qml.QubitUnitary(U_pad, wires=range(n_qubits))
            if noise_active:
                apply_noise_layer(noise, n_qubits)
        return qml.probs(wires=range(n_qubits))

    probs_padded = np.asarray(circuit(), dtype=np.float64)

    # Truncar y renormalizar
    probs_truncated = probs_padded[:dim]
    total = float(probs_truncated.sum())
    if total > 0.0:
        probs_truncated = probs_truncated / total

    # Marginalizar sobre el registro de moneda
    M = pmap.M
    out = np.zeros(M, dtype=np.float64)
    for i in range(M):
        base = i * pmap.d_max
        out[i] = float(probs_truncated[base : base + pmap.d_max].sum())
    # Renormalización final defensiva (por errores numéricos del marginalizado)
    final_sum = float(out.sum())
    if final_sum > 0.0:
        out /= final_sum
    return out
