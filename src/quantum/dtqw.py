"""Caminata cuántica de tiempo discreto sobre un subgrafo (Sec. 7.14-7.17).

Implementación matricial densa en ``complex128``. Para M ≤ 8 y d_max ≤ 8
el Hilbert es ≤ 64-dim, perfectamente manejable.

Estructura:
    U_t = S_t · C_t
    |ψ_k⟩ = U_t^k |ψ_0⟩
    P_k(v_i) = Σ_c |⟨v_i, c | ψ_k⟩|²

C_t es la suma directa de reflexiones locales tipo Householder:
    C_{i,t} = 2 |w_i⟩⟨w_i| - I_{d_i}
con |w_i⟩ = (1/√Σ W_ij) Σ_j √W_ij |c_{i→j}⟩.

S_t conecta puertos recíprocos: S |i, c_{i→j}⟩ = |j, c_{j→i}⟩, dejando
puntos fijos en puertos de padding.

Renormalización defensiva: si |‖ψ‖² - 1| > tol, se renormaliza el vector.
"""

from __future__ import annotations

import numpy as np

from src.quantum.encoding import (
    PortMap,
    build_port_map,
    hilbert_dim,
    index_of,
    initial_state_seed_centered,
    initial_state_uniform,
)


def build_coin(pmap: PortMap, coin_type: str = "weighted_householder") -> np.ndarray:
    """Construir C_t como matriz unitaria densa.

    Argumentos:
        pmap: PortMap del subgrafo.
        coin_type: tipo de moneda:
            - ``"weighted_householder"`` (default, sweet_spot): para cada
              nodo i, $C_{i,t} = 2|w_i\\rangle\\langle w_i| - I_{d_i}$
              donde $|w_i\\rangle$ es la amplitud ponderada por los pesos
              de aristas. Generalización de Grover con pesos.
            - ``"grover"``: $C_{i,t} = 2|u\\rangle\\langle u| - I_{d_i}$
              donde $|u\\rangle = \\frac{1}{\\sqrt{d_i}} \\sum_c |c\\rangle$
              (uniforme; ignora los pesos). Es el caso límite con
              afinidad constante.
            - ``"fourier"``: $C_{i,t} = F_{d_i}$ donde $F$ es la DFT
              normalizada. No es reflexión ($C^2 \\neq I$); produce
              dispersión más uniforme entre puertos sin preferencia
              por las afinidades.

    Returns:
        Matriz ``(M*d_max, M*d_max)`` compleja unitaria.
    """
    if coin_type not in {"weighted_householder", "grover", "fourier"}:
        raise ValueError(
            f"coin_type='{coin_type}' no reconocido. "
            f"Opciones: weighted_householder, grover, fourier."
        )
    dim = hilbert_dim(pmap)
    C = np.eye(dim, dtype=np.complex128)
    for i in range(pmap.M):
        d_i = int(pmap.degrees[i])
        if d_i == 0:
            continue  # nodo aislado: identidad pura (no aplica reflexión)
        base = i * pmap.d_max
        local = np.eye(pmap.d_max, dtype=np.complex128)
        if coin_type == "weighted_householder":
            weights = pmap.edge_weights[i, :d_i]
            total = float(weights.sum())
            if total <= 0.0:
                continue
            amps = np.sqrt(weights / total).astype(np.complex128)
            local[:d_i, :d_i] = (
                2.0 * np.outer(amps, amps.conj()) - np.eye(d_i, dtype=np.complex128)
            )
        elif coin_type == "grover":
            amps = np.full(d_i, 1.0 / np.sqrt(d_i), dtype=np.complex128)
            local[:d_i, :d_i] = (
                2.0 * np.outer(amps, amps.conj()) - np.eye(d_i, dtype=np.complex128)
            )
        else:  # fourier
            # DFT normalizada de tamaño d_i. F_{kn} = (1/sqrt(d_i)) * exp(-2 pi i k n / d_i).
            n = np.arange(d_i)
            k = n.reshape(-1, 1)
            local[:d_i, :d_i] = (
                np.exp(-2.0j * np.pi * k * n / d_i) / np.sqrt(d_i)
            ).astype(np.complex128)
        C[base : base + pmap.d_max, base : base + pmap.d_max] = local
    return C


def build_shift(pmap: PortMap) -> np.ndarray:
    """Construir S_t como matriz de permutación 0/1.

    Para cada arista (i, j) activa, intercambia |i, c_{i→j}⟩ con |j, c_{j→i}⟩.
    Los puertos de padding son puntos fijos.

    Returns:
        Matriz ``(M*d_max, M*d_max)`` real (representada como complex128).
    """
    dim = hilbert_dim(pmap)
    S = np.zeros((dim, dim), dtype=np.complex128)
    used = np.zeros(dim, dtype=bool)
    for i in range(pmap.M):
        d_i = int(pmap.degrees[i])
        for c in range(d_i):
            j = int(pmap.port_to_neighbor[i, c])
            c_rec = int(pmap.neighbor_to_port[j, i])
            if c_rec < 0:
                raise ValueError(
                    f"Puerto recíproco no encontrado para arista ({i},{j}). "
                    f"Verifica simetría de W_local."
                )
            src = index_of(i, c, pmap.d_max)
            dst = index_of(j, c_rec, pmap.d_max)
            S[dst, src] = 1.0
            used[src] = True
    # Puntos fijos para puertos no usados (padding)
    for i in range(pmap.M):
        for c in range(pmap.degrees[i], pmap.d_max):
            idx = index_of(i, c, pmap.d_max)
            S[idx, idx] = 1.0
            used[idx] = True
    # Verificación: cada columna usada debe ser una permutación
    if not used.all():
        # Los no-usados son padding sin mapeo explícito; les damos identidad.
        for idx in np.flatnonzero(~used):
            S[idx, idx] = 1.0
    return S


def evolve(
    initial_state: np.ndarray,
    coin: np.ndarray,
    shift: np.ndarray,
    k: int,
    *,
    renormalize_threshold: float = 1e-9,
) -> np.ndarray:
    """Aplicar ``(S · C)^k`` a ``initial_state``.

    Args:
        initial_state: vector ``(dim,)`` complejo unitario.
        coin: matriz C_t.
        shift: matriz S_t.
        k: número de pasos.
        renormalize_threshold: tolerancia ``||ψ||² - 1`` antes de renormalizar.

    Returns:
        ``|ψ_k⟩`` complejo unitario.
    """
    if k < 0:
        raise ValueError(f"k debe ser >= 0; recibido {k}")
    psi = initial_state.copy()
    if k == 0:
        return psi
    for _ in range(k):
        psi = shift @ (coin @ psi)
        norm_sq = float(np.real(np.vdot(psi, psi)))
        if abs(norm_sq - 1.0) > renormalize_threshold:
            psi /= np.sqrt(norm_sq)
    return psi


def measure_position(psi: np.ndarray, pmap: PortMap) -> np.ndarray:
    """Marginalizar sobre el registro de moneda para obtener P_k(v_i).

    Returns:
        Vector ``(M,)`` real no negativo que suma 1.
    """
    if psi.shape != (hilbert_dim(pmap),):
        raise ValueError(f"psi shape {psi.shape} no coincide con hilbert_dim={hilbert_dim(pmap)}")
    probs = np.zeros(pmap.M, dtype=np.float64)
    for i in range(pmap.M):
        base = i * pmap.d_max
        amps = psi[base : base + pmap.d_max]
        probs[i] = float(np.real(np.vdot(amps, amps)))
    # Renormalización defensiva
    total = float(probs.sum())
    if total > 0.0:
        probs /= total
    return probs


def apply_dtqw(
    W_local: np.ndarray,
    seed_idx: int,
    k: int,
    *,
    init_mode: str = "uniform",
    renormalize_threshold: float = 1e-9,
    coin_type: str = "weighted_householder",
) -> np.ndarray:
    """Función pública de alto nivel: subgrafo → P_k.

    Args:
        W_local: matriz simétrica ``(M, M)`` con diagonal 0.
        seed_idx: índice local del seed (relevante si ``init_mode='seed_centered'``).
        k: número de pasos.
        init_mode: ``"uniform"`` o ``"seed_centered"``.
        coin_type: tipo de moneda (ver ``build_coin``).

    Returns:
        Vector ``(M,)`` con P_k(v_i).
    """
    pmap = build_port_map(W_local)
    if init_mode == "uniform":
        psi_0 = initial_state_uniform(pmap)
    elif init_mode == "seed_centered":
        psi_0 = initial_state_seed_centered(pmap, seed_idx)
    else:
        raise ValueError(f"init_mode desconocido: {init_mode!r}")
    C = build_coin(pmap, coin_type=coin_type)
    S = build_shift(pmap)
    psi_k = evolve(psi_0, C, S, k, renormalize_threshold=renormalize_threshold)
    return measure_position(psi_k, pmap)
