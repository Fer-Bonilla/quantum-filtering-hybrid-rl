"""Caminata aleatoria clásica ponderada sobre un subgrafo (Sec. 7.18).

Implementa el baseline local del Modelo C:

    P = D^{-1} W   (matriz de transición row-stochastic)
    p_k = (P^T)^k e_seed
    C_t^(loc) = TopM(p_k, m)

Esta caminata es el comparable EXACTO de la DTQW: mismo subgrafo, mismo
seed, mismo número de pasos k.
"""

from __future__ import annotations

import numpy as np

from src.graph.sparsify import degree_vector


def transition_matrix(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Matriz de transición row-stochastic P = D^{-1} W.

    Nodos aislados (grado 0) reciben una fila uniforme para evitar NaN.
    """
    d = degree_vector(W)
    P = np.empty_like(W, dtype=np.float64)
    isolated = d < eps
    safe_d = np.where(isolated, 1.0, d)
    P[:] = W / safe_d[:, None]
    if isolated.any():
        n = W.shape[0]
        uniform = np.full(n, 1.0 / n)
        for i in np.flatnonzero(isolated):
            P[i] = uniform
    return P


def random_walk_distribution(W: np.ndarray, seed_idx: int, k: int) -> np.ndarray:
    """Distribución tras k pasos de caminata aleatoria ponderada.

    Args:
        W: matriz de adyacencia ``(M, M)`` simétrica no negativa.
        seed_idx: índice local del nodo inicial en ``[0, M)``.
        k: número de pasos.

    Returns:
        Vector ``(M,)`` de probabilidades que suma 1.
    """
    if k < 0:
        raise ValueError(f"k debe ser >= 0; recibido {k}")
    n = W.shape[0]
    if not (0 <= seed_idx < n):
        raise ValueError(f"seed_idx fuera de rango: {seed_idx} (M={n})")
    p = np.zeros(n, dtype=np.float64)
    p[seed_idx] = 1.0
    if k == 0:
        return p
    P = transition_matrix(W)
    Pt = P.T  # (P^T)^k aplicado a p^0
    for _ in range(k):
        p = Pt @ p
    # Renormalización defensiva ante errores numéricos
    s = float(p.sum())
    if s > 0.0:
        p /= s
    return p
