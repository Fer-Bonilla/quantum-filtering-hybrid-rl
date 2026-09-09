"""Sparsificación del grafo por k-NN simetrizado (Sec. 7.7).

Para cada nodo i se conservan sus k vecinos con mayor afinidad. La regla
de simetrización determina si una arista (i, j) sobrevive cuando j es
vecino de i pero i no es vecino de j:

    - ``avg`` (por defecto): la arista sobrevive si CUALQUIERA de los dos
      la mantiene; el peso final es ``(A_t(i,j) + A_t(j,i)) / 2``.
    - ``mutual``: solo sobrevive si AMBOS la mantienen (k-NN mutuo).
    - ``max``: como avg pero el peso es ``max(A_t(i,j), A_t(j,i))``.

La diagonal se fuerza a 0 (sin lazos).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

SymMode = Literal["avg", "mutual", "max"]


def knn_sparsify(
    affinity: np.ndarray,
    k: int,
    sym_mode: SymMode = "avg",
) -> np.ndarray:
    """Sparsificar una matriz de afinidad por k-NN simetrizado.

    Args:
        affinity: Matriz ``(N, N)`` simétrica no negativa con diagonal 1
            (o cualquier valor; será forzada a 0 en el resultado).
        k: Número de vecinos a conservar por nodo.
        sym_mode: Modo de simetrización.

    Returns:
        Matriz ``(N, N)`` esparsificada con peso final, diagonal 0 y simetría
        exacta (``W == W.T`` a tolerancia numérica).
    """
    n = affinity.shape[0]
    if affinity.shape != (n, n):
        raise ValueError(f"affinity debe ser cuadrada; recibido {affinity.shape}")
    if k < 1 or k >= n:
        raise ValueError(f"k debe estar en [1, N-1]; recibido k={k}, N={n}")

    # Para cada fila, identificar los k vecinos con mayor afinidad EXCLUYENDO
    # la diagonal (auto-loop no es un vecino real).
    a = affinity.copy()
    np.fill_diagonal(a, -np.inf)
    # argpartition encuentra los k mayores sin ordenarlos completamente.
    top_k_idx = np.argpartition(-a, kth=k - 1, axis=1)[:, :k]
    mask_in = np.zeros_like(affinity, dtype=bool)
    rows = np.arange(n)[:, None]
    mask_in[rows, top_k_idx] = True

    if sym_mode == "mutual":
        edge_mask = mask_in & mask_in.T
        weights = 0.5 * (affinity + affinity.T)
    elif sym_mode == "avg":
        edge_mask = mask_in | mask_in.T
        weights = 0.5 * (affinity + affinity.T)
    elif sym_mode == "max":
        edge_mask = mask_in | mask_in.T
        weights = np.maximum(affinity, affinity.T)
    else:
        raise ValueError(f"sym_mode desconocido: {sym_mode!r}")

    W = np.where(edge_mask, weights, 0.0)
    np.fill_diagonal(W, 0.0)
    return W


def degree_vector(W: np.ndarray) -> np.ndarray:
    """Grados ponderados (suma por fila)."""
    return W.sum(axis=1)
