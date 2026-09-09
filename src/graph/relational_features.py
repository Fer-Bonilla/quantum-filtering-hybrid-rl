"""Rasgos relacionales del grafo para el Modelo B (Sec. 5.5.3).

Para cada nodo se computan rasgos agregados que enriquecen el estado clásico
sin necesidad de operar sobre subgrafos:

- ``degree``: grado ponderado (suma de pesos de aristas incidentes).
- ``clustering``: coeficiente de clustering local ponderado (proxy
  simple basado en triángulos ponderados).
- ``spectral_top_k``: k primeros eigenvectores del Laplaciano normalizado
  como embedding posicional.

Resultado: matriz ``(N, d_rel)`` donde ``d_rel = 2 + spectral_top_k``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RelationalSpec:
    """Configuración de los rasgos relacionales."""

    spectral_top_k: int = 2
    """Número de eigenvectores del Laplaciano a usar como embedding."""


def degree(W: np.ndarray) -> np.ndarray:
    """Grado ponderado por nodo."""
    return W.sum(axis=1)


def weighted_clustering(W: np.ndarray) -> np.ndarray:
    """Coeficiente de clustering ponderado simple.

    Cuenta triángulos pesados normalizados por el número máximo posible:

        C_i = sum_{j,k} W_ij * W_jk * W_ki / (d_i * (d_i - 1))

    donde d_i es el grado de i. Devuelve 0 cuando d_i < 2.
    """
    n = W.shape[0]
    W3 = np.linalg.matrix_power(W, 3)
    diag = np.diag(W3).astype(np.float64)
    d = degree(W)
    out = np.zeros(n, dtype=np.float64)
    safe = d > 1.0
    out[safe] = diag[safe] / (d[safe] * (d[safe] - 1.0))
    return out


def laplacian_spectral_embedding(W: np.ndarray, k: int) -> np.ndarray:
    """Eigenvectores k más pequeños del Laplaciano normalizado L_sym.

    L_sym = I - D^{-1/2} W D^{-1/2}.

    Returns:
        Matriz ``(N, k)``. Si W es de rango bajo, se rellenan ceros.
    """
    n = W.shape[0]
    if k <= 0:
        return np.zeros((n, 0), dtype=np.float64)
    d = degree(W)
    d_safe = np.where(d > 0.0, d, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(d_safe)
    L = np.eye(n) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
    # eigsh para matrices densas pequeñas: usamos la versión densa eigh.
    _eigvals, eigvecs = np.linalg.eigh(L)
    # Tomar los k más pequeños (eigh devuelve ordenado ascendente)
    return eigvecs[:, :k].astype(np.float64, copy=False)


def relational_features(W: np.ndarray, spec: RelationalSpec) -> np.ndarray:
    """Combinar todos los rasgos relacionales en una matriz ``(N, d_rel)``."""
    deg = degree(W).reshape(-1, 1)
    clust = weighted_clustering(W).reshape(-1, 1)
    spec_emb = laplacian_spectral_embedding(W, spec.spectral_top_k)
    return np.concatenate([deg, clust, spec_emb], axis=1).astype(np.float32)
