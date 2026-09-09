"""Constructor del grafo dinámico G_t = (V, E_t, W_t).

El builder es STATELESS: dada una ventana de retornos y sectores, produce
una matriz W_t. Para uso eficiente durante PPO se mantiene una caché
externa basada en hash de la ventana (en ``hybrid_agent``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.graph.affinity import affinity_matrix, normalize_affinity
from src.graph.sparsify import SymMode, knn_sparsify


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """Especificación del grafo (Sec. 6.4 / 7.5-7.7)."""

    alpha: float = 1.0
    beta: float = 0.0
    eps: float = 1e-8
    k_neighbors: int = 5
    sym_mode: SymMode = "avg"


@dataclass(frozen=True, slots=True)
class Graph:
    """Representación inmutable de G_t."""

    W: np.ndarray
    """Matriz de adyacencia ponderada ``(N, N)`` simétrica, diagonal 0."""

    node_names: tuple[str, ...]
    """Tickers en orden posicional."""

    @property
    def n_nodes(self) -> int:
        return len(self.node_names)


def build_graph(
    returns_window: np.ndarray,
    node_names: list[str] | tuple[str, ...],
    sectors: list[str] | None,
    spec: GraphSpec,
) -> Graph:
    """Construir G_t a partir de una ventana de retornos.

    Args:
        returns_window: Matriz ``(L_t, N)`` con la ventana móvil de retornos.
        node_names: lista de N tickers en orden.
        sectors: lista de N sectores (requerido si ``spec.beta > 0``).
        spec: Hiperparámetros.

    Returns:
        ``Graph`` inmutable con ``W_t`` esparsa.
    """
    A = affinity_matrix(
        returns_window,
        sectors,
        alpha=spec.alpha,
        beta=spec.beta,
    )
    A_norm = normalize_affinity(A, eps=spec.eps)
    W = knn_sparsify(A_norm, k=spec.k_neighbors, sym_mode=spec.sym_mode)
    return Graph(W=W, node_names=tuple(node_names))
