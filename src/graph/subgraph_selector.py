"""Selector de subgrafos H_t (Sec. 6.5 / 7.9).

Nodo semilla v_seed,t = argmax_i b_t(i) con b_t(i) = mean(r_i) / (std(r_i) + eps).
H_t se construye por BFS ponderada desde v_seed,t hasta tamaño máximo M.

El selector es PURO. Devuelve una estructura ``Subgraph`` con la
submatriz densa W_local, los IDs globales de los nodos y el índice local
del seed (necesario para el módulo cuántico).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from src.graph.graph_builder import Graph


@dataclass(frozen=True, slots=True)
class Subgraph:
    """Subgrafo H_t para el módulo local (Sec. 7.9)."""

    W_local: np.ndarray
    """Adyacencia densa simétrica ``(M, M)``, diagonal 0."""

    global_node_ids: np.ndarray
    """Índices globales (en el universo U) de los M nodos. dtype int64."""

    seed_idx_local: int
    """Posición local (en ``[0, M)``) del nodo semilla."""

    @property
    def size(self) -> int:
        return int(self.W_local.shape[0])

    @property
    def seed_global(self) -> int:
        return int(self.global_node_ids[self.seed_idx_local])


def seed_score(returns_window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Score reproducible para elegir v_seed,t (Sec. 6.5):

        b_t(i) = mean(r_i) / (std(r_i) + eps)

    Args:
        returns_window: ``(L_s, N)``.
        eps: estabilizador del divisor.

    Returns:
        Vector ``(N,)``.
    """
    if returns_window.ndim != 2:
        raise ValueError(f"returns_window debe ser 2D; recibido {returns_window.shape}")
    if returns_window.shape[0] < 2:
        raise ValueError("Se necesitan al menos 2 observaciones.")
    mean = np.mean(returns_window, axis=0)
    std = np.std(returns_window, axis=0, ddof=1)
    return mean / (std + eps)


def select_subgraph(
    graph: Graph,
    seed_scores: np.ndarray,
    max_size: int,
    *,
    eligible_mask: np.ndarray | None = None,
) -> Subgraph:
    """Seleccionar H_t alrededor de argmax(seed_scores) por BFS ponderada.

    Args:
        graph: ``Graph`` con matriz ``W`` (N, N).
        seed_scores: vector ``(N,)`` con b_t(i).
        max_size: M (tamaño máximo del subgrafo).
        eligible_mask: máscara opcional de activos elegibles; el seed se
            elige solo entre los elegibles. Si None, todos elegibles.

    Returns:
        ``Subgraph`` de tamaño ``min(M, número de nodos alcanzables)``.

    Raises:
        ValueError: Si el seed no tiene vecinos en el grafo (subgrafo
            quedaría con un único nodo). En este caso el caller decide si
            saltar el step o regresar mask all-True como fallback.
    """
    if max_size < 2:
        raise ValueError(f"max_size debe ser >= 2; recibido {max_size}")

    n = graph.n_nodes
    if seed_scores.shape != (n,):
        raise ValueError(f"seed_scores shape {seed_scores.shape} no coincide con N={n}")

    if eligible_mask is None:
        eligible_mask = np.ones(n, dtype=bool)
    if not eligible_mask.any():
        raise ValueError("eligible_mask completamente False.")

    masked_scores = np.where(eligible_mask, seed_scores, -np.inf)
    seed_global = int(np.argmax(masked_scores))

    # BFS ponderada: cola por afinidad descendente (best-first).
    visited: set[int] = {seed_global}
    # heap de tuplas (-peso, nodo); usamos negativo porque heapq es min-heap.
    heap: list[tuple[float, int]] = []
    for j in range(n):
        w = float(graph.W[seed_global, j])
        if w > 0.0 and j != seed_global:
            heapq.heappush(heap, (-w, j))

    order: list[int] = [seed_global]
    while heap and len(order) < max_size:
        _, j = heapq.heappop(heap)
        if j in visited:
            continue
        visited.add(j)
        order.append(j)
        # Expandir vecinos de j
        for k in range(n):
            if k not in visited:
                w_jk = float(graph.W[j, k])
                if w_jk > 0.0:
                    heapq.heappush(heap, (-w_jk, k))

    if len(order) < 2:
        raise ValueError(
            f"Subgrafo aislado: el nodo semilla {seed_global} no tiene "
            f"vecinos en W. Considera aumentar k_neighbors."
        )

    global_ids = np.array(order, dtype=np.int64)
    W_local = graph.W[np.ix_(global_ids, global_ids)].astype(np.float64, copy=True)
    np.fill_diagonal(W_local, 0.0)
    return Subgraph(W_local=W_local, global_node_ids=global_ids, seed_idx_local=0)
