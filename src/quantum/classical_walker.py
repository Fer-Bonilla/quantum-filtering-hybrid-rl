"""Fachada del Modelo C: cumple el protocolo ``LocalModule``.

Convierte la distribución de la caminata aleatoria clásica en el conjunto
top-m candidato (indices GLOBALES de los activos).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.graph.classical_walk import random_walk_distribution
from src.graph.subgraph_selector import Subgraph


@dataclass(frozen=True, slots=True)
class ClassicalWalker:
    """Implementación del protocolo ``LocalModule`` para Modelo C."""

    name: str = "classical"

    def candidate_set(
        self,
        subgraph: Subgraph,
        k: int,
        m: int,
    ) -> np.ndarray:
        """Top-m índices globales según la caminata clásica ponderada.

        Args:
            subgraph: H_t.
            k: pasos de caminata.
            m: tamaño del conjunto candidato (recortado a ``min(m, M)``).

        Returns:
            Array ``(m_eff,)`` con índices globales (en U), ordenados por
            probabilidad descendente.
        """
        M = subgraph.size
        m_eff = min(m, M)
        p = random_walk_distribution(subgraph.W_local, subgraph.seed_idx_local, k)
        # argpartition es más rápido que argsort para top-m
        top_local = np.argpartition(-p, kth=m_eff - 1)[:m_eff]
        # Ordenar exhaustivamente esos m
        top_local = top_local[np.argsort(-p[top_local])]
        return subgraph.global_node_ids[top_local].astype(np.int64, copy=False)
