"""Fachada del Modelo R: máscara aleatoria de tamaño m (v6 — Punto 2).

Implementa el protocolo ``LocalModule`` devolviendo ``m`` nodos
ALEATORIOS del subgrafo ``H_t``, sin usar la estructura del grafo ni
ninguna caminata. Es el control crítico para descomponer el aporte:

    A  (RL puro)            →  sin máscara
    R  (RL + máscara random) →  restricción del espacio de acción SIN
                                 información de la caminata
    C  (RL + caminata clásica)
    D  (RL + DTQW)

Si R ≈ C ≈ D en las métricas relevantes, la ventaja proviene de la
formulación RL + restricción del espacio de acción (subgrafo + top-m),
no de la caminata informada (clásica o cuántica).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.graph.subgraph_selector import Subgraph


@dataclass(slots=True)
class RandomWalker:
    """Implementación del protocolo ``LocalModule`` para el Modelo R.

    Mantiene un ``Generator`` propio sembrado en la construcción para
    reproducibilidad por corrida. NO es frozen porque el generador
    avanza su estado interno en cada llamada.
    """

    seed: int = 0
    name: str = "random_mask"
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def candidate_set(
        self,
        subgraph: Subgraph,
        k: int,
        m: int,
    ) -> np.ndarray:
        """``m`` índices globales elegidos uniformemente sin reemplazo.

        Args:
            subgraph: H_t (solo se usa su lista de nodos, no su estructura).
            k: ignorado (no hay caminata).
            m: tamaño del conjunto candidato (recortado a ``min(m, M)``).

        Returns:
            Array ``(m_eff,)`` de índices globales, orden aleatorio.
        """
        M = subgraph.size
        m_eff = min(m, M)
        local = self._rng.choice(M, size=m_eff, replace=False)
        return subgraph.global_node_ids[local].astype(np.int64, copy=False)
