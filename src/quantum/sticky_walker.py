"""Fachada del Modelo S: máscara pegajosa con rotación parametrizada (v7).

Experimento **dosis-respuesta** del mecanismo de rotación identificado en v6
(Punto 2). El Modelo R (máscara aleatoria, rotación máxima) igualó o superó a
las caminatas informadas en ``candidate_hit_rate``, y el Modelo Q (óptimo QUBO,
determinista → rotación mínima) resultó el PEOR selector. La inferencia fue:

    *el mecanismo operativo es la ROTACIÓN de la máscara top-m, no la calidad
    (información) de la selección.*

Esa inferencia se apoyaba en un gradiente observacional (R>D>C>Q). El
``StickyWalker`` la convierte en una variable **manipulable**: mantiene la
selección del paso anterior y re-sortea cada ranura con probabilidad
``rotation_p`` independiente.

    rotation_p = 0.0  →  máscara CONGELADA (rotación mínima: solo cambia cuando
                          un nodo abandona el subgrafo H_t). Información cero,
                          rotación cero → análogo "limpio" del determinismo del
                          Modelo Q, pero sin su información.
    rotation_p = 1.0  →  re-sorteo total cada paso ≡ ``RandomWalker`` (Modelo R).

La INFORMACIÓN se mantiene en cero a lo largo de todo el barrido (todas las
elecciones son uniformes); SOLO varía la rotación. Si ``candidate_hit_rate``
crece de forma monótona con ``rotation_p``, el mecanismo queda demostrado de
forma CAUSAL (curva dosis-respuesta) y no meramente inferido del gradiente
R>D>C>Q.

Diseño paralelo a :class:`src.quantum.random_walker.RandomWalker`: no usa la
estructura del grafo ni ninguna caminata; ``k`` se ignora. A diferencia de R,
mantiene estado (la selección retenida) entre llamadas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.graph.subgraph_selector import Subgraph


@dataclass(slots=True)
class StickyWalker:
    """Selector pegajoso con probabilidad de re-sorteo ``rotation_p``.

    Implementa el protocolo ``LocalModule``. Mantiene un ``Generator`` propio
    sembrado en la construcción para reproducibilidad por corrida y la última
    selección retenida ``_held``. NO es frozen porque tanto el generador como
    ``_held`` avanzan su estado en cada llamada.

    Args:
        seed: semilla del generador propio.
        rotation_p: probabilidad independiente de re-sortear cada ranura de la
            máscara en cada llamada. ``0.0`` = congelada; ``1.0`` = aleatoria
            pura (≡ ``RandomWalker``). Debe estar en ``[0, 1]``.
    """

    seed: int = 0
    rotation_p: float = 1.0
    name: str = field(default="", init=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _held: np.ndarray = field(init=False, repr=False)
    _turnover_sum: float = field(default=0.0, init=False, repr=False)
    _turnover_n: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rotation_p <= 1.0:
            raise ValueError(f"rotation_p debe estar en [0, 1]; recibido {self.rotation_p}")
        self._rng = np.random.default_rng(self.seed)
        self._held = np.empty(0, dtype=np.int64)
        self.name = f"sticky_p{self.rotation_p:.2f}"

    def candidate_set(
        self,
        subgraph: Subgraph,
        k: int,
        m: int,
    ) -> np.ndarray:
        """``m`` índices globales con rotación parcial respecto al paso previo.

        Cada nodo retenido que sigue presente en ``H_t`` se conserva con
        probabilidad ``1 - rotation_p``; el resto de ranuras (incluidas las de
        nodos que abandonaron el subgrafo) se rellena con muestreo uniforme sin
        reemplazo de los nodos del subgrafo no retenidos.

        Args:
            subgraph: H_t (solo se usa su lista de nodos, no su estructura).
            k: ignorado (no hay caminata).
            m: tamaño del conjunto candidato (recortado a ``min(m, M)``).

        Returns:
            Array ``(m_eff,)`` de índices globales (en U), dtype int64.
        """
        nodes = subgraph.global_node_ids.astype(np.int64, copy=False)
        m_eff = min(m, nodes.size)
        node_set = set(nodes.tolist())

        # Nodos retenidos que siguen presentes en H_t (preserva el orden previo).
        held_valid = [g for g in self._held.tolist() if g in node_set]

        # Decidir por ranura: conservar con prob (1 - rotation_p), si no rotar.
        # Los nodos que abandonaron H_t no están en held_valid → rotación forzada.
        keep = [g for g in held_valid if self._rng.random() >= self.rotation_p]
        keep = keep[:m_eff]  # defensivo si el subgrafo encogió
        keep_set = set(keep)

        n_needed = m_eff - len(keep)
        if n_needed > 0:
            available = np.fromiter(
                (g for g in nodes.tolist() if g not in keep_set),
                dtype=np.int64,
            )
            chosen = self._rng.choice(available, size=n_needed, replace=False)
            result = np.concatenate([np.asarray(keep, dtype=np.int64), chosen])
        else:
            result = np.asarray(keep, dtype=np.int64)

        # Diagnóstico: rotación REALIZADA (distancia de Jaccard consecutiva).
        if self._held.size:
            prev = set(self._held.tolist())
            cur = set(result.tolist())
            union = len(prev | cur)
            self._turnover_sum += 1.0 - len(prev & cur) / union if union else 0.0
            self._turnover_n += 1

        self._held = result
        return result

    @property
    def mean_turnover(self) -> float:
        """Rotación realizada media (Jaccard) entre selecciones consecutivas.

        Diagnóstico para verificar que ``rotation_p`` controla efectivamente la
        rotación. ``nan`` si aún no hubo dos llamadas.
        """
        if self._turnover_n == 0:
            return float("nan")
        return self._turnover_sum / self._turnover_n
