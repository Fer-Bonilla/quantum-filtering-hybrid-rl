"""Selectores clásicos informados-con-rotación (v8 — EXP-7).

El gradiente de v6/v7 compara rotación-sin-información (R) contra
información-sin-rotación (Q); estos selectores completan la celda
información+rotación:

  - ``SoftmaxWalker``        : muestreo de m nodos sin reemplazo desde
                               softmax(p_k clásica / τ) — información de la
                               caminata + rotación estocástica.
  - ``MomentumRefreshWalker``: top-m por momentum 20d sobre H_t, recalculado
                               cada paso — información de precio + rotación
                               por refresco.
  - ``ThompsonWalker``       : Thompson sampling Beta sobre P(activo
                               prometedor), actualizado online SOLO con
                               información pasada resuelta (lag = horizonte).

Los tres implementan el protocolo ``LocalModule``. Los que necesitan el
instante ``t`` lo reciben vía el atributo público ``t``, actualizado por el
hook del runner antes de cada llamada (la interfaz ``candidate_set`` queda
intacta).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.graph.classical_walk import random_walk_distribution
from src.graph.subgraph_selector import Subgraph
from src.training.evaluate import PromisingSpec, compute_promising_matrix


@dataclass(slots=True)
class SoftmaxWalker:
    """Muestreo softmax sin reemplazo sobre la distribución clásica p_k."""

    seed: int = 0
    tau: float = 1.0
    name: str = field(default="softmax", init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau debe ser > 0")
        self._rng = np.random.default_rng(self.seed)

    def candidate_set(self, subgraph: Subgraph, k: int, m: int) -> np.ndarray:
        p = random_walk_distribution(subgraph.W_local, subgraph.seed_idx_local, k)
        logits = p / self.tau
        # Gumbel top-k = muestreo sin reemplazo proporcional a softmax(logits).
        gumbel = -np.log(-np.log(self._rng.uniform(1e-12, 1.0, size=len(logits))))
        m_eff = min(m, subgraph.size)
        top_local = np.argsort(-(logits + gumbel))[:m_eff]
        return subgraph.global_node_ids[top_local].astype(np.int64, copy=False)


@dataclass(slots=True)
class MomentumRefreshWalker:
    """Top-m por momentum reciente sobre los nodos de H_t (refresco por paso)."""

    returns_panel: np.ndarray
    window: int = 20
    name: str = field(default="momentum_refresh", init=False)
    t: int = field(default=0)

    def candidate_set(self, subgraph: Subgraph, k: int, m: int) -> np.ndarray:
        del k
        ids = subgraph.global_node_ids
        start = max(0, self.t - self.window + 1)
        ventana = self.returns_panel[start : self.t + 1, ids]
        mom = ventana.mean(axis=0) if len(ventana) else np.zeros(len(ids))
        m_eff = min(m, subgraph.size)
        top_local = np.argsort(-mom)[:m_eff]
        return ids[top_local].astype(np.int64, copy=False)


@dataclass(slots=True)
class ThompsonWalker:
    """Thompson sampling Beta sobre P(prometedor), con lag causal = horizonte."""

    returns_panel: np.ndarray
    seed: int = 0
    horizon: int = 5
    name: str = field(default="thompson", init=False)
    t: int = field(default=0)
    _rng: np.random.Generator = field(init=False, repr=False)
    _promising: np.ndarray = field(init=False, repr=False)
    _alpha: np.ndarray = field(init=False, repr=False)
    _beta: np.ndarray = field(init=False, repr=False)
    _resolved: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._promising = compute_promising_matrix(
            self.returns_panel, PromisingSpec(horizon=self.horizon))
        n = self.returns_panel.shape[1]
        self._alpha = np.ones(n)
        self._beta = np.ones(n)

    def _update(self) -> None:
        limite = self.t - self.horizon  # promising[t-h] usa retornos hasta t
        while self._resolved < limite:
            self._resolved += 1
            if 0 <= self._resolved < self._promising.shape[0]:
                fila = self._promising[self._resolved]
                self._alpha += fila
                self._beta += ~fila

    def candidate_set(self, subgraph: Subgraph, k: int, m: int) -> np.ndarray:
        del k
        self._update()
        ids = subgraph.global_node_ids
        theta = self._rng.beta(self._alpha[ids], self._beta[ids])
        m_eff = min(m, subgraph.size)
        top_local = np.argsort(-theta)[:m_eff]
        return ids[top_local].astype(np.int64, copy=False)
