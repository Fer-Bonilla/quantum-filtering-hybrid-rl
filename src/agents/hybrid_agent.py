"""Agente híbrido para los modelos C y D.

Compone:
- ``ClassicalAgent`` (núcleo PPO discreto, la única parte que aprende).
- ``LocalModule`` (clásico o cuántico) que produce el conjunto candidato.

En cada step:
    1. Recibe (obs, info) del entorno.
    2. Construye G_t a partir de la ventana de retornos.
    3. Selecciona H_t alrededor del seed.
    4. Pide al LocalModule top-m índices globales.
    5. Construye una máscara bool[N] con True solo en esos m índices.
    6. Pasa la máscara al ClassicalAgent.act, que muestrea u_t bajo π_eff.

Diferencia C ↔ D: SOLO el ``LocalModule`` cambia. La arquitectura es
intercambiable vía configuración.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.agents.classical_agent import ClassicalAgent
from src.env.market_env import MarketEnv
from src.graph.graph_builder import GraphSpec, build_graph
from src.graph.subgraph_selector import Subgraph, seed_score, select_subgraph


class LocalModule(Protocol):
    """Protocolo común a Modelos C y D."""

    @property
    def name(self) -> str: ...

    def candidate_set(self, subgraph: Subgraph, k: int, m: int) -> np.ndarray: ...


@dataclass(slots=True)
class HybridSpec:
    """Hiperparámetros del agente híbrido."""

    graph_spec: GraphSpec
    """Hiperparámetros del grafo."""

    subgraph_max_size: int
    """M (tamaño máximo del subgrafo)."""

    seed_score_window: int
    """L_s (ventana del score b_t)."""

    graph_lookback: int
    """L_t (ventana de afinidad del grafo)."""

    k_steps: int
    """Pasos de la caminata (clásica o cuántica)."""

    m_top: int
    """Tamaño del conjunto candidato top-m."""

    update_frequency: int = 1
    """Reconstruir G_t cada N steps (1 = cada step)."""


class HybridAgent:
    """Agente para modelos C y D.

    Mantiene un cache rotatorio del último G_t/H_t para ahorrar cómputo
    cuando ``update_frequency > 1``.
    """

    def __init__(
        self,
        classical_agent: ClassicalAgent,
        local_module: LocalModule,
        hybrid_spec: HybridSpec,
        *,
        returns_panel: np.ndarray,
        ticker_to_idx: dict[str, int],
        sectors: list[str] | None = None,
    ) -> None:
        """
        Args:
            classical_agent: PPO core que aprende.
            local_module: ``ClassicalWalker`` o ``QuantumWalker``.
            hybrid_spec: hiperparámetros.
            returns_panel: matriz ``(T, N)`` precomputada con retornos (vista
                de la feature 'return' del entorno).
            ticker_to_idx: mapeo de ticker a índice posicional en la dim N.
            sectors: lista de N sectores (requerido si beta > 0).
        """
        self._classical = classical_agent
        self._local = local_module
        self._spec = hybrid_spec
        self._returns_panel = returns_panel
        self._tickers = sorted(ticker_to_idx.keys(), key=lambda t: ticker_to_idx[t])
        self._sectors = sectors
        self._n_tickers = len(self._tickers)
        self._step_counter = 0
        self._cached_mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Step hook (firma compatible con PPOTrainer.step_hook)
    # ------------------------------------------------------------------

    def make_mask(
        self,
        env: MarketEnv,
        obs: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Calcular la máscara top-m para el step actual."""
        if self._cached_mask is not None and self._step_counter % self._spec.update_frequency != 0:
            self._step_counter += 1
            return self._cached_mask.copy()

        t = int(env.current_t)
        graph_start = max(0, t - self._spec.graph_lookback + 1)
        seed_start = max(0, t - self._spec.seed_score_window + 1)
        if graph_start == t or seed_start == t:
            # Histórico insuficiente: fallback a all-True (modelo A)
            self._step_counter += 1
            mask = np.ones(self._n_tickers, dtype=bool)
            self._cached_mask = mask
            return mask

        returns_window = self._returns_panel[graph_start : t + 1]
        seed_window = self._returns_panel[seed_start : t + 1]

        graph = build_graph(
            returns_window,
            self._tickers,
            self._sectors,
            self._spec.graph_spec,
        )
        scores = seed_score(seed_window)
        try:
            subgraph = select_subgraph(
                graph,
                scores,
                max_size=self._spec.subgraph_max_size,
                eligible_mask=info["candidate_mask"].astype(bool, copy=False),
            )
        except ValueError:
            # Subgrafo aislado: fallback a all-True
            self._step_counter += 1
            mask = np.ones(self._n_tickers, dtype=bool)
            self._cached_mask = mask
            return mask

        global_indices = self._local.candidate_set(
            subgraph,
            k=self._spec.k_steps,
            m=self._spec.m_top,
        )
        mask = np.zeros(self._n_tickers, dtype=bool)
        mask[global_indices] = True
        if not mask.any():
            mask = np.ones(self._n_tickers, dtype=bool)
        self._cached_mask = mask
        self._step_counter += 1
        return mask

    @property
    def classical(self) -> ClassicalAgent:
        return self._classical

    @property
    def local_module_name(self) -> str:
        return self._local.name


def build_returns_panel(features: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    """Extraer matriz ``(T, N)`` con la feature ``return`` para una lista de tickers.

    Garantiza orden posicional consistente con ``HybridAgent``.
    """
    cols = [(t, "return") for t in tickers]
    return features.loc[:, cols].to_numpy(dtype=np.float64, copy=True)


def build_ticker_index(tickers: list[str]) -> dict[str, int]:
    return {t: i for i, t in enumerate(tickers)}
