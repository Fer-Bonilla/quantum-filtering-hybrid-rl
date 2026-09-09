"""Constructor de rasgos relacionales para el Modelo B (rev. v2).

Genera, para cada índice temporal ``t``, una matriz ``(N, d_rel)`` con
características agregadas del grafo dinámico ``G_t`` (degree ponderado,
clustering, embedding espectral del Laplaciano normalizado). El
``MarketEnv`` invoca este constructor en cada step y concatena el
resultado al estado clásico, materializando el Modelo B descrito en el
Cap.~5.5.3 del documento.

Diseño:
- El builder es PURO y reproducible: misma ventana de retornos →
  mismos rasgos. Sin estado entre invocaciones.
- Sólo se usan retornos hasta ``t`` (sin fuga temporal).
- Cuando la ventana es insuficiente al inicio del episodio, se
  devuelve una matriz de ceros para no romper el shape; el caller
  decide si filtra esos steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.graph.graph_builder import GraphSpec, build_graph
from src.graph.relational_features import RelationalSpec, relational_features


@dataclass(frozen=True, slots=True)
class RelationalStateConfig:
    """Configuración del constructor de rasgos relacionales."""

    graph_spec: GraphSpec
    """Hiperparámetros del grafo G_t."""

    relational_spec: RelationalSpec
    """Hiperparámetros de los rasgos (spectral_top_k, etc.)."""

    lookback_window: int
    """Ventana de retornos que alimenta la construcción del grafo (Lt)."""

    feature_dim: int
    """d_rel — número de columnas por nodo. Debe coincidir con
    ``2 + relational_spec.spectral_top_k``."""


class RelationalStateBuilder:
    """Genera rasgos relacionales por step para el Modelo B.

    Mantiene una referencia al panel de retornos y a un mapeo de
    tickers para garantizar consistencia con el orden del ``MarketEnv``.
    Construye el grafo ``G_t`` solo con datos ``[t-Lt+1 : t+1]`` —
    nunca futuros.
    """

    def __init__(
        self,
        returns_panel: np.ndarray,
        node_names: list[str],
        cfg: RelationalStateConfig,
        sectors: list[str] | None = None,
    ) -> None:
        if returns_panel.shape[1] != len(node_names):
            raise ValueError(
                f"returns_panel tiene {returns_panel.shape[1]} columnas, "
                f"node_names tiene {len(node_names)}"
            )
        self._returns = returns_panel
        self._node_names = list(node_names)
        self._cfg = cfg
        self._sectors = sectors

    @property
    def feature_dim(self) -> int:
        return self._cfg.feature_dim

    @property
    def n_tickers(self) -> int:
        return len(self._node_names)

    def __call__(self, t_index: int) -> np.ndarray:
        """Construir rasgos relacionales para el paso ``t_index``.

        Args:
            t_index: posición temporal (0-indexada). El builder usa
                ``self._returns[t_index - Lt + 1 : t_index + 1]``.

        Returns:
            Matriz ``(N, d_rel)`` con los rasgos. Si la ventana no
            encaja (inicio del horizonte), retorna ceros.
        """
        Lt = self._cfg.lookback_window
        start = t_index - Lt + 1
        if start < 0 or t_index + 1 > self._returns.shape[0]:
            return np.zeros((self.n_tickers, self.feature_dim), dtype=np.float32)
        window = self._returns[start : t_index + 1]
        # Garantía de NO fuga temporal: window incluye sólo data[..t].
        graph = build_graph(
            window, self._node_names, sectors=self._sectors, spec=self._cfg.graph_spec
        )
        feats = relational_features(graph.W, self._cfg.relational_spec)
        # Garantizar shape (N, d_rel)
        if feats.shape != (self.n_tickers, self.feature_dim):
            raise ValueError(
                f"relational_features devolvió shape {feats.shape}; "
                f"esperado ({self.n_tickers}, {self.feature_dim})"
            )
        return feats
