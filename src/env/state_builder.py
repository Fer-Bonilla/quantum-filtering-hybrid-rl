"""Construcción del estado clásico s_t = phi(X_t, F_t, Z_t).

Para el MVP usamos solo X_t (rasgos temporales por ticker). F_t y Z_t
(fundamentales y relacionales del grafo) se introducirán en Semanas 4-5.

El estado es un vector NumPy 1D obtenido aplanando una ventana deslizante
de longitud L sobre las features por ticker:

    s_t = vec([ X_{t-L+1}, ..., X_t ])  in R^{L * N * d}

con N tickers y d features. El builder es PURO: no realiza I/O ni mantiene
estado entre invocaciones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class StateBuilder:
    """Construye s_t a partir de un DataFrame de features (MultiIndex columns)."""

    window_length: int
    """Longitud L de la ventana deslizante."""

    def state_shape(self, n_tickers: int, n_features: int) -> tuple[int, ...]:
        """Devolver la forma plana del vector estado."""
        return (self.window_length * n_tickers * n_features,)

    def build(
        self,
        features: pd.DataFrame,
        t_index: int,
        relational_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Construir s_t desde el índice posicional `t_index`.

        Args:
            features: DataFrame con MultiIndex (ticker, feature_name) e
                índice temporal monotónicamente creciente. Cada valor en
                fila ``t`` depende SOLO de ``data[:t]`` (garantizado por
                ``compute_features``).
            t_index: Posición temporal `t` (0-indexada). El estado usa la
                ventana ``features.iloc[t-L+1 : t+1]`` (`t` incluido, pero
                las features en `t` ya están desfasadas por ``shift(1)``).
            relational_features: matriz opcional ``(N, d_rel)`` con rasgos
                agregados del grafo dinámico ``G_t`` (degree ponderado,
                clustering, embedding espectral del Laplaciano). Cuando
                se proporciona se concatena al vector aplanado del estado,
                resultando en una observación de dimensión
                ``L·N·d + N·d_rel`` (usado por el Modelo B v2).

        Returns:
            Vector 1D ``np.float32`` aplanado. Sin ``relational_features``
            la forma es ``(L·N·d,)``; con ellos, ``(L·N·d + N·d_rel,)``.

        Raises:
            IndexError: Si la ventana se sale por la izquierda del DataFrame.
        """
        L = self.window_length
        if t_index + 1 < L:
            raise IndexError(
                f"Ventana de tamaño {L} no encaja en t_index={t_index} "
                f"(disponibles {t_index + 1} filas)."
            )
        window = features.iloc[t_index - L + 1 : t_index + 1]
        arr = window.to_numpy(dtype=np.float32, copy=False)
        if relational_features is not None:
            # Concatenar al vector aplanado tras la ventana temporal.
            rel = np.ascontiguousarray(relational_features, dtype=np.float32)
            return np.concatenate(
                [np.ascontiguousarray(arr.reshape(-1)), rel.reshape(-1)]
            )
        # Garantizar contiguidad y forma plana
        return np.ascontiguousarray(arr.reshape(-1))


def infer_n_tickers_and_features(features: pd.DataFrame) -> tuple[int, int]:
    """Inferir (n_tickers, n_features_por_ticker) desde el MultiIndex."""
    n_tickers = len({c[0] for c in features.columns})
    n_features = len(features.columns) // n_tickers
    if n_tickers * n_features != len(features.columns):
        raise ValueError(
            f"Columnas no balanceadas: {len(features.columns)} columnas, "
            f"{n_tickers} tickers detectados."
        )
    return n_tickers, n_features
