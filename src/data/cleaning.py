"""Limpieza y alineación temporal de OHLCV multi-ticker.

El resultado es un DataFrame con:
    - Índice temporal único (intersección de fechas válidas) ordenado.
    - Columnas MultiIndex (ticker, ohlcv_field).

Política de NaN configurable; por defecto se descartan tickers con
``> max_nan_ratio`` de huecos y se elimina cualquier fecha con NaN en
``close``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CleaningPolicy:
    """Política de limpieza."""

    max_nan_ratio: float = 0.05
    """Fracción máxima de NaN por columna 'close' antes de descartar el ticker."""

    drop_dates_with_any_nan: bool = True
    """Si True, se eliminan fechas donde algún ticker tenga NaN en 'close'."""

    forward_fill_limit: int = 0
    """Forward-fill controlado (en días). 0 = no aplicar ffill."""


def clean(
    raw: dict[str, pd.DataFrame],
    policy: CleaningPolicy | None = None,
) -> pd.DataFrame:
    """Limpiar y alinear datos OHLCV multi-ticker.

    Args:
        raw: Dict ticker → DataFrame OHLCV individual (índice fecha).
        policy: Política de limpieza. Si None, usa la política por defecto.

    Returns:
        DataFrame con índice temporal único y columnas MultiIndex
        (ticker, field) ordenadas. Tipos numéricos (float64) excepto volume.

    Raises:
        ValueError: Si tras la limpieza no quedan tickers o filas.
    """
    if policy is None:
        policy = CleaningPolicy()
    if not raw:
        raise ValueError("Diccionario de entrada vacío.")

    # 1. Validar columnas mínimas y filtrar tickers con demasiados NaN
    valid: dict[str, pd.DataFrame] = {}
    for ticker, df in raw.items():
        if "close" not in df.columns:
            _log.warning("%s sin columna 'close'; descartado.", ticker)
            continue
        nan_ratio = df["close"].isna().mean()
        if nan_ratio > policy.max_nan_ratio:
            _log.warning(
                "%s descartado por NaN ratio %.3f > %.3f",
                ticker,
                nan_ratio,
                policy.max_nan_ratio,
            )
            continue
        valid[ticker] = df

    if not valid:
        raise ValueError("Ningún ticker pasó el filtro de NaN.")

    # 2. Forward fill opcional
    if policy.forward_fill_limit > 0:
        valid = {t: df.ffill(limit=policy.forward_fill_limit) for t, df in valid.items()}

    # 3. Construir DataFrame con columnas MultiIndex (ticker, field)
    pieces = []
    for ticker, df in valid.items():
        df = df.copy()
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        pieces.append(df)
    merged = pd.concat(pieces, axis=1)
    merged = merged.sort_index()
    merged = merged.sort_index(axis=1)

    # 4. Eliminar fechas con NaN si así lo pide la política
    if policy.drop_dates_with_any_nan:
        close_cols = [c for c in merged.columns if c[1] == "close"]
        before = len(merged)
        merged = merged.dropna(subset=close_cols, how="any")
        dropped = before - len(merged)
        if dropped > 0:
            _log.info("Eliminadas %d fechas con NaN en close.", dropped)

    if merged.empty:
        raise ValueError("Tras limpieza no quedan fechas válidas.")

    # 5. Tipos numéricos consistentes
    merged = merged.astype("float64", copy=False)
    merged.index.name = "date"

    _log.info(
        "Limpieza completa: %d tickers, %d fechas [%s, %s].",
        len({c[0] for c in merged.columns}),
        len(merged),
        merged.index.min().date(),
        merged.index.max().date(),
    )
    return merged


def get_tickers(clean_df: pd.DataFrame) -> list[str]:
    """Extraer la lista ordenada de tickers de un DataFrame limpio."""
    return sorted({c[0] for c in clean_df.columns})


def get_close(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Extraer la matriz de precios de cierre (fecha x ticker)."""
    close_cols = [c for c in clean_df.columns if c[1] == "close"]
    df = clean_df[close_cols].copy()
    df.columns = [c[0] for c in df.columns]
    return df
