"""Generación de features X_t a partir de OHLCV multi-ticker.

REGLA CRÍTICA — sin fuga temporal: toda feature en `t` solo depende de
``data[:t]`` (i.e., no usa el valor de `t` ni futuro). Esto se logra con
``shift(1)`` DESPUÉS de cualquier operación rolling y se verifica con
``test_features_no_leak``.

Features producidas (configurables vía :class:`FeatureSpec`):
- Retornos (simples o logarítmicos), shift(1) tras computar diff/log.
- Volatilidad histórica (std rolling, anualizada).
- Volumen relativo (volume / volume.rolling.mean()).
- Indicadores técnicos opcionales: RSI, MACD, BB-width.

Salida: DataFrame con MultiIndex de columnas (ticker, feature_name) e índice
temporal alineado al input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

_log = get_logger(__name__)

ReturnType = Literal["simple", "log"]
IndicatorName = Literal["rsi", "macd", "bb_width"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Especificación de features a calcular."""

    return_type: ReturnType = "log"
    volatility_window: int = 20
    volume_window: int = 20
    indicators: tuple[IndicatorName, ...] = field(default_factory=tuple)
    annualize_volatility: bool = True
    trading_days_per_year: int = 252


def compute_features(clean_df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Calcular el panel de features X_t.

    Args:
        clean_df: DataFrame con MultiIndex (ticker, ohlcv_field) e índice fecha.
        spec: Especificación de features.

    Returns:
        DataFrame con MultiIndex (ticker, feature_name). Cada feature en
        fecha `t` depende **solo** de ``data[:t]`` (i.e., el último dato
        observable es de `t-1`).
    """
    tickers = sorted({c[0] for c in clean_df.columns})
    pieces: list[pd.DataFrame] = []

    for ticker in tickers:
        close = clean_df[(ticker, "close")]
        volume = clean_df[(ticker, "volume")] if (ticker, "volume") in clean_df.columns else None
        features = _features_per_ticker(close, volume, spec)
        features.columns = pd.MultiIndex.from_product([[ticker], features.columns])
        pieces.append(features)

    out = pd.concat(pieces, axis=1)
    out = out.sort_index(axis=1)
    out.index.name = clean_df.index.name

    # Eliminar las primeras filas que tienen NaN por warm-up de ventanas
    out = out.dropna(how="any")
    _log.info(
        "Features generadas: %d tickers x %d features (%d filas tras warm-up).",
        len(tickers),
        len({c[1] for c in out.columns}),
        len(out),
    )
    return out


def _features_per_ticker(
    close: pd.Series,
    volume: pd.Series | None,
    spec: FeatureSpec,
) -> pd.DataFrame:
    """Calcular features para un único ticker.

    Cada output en t depende SOLO de close[:t] (shift(1) defensivo aplicado
    al final de cada cómputo).
    """
    result: dict[str, pd.Series] = {}

    # 1. Retorno: por definición r_t = ln(close_t / close_{t-1}) se conoce en t,
    # pero como feature de entrada al estado, se desfasa un paso para que
    # represente "información disponible al cierre de t-1".
    ret = np.log(close / close.shift(1)) if spec.return_type == "log" else close.pct_change()
    result["return"] = ret.shift(1)

    # 2. Volatilidad rolling sobre retornos (anualizada opcional)
    vol = ret.rolling(spec.volatility_window, min_periods=spec.volatility_window).std()
    if spec.annualize_volatility:
        vol = vol * np.sqrt(spec.trading_days_per_year)
    result["volatility"] = vol.shift(1)

    # 3. Volumen relativo
    if volume is not None:
        vol_mean = volume.rolling(spec.volume_window, min_periods=spec.volume_window).mean()
        rel_vol = volume / vol_mean.replace(0.0, np.nan)
        result["volume_rel"] = rel_vol.shift(1)

    # 4. Indicadores técnicos opcionales
    for ind in spec.indicators:
        if ind == "rsi":
            result["rsi"] = _rsi(close, window=14).shift(1)
        elif ind == "macd":
            result["macd"] = _macd(close).shift(1)
        elif ind == "bb_width":
            result["bb_width"] = _bb_width(close, window=20).shift(1)
        else:
            raise ValueError(f"Indicador no soportado: {ind}")

    return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Indicadores técnicos (implementaciones puras NumPy/pandas, sin TA-Lib)
# ---------------------------------------------------------------------------


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD - signal (histograma normalizado por precio)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - signal_line) / close.replace(0.0, np.nan)


def _bb_width(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Ancho relativo de las Bandas de Bollinger ((upper - lower) / mid)."""
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    return (n_std * 2.0 * std) / mid.replace(0.0, np.nan)


def feature_names(spec: FeatureSpec, *, has_volume: bool = True) -> list[str]:
    """Listar los nombres de features que producirá la spec dada."""
    names = ["return", "volatility"]
    if has_volume:
        names.append("volume_rel")
    names.extend(spec.indicators)
    return names
