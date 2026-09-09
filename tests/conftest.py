"""Fixtures comunes para la suite de tests.

Provee:
- ``synthetic_ohlcv``: dict ticker → DataFrame OHLCV sintético determinista.
- ``synthetic_clean_df``: DataFrame multi-ticker ya limpio.
- ``rng``: NumPy Generator con semilla fija.
- Fixtures de subgrafos sintéticos (M=4, 6, 8) — se agregarán cuando se
  implemente el módulo cuántico (Semana 6).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from src.data.cleaning import CleaningPolicy, clean


@pytest.fixture
def rng() -> np.random.Generator:
    """RNG con semilla fija para tests deterministas."""
    return np.random.default_rng(seed=12345)


def _synth_one_ticker(
    seed: int, n_days: int = 504, start: date = date(2022, 1, 3), drift: float = 0.0004
) -> pd.DataFrame:
    """Generar OHLCV sintético determinista (precios geométricos brownianos)."""
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(loc=drift, scale=0.012, size=n_days)
    close = 100.0 * np.exp(np.cumsum(daily_ret))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.003, n_days)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.003, n_days)))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, n_days))
    volume = rng.integers(1_000_000, 5_000_000, size=n_days).astype(np.float64)

    idx = pd.bdate_range(start=start, periods=n_days, freq="B")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([open_, high, close]),
            "low": np.minimum.reduce([open_, low, close]),
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


@pytest.fixture
def synthetic_ohlcv() -> dict[str, pd.DataFrame]:
    """Dict ticker → DataFrame OHLCV sintético determinista (4 tickers, 504 días)."""
    return {
        "AAA": _synth_one_ticker(seed=1, drift=0.0005),
        "BBB": _synth_one_ticker(seed=2, drift=0.0003),
        "CCC": _synth_one_ticker(seed=3, drift=0.0007),
        "DDD": _synth_one_ticker(seed=4, drift=0.0002),
    }


@pytest.fixture
def synthetic_clean_df(synthetic_ohlcv: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """DataFrame multi-ticker ya limpio listo para tests de features/splits."""
    return clean(synthetic_ohlcv, CleaningPolicy())
