"""Tests de src/data/cleaning.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.data.cleaning import CleaningPolicy, clean, get_close, get_tickers


def test_clean_aligns_indices(synthetic_ohlcv: dict[str, pd.DataFrame]) -> None:
    out = clean(synthetic_ohlcv, CleaningPolicy())
    # Índice debe ser monotónico creciente y único
    assert out.index.is_monotonic_increasing
    assert out.index.is_unique
    # Columnas MultiIndex (ticker, field)
    assert isinstance(out.columns, pd.MultiIndex)
    assert {c[0] for c in out.columns} == set(synthetic_ohlcv.keys())


def test_clean_no_nan_in_close(synthetic_ohlcv: dict[str, pd.DataFrame]) -> None:
    out = clean(synthetic_ohlcv, CleaningPolicy())
    close_cols = [c for c in out.columns if c[1] == "close"]
    assert out[close_cols].isna().sum().sum() == 0


def test_clean_drops_high_nan_ticker(synthetic_ohlcv: dict[str, pd.DataFrame]) -> None:
    # Inyectar NaN en uno de los tickers
    contaminated = dict(synthetic_ohlcv)
    bad = contaminated["AAA"].copy()
    bad.iloc[:300, bad.columns.get_loc("close")] = np.nan
    contaminated["AAA"] = bad

    out = clean(contaminated, CleaningPolicy(max_nan_ratio=0.05))
    tickers = get_tickers(out)
    assert "AAA" not in tickers
    assert {"BBB", "CCC", "DDD"}.issubset(set(tickers))


def test_clean_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        clean({}, CleaningPolicy())


def test_get_close_shapes(synthetic_clean_df: pd.DataFrame) -> None:
    close = get_close(synthetic_clean_df)
    assert close.shape[1] == 4
    assert list(close.columns) == ["AAA", "BBB", "CCC", "DDD"]
    assert close.notna().all().all()
