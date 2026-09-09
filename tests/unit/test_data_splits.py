"""Tests de src/data/splits.py."""

from __future__ import annotations

import pandas as pd
import pytest
from src.data.splits import SplitSpec, chronological_split


def test_split_no_overlap(synthetic_clean_df: pd.DataFrame) -> None:
    spec = SplitSpec(train_frac=0.6, val_frac=0.2)
    out = chronological_split(synthetic_clean_df, spec)
    # Sin solapamiento
    assert out["train"].index.max() < out["val"].index.min()
    assert out["val"].index.max() < out["test"].index.min()
    # Reconstrucción exacta
    total = len(out["train"]) + len(out["val"]) + len(out["test"])
    assert total == len(synthetic_clean_df)


def test_split_fractions_approx(synthetic_clean_df: pd.DataFrame) -> None:
    spec = SplitSpec(train_frac=0.7, val_frac=0.15)
    out = chronological_split(synthetic_clean_df, spec)
    n = len(synthetic_clean_df)
    # Tolerancia ±1 por redondeo entero
    assert abs(len(out["train"]) - int(n * 0.7)) <= 1
    assert abs(len(out["val"]) - int(n * 0.15)) <= 1


def test_split_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        SplitSpec(train_frac=0.7, val_frac=0.4)  # suma >= 1
    with pytest.raises(ValueError):
        SplitSpec(train_frac=0.0, val_frac=0.5)


def test_split_rejects_unordered_index() -> None:
    df = pd.DataFrame({"x": range(10)}, index=pd.date_range("2020-01-01", periods=10)[::-1])
    with pytest.raises(ValueError):
        chronological_split(df, SplitSpec())


def test_split_rejects_empty() -> None:
    df = pd.DataFrame({"x": []}, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError):
        chronological_split(df, SplitSpec())
