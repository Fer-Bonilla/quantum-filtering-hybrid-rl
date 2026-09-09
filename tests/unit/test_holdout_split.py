"""Tests del hold-out temporal (rev. v2 — Problema 1.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splits import holdout_split


def _make_panel(start: str = "2018-01-02", end: str = "2024-12-31") -> pd.DataFrame:
    """Panel sintético con índice de business days."""
    idx = pd.bdate_range(start=start, end=end, freq="B")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.standard_normal(len(idx))}, index=idx)


def test_holdout_split_no_overlap() -> None:
    """tuning ∩ holdout = ∅."""
    df = _make_panel()
    out = holdout_split(df, tuning_end="2022-12-31")
    overlap = set(out["tuning"].index) & set(out["holdout"].index)
    assert len(overlap) == 0


def test_holdout_split_covers_panel() -> None:
    """tuning ∪ holdout ≈ df (puede haber gap mínimo si holdout_start se infiere)."""
    df = _make_panel()
    out = holdout_split(df, tuning_end="2022-12-31")
    coverage = len(out["tuning"]) + len(out["holdout"])
    # Tolerar 0 ó 1 día perdido (el siguiente business day tras tuning_end)
    assert coverage >= len(df) - 1


def test_holdout_split_tuning_before_holdout() -> None:
    """Última fecha de tuning < primera fecha de holdout."""
    df = _make_panel()
    out = holdout_split(df, tuning_end="2022-12-31")
    assert out["tuning"].index[-1] < out["holdout"].index[0]


def test_holdout_split_explicit_dates() -> None:
    """Con holdout_start explícito puede haber gap."""
    df = _make_panel()
    out = holdout_split(df, tuning_end="2022-06-30", holdout_start="2023-01-02")
    assert out["tuning"].index[-1] <= pd.Timestamp("2022-06-30")
    assert out["holdout"].index[0] >= pd.Timestamp("2023-01-02")


def test_holdout_split_rejects_overlap_in_dates() -> None:
    df = _make_panel()
    with pytest.raises(ValueError):
        holdout_split(df, tuning_end="2022-12-31", holdout_start="2022-06-30")


def test_holdout_split_rejects_empty_holdout() -> None:
    df = _make_panel("2018-01-02", "2022-12-31")
    with pytest.raises(ValueError):
        holdout_split(df, tuning_end="2022-12-31")


def test_holdout_split_uses_2018_2022_for_tuning() -> None:
    """Caso canónico del TFE v2: tuning 2018-2022, hold-out 2023-2024."""
    df = _make_panel("2018-01-02", "2024-12-31")
    out = holdout_split(df, tuning_end="2022-12-31", holdout_start="2023-01-02")
    assert out["tuning"].index[0].year == 2018
    assert out["tuning"].index[-1].year == 2022
    assert out["holdout"].index[0].year == 2023
    assert out["holdout"].index[-1].year == 2024
    # Aproximadamente 5 años tuning, 2 años holdout
    tuning_years = (out["tuning"].index[-1] - out["tuning"].index[0]).days / 365.0
    holdout_years = (out["holdout"].index[-1] - out["holdout"].index[0]).days / 365.0
    assert 4.5 < tuning_years < 5.5
    assert 1.5 < holdout_years < 2.5
