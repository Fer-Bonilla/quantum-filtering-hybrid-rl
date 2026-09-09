"""Tests de src/env/state_builder.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.data.features import FeatureSpec, compute_features
from src.env.state_builder import StateBuilder, infer_n_tickers_and_features


def test_state_shape(synthetic_clean_df: pd.DataFrame) -> None:
    features = compute_features(synthetic_clean_df, FeatureSpec(volatility_window=10))
    n_tickers, n_features = infer_n_tickers_and_features(features)
    sb = StateBuilder(window_length=5)
    shape = sb.state_shape(n_tickers, n_features)
    assert shape == (5 * n_tickers * n_features,)


def test_build_returns_float32(synthetic_clean_df: pd.DataFrame) -> None:
    features = compute_features(synthetic_clean_df, FeatureSpec(volatility_window=5))
    sb = StateBuilder(window_length=5)
    s = sb.build(features, t_index=len(features) - 1)
    assert s.dtype == np.float32
    assert s.ndim == 1


def test_build_uses_last_L_rows(synthetic_clean_df: pd.DataFrame) -> None:
    features = compute_features(synthetic_clean_df, FeatureSpec(volatility_window=3))
    sb = StateBuilder(window_length=3)
    t = len(features) - 1
    s = sb.build(features, t_index=t)
    expected = features.iloc[t - 2 : t + 1].to_numpy(dtype=np.float32).reshape(-1)
    np.testing.assert_array_equal(s, expected)


def test_build_raises_for_insufficient_history(synthetic_clean_df: pd.DataFrame) -> None:
    features = compute_features(synthetic_clean_df, FeatureSpec(volatility_window=3))
    sb = StateBuilder(window_length=5)
    with pytest.raises(IndexError):
        sb.build(features, t_index=2)
