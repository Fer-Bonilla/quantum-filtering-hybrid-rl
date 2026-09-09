"""Tests adicionales de src/data/features.py (más allá de no-fuga)."""

from __future__ import annotations

import pandas as pd
import pytest
from src.data.features import FeatureSpec, compute_features


def test_compute_features_smoke(synthetic_clean_df: pd.DataFrame) -> None:
    """Sanity check: produce un DataFrame no vacío con MultiIndex."""
    spec = FeatureSpec(volatility_window=10, volume_window=10)
    out = compute_features(synthetic_clean_df, spec)
    assert isinstance(out.columns, pd.MultiIndex)
    assert not out.empty
    assert {c[1] for c in out.columns} == {"return", "volatility", "volume_rel"}


def test_volatility_positive(synthetic_clean_df: pd.DataFrame) -> None:
    """Volatilidad debe ser estrictamente positiva (std >= 0)."""
    spec = FeatureSpec(volatility_window=10)
    out = compute_features(synthetic_clean_df, spec)
    vol_cols = [c for c in out.columns if c[1] == "volatility"]
    assert (out[vol_cols] >= 0.0).all().all()


def test_volume_rel_around_one(synthetic_clean_df: pd.DataFrame) -> None:
    """Volumen relativo debe oscilar alrededor de 1 con varianza acotada."""
    spec = FeatureSpec(volume_window=20)
    out = compute_features(synthetic_clean_df, spec)
    rel_cols = [c for c in out.columns if c[1] == "volume_rel"]
    rel = out[rel_cols]
    assert (rel > 0).all().all()
    # Media debería estar cerca de 1
    assert (rel.mean().abs() - 1.0).abs().max() < 0.1


def test_log_return_consistency(synthetic_clean_df: pd.DataFrame) -> None:
    """Log returns deben ser pequeños en magnitud (< 50% por día típico)."""
    spec = FeatureSpec(return_type="log", volatility_window=5)
    out = compute_features(synthetic_clean_df, spec)
    ret_cols = [c for c in out.columns if c[1] == "return"]
    assert out[ret_cols].abs().max().max() < 0.5


def test_indicators_optional(synthetic_clean_df: pd.DataFrame) -> None:
    """Habilitar indicadores agrega exactamente esas columnas."""
    spec = FeatureSpec(indicators=("rsi", "macd"))
    out = compute_features(synthetic_clean_df, spec)
    names = {c[1] for c in out.columns}
    assert {"rsi", "macd"}.issubset(names)
    assert "bb_width" not in names


def test_unknown_indicator_raises(synthetic_clean_df: pd.DataFrame) -> None:
    spec = FeatureSpec(indicators=("not_a_real_indicator",))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compute_features(synthetic_clean_df, spec)
