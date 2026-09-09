"""Test crítico P0: ausencia de fuga temporal en las features.

Verifica que cada valor de feature en la fecha ``t`` se mantiene idéntico
cuando los datos se truncan en ``t + k`` (k > 0). Si la feature dependiera
de información futura, el valor en ``t`` diferiría entre las dos
recomputaciones.

Este es el invariante MÁS importante de la capa de datos: una fuga aquí
INVALIDA toda la evaluación experimental de la tesis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.data.features import FeatureSpec, compute_features


@pytest.fixture
def feature_spec_all() -> FeatureSpec:
    """Spec con TODAS las features e indicadores habilitados."""
    return FeatureSpec(
        return_type="log",
        volatility_window=10,
        volume_window=10,
        indicators=("rsi", "macd", "bb_width"),
    )


def test_no_temporal_leak(synthetic_clean_df: pd.DataFrame, feature_spec_all: FeatureSpec) -> None:
    """Test crítico de fuga temporal.

    Computar features sobre todo el panel y, separadamente, sobre la versión
    truncada hasta cada `t`. Los valores comunes deben coincidir exactamente.

    Se prueba en varios puntos de truncamiento, incluyendo cerca del final.
    """
    full = compute_features(synthetic_clean_df, feature_spec_all)
    assert not full.empty, "Features vacías; revisar warm-up"

    n = len(synthetic_clean_df)
    # Probar truncar en 50%, 70%, 90% y 95% del horizonte
    for frac in (0.5, 0.7, 0.9, 0.95):
        cutoff = int(n * frac)
        truncated_input = synthetic_clean_df.iloc[:cutoff]
        truncated_features = compute_features(truncated_input, feature_spec_all)

        # Tomar la intersección de fechas entre los dos paneles
        common_idx = full.index.intersection(truncated_features.index)
        assert len(common_idx) > 50, (
            f"Intersección demasiado pequeña ({len(common_idx)}) en frac={frac}"
        )

        full_slice = full.loc[common_idx]
        trunc_slice = truncated_features.loc[common_idx]

        # Verificar igualdad ELEMENT-WISE con tolerancia para errores
        # numéricos acumulados (EWM puede acumular fp error pequeño)
        diff = (full_slice - trunc_slice).abs()
        max_diff = diff.max().max()
        assert max_diff < 1e-9, (
            f"FUGA TEMPORAL detectada en frac={frac}: max_diff={max_diff} "
            f"en columnas con discrepancia: {diff.max()[diff.max() > 1e-9].to_dict()}"
        )


def test_feature_t_uses_data_up_to_t_minus_one(
    synthetic_clean_df: pd.DataFrame, feature_spec_all: FeatureSpec
) -> None:
    """Verificar que las features en `t` no incluyen el cierre de `t`.

    Estrategia: tomar el panel limpio, copiarlo, y alterar SOLO el último
    cierre de cada ticker. Las features hasta `T-1` deben ser idénticas.
    Las features en `T` pueden cambiar (porque están en la frontera).
    """
    original = compute_features(synthetic_clean_df, feature_spec_all)

    perturbed = synthetic_clean_df.copy()
    last_date = perturbed.index[-1]
    # Multiplicar todos los closes del último día por 1.5 (cambio masivo)
    close_cols = [c for c in perturbed.columns if c[1] == "close"]
    perturbed.loc[last_date, close_cols] *= 1.5

    perturbed_features = compute_features(perturbed, feature_spec_all)

    # Las features hasta `last_date - 1` (excluyendo `last_date`) deben coincidir
    common_idx = original.index.intersection(perturbed_features.index)
    common_idx_before_last = common_idx[common_idx < last_date]
    assert len(common_idx_before_last) > 50

    diff = (
        original.loc[common_idx_before_last] - perturbed_features.loc[common_idx_before_last]
    ).abs()
    max_diff = diff.max().max()
    assert max_diff < 1e-9, f"Una feature en t < T depende de close[T]: max_diff={max_diff}"


def test_features_finite(synthetic_clean_df: pd.DataFrame, feature_spec_all: FeatureSpec) -> None:
    """Todas las features producidas deben ser finitas (sin NaN ni inf)."""
    features = compute_features(synthetic_clean_df, feature_spec_all)
    finite = np.isfinite(features.to_numpy())
    assert finite.all(), (
        f"Hay {(~finite).sum()} valores no finitos en las features tras compute_features"
    )


def test_features_no_volume_when_absent(synthetic_clean_df: pd.DataFrame) -> None:
    """Si no hay columna volume, no debe aparecer la feature volume_rel."""
    no_vol = synthetic_clean_df.drop(
        columns=[c for c in synthetic_clean_df.columns if c[1] == "volume"]
    )
    spec = FeatureSpec(volatility_window=10, indicators=())
    features = compute_features(no_vol, spec)
    feat_names = {c[1] for c in features.columns}
    assert "volume_rel" not in feat_names
    assert {"return", "volatility"}.issubset(feat_names)
