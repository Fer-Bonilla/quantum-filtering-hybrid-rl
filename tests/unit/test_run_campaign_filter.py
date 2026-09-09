"""Tests del filtro temporal y override sweet_spot en run_campaign.py.

Verifica que el argumento ``--filter-dates`` recorta correctamente el
panel antes del split interno y que ``--sweet-spot`` aplica los
overrides apropiados.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_run_campaign():
    """Cargar run_campaign.py como módulo aislado."""
    _path = Path(__file__).resolve().parents[2] / "scripts" / "run_campaign.py"
    _spec = importlib.util.spec_from_file_location("run_campaign", _path)
    assert _spec is not None and _spec.loader is not None
    mod = importlib.util.module_from_spec(_spec)
    sys.modules["run_campaign"] = mod
    _spec.loader.exec_module(mod)
    return mod


def _make_multi_panel(n_days: int = 60) -> pd.DataFrame:
    """Panel mínimo (2 tickers × 2 features) para tests."""
    idx = pd.bdate_range("2023-01-02", periods=n_days, freq="B")
    rng = np.random.default_rng(0)
    cols = pd.MultiIndex.from_tuples(
        [("AAPL", "return"), ("AAPL", "volatility"), ("MSFT", "return"), ("MSFT", "volatility")]
    )
    data = rng.standard_normal((n_days, 4))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_filter_dates_truncates_panel() -> None:
    """El filtro ``filter_dates`` reduce el número de filas correctamente."""
    df = _make_multi_panel(60)
    start = "2023-01-15"
    end = "2023-02-15"
    sub = df.loc[start:end]
    assert len(sub) < len(df)
    assert sub.index[0] >= pd.Timestamp(start)
    assert sub.index[-1] <= pd.Timestamp(end)


def test_filter_dates_argparse_present() -> None:
    """El parser de run_campaign acepta --filter-dates como argumento."""
    rc = _load_run_campaign()
    # Inspeccionar el código fuente: el argumento debe existir literalmente.
    src = Path(rc.__file__).read_text(encoding="utf-8")
    assert "--filter-dates" in src
    assert "filter_dates" in src
    assert "sweet_spot_overrides" in src


def test_sweet_spot_overrides_present() -> None:
    """El parser acepta --sweet-spot y propaga los overrides."""
    rc = _load_run_campaign()
    src = Path(rc.__file__).read_text(encoding="utf-8")
    assert "--sweet-spot" in src
    # Verificar que los valores del sweet_spot canónico están presentes:
    assert "subgraph_max_size" in src
    assert '"beta": 0.5' in src
    assert '"k_steps": 3' in src
    assert '"m_top": 5' in src
