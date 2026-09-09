"""Tests del helper run_ablation (importado como módulo)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_path = Path(__file__).resolve().parents[2] / "scripts" / "run_ablation.py"
_spec = importlib.util.spec_from_file_location("run_ablation_test", _path)
assert _spec is not None and _spec.loader is not None
run_ablation = importlib.util.module_from_spec(_spec)
sys.modules["run_ablation_test"] = run_ablation
_spec.loader.exec_module(run_ablation)


def test_row_to_dict_adds_metadata() -> None:
    """_row_to_dict añade las columnas ablation_kind y ablation_value."""

    # Construir un RunRow real (importado del run_campaign vía run_ablation)
    rc = run_ablation.run_campaign
    row = rc.RunRow(
        campaign_id="x",
        model="D",
        seed=42,
        universe="u",
        total_steps=100,
        cumulative_return=0.1,
        sharpe_ratio=0.2,
        max_drawdown=0.05,
        mean_reward=0.001,
        asset_coverage=0.5,
        topm_hit_rate=0.3,
        candidate_hit_rate=0.4,
        time_to_first_promising=10.0,
        mean_latency_ms=1.5,
        episodes_to_convergence=3.0,
        train_reward_last=0.001,
        train_reward_mean=0.001,
        n_train_steps=100,
        timestamp_utc="2026-01-01T00:00:00Z",
        duration_seconds=2.0,
    )
    d = run_ablation._row_to_dict(row, ablation_kind="k", ablation_value="3")
    assert d["ablation_kind"] == "k"
    assert d["ablation_value"] == "3"
    assert d["cumulative_return"] == 0.1
    assert d["model"] == "D"


def test_write_csv_includes_metadata_columns(tmp_path: Path) -> None:
    out = tmp_path / "abl.csv"
    rows = [
        {
            "model": "D",
            "seed": 42,
            "cumulative_return": 0.1,
            "ablation_kind": "k",
            "ablation_value": "3",
        },
        {
            "model": "D",
            "seed": 42,
            "cumulative_return": 0.2,
            "ablation_kind": "k",
            "ablation_value": "5",
        },
    ]
    run_ablation._write_csv(rows, out)
    text = out.read_text(encoding="utf-8")
    assert "ablation_kind" in text
    assert "ablation_value" in text
    assert "3" in text
    assert "5" in text


def test_write_csv_handles_empty() -> None:
    """Llamar _write_csv con una lista vacía no debe explotar."""

    # Simplemente no debe lanzar excepción; el archivo no se escribe.
    run_ablation._write_csv([], Path("nonexistent.csv"))
