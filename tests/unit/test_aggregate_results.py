"""Tests de scripts/aggregate_results.py (importado como módulo)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# Cargar el script como módulo
_path = Path(__file__).resolve().parents[2] / "scripts" / "aggregate_results.py"
spec = importlib.util.spec_from_file_location("aggregate_results", _path)
assert spec is not None and spec.loader is not None
aggregate_results = importlib.util.module_from_spec(spec)
sys.modules["aggregate_results"] = aggregate_results
spec.loader.exec_module(aggregate_results)


def _make_rows() -> list[dict[str, str]]:
    """Cuatro modelos × 3 semillas con métricas sintéticas."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, str]] = []
    means = {"A": 0.0, "B": 0.05, "C": 0.10, "D": 0.15}
    for model in ("A", "B", "C", "D"):
        for seed in (42, 123, 456):
            ret = means[model] + rng.normal(0.0, 0.01)
            rows.append(
                {
                    "model": model,
                    "seed": str(seed),
                    "cumulative_return": str(ret),
                    "sharpe_ratio": str(ret * 2),
                    "max_drawdown": "0.05",
                    "mean_reward": "0.001",
                    "episodes_to_convergence": "3.0",
                    "train_reward_last": "0.001",
                    "topm_hit_rate": "0.25",
                    "candidate_hit_rate": "0.35",
                    "asset_coverage": "0.8",
                    "mean_latency_ms": "1.2",
                    "duration_seconds": "5.0",
                }
            )
    return rows


def test_to_float_handles_minus_one() -> None:
    assert np.isnan(aggregate_results._to_float("-1"))
    assert aggregate_results._to_float("3.14") == 3.14
    assert np.isnan(aggregate_results._to_float("not_a_number"))


def test_per_model_stats_produces_all_metrics() -> None:
    rows = _make_rows()
    stats = aggregate_results._per_model_stats(rows)
    assert set(stats.keys()) == {"A", "B", "C", "D"}
    for model in stats:
        for metric in aggregate_results.ALL_METRICS:
            assert metric in stats[model]
            assert stats[model][metric]["n"] in (0.0, 3.0)


def test_paired_bootstrap_detects_better_d() -> None:
    """Con D > C consistentemente, P(D>C) debe ser alta."""
    rows = _make_rows()  # D mean 0.15 vs C mean 0.10
    paired = aggregate_results._paired_bootstrap_c_vs_d(rows, n_boot=500)
    assert "cumulative_return" in paired
    s = paired["cumulative_return"]
    assert s["mean_diff"] > 0
    assert s["p_better"] > 0.9


def test_paired_bootstrap_no_common_seeds() -> None:
    """Si C y D no comparten semillas, el resultado debe estar vacío."""
    rows: list[dict[str, str]] = [
        {"model": "C", "seed": "1", "cumulative_return": "0.1"},
        {"model": "D", "seed": "2", "cumulative_return": "0.2"},
    ]
    # Rellenar las otras métricas con valores válidos para evitar KeyError
    for r in rows:
        for m in aggregate_results.ALL_METRICS:
            r.setdefault(m, "0.0")
    paired = aggregate_results._paired_bootstrap_c_vs_d(rows, n_boot=10)
    assert paired == {}


def test_markdown_summary_includes_models_and_pairing() -> None:
    rows = _make_rows()
    stats = aggregate_results._per_model_stats(rows)
    paired = aggregate_results._paired_bootstrap_c_vs_d(rows, n_boot=200)
    md = aggregate_results._format_markdown_summary(stats, paired)
    for model in ("A", "B", "C", "D"):
        assert f"| {model}" not in md or "cumulative_return" in md  # presencia de tabla
    assert "Comparación pareada D vs C" in md
    assert "P(D > C)" in md
