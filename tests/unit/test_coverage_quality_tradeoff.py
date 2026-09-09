"""Tests del trade-off cobertura ↔ calidad (rev. v2 — Problema 1.6).

Verifica que las métricas derivadas se computan correctamente y son
consistentes con la narrativa del informe.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_path = Path(__file__).resolve().parents[2] / "scripts" / "coverage_quality_tradeoff.py"
_spec = importlib.util.spec_from_file_location("coverage_quality_tradeoff", _path)
assert _spec is not None and _spec.loader is not None
cqt = importlib.util.module_from_spec(_spec)
sys.modules["coverage_quality_tradeoff"] = cqt
_spec.loader.exec_module(cqt)


def _row(model: str, seed: int, topm: float, coverage: float, cand: float) -> dict[str, str]:
    return {
        "campaign_id": "test",
        "model": model,
        "seed": str(seed),
        "topm_hit_rate": str(topm),
        "asset_coverage": str(coverage),
        "candidate_hit_rate": str(cand),
    }


def test_precision_per_visited_equals_topm_when_full_coverage() -> None:
    """ppv = topm_hit / coverage; con coverage=1.0 ppv == topm_hit."""
    rows = [_row("A", 42, topm=0.21, coverage=1.0, cand=1.0)]
    enriched = cqt.compute_tradeoff_metrics(rows)
    assert abs(float(enriched[0]["precision_per_visited"]) - 0.21) < 1e-9


def test_precision_per_visited_higher_when_lower_coverage_same_topm() -> None:
    """Con mismo topm_hit, menor coverage ⇒ mayor ppv."""
    rows = [
        _row("C", 42, topm=0.20, coverage=1.00, cand=0.5),
        _row("D", 42, topm=0.20, coverage=0.50, cand=0.5),
    ]
    enriched = cqt.compute_tradeoff_metrics(rows)
    ppv_c = float(enriched[0]["precision_per_visited"])
    ppv_d = float(enriched[1]["precision_per_visited"])
    assert ppv_d > ppv_c, f"ppv_D ({ppv_d}) debería ser mayor que ppv_C ({ppv_c})"
    assert abs(ppv_d - 2.0 * ppv_c) < 1e-6


def test_recall_proxy_uses_m_top_and_universe() -> None:
    """recall_proxy = candidate_hit_rate × (m_top / n_promising)."""
    rows = [_row("D", 42, topm=0.20, coverage=0.6, cand=0.50)]
    enriched = cqt.compute_tradeoff_metrics(rows, m_top=3, n_promising_total=6)
    expected = 0.50 * (3 / 6)
    assert abs(float(enriched[0]["recall_proxy"]) - expected) < 1e-9


def test_recall_proxy_capped_at_one() -> None:
    """Si cand_hit · m/N supera 1, se capa a 1."""
    rows = [_row("D", 42, topm=0.50, coverage=0.6, cand=1.0)]
    enriched = cqt.compute_tradeoff_metrics(rows, m_top=10, n_promising_total=6)
    assert float(enriched[0]["recall_proxy"]) == 1.0


def test_aggregate_groups_by_model() -> None:
    """aggregate_by_model agrupa filas por modelo y computa media."""
    rows = [
        _row("D", 42, topm=0.20, coverage=0.60, cand=0.45),
        _row("D", 123, topm=0.22, coverage=0.58, cand=0.48),
        _row("C", 42, topm=0.19, coverage=0.92, cand=0.40),
    ]
    enriched = cqt.compute_tradeoff_metrics(rows)
    summary = cqt.aggregate_by_model(enriched)
    assert set(summary.keys()) == {"D", "C"}
    assert abs(summary["D"]["topm_hit_rate"]["mean"] - 0.21) < 1e-9
    assert abs(summary["D"]["asset_coverage"]["mean"] - 0.59) < 1e-9


def test_ppv_stable_when_coverage_near_zero() -> None:
    """ppv no diverge con coverage = 0 (división protegida por eps)."""
    rows = [_row("D", 42, topm=0.0, coverage=0.0, cand=0.0)]
    enriched = cqt.compute_tradeoff_metrics(rows)
    ppv = float(enriched[0]["precision_per_visited"])
    assert np.isfinite(ppv) and ppv == 0.0


def test_d_beats_c_in_ppv_typical_pattern() -> None:
    """Patrón esperado: D tiene mayor ppv que C aunque menor coverage."""
    # Replicar valores reales de campaign_1
    rows = [
        _row("C", 42, topm=0.196, coverage=0.933, cand=0.470),
        _row("C", 123, topm=0.196, coverage=0.933, cand=0.470),
        _row("D", 42, topm=0.201, coverage=0.880, cand=0.450),
        _row("D", 123, topm=0.201, coverage=0.880, cand=0.450),
    ]
    enriched = cqt.compute_tradeoff_metrics(rows)
    summary = cqt.aggregate_by_model(enriched)
    assert summary["D"]["precision_per_visited"]["mean"] > summary["C"]["precision_per_visited"]["mean"]
    assert summary["D"]["asset_coverage"]["mean"] < summary["C"]["asset_coverage"]["mean"]
