"""Tests de las correcciones por múltiples comparaciones (rev. v2).

Verifica las propiedades matemáticas de Bonferroni, Holm-Bonferroni
step-down y Benjamini-Hochberg FDR aplicadas a la salida del
bootstrap pareado.

Referencia: Problema 1.2 del revisor del TFE.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_path = Path(__file__).resolve().parents[2] / "scripts" / "aggregate_results.py"
_spec = importlib.util.spec_from_file_location("aggregate_results", _path)
assert _spec is not None and _spec.loader is not None
ar = importlib.util.module_from_spec(_spec)
sys.modules["aggregate_results"] = ar
_spec.loader.exec_module(ar)


def _fake_bootstrap_dict(p_two_sided: list[float]) -> dict[str, dict[str, float]]:
    """Construir un dict equivalente al output del bootstrap con p_values dados."""
    return {
        f"metric_{i}": {
            "mean_diff": 0.0,
            "ci_lo": -1.0,
            "ci_hi": 1.0,
            "n_pairs": 5.0,
            "p_better": 1.0 - p / 2,  # invertir para p_two_sided=p
            "p_two_sided": p,
        }
        for i, p in enumerate(p_two_sided)
    }


def test_bonferroni_more_conservative_than_raw() -> None:
    """p_bonferroni nunca es menor que p_two_sided."""
    p_values = [0.001, 0.01, 0.03, 0.05, 0.1, 0.5, 1.0]
    d = _fake_bootstrap_dict(p_values)
    ar._apply_multiple_testing_correction(d)
    for v in d.values():
        assert v["p_bonferroni"] >= v["p_two_sided"] - 1e-12, (
            f"Bonferroni {v['p_bonferroni']} < raw {v['p_two_sided']}"
        )


def test_bonferroni_multiplies_by_k() -> None:
    """p_bonferroni = min(p_raw * k, 1)."""
    p_values = [0.01, 0.05, 0.1]
    d = _fake_bootstrap_dict(p_values)
    ar._apply_multiple_testing_correction(d)
    k = 3
    for (_, v), p in zip(sorted(d.items()), p_values, strict=True):
        expected = min(p * k, 1.0)
        assert abs(v["p_bonferroni"] - expected) < 1e-12


def test_holm_between_raw_and_bonferroni() -> None:
    """raw ≤ Holm ≤ Bonferroni para cada métrica."""
    p_values = [0.001, 0.005, 0.01, 0.04, 0.06, 0.5]
    d = _fake_bootstrap_dict(p_values)
    ar._apply_multiple_testing_correction(d)
    for v in d.values():
        assert v["p_two_sided"] - 1e-12 <= v["p_holm"] <= v["p_bonferroni"] + 1e-12, (
            f"Holm {v['p_holm']} fuera de [{v['p_two_sided']}, {v['p_bonferroni']}]"
        )


def test_holm_monotone_when_sorted() -> None:
    """Al ordenar por p_two_sided, p_holm es no-decreciente."""
    p_values = [0.001, 0.01, 0.05, 0.1, 0.3]
    d = _fake_bootstrap_dict(p_values)
    ar._apply_multiple_testing_correction(d)
    sorted_items = sorted(d.values(), key=lambda v: v["p_two_sided"])
    holm_seq = [v["p_holm"] for v in sorted_items]
    assert all(a <= b + 1e-12 for a, b in zip(holm_seq, holm_seq[1:], strict=False))


def test_fdr_less_conservative_than_bonferroni() -> None:
    """FDR ≤ Bonferroni para cada métrica (controla FDR, no FWER)."""
    p_values = [0.001, 0.005, 0.01, 0.04, 0.06, 0.5]
    d = _fake_bootstrap_dict(p_values)
    ar._apply_multiple_testing_correction(d)
    for v in d.values():
        assert v["p_fdr"] <= v["p_bonferroni"] + 1e-12, (
            f"FDR {v['p_fdr']} > Bonferroni {v['p_bonferroni']}"
        )


def test_fdr_monotone_when_sorted() -> None:
    """Al ordenar por p_two_sided, p_fdr es no-decreciente."""
    p_values = [0.001, 0.01, 0.05, 0.1, 0.3]
    d = _fake_bootstrap_dict(p_values)
    ar._apply_multiple_testing_correction(d)
    sorted_items = sorted(d.values(), key=lambda v: v["p_two_sided"])
    fdr_seq = [v["p_fdr"] for v in sorted_items]
    assert all(a <= b + 1e-12 for a, b in zip(fdr_seq, fdr_seq[1:], strict=False))


def test_candidate_hit_rate_survives_bonferroni_with_k11() -> None:
    """Caso real del revisor: con P(D>C)=1.000 y k=11, p_two_sided ≈ 0.

    candidate_hit_rate del informe original tenía P(D>C)=1.000, que con
    p_two_sided = min(p, 1-p) * 2 da p_two_sided=0. Bonferroni × 11 sigue
    siendo 0. Debe seguir significativo.
    """
    # Simulamos 11 métricas; 1 con p_two_sided ≈ 0, 10 con p ~ 0.5
    p_values = [0.0001] + [0.5] * 10
    d = _fake_bootstrap_dict(p_values)
    ar._apply_multiple_testing_correction(d)
    # La primera métrica debe sobrevivir Bonferroni
    first = d["metric_0"]
    assert first["p_bonferroni"] < 0.05, (
        f"candidate_hit_rate debe sobrevivir Bonferroni con k=11; "
        f"p_bonferroni = {first['p_bonferroni']}"
    )


def test_topm_hit_rate_does_not_survive_bonferroni() -> None:
    """Caso real del revisor: P(D>C)=0.957, p_two_sided≈0.086, k=11.

    Bonferroni × 11 = 0.95: NO significativo a α=0.05.
    """
    # topm_hit_rate del informe original
    p_values = [0.086] + [0.5] * 10
    d = _fake_bootstrap_dict(p_values)
    ar._apply_multiple_testing_correction(d)
    first = d["metric_0"]
    # Bonferroni: 0.086 * 11 ≈ 0.946 → NO significativo
    assert first["p_bonferroni"] > 0.05, (
        f"topm_hit_rate NO debe sobrevivir Bonferroni con k=11; "
        f"p_bonferroni = {first['p_bonferroni']}"
    )
