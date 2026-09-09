"""Comparar resultados sweet_spot en bloques tuning vs hold-out (rev. v2 — Anexo M).

Lee dos CSVs ``sweet_spot_<block>_v2.csv`` y produce una tabla resumen
con métricas clave por bloque, además de la diferencia D-C en cada
uno. Diseñado para rellenar la Tabla M del informe.

Uso::

    uv run python scripts/compare_tuning_vs_holdout.py \
        --tuning outputs/tables/sweet_spot_tuning_v2.csv \
        --holdout outputs/tables/sweet_spot_holdout_v2.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def _f(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _summarize(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, float]]:
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    out: dict[str, dict[str, float]] = {}
    metrics = [
        "sharpe_ratio",
        "cumulative_return",
        "topm_hit_rate",
        "candidate_hit_rate",
        "asset_coverage",
    ]
    for model, model_rows in by_model.items():
        d = {}
        for m in metrics:
            vals = np.array([_f(r[m]) for r in model_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            d[m + "_mean"] = float(vals.mean()) if vals.size else float("nan")
            d[m + "_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            d[m + "_n"] = int(vals.size)
        out[model] = d
    return out


def _paired_d_minus_c(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Bootstrap pareado D-C por semilla."""
    by_seed: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        by_seed[r["seed"]][r["model"]] = r
    metrics = ["sharpe_ratio", "candidate_hit_rate", "topm_hit_rate", "asset_coverage"]
    paired_seeds = sorted(
        s for s, models in by_seed.items() if "C" in models and "D" in models
    )
    out: dict[str, dict[str, float]] = {}
    rng = np.random.default_rng(2026)
    for m in metrics:
        diffs = np.array(
            [_f(by_seed[s]["D"][m]) - _f(by_seed[s]["C"][m]) for s in paired_seeds],
            dtype=np.float64,
        )
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size < 2:
            out[m] = {
                "mean_diff": float("nan"),
                "p_d_better": float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
            }
            continue
        boot = np.array(
            [rng.choice(diffs, size=diffs.size, replace=True).mean() for _ in range(5000)]
        )
        out[m] = {
            "mean_diff": float(diffs.mean()),
            "p_d_better": float((boot > 0).mean()),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
            "n_pairs": float(diffs.size),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    args = parser.parse_args()

    rows_t = _read(args.tuning)
    rows_h = _read(args.holdout)

    sum_t = _summarize(rows_t, "tuning")
    sum_h = _summarize(rows_h, "holdout")
    pair_t = _paired_d_minus_c(rows_t)
    pair_h = _paired_d_minus_c(rows_h)

    print("\n=== Resumen por bloque ===\n")
    print(f"{'Block':<10}{'Model':<6}{'Sharpe':>10}{'cand_hit':>10}{'topm_hit':>10}{'coverage':>10}")
    for label, summary in [("Tuning", sum_t), ("Holdout", sum_h)]:
        for model in sorted(summary.keys()):
            d = summary[model]
            print(
                f"{label:<10}{model:<6}"
                f"{d['sharpe_ratio_mean']:>+7.4f} ± {d['sharpe_ratio_std']:.3f}"
                f"{d['candidate_hit_rate_mean']:>+7.4f}{'':<3}"
                f"{d['topm_hit_rate_mean']:>+7.4f}{'':<3}"
                f"{d['asset_coverage_mean']:>+7.4f}{'':<3}"
            )

    print("\n=== Comparación pareada D-C ===\n")
    print(f"{'Block':<10}{'metric':<22}{'mean_diff':>12}{'IC95%':>22}{'P(D>C)':>10}")
    for label, pair in [("Tuning", pair_t), ("Holdout", pair_h)]:
        for m, d in pair.items():
            ci = f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]"
            print(
                f"{label:<10}{m:<22}"
                f"{d['mean_diff']:>+12.4f}"
                f"{ci:>22}"
                f"{d['p_d_better']:>10.3f}"
            )


if __name__ == "__main__":
    main()
