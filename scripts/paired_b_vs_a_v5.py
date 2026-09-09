"""Bootstrap pareado B vs A para campaign_1_v3 (v5 — punto 2.2 del revisor).

Análisis post-hoc sobre datos existentes en `campaign_1_v3.csv`. Computa
la diferencia $B - A$ por semilla con bootstrap pareado de 5000
iteraciones y aplica Bonferroni $k=11$ y $k=5$ para coherencia con el
resto del análisis (Anexo K).

El objetivo es responder a la pregunta del revisor: ¿el estado relacional
del Modelo B aporta algo medible sobre el Modelo A baseline?
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from src.utils.paired_stats import paired_bootstrap

CSV_PATH = Path("outputs/tables/campaign_1_v3.csv")
OUT_PATH = Path("outputs/tables/campaign_1_v3_paired_b_vs_a.csv")

METRICS_FIVE = (
    "candidate_hit_rate",
    "topm_hit_rate",
    "sharpe_ratio",
    "cumulative_return",
    "episodes_to_convergence",
)
METRICS_ALL = (
    "cumulative_return",
    "sharpe_ratio",
    "max_drawdown",
    "mean_reward",
    "episodes_to_convergence",
    "train_reward_last",
    "topm_hit_rate",
    "candidate_hit_rate",
    "asset_coverage",
    "mean_latency_ms",
    "duration_seconds",
)


def main() -> int:
    rows = list(csv.DictReader(CSV_PATH.open("r", encoding="utf-8")))
    by_seed: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        by_seed[r["seed"]][r["model"]] = r

    paired_seeds = sorted(
        s for s, models in by_seed.items() if "A" in models and "B" in models
    )
    print(f"Pares B-A disponibles: {len(paired_seeds)} semillas comunes")

    rng = np.random.default_rng(2026)
    out_rows: list[dict[str, float | str]] = []

    print(f"{'metric':<26}{'mean(B-A)':>12}{'IC95%':>26}"
          f"{'P(B>A)':>10}{'p_raw':>10}{'p_Bk=11':>10}{'p_Bk=5':>10}")
    print("-" * 105)

    for m in METRICS_ALL:

        def to_float(s: str) -> float:
            try:
                v = float(s)
                if v == -1.0:
                    return float("nan")
                return v
            except (ValueError, TypeError):
                return float("nan")

        diffs = []
        for s in paired_seeds:
            ba = to_float(by_seed[s]["B"][m])
            aa = to_float(by_seed[s]["A"][m])
            if np.isfinite(ba) and np.isfinite(aa):
                diffs.append(ba - aa)
        diffs = np.asarray(diffs, dtype=np.float64)
        if diffs.size < 2:
            print(f"{m:<26}{'(insufficient data)':>50}")
            continue

        bs = paired_bootstrap(diffs, n_boot=5000, rng=rng)
        ci_lo, ci_hi = bs.ci_lo, bs.ci_hi
        p_better = bs.p_positive
        p_two = bs.p_two          # nan si degenerado (identico por construccion)
        p_bonf_k11 = min(p_two * 11, 1.0) if np.isfinite(p_two) else float("nan")
        is_h = m in METRICS_FIVE
        p_bonf_k5 = (min(p_two * 5, 1.0) if (is_h and np.isfinite(p_two))
                     else float("nan"))
        out_rows.append(
            {
                "metric": m,
                "mean_diff_b_minus_a": float(diffs.mean()),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_pairs": int(diffs.size),
                "p_b_better_than_a": p_better,
                "p_two_sided": p_two,
                "p_bonferroni_k11": p_bonf_k11,
                "p_bonferroni_k5": p_bonf_k5,
                "hypothesis": "H1-H5" if is_h else "descriptiva",
            }
        )
        ci_str = f"[{ci_lo:+.4f},{ci_hi:+.4f}]"
        bk5 = f"{p_bonf_k5:.3f}" if is_h else "---"
        print(
            f"{m:<26}{diffs.mean():+12.4f}{ci_str:>26}"
            f"{p_better:>10.3f}{p_two:>10.3f}{p_bonf_k11:>10.3f}{bk5:>10}"
        )

    if out_rows:
        with OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print()
        print(f"[OK] Escrito {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
