"""Bootstrap pareado de las variantes v6 contra C y D (puntos 2-3 director).

Compara, por semilla común y con 5000 iteraciones bootstrap:

    R vs C   — ¿la caminata clásica informada aporta sobre máscara random?
    R vs D   — ¿la DTQW aporta sobre máscara random?
    Q vs C   — ¿el annealing aporta sobre caminata clásica?
    Q vs D   — ¿annealing vs DTQW?

Las filas C y D provienen de ``campaign_1_v3.csv`` (misma configuración
por defecto M=8, mismas 10 semillas, mismos 50k steps). Bonferroni k=11.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path("outputs/tables")
METRICS = (
    "cumulative_return", "sharpe_ratio", "max_drawdown", "mean_reward",
    "episodes_to_convergence", "train_reward_last", "topm_hit_rate",
    "candidate_hit_rate", "asset_coverage", "mean_latency_ms",
    "duration_seconds",
)


def _to_float(s: str) -> float:
    try:
        v = float(s)
        return float("nan") if v == -1.0 else v
    except (TypeError, ValueError):
        return float("nan")


def _load(path: Path, model: str) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    return {r["seed"]: r for r in rows if r["model"] == model}


def paired(label: str, lhs: dict, rhs: dict, out_rows: list[dict]) -> None:
    seeds = sorted(set(lhs) & set(rhs))
    rng = np.random.default_rng(2026)
    print(f"\n=== {label} (n={len(seeds)} semillas comunes) ===")
    print(f"{'metric':<26}{'mean diff':>12}{'IC95%':>26}{'P(>0)':>9}{'p_Bonf':>9}")
    for m in METRICS:
        diffs = np.array([
            _to_float(lhs[s][m]) - _to_float(rhs[s][m]) for s in seeds
        ])
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size < 2:
            continue
        boot = np.array([
            rng.choice(diffs, diffs.size, replace=True).mean() for _ in range(5000)
        ])
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        p_pos = float((boot > 0).mean())
        p_two = 2.0 * min(p_pos, 1.0 - p_pos)
        p_bonf = min(p_two * 11, 1.0)
        out_rows.append({
            "comparison": label, "metric": m,
            "mean_diff": float(diffs.mean()),
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "n_pairs": int(diffs.size), "p_positive": p_pos,
            "p_two_sided": p_two, "p_bonferroni_k11": p_bonf,
        })
        flag = " *" if p_bonf < 0.05 else ""
        print(f"{m:<26}{diffs.mean():+12.4f}"
              f"  [{ci_lo:+.4f},{ci_hi:+.4f}]"
              f"{p_pos:>9.3f}{p_bonf:>9.3f}{flag}")


def main() -> None:
    c_rows = _load(BASE / "campaign_1_v3.csv", "C")
    d_rows = _load(BASE / "campaign_1_v3.csv", "D")
    r_rows = _load(BASE / "variant_R_v6.csv", "R")
    q_rows = _load(BASE / "variant_Q_v6.csv", "Q")

    out: list[dict] = []
    paired("R-C", r_rows, c_rows, out)
    paired("R-D", r_rows, d_rows, out)
    paired("Q-C", q_rows, c_rows, out)
    paired("Q-D", q_rows, d_rows, out)
    paired("Q-R", q_rows, r_rows, out)

    out_path = BASE / "paired_variants_v6.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\n[OK] {out_path}")


if __name__ == "__main__":
    main()
