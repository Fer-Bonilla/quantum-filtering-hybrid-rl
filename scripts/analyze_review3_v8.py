"""Análisis de las comprobaciones de la 3ª revisión (v8).

Contrastes:
  (1) Hallazgo 1 — C vs D_centered (init emparejada): dinámica como único factor.
      También D_centered vs D_uniforme (efecto de la condición inicial).
  (2) Hallazgo 3 — sensibilidad al coste: {C, D, R} con 5 bp y 10 bp por cambio;
      contrastes pareados D-C y R-D dentro de cada nivel, y efecto del coste
      dentro de cada brazo frente a la base (5e-7).

Uso::

    python scripts/analyze_review3_v8.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

TABLES = "outputs/tables"
SEEDS = [7, 42, 99, 123, 314, 456, 789, 1024, 1729, 65535]
METRICS = ["candidate_hit_rate", "topm_hit_rate", "sharpe_ratio"]


def paired(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> dict:
    """Diferencia pareada por semilla a - b con t de Student e IC95."""
    m = a.set_index("seed")[metric].loc[SEEDS].to_numpy(dtype=float)
    n = b.set_index("seed")[metric].loc[SEEDS].to_numpy(dtype=float)
    d = m - n
    t, p2 = stats.ttest_rel(m, n)
    se = d.std(ddof=1) / np.sqrt(len(d))
    tc = stats.t.ppf(0.975, len(d) - 1)
    return dict(mean=d.mean(), lo=d.mean() - tc * se, hi=d.mean() + tc * se,
                p2=p2, n=len(d))


def fmt(r: dict) -> str:
    return (f"{r['mean']:+.4f}  IC95 [{r['lo']:+.4f},{r['hi']:+.4f}]  "
            f"p2={r['p2']:.4f}  n={r['n']}")


def main() -> None:
    rev = pd.read_csv(f"{TABLES}/review3_checks_v8.csv")
    rev["seed"] = rev["seed"].astype(int)
    base = pd.read_csv(f"{TABLES}/campaign_1_v3.csv")
    base = base[base["model"].isin(["C", "D"])].copy()
    base["seed"] = base["seed"].astype(int)
    base = base[base["seed"].isin(SEEDS)]
    c0 = base[base["model"] == "C"]
    d0 = base[base["model"] == "D"]
    r0 = pd.read_csv(f"{TABLES}/variant_R_v6.csv")
    r0["seed"] = r0["seed"].astype(int)
    r0 = r0[r0["seed"].isin(SEEDS)].drop_duplicates("seed")

    dc = rev[rev["tag"] == "Dcentered"]

    print("=" * 72)
    print("HALLAZGO 1 — condicion inicial emparejada (n=10 semillas pareadas)")
    print("=" * 72)
    for met in METRICS:
        print(f"\n  {met}:")
        print(f"    D_centered - C          : {fmt(paired(dc, c0, met))}")
        print(f"    D_centered - D_uniforme : {fmt(paired(dc, d0, met))}")
        print(f"    (referencia D_unif - C) : {fmt(paired(d0, c0, met))}")

    print()
    print("=" * 72)
    print("HALLAZGO 3 — sensibilidad al coste de transaccion (n=10 por brazo)")
    print("=" * 72)
    arms = {"C": c0, "D": d0, "R": r0}
    for level in ("5bp", "10bp"):
        sub = {v: rev[rev["tag"] == f"{v}_cost{level}"] for v in ("C", "D", "R")}
        print(f"\n--- coste {level} por cambio ---")
        for met in METRICS:
            print(f"  {met}:")
            print(f"    D - C : {fmt(paired(sub['D'], sub['C'], met))}")
            print(f"    R - D : {fmt(paired(sub['R'], sub['D'], met))}")
        print("  efecto del coste dentro de cada brazo (coste - base):")
        for v in ("C", "D", "R"):
            for met in ("sharpe_ratio", "candidate_hit_rate"):
                print(f"    {v} {met:<20}: {fmt(paired(sub[v], arms[v], met))}")

    print()
    print("Medias por brazo (sharpe / cand_hit):")
    for tag, g in rev.groupby("tag"):
        print(f"  {tag:<12} sharpe={g['sharpe_ratio'].mean():+.4f}  "
              f"cand={g['candidate_hit_rate'].mean():.4f}")
    print(f"  {'C_base':<12} sharpe={c0['sharpe_ratio'].mean():+.4f}  "
          f"cand={c0['candidate_hit_rate'].mean():.4f}")
    print(f"  {'D_base':<12} sharpe={d0['sharpe_ratio'].mean():+.4f}  "
          f"cand={d0['candidate_hit_rate'].mean():.4f}")
    print(f"  {'R_base':<12} sharpe={r0['sharpe_ratio'].mean():+.4f}  "
          f"cand={r0['candidate_hit_rate'].mean():.4f}")


if __name__ == "__main__":
    main()
