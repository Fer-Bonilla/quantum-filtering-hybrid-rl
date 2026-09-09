"""Calibración bajo el nulo del valor p bootstrap y cruce con pruebas
calibradas (7ª revisión, hallazgo 2.7).

(1) Tasa de rechazo al 5 % bajo H0 (diferencias N(0,1)) para n en {10, 40}:
    bootstrap percentil (especificación del Anexo B), t pareada y Wilcoxon.
(2) Cruce de los contrastes pareados del cuerpo con las tres pruebas.

Salidas:
  outputs/tables/bootstrap_calibration.csv
  outputs/tables/paired_crosscheck.csv

Uso::

    uv run python scripts/bootstrap_calibration_review.py
"""

from __future__ import annotations

import csv
import sys

import numpy as np
import pandas as pd
from scipy import stats
from src.utils.paired_stats import paired_bootstrap
from src.utils.paths import TABLES_DIR

sys.stdout.reconfigure(encoding="utf-8")

R_REPS = 2000
B_NULL = 2000
SEED_NULL = 20260909
OUT_CAL = TABLES_DIR / "bootstrap_calibration.csv"
OUT_CROSS = TABLES_DIR / "paired_crosscheck.csv"
METRICS = ["candidate_hit_rate", "topm_hit_rate", "sharpe_ratio"]


def calibration() -> list[dict]:
    rng = np.random.default_rng(SEED_NULL)
    rows = []
    for n in (10, 40):
        rej = {"bootstrap_percentil": 0, "t_pareada": 0, "wilcoxon": 0}
        for _ in range(R_REPS):
            d = rng.normal(0.0, 1.0, n)
            rej["bootstrap_percentil"] += paired_bootstrap(d, n_boot=B_NULL, rng=rng).p_two < 0.05
            rej["t_pareada"] += stats.ttest_1samp(d, 0.0).pvalue < 0.05
            rej["wilcoxon"] += stats.wilcoxon(d).pvalue < 0.05
        for k, v in rej.items():
            rows.append({"n": n, "prueba": k, "alpha": 0.05,
                         "tasa_rechazo_H0": round(v / R_REPS, 4),
                         "replicas": R_REPS, "B": B_NULL, "rng_seed": SEED_NULL})
            print(f"  n={n:<3} {k:<20} rechazo bajo H0 = {v / R_REPS:.3f}")
    return rows


def _arm(path: str, model: str) -> pd.DataFrame:
    df = pd.read_csv(TABLES_DIR / path)
    df["seed"] = df["seed"].astype(int)
    return df[df["model"] == model].set_index("seed")


def crosscheck() -> list[dict]:
    A, B, C, D = [_arm("campaign_1_v3.csv", m) for m in "ABCD"]
    R = _arm("variant_R_v6.csv", "R")
    Q = _arm("variant_Q_v6.csv", "Q")
    Cr = _arm("crel_variant_v8.csv", "Crel")
    seeds = sorted(set(C.index) & set(D.index) & set(R.index))
    pares = {"D-C": (D, C), "B-A": (B, A), "C-B": (C, B), "R-D": (R, D),
             "R-C": (R, C), "Q-R": (Q, R), "Crel-C": (Cr, C), "Crel-B": (Cr, B)}
    rows = []
    for name, (X, Y) in pares.items():
        for met in METRICS:
            d = (X.loc[seeds, met] - Y.loc[seeds, met]).to_numpy(float)
            bs = paired_bootstrap(d, seed=2026)
            if bs.degenerate:
                rows.append({"contraste": name, "metrica": met, "n": bs.n,
                             "mean_diff": round(bs.mean, 6), "p_bootstrap": "",
                             "p_t_pareada": "", "p_wilcoxon": "",
                             "nota": "identico por construccion (degenerado)"})
                continue
            pt = stats.ttest_1samp(d, 0.0).pvalue
            pw = stats.wilcoxon(d).pvalue
            rows.append({"contraste": name, "metrica": met, "n": bs.n,
                         "mean_diff": round(bs.mean, 6),
                         "p_bootstrap": round(bs.p_two, 5),
                         "p_t_pareada": round(float(pt), 5),
                         "p_wilcoxon": round(float(pw), 5), "nota": ""})
            print(f"  {name:<7}{met:<20}{bs.mean:+.4f}  boot={bs.p_two:.4f}  "
                  f"t={pt:.4f}  W={pw:.4f}")
    return rows


def main() -> None:
    print("=== Calibracion bajo H0 (alpha=0.05) ===")
    cal = calibration()
    print("\n=== Cruce de contrastes pareados ===")
    cross = crosscheck()
    for path, rows in ((OUT_CAL, cal), (OUT_CROSS, cross)):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()
