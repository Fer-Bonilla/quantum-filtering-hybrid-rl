"""Sensibilidad del DSR al numero de ensayos K = 2..39 (hallazgo 1, 6a rev.).

Misma formula que recalculate_dsr_review.py; para cada estrategia y cada K
reporta SR* y DSR, y el primer K a partir del cual DSR < 0,95 (punto de cruce).
Salida: outputs/tables/dsr_k_sensitivity.csv.

Uso::

    uv run python scripts/dsr_k_sensitivity_review.py
"""

from __future__ import annotations

import csv
import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.paths import TABLES_DIR

sys.stdout.reconfigure(encoding="utf-8")

EULER = 0.5772156649015329
OUT = TABLES_DIR / "dsr_k_sensitivity.csv"


def sr_star(sd: float, k: int) -> float:
    z1 = stats.norm.ppf(1 - 1.0 / k)
    z2 = stats.norm.ppf(1 - 1.0 / (k * np.e))
    return float(sd * ((1 - EULER) * z1 + EULER * z2))


def dsr(sr: float, T: int, g3: float, g4: float, star: float) -> float:
    num = (sr - star) * np.sqrt(T - 1)
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2, 1e-12))
    return float(stats.norm.cdf(num / den))


def main() -> None:
    d = pd.read_csv(TABLES_DIR / "dsr_recalculated.csv")
    d = d[d["escenario"] == "K=39_sd_candidatas"]
    fam = pd.read_csv(TABLES_DIR / "dsr_trial_sharpes.csv")
    sd_c = float(fam[fam["family"] != "diagnostico_no_candidato"]
                 ["daily_sharpe"].std(ddof=1))
    sd_s = float(fam[fam["family"] == "seleccion_original"]
                 ["daily_sharpe"].std(ddof=1))
    rows = []
    for sd_name, sd in [("candidatas_39", sd_c), ("seleccion_19", sd_s)]:
        for _, r in d.iterrows():
            cruce = None
            for k in range(2, 40):
                star = sr_star(sd, k)
                v = dsr(r["sharpe"], int(r["T"]), r["skew"],
                        r["kurt_pearson"], star)
                if v < 0.95 and cruce is None:
                    cruce = k
                rows.append({"estrategia": r["estrategia"],
                             "sd_familia": sd_name, "sd": round(sd, 6),
                             "K": k, "sr_star": round(star, 6),
                             "dsr": round(v, 6),
                             "supera_095": int(v >= 0.95)})
            print(f"{r['estrategia']:<14} sd={sd_name:<14} "
                  f"primer K con DSR<0,95: {cruce}")
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
