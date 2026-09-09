"""Diagnósticos de la 4ª revisión para los TOST v8 (hallazgo 5).

1. Normalidad de las diferencias pareadas (Shapiro–Wilk).
2. Sensibilidad no paramétrica: TOST de Wilcoxon (dos unilaterales con
   desplazamiento en los márgenes).
3. Sensibilidad del veredicto de H-v8.1 a márgenes alternativos.

Uso::

    uv run python scripts/tost_diagnostics_v8.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

TABLES = "outputs/tables"
SEEDS_OLD = [7, 42, 99, 123, 314, 456, 789, 1024, 1729, 65535]


def tost_t(d: np.ndarray, marg: float) -> float:
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    p_inf = 1 - stats.t.cdf((d.mean() + marg) / se, n - 1)
    p_sup = stats.t.cdf((d.mean() - marg) / se, n - 1)
    return max(p_inf, p_sup)


def tost_wilcoxon(d: np.ndarray, marg: float) -> float:
    """TOST no paramétrico: Wilcoxon de rangos con signo desplazado."""
    p_inf = stats.wilcoxon(d + marg, alternative="greater").pvalue
    p_sup = stats.wilcoxon(d - marg, alternative="less").pvalue
    return max(p_inf, p_sup)


def main() -> None:
    r0 = pd.read_csv(f"{TABLES}/variant_R_v6.csv")
    d0 = pd.read_csv(f"{TABLES}/campaign_1_v3.csv")
    d0 = d0[d0["model"] == "D"]
    ext = pd.read_csv(f"{TABLES}/tost_extension_v8.csv")
    stk = pd.read_csv(f"{TABLES}/sticky_calibrated_v8.csv")
    for df in (r0, d0, ext, stk):
        df["seed"] = df["seed"].astype(int)

    def pairs(metric):
        rm = pd.concat([r0, ext[ext["model"] == "R"]]).set_index("seed")[metric]
        dm = pd.concat([d0, ext[ext["model"] == "D"]]).set_index("seed")[metric]
        common = sorted(set(rm.index) & set(dm.index))
        return (rm.loc[common] - dm.loc[common]).to_numpy(dtype=float)

    contrastes = {
        "H-v8.1 R-D cand_hit (n=40)": (pairs("candidate_hit_rate"), 0.019),
        "H-v8.1 R-D topm (n=40)": (pairs("topm_hit_rate"), 0.010),
        "H-v8.1 R-D Sharpe (n=40)": (pairs("sharpe_ratio"), 0.020),
    }
    sm = stk.set_index("seed")["candidate_hit_rate"]
    dm_new = ext[ext["model"] == "D"].set_index("seed")["candidate_hit_rate"]
    dm_old = d0.set_index("seed")["candidate_hit_rate"]
    seeds_new = sorted(set(sm.index) - set(SEEDS_OLD))
    contrastes["H-v8.2 D-S(p*) replica (n=30)"] = (
        (dm_new.loc[seeds_new] - sm.loc[seeds_new]).to_numpy(float), 0.019)
    contrastes["H-v8.2 D-S(p*) prereg (n=10)"] = (
        (dm_old.loc[SEEDS_OLD] - sm.loc[SEEDS_OLD]).to_numpy(float), 0.019)

    print("=== Normalidad (Shapiro-Wilk) y TOST parametrico vs Wilcoxon ===")
    for nombre, (d, marg) in contrastes.items():
        W, p_sh = stats.shapiro(d)
        p_t = tost_t(d, marg)
        p_w = tost_wilcoxon(d, marg)
        print(f"  {nombre:<34} Shapiro p={p_sh:.3f}  "
              f"p_TOST(t)={p_t:.2e}  p_TOST(Wilcoxon)={p_w:.2e}")

    print("\n=== Sensibilidad de H-v8.1 cand_hit al margen ===")
    d = contrastes["H-v8.1 R-D cand_hit (n=40)"][0]
    for marg in (0.010, 0.014, 0.019, 0.025):
        p = tost_t(d, marg)
        v = "equivalentes" if p < 0.05 else "no concluyente"
        print(f"  margen ±{marg:.3f}: p_TOST={p:.2e}  -> {v}")
    print(f"  (diferencia media {d.mean():+.4f}; el IC90 debe caber en el margen)")

    print("\n=== Sensibilidad de H-v8.2 replica (n=30) al margen ===")
    d = contrastes["H-v8.2 D-S(p*) replica (n=30)"][0]
    for marg in (0.005, 0.010, 0.019):
        p = tost_t(d, marg)
        v = "equivalentes" if p < 0.05 else "no concluyente"
        print(f"  margen ±{marg:.3f}: p_TOST={p:.2e}  -> {v}")


if __name__ == "__main__":
    main()
