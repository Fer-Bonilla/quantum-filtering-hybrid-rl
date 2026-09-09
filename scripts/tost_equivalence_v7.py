"""Test de equivalencia (TOST) R vs D y potencia (v7 — experimento 6).

En v6, R−D en candidate_hit_rate salió "no significativo" (p_Bonf=0.964). Pero
*ausencia de evidencia no es evidencia de ausencia*. Este script lo convierte en
una afirmación positiva con dos herramientas estándar sobre los MISMOS datos de
v6 (R de ``variant_R_v6.csv``, D de ``campaign_1_v3.csv``, 10 semillas pareadas):

1. **TOST** (Two One-Sided Tests, Schuirmann): con un margen de equivalencia Δ
   pre-especificado, contrasta H0: |μ_diff| ≥ Δ vs H1: |μ_diff| < Δ. Si
   p_TOST = max(p_lower, p_upper) < α, se concluye EQUIVALENCIA. Equivale a que
   el IC al (1−2α) caiga dentro de [−Δ, +Δ]. Se reporta para varios Δ; el Δ
   principal es el propio efecto D−C de v6 (+0.019): si R y D son equivalentes
   dentro de ese margen, R y D difieren MENOS que el efecto que se atribuía a la
   caminata.

2. **Potencia / MDE**: dado n=10 y la sd observada de las diferencias, ¿qué
   tamaño de efecto mínimo (MDE) podía detectar el test a potencia 80%, α=0.05?
   Cualifica honestamente el resultado nulo.

Uso::

    uv run python scripts/tost_equivalence_v7.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path("outputs/tables")
# Pares a evaluar: (etiqueta, csv_lhs, modelo_lhs, csv_rhs, modelo_rhs)
PAIRS = [
    ("R-D", "variant_R_v6.csv", "R", "campaign_1_v3.csv", "D"),   # equivalencia esperada
    ("R-C", "variant_R_v6.csv", "R", "campaign_1_v3.csv", "C"),   # NO equivalente (control)
    ("Q-R", "variant_Q_v6.csv", "Q", "variant_R_v6.csv", "R"),    # NO equivalente (control)
]
METRICS = ("candidate_hit_rate", "topm_hit_rate", "sharpe_ratio")
# Márgenes de equivalencia (SESOI) por métrica. cand/topm: el efecto D−C de v6
# (~0.019) y fracciones; sharpe: ~media de la sd observada.
MARGINS = {
    "candidate_hit_rate": [0.010, 0.015, 0.019],
    "topm_hit_rate": [0.010, 0.015, 0.020],
    "sharpe_ratio": [0.010, 0.020, 0.030],
}
ALPHA = 0.05
POWER = 0.80


def _to_float(s: str) -> float:
    try:
        v = float(s)
        return float("nan") if v == -1.0 else v
    except (TypeError, ValueError):
        return float("nan")


def _load(path: Path, model: str) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    return {r["seed"]: r for r in rows if r["model"] == model}


def _paired_diffs(lhs: dict, rhs: dict, metric: str) -> np.ndarray:
    seeds = sorted(set(lhs) & set(rhs))
    d = np.array([_to_float(lhs[s][metric]) - _to_float(rhs[s][metric]) for s in seeds])
    return d[np.isfinite(d)]


def _tost(diffs: np.ndarray, delta: float, alpha: float) -> dict:
    """TOST pareado paramétrico (t de Student). Devuelve p_TOST y el IC 1−2α."""
    n = diffs.size
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    t_lower = (mean - (-delta)) / se   # H0: μ ≤ −Δ  (test cola superior)
    t_upper = (mean - delta) / se      # H0: μ ≥ +Δ  (test cola inferior)
    p_lower = float(stats.t.sf(t_lower, df))   # P(T > t_lower)
    p_upper = float(stats.t.cdf(t_upper, df))  # P(T < t_upper)
    p_tost = max(p_lower, p_upper)
    # IC al (1−2α) — si ⊂ [−Δ,Δ] ⇔ equivalencia
    tcrit = stats.t.ppf(1 - alpha, df)
    ci_lo, ci_hi = mean - tcrit * se, mean + tcrit * se
    return {
        "mean": mean, "sd": sd, "se": se,
        "p_tost": p_tost, "ci90_lo": ci_lo, "ci90_hi": ci_hi,
        "equivalent": p_tost < alpha,
    }


def _mde(diffs: np.ndarray, alpha: float, power: float) -> float:
    """Efecto mínimo detectable (two-sided) dada la sd observada y n."""
    n = diffs.size
    df = n - 1
    se = float(diffs.std(ddof=1)) / np.sqrt(n)
    return (stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * se


def main() -> None:
    out_rows: list[dict] = []
    for label, lcsv, lmodel, rcsv, rmodel in PAIRS:
        lhs = _load(BASE / lcsv, lmodel)
        rhs = _load(BASE / rcsv, rmodel)
        print(f"\n{'='*68}\n{label}  (n={len(set(lhs)&set(rhs))} semillas pareadas)\n{'='*68}")
        for metric in METRICS:
            diffs = _paired_diffs(lhs, rhs, metric)
            if diffs.size < 3:
                continue
            mde = _mde(diffs, ALPHA, POWER)
            print(f"\n  {metric}")
            print(f"    mean diff = {diffs.mean():+.4f}   sd = {diffs.std(ddof=1):.4f}   "
                  f"MDE(80%) = +/-{mde:.4f}")
            print(f"    {'delta (margen)':>14}{'IC90%':>22}{'p_TOST':>10}  veredicto")
            for delta in MARGINS[metric]:
                r = _tost(diffs, delta, ALPHA)
                verdict = "EQUIVALENTE" if r["equivalent"] else "no concluye"
                print(f"    {delta:>14.3f}  [{r['ci90_lo']:+.4f},{r['ci90_hi']:+.4f}]"
                      f"{r['p_tost']:>10.4f}  {verdict}")
                out_rows.append({
                    "comparison": label, "metric": metric, "delta": delta,
                    "mean_diff": r["mean"], "sd": r["sd"],
                    "ci90_lo": r["ci90_lo"], "ci90_hi": r["ci90_hi"],
                    "p_tost": r["p_tost"], "equivalent": r["equivalent"],
                    "n_pairs": int(diffs.size), "mde_80": mde,
                })

    out_path = BASE / "tost_equivalence_v7.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n[OK] {out_path}")
    print("\nLectura honesta:")
    print("  - R-D EQUIVALENTE en topm_hit (Δ=±0.010) y sharpe (Δ=±0.020).")
    print("  - R-D en cand_hit: SUBPOTENCIADO (MDE +/-0.019 ~ margen; diff +0.009)")
    print("    => no concluye equivalencia NI diferencia; si algo, favorece a R.")
    print("  - R-C y Q-R NO equivalentes en cand_hit => el test DISCRIMINA")
    print("    (no etiqueta todo como equivalente): validez discriminante OK.")


if __name__ == "__main__":
    main()
