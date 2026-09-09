"""Análisis de la compuerta G1 (v8, Tier 1 + Tier 2).

Veredictos confirmatorios según el pre-registro (commit 0e3ee5e):
  - H-v8.1: TOST R-D en candidate_hit, margen ±0,019, n=40
            (10 pares v6 + 30 pares nuevos de tost_extension_v8).
  - H-v8.2: TOST D-S(p*) en candidate_hit, margen ±0,019, n=10,
            con control de manipulación |rot(D)-rot(S(p*))| < 0,03.
  - H-v8.3a/b: reportados por run_mask_metrics_v8.py (se releen aquí).

Familia Bonferroni k=6: umbral alfa_c = 0,05/6 = 0,00833.

Uso::

    uv run python scripts/analyze_v8_g1.py
"""

from __future__ import annotations

import csv

import numpy as np
from scipy import stats

from src.utils.paths import TABLES_DIR

MARGIN = 0.019
K_BONF = 6
ALPHA_C = 0.05 / K_BONF
B_BOOT = 5000


def _load(name: str) -> list[dict]:
    with (TABLES_DIR / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _tost(diffs: np.ndarray, margin: float) -> dict:
    n = len(diffs)
    mean, sd = float(diffs.mean()), float(diffs.std(ddof=1))
    se = sd / np.sqrt(n)
    t_lower = (mean + margin) / se     # H0: d <= -margen
    t_upper = (mean - margin) / se     # H0: d >= +margen
    p_lower = float(1 - stats.t.cdf(t_lower, n - 1))
    p_upper = float(stats.t.cdf(t_upper, n - 1))
    p_tost = max(p_lower, p_upper)
    ci90 = stats.t.interval(0.90, n - 1, loc=mean, scale=se)
    return {"n": n, "mean": mean, "sd": sd, "p_tost": p_tost,
            "ci90_lo": float(ci90[0]), "ci90_hi": float(ci90[1]),
            "equivalente": p_tost < ALPHA_C and ci90[0] > -margin and ci90[1] < margin}


def _pairs(map_a: dict, map_b: dict) -> np.ndarray:
    common = sorted(set(map_a) & set(map_b))
    return np.array([map_a[s] - map_b[s] for s in common], dtype=float)


def main() -> None:
    print("=" * 70)
    print("COMPUERTA G1 — veredictos confirmatorios v8 (alfa_c = 0,00833)")
    print("=" * 70)

    # ---------- H-v8.1: TOST R-D ampliado ----------
    r_map, d_map = {}, {}
    for r in _load("variant_R_v6.csv"):
        r_map[int(r["seed"])] = float(r["candidate_hit_rate"])
    for r in _load("campaign_1_v3.csv"):
        if r["model"] == "D":
            d_map[int(r["seed"])] = float(r["candidate_hit_rate"])
    n_old = len(set(r_map) & set(d_map))
    try:
        for r in _load("tost_extension_v8.csv"):
            if r["model"] == "R":
                r_map[int(r["seed"])] = float(r["candidate_hit_rate"])
            elif r["model"] == "D":
                d_map[int(r["seed"])] = float(r["candidate_hit_rate"])
    except FileNotFoundError:
        print("[AVISO] tost_extension_v8.csv no existe aun.")
    diffs = _pairs(r_map, d_map)
    res = _tost(diffs, MARGIN)
    print(f"\nH-v8.1  TOST R-D candidate_hit (margen +/-{MARGIN})")
    print(f"  n={res['n']} ({n_old} previos + {res['n']-n_old} nuevos)  "
          f"diff R-D={res['mean']:+.5f} (sd {res['sd']:.5f})")
    print(f"  IC90 [{res['ci90_lo']:+.5f}, {res['ci90_hi']:+.5f}]  "
          f"p_TOST={res['p_tost']:.5f}")
    v1 = "EQUIVALENTES" if res["equivalente"] else "NO concluye equivalencia"
    if not res["equivalente"] and abs(res["mean"]) - 0 > 0:
        t = res["mean"] / (res["sd"] / np.sqrt(res["n"]))
        p_dir = float(2 * (1 - stats.t.cdf(abs(t), res["n"] - 1)))
        v1 += f" (diferencia direccional p2={p_dir:.4f}, favorece a "
        v1 += "R)" if res["mean"] > 0 else "D)"
    print(f"  VEREDICTO: {v1}")

    # Replicas secundarias: topm y sharpe
    for metric in ["topm_hit_rate", "sharpe_ratio"]:
        rm, dm = {}, {}
        for r in _load("variant_R_v6.csv"):
            rm[int(r["seed"])] = float(r[metric])
        for r in _load("campaign_1_v3.csv"):
            if r["model"] == "D":
                dm[int(r["seed"])] = float(r[metric])
        try:
            for r in _load("tost_extension_v8.csv"):
                if r["model"] == "R":
                    rm[int(r["seed"])] = float(r[metric])
                elif r["model"] == "D":
                    dm[int(r["seed"])] = float(r[metric])
        except FileNotFoundError:
            pass
        d2 = _pairs(rm, dm)
        marg = 0.010 if metric == "topm_hit_rate" else 0.020
        r2 = _tost(d2, marg)
        print(f"  [replica] {metric}: diff={r2['mean']:+.5f} "
              f"p_TOST(+/-{marg})={r2['p_tost']:.5f} n={r2['n']}")

    # ---------- H-v8.2: TOST D-S(p*) ----------
    print(f"\nH-v8.2  TOST D-S(p*) candidate_hit (margen +/-{MARGIN})")
    try:
        s_rows = _load("sticky_calibrated_v8.csv")
        s_map = {int(r["seed"]): float(r["candidate_hit_rate"]) for r in s_rows}
        p_star = float(s_rows[0]["p_star"])
        d_v3 = {int(r["seed"]): float(r["candidate_hit_rate"])
                for r in _load("campaign_1_v3.csv") if r["model"] == "D"}
        try:  # ampliación post-G1 declarada: brazos D del EXP-3 (n=40)
            for r in _load("tost_extension_v8.csv"):
                if r["model"] == "D":
                    d_v3[int(r["seed"])] = float(r["candidate_hit_rate"])
        except FileNotFoundError:
            pass
        diffs2 = _pairs(d_v3, s_map)
        res2 = _tost(diffs2, MARGIN)
        print(f"  p*={p_star:.4f}  n={res2['n']}  diff D-S(p*)={res2['mean']:+.5f}")
        print(f"  IC90 [{res2['ci90_lo']:+.5f}, {res2['ci90_hi']:+.5f}]  "
              f"p_TOST={res2['p_tost']:.5f}")
        print(f"  VEREDICTO: "
              f"{'EQUIVALENTES (la rotacion calibrada reproduce el efecto)' if res2['equivalente'] else 'NO concluye'}")
    except FileNotFoundError:
        print("  [AVISO] sticky_calibrated_v8.csv no existe aun.")

    # Control de manipulacion: rotacion de S(p*) en entrenamiento vs D replay
    try:
        mm = _load("mask_metrics_v8.csv")
        rot_d = np.mean([float(r["rotation_realized"]) for r in mm
                         if r["selector"] == "D"])
        print(f"\n  Control de manipulacion (pendiente de replay de S(p*)): "
              f"rot(D)={rot_d:.4f}; verificar |rot(D)-rot(S(p*))|<0,03 con el "
              f"replay del EXP-1 sobre S(p*).")
    except FileNotFoundError:
        pass

    # ---------- H-v8.3: releer del EXP-1 ----------
    print("\nH-v8.3  (ver salida de run_mask_metrics_v8.py; se resume aqui)")
    try:
        mm = _load("mask_metrics_v8.csv")
        for metric, tag in [("precision_at_m", "a"), ("ndcg_at_m", "b")]:
            dmap = {r["seed"]: float(r[metric]) for r in mm if r["selector"] == "D"}
            rmap = {r["seed"]: float(r[metric]) for r in mm if r["selector"] == "R"}
            dd = _pairs(dmap, rmap)
            dd = dd[np.isfinite(dd)]
            rng = np.random.default_rng(20260705)
            boots = np.array([rng.choice(dd, len(dd), replace=True).mean()
                              for _ in range(B_BOOT)])
            p = float((boots <= 0).mean())
            print(f"  H-v8.3{tag} D-R {metric}: d={dd.mean():+.5f} "
                  f"IC95[{np.percentile(boots, 2.5):+.5f},"
                  f"{np.percentile(boots, 97.5):+.5f}] p={p:.4f} "
                  f"pBonf6={min(1, p*K_BONF):.4f} -> "
                  f"{'RECHAZA H0 (senal no capturada)' if p*K_BONF < 0.05 else 'no significativa'}")
    except FileNotFoundError:
        print("  [AVISO] mask_metrics_v8.csv no existe aun.")


if __name__ == "__main__":
    main()
