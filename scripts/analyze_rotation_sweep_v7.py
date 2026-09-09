"""Análisis dosis-respuesta del barrido de rotación (v7 — experimento 1).

Lee ``rotation_sweep_v7.csv`` (producido por ``run_rotation_sweep_v7.py``) y
contrasta la hipótesis causal del mecanismo de v6:

    H_rot: ``candidate_hit_rate`` crece monótonamente con ``rotation_p``.

Estadística (estilo pareado de ``paired_variants_v6.py``, numpy + 5000 boot):

1. **Tabla dosis-respuesta**: media ± sd por nivel de ``rotation_p`` de
   cand_hit, topm, sharpe, convergencia y cobertura.
2. **Pendiente pareada por semilla**: para cada semilla se ajusta la pendiente
   OLS de la métrica frente a ``rotation_p`` (las 6 dosis comparten semilla →
   diseño pareado). Se bootstrapea la media de las pendientes entre semillas →
   IC95 + P(>0) + p de dos colas. Pendiente > 0 con IC que excluye 0 ⇒ efecto
   dosis-respuesta confirmado controlando por semilla.
3. **Spearman ρ(p, cand_hit)** sobre los 60 runs (monotonicidad agregada).
4. **Sonda de rotación realizada**: simula ``StickyWalker`` sobre un subgrafo
   estable y reporta la rotación (Jaccard) por nivel de p, confirmando que la
   perilla controla la rotación con independencia del RL.

Uso::

    uv run python scripts/analyze_rotation_sweep_v7.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.graph.subgraph_selector import Subgraph
from src.quantum.sticky_walker import StickyWalker

BASE = Path("outputs/tables")
SWEEP = BASE / "rotation_sweep_v7.csv"
N_BOOT = 5000
RNG = np.random.default_rng(2026)

# Métricas a tabular; convergencia usa el sentinel -1 → NaN.
LEVEL_METRICS = (
    "candidate_hit_rate", "topm_hit_rate", "sharpe_ratio",
    "asset_coverage", "episodes_to_convergence",
)
# Métricas para el contraste de pendiente dosis-respuesta y su signo esperado.
SLOPE_METRICS = {
    "candidate_hit_rate": "+",   # más rotación → más cobertura de prometedores
    "topm_hit_rate": "+",
    "episodes_to_convergence": "-",  # más rotación → converge antes
    "sharpe_ratio": "0",         # se espera neutro (ns)
}


def _to_float(s: str) -> float:
    try:
        v = float(s)
        return float("nan") if v == -1.0 else v
    except (TypeError, ValueError):
        return float("nan")


def _load() -> list[dict]:
    if not SWEEP.exists():
        raise SystemExit(f"No existe {SWEEP}; ejecuta run_rotation_sweep_v7.py primero.")
    return list(csv.DictReader(SWEEP.open("r", encoding="utf-8")))


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Pendiente OLS de y~x sobre puntos finitos (NaN si <2 puntos o var(x)=0)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    xx, yy = x[mask], y[mask]
    vx = np.var(xx)
    if vx == 0.0:
        return float("nan")
    return float(np.cov(xx, yy, ddof=0)[0, 1] / vx)


def _boot_mean_ci(values: np.ndarray) -> tuple[float, float, float, float, float]:
    """Bootstrap de la media: (mean, ci_lo, ci_hi, P(>0), p_two)."""
    values = values[np.isfinite(values)]
    if values.size < 2:
        return (float("nan"),) * 5
    boot = np.array([
        RNG.choice(values, values.size, replace=True).mean() for _ in range(N_BOOT)
    ])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    p_pos = float((boot > 0).mean())
    p_two = 2.0 * min(p_pos, 1.0 - p_pos)
    return float(values.mean()), float(ci_lo), float(ci_hi), p_pos, p_two


def dose_response_table(rows: list[dict]) -> list[float]:
    """Imprime media±sd por nivel de p. Devuelve los niveles ordenados."""
    by_p: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        p = float(r["rotation_p"])
        for m in LEVEL_METRICS:
            by_p[p][m].append(_to_float(r[m]))
    levels = sorted(by_p)

    print("\n=== Tabla dosis-respuesta (media +/- sd entre semillas) ===")
    hdr = f"{'p':>5} {'n':>3}  " + "  ".join(f"{m.split('_')[0]:>16}" for m in LEVEL_METRICS)
    print(hdr)
    for p in levels:
        cells = []
        n_seeds = 0
        for m in LEVEL_METRICS:
            arr = np.array(by_p[p][m]); arr = arr[np.isfinite(arr)]
            n_seeds = max(n_seeds, arr.size)
            if arr.size:
                cells.append(f"{arr.mean():+8.3f}+/-{arr.std(ddof=1):.3f}")
            else:
                cells.append(f"{'nan':>16}")
        print(f"{p:>5.2f} {n_seeds:>3}  " + "  ".join(f"{c:>16}" for c in cells))
    return levels


def paired_slope_test(rows: list[dict]) -> list[dict]:
    """Pendiente OLS por semilla + bootstrap de su media entre semillas."""
    seeds = sorted({int(r["seed"]) for r in rows})
    out: list[dict] = []
    print("\n=== Pendiente dosis-respuesta pareada por semilla (bootstrap n_seeds) ===")
    print(f"{'metric':<26}{'mean slope':>12}{'IC95%':>24}{'P(>0)':>8}{'p_two':>8}  signo")
    for metric, expected in SLOPE_METRICS.items():
        slopes = []
        for s in seeds:
            srows = [r for r in rows if int(r["seed"]) == s]
            x = np.array([float(r["rotation_p"]) for r in srows])
            y = np.array([_to_float(r[metric]) for r in srows])
            slopes.append(_ols_slope(x, y))
        slopes = np.array(slopes)
        mean, lo, hi, p_pos, p_two = _boot_mean_ci(slopes)
        # consistencia con el signo esperado
        if expected == "+":
            ok = "OK" if (mean > 0 and lo > 0) else ".."
        elif expected == "-":
            ok = "OK" if (mean < 0 and hi < 0) else ".."
        else:
            ok = "OK" if not (lo > 0 or hi < 0) else ".."  # neutro: IC incluye 0
        out.append({
            "metric": metric, "mean_slope": mean, "ci_lo": lo, "ci_hi": hi,
            "p_positive": p_pos, "p_two_sided": p_two,
            "expected_sign": expected, "consistent": ok == "OK",
            "n_seeds": int(np.isfinite(slopes).sum()),
        })
        print(f"{metric:<26}{mean:+12.4f}  [{lo:+.4f},{hi:+.4f}]"
              f"{p_pos:>8.3f}{p_two:>8.3f}  {expected:>2} {ok}")
    return out


def spearman_monotonicity(rows: list[dict]) -> None:
    p = np.array([float(r["rotation_p"]) for r in rows])
    y = np.array([_to_float(r["candidate_hit_rate"]) for r in rows])
    mask = np.isfinite(p) & np.isfinite(y)
    rho, pval = spearmanr(p[mask], y[mask])
    print(f"\n=== Spearman rho(rotation_p, candidate_hit_rate) sobre {int(mask.sum())} runs ===")
    print(f"  rho = {rho:+.4f}   p = {pval:.2e}")


def turnover_probe(levels: list[float]) -> dict[float, float]:
    """Rotación realizada (Jaccard) por nivel de p sobre un subgrafo estable.

    Demuestra que ``rotation_p`` controla la rotación con independencia del RL
    (mismo M=8, m=3 que la campaña). Subgrafo fijo → la rotación procede SOLO
    de la perilla, no de la deriva de H_t.
    """
    M, m, n_steps = 8, 3, 1000
    rng = np.random.default_rng(0)
    W = rng.random((M, M)); W = 0.5 * (W + W.T); np.fill_diagonal(W, 0.0)
    sub = Subgraph(W_local=W, global_node_ids=np.arange(M, dtype=np.int64),
                   seed_idx_local=0)
    realized: dict[float, float] = {}
    print("\n=== Sonda de rotacion realizada (M=8, m=3, subgrafo estable) ===")
    print(f"{'p':>5}{'realized_turnover (Jaccard)':>30}")
    for p in levels:
        w = StickyWalker(seed=0, rotation_p=p)
        for _ in range(n_steps):
            w.candidate_set(sub, k=3, m=m)
        realized[p] = w.mean_turnover
        print(f"{p:>5.2f}{w.mean_turnover:>30.4f}")
    return realized


def main() -> None:
    rows = _load()
    levels = dose_response_table(rows)
    slope_out = paired_slope_test(rows)
    spearman_monotonicity(rows)
    realized = turnover_probe(levels)

    # Veredicto
    cand = next(r for r in slope_out if r["metric"] == "candidate_hit_rate")
    verdict = (
        "CONFIRMADA" if cand["consistent"]
        else "NO confirmada (IC de la pendiente incluye 0)"
    )
    print(f"\n=== Veredicto H_rot (candidate_hit_rate sube con rotation_p): {verdict} ===")

    # Persistir resumen por nivel + pendientes
    out_path = BASE / "rotation_sweep_v7_summary.csv"
    by_p: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        p = float(r["rotation_p"])
        for mtr in LEVEL_METRICS:
            by_p[p][mtr].append(_to_float(r[mtr]))
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        cols = ["rotation_p", "n_seeds", *[f"{m}_mean" for m in LEVEL_METRICS],
                *[f"{m}_sd" for m in LEVEL_METRICS], "realized_turnover"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for p in levels:
            row = {"rotation_p": p, "realized_turnover": realized.get(p, float("nan"))}
            n_seeds = 0
            for mtr in LEVEL_METRICS:
                arr = np.array(by_p[p][mtr]); arr = arr[np.isfinite(arr)]
                n_seeds = max(n_seeds, arr.size)
                row[f"{mtr}_mean"] = arr.mean() if arr.size else float("nan")
                row[f"{mtr}_sd"] = arr.std(ddof=1) if arr.size > 1 else float("nan")
            row["n_seeds"] = n_seeds
            w.writerow(row)
    print(f"\n[OK] resumen -> {out_path}")


if __name__ == "__main__":
    main()
