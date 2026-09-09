"""Análisis de la máscara-oráculo (v7 — experimento 4: techo del canal selector).

Lee ``oracle_v7.csv`` y contesta: ¿el canal selector puede mover el Sharpe?

1. Bootstrap pareado ORACLE vs A y ORACLE vs R (5000 iter) en sharpe, cand_hit,
   topm y retorno acumulado.
2. "Escalera de Sharpe": sitúa la oracle-máscara entre los agentes RL reales y
   la cota ``oracle_expost`` de v6, con momentum y 1/N de referencia.
3. Figura ``fig_oracle_techo.png`` (barh ordenado, estilo v6).

Veredicto:
  - ORACLE >> A  =>  el canal selector TIENE techo; la política sí explota una
    buena máscara. El cuello de botella de C/D/R/Q/S es la CALIDAD de selección
    (no mejor que el azar), no la arquitectura.
  - ORACLE ~ A   =>  el límite sería la política/recompensa.

Uso::

    uv run python scripts/analyze_oracle_v7.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("outputs/tables")
FIG_DIR = Path("outputs/figures/v7")
N_BOOT = 5000


def _to_float(s: str) -> float:
    try:
        v = float(s)
        return float("nan") if v == -1.0 else v
    except (TypeError, ValueError):
        return float("nan")


def _load(path: Path, model: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {r["seed"]: r for r in csv.DictReader(path.open(encoding="utf-8"))
            if r["model"] == model}


def _paired(lhs: dict, rhs: dict, metric: str) -> tuple[float, float, float, float]:
    seeds = sorted(set(lhs) & set(rhs))
    diffs = np.array([_to_float(lhs[s][metric]) - _to_float(rhs[s][metric]) for s in seeds])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 2:
        return (float("nan"),) * 4
    rng = np.random.default_rng(2026)
    boot = np.array([rng.choice(diffs, diffs.size, replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_pos = float((boot > 0).mean())
    return float(diffs.mean()), float(lo), float(hi), 2.0 * min(p_pos, 1.0 - p_pos)


def _mean(path: Path, model: str, metric: str) -> float:
    rows = _load(path, model)
    vals = np.array([_to_float(r[metric]) for r in rows.values()])
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _bench_sharpe(strategy: str) -> float:
    p = BASE / "benchmarks_v6.csv"
    if not p.exists():
        return float("nan")
    vals = [_to_float(r["sharpe_ratio"]) for r in csv.DictReader(p.open(encoding="utf-8"))
            if r["strategy"] == strategy]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    oracle = _load(BASE / "oracle_v7.csv", "ORACLE")
    if not oracle:
        raise SystemExit("No existe oracle_v7.csv; ejecuta run_oracle_campaign_v7.py primero.")
    a = _load(BASE / "campaign_1_v3.csv", "A")
    r = _load(BASE / "variant_R_v6.csv", "R")

    print(f"ORACLE n={len(oracle)}  ·  A n={len(a)}  ·  R n={len(r)}\n")
    print("=== Bootstrap pareado (5000 iter) ===")
    print(f"{'comparación':<14}{'métrica':<22}{'mean diff':>12}{'IC95%':>24}{'p_two':>8}")
    for label, rhs in (("ORACLE-A", a), ("ORACLE-R", r)):
        for metric in ("sharpe_ratio", "candidate_hit_rate", "topm_hit_rate", "cumulative_return"):
            md, lo, hi, p = _paired(oracle, rhs, metric)
            flag = " *" if p < 0.05 else ""
            print(f"{label:<14}{metric:<22}{md:+12.4f}  [{lo:+.4f},{hi:+.4f}]{p:>8.3f}{flag}")

    # Escalera de Sharpe
    ladder = [
        ("Modelo A (PPO puro)", _mean(BASE / "campaign_1_v3.csv", "A", "sharpe_ratio"), "rl"),
        ("Modelo R (máscara azar)", _mean(BASE / "variant_R_v6.csv", "R", "sharpe_ratio"), "rl"),
        ("Modelo D (DTQW)", _mean(BASE / "campaign_1_v3.csv", "D", "sharpe_ratio"), "rl"),
        ("equal_weight (1/N)", _bench_sharpe("equal_weight"), "bench"),
        ("momentum_20d", _bench_sharpe("momentum_20d"), "bench"),
        ("ORACLE-máscara + PPO", _mean(BASE / "oracle_v7.csv", "ORACLE", "sharpe_ratio"), "oracle"),
        ("oracle_expost (cota)", _bench_sharpe("oracle_expost"), "ceiling"),
    ]
    print("\n=== Escalera de Sharpe (media por step) ===")
    for name, v, _ in ladder:
        print(f"  {name:<26}{v:+.4f}")

    oracle_sharpe = ladder[5][1]
    ceiling = ladder[6][1]
    a_sharpe = ladder[0][1]
    if np.isfinite(ceiling) and ceiling != 0:
        pct = 100.0 * (oracle_sharpe - a_sharpe) / (ceiling - a_sharpe)
        print(f"\nLa oracle-máscara recupera el {pct:.0f}% del trayecto A -> cota "
              f"({a_sharpe:+.3f} -> {ceiling:+.3f}); su Sharpe = {oracle_sharpe:+.3f}.")
    verdict = "TIENE TECHO (la política SÍ explota una buena máscara)" \
        if oracle_sharpe > a_sharpe + 0.2 else "limitado por la política"
    print(f"Veredicto: el canal selector {verdict}.")

    _figure(ladder)
    _save(ladder, oracle, a, r)


def _figure(ladder) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ladder_sorted = sorted(ladder, key=lambda e: e[1])
    names = [e[0] for e in ladder_sorted]
    vals = [e[1] for e in ladder_sorted]
    cmap = {"rl": "#d62728", "bench": "#1f77b4", "oracle": "#ff7f0e", "ceiling": "#2ca02c"}
    colors = [cmap[e[2]] for e in ladder_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    for yi, v in zip(y, vals, strict=False):
        ax.annotate(f"{v:+.3f}", (v, yi), va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5,
                    xytext=(3 if v >= 0 else -3, 0), textcoords="offset points")
    ax.set_xlabel("Sharpe ratio (por step)")
    ax.set_title("Experimento 4 — techo del canal selector: con la máscara perfecta el\n"
                 "PPO alcanza casi la cota oracle_expost => el límite es la SELECCIÓN, "
                 "no la política",
                 fontsize=11.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    # leyenda de colores
    from matplotlib.patches import Patch
    leg = [Patch(facecolor="#d62728", label="agente RL real"),
           Patch(facecolor="#1f77b4", label="benchmark clásico"),
           Patch(facecolor="#ff7f0e", label="oracle-máscara + PPO (este exp.)"),
           Patch(facecolor="#2ca02c", label="cota oracle_expost")]
    ax.legend(handles=leg, loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_oracle_techo.png", dpi=150)
    plt.close(fig)
    print(f"[OK] {FIG_DIR}/fig_oracle_techo.png")


def _save(ladder, oracle, a, r) -> None:
    out = BASE / "oracle_v7_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "sharpe_mean", "kind"])
        for name, v, kind in ladder:
            w.writerow([name, f"{v:.4f}", kind])
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
