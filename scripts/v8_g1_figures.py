"""Figuras de la compuerta G1 (campaña v8, Tier 1 + Tier 2).

Genera en outputs/figures/v8/:
  - fig_g1_tost.png      : TOSTs con IC90 frente a los márgenes de equivalencia.
  - fig_g1_metricas.png  : precision@m / NDCG@m / MI por selector.

Uso::

    uv run python scripts/v8_g1_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.utils.paths import TABLES_DIR

FIG = Path("outputs/figures/v8")
MARGIN = 0.019


def _load(name):
    with (TABLES_DIR / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pairs(map_a, map_b):
    common = sorted(set(map_a) & set(map_b))
    return np.array([map_a[s] - map_b[s] for s in common])


def _ci90(d):
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.90, n - 1, loc=d.mean(), scale=se)
    return d.mean(), lo, hi, n


def fig_tost() -> None:
    r_map, d_map = {}, {}
    for r in _load("variant_R_v6.csv"):
        r_map[int(r["seed"])] = float(r["candidate_hit_rate"])
    for r in _load("campaign_1_v3.csv"):
        if r["model"] == "D":
            d_map[int(r["seed"])] = float(r["candidate_hit_rate"])
    for r in _load("tost_extension_v8.csv"):
        (r_map if r["model"] == "R" else d_map)[int(r["seed"])] = \
            float(r["candidate_hit_rate"])
    s_map = {int(r["seed"]): float(r["candidate_hit_rate"])
             for r in _load("sticky_calibrated_v8.csv")}
    d_all = dict(d_map)  # incluye brazos D del EXP-3 (ampliación n=40)

    entradas = [
        ("H-v8.1  R − D  (n=40)", _ci90(_pairs(r_map, d_map)), MARGIN),
        ("H-v8.2  D − S(p*)  (n=40)", _ci90(_pairs(d_all, s_map)), MARGIN),
    ]
    # réplicas
    for metric, marg, lab in [("topm_hit_rate", 0.010, "réplica  R − D topm (n=40)"),
                              ("sharpe_ratio", 0.020, "réplica  R − D Sharpe (n=40)")]:
        rm, dm = {}, {}
        for r in _load("variant_R_v6.csv"):
            rm[int(r["seed"])] = float(r[metric])
        for r in _load("campaign_1_v3.csv"):
            if r["model"] == "D":
                dm[int(r["seed"])] = float(r[metric])
        for r in _load("tost_extension_v8.csv"):
            (rm if r["model"] == "R" else dm)[int(r["seed"])] = float(r[metric])
        entradas.append((lab, _ci90(_pairs(rm, dm)), marg))

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ys = np.arange(len(entradas))[::-1]
    for y, (nombre, (mean, lo, hi, n), marg) in zip(ys, entradas):
        ax.plot([-marg, -marg], [y - 0.3, y + 0.3], color="#c04040", lw=1.2)
        ax.plot([marg, marg], [y - 0.3, y + 0.3], color="#c04040", lw=1.2)
        ax.fill_betweenx([y - 0.3, y + 0.3], -marg, marg, color="#c04040",
                         alpha=0.06)
        dentro = lo > -marg and hi < marg
        color = "#2ca02c" if dentro else "#9a9a9a"
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=color, ecolor=color, elinewidth=2, capsize=4,
                    markersize=8)
        ax.annotate(f"{mean:+.4f}".replace(".", ","), (mean, y),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8)
    ax.axvline(0, color="#444444", lw=0.8, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in entradas], fontsize=9)
    ax.set_xlabel("Diferencia media pareada (IC 90 %); banda roja = margen de equivalencia")
    ax.set_title("Compuerta G1 — TOSTs pre-registrados: IC90 dentro del margen ⇒ equivalencia\n"
                 "(verde: IC90 íntegramente dentro del margen)")
    ax.grid(axis="x", alpha=0.3)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_g1_tost.png", dpi=150)
    plt.close(fig)
    print(f"[OK] {FIG}/fig_g1_tost.png")


def fig_metricas() -> None:
    mm = _load("mask_metrics_v8.csv")
    mi = {r["selector"]: r for r in _load("mask_information_v8.csv")}
    orden = ["R", "D", "C", "Q"]
    color = {"R": "#ff7f0e", "D": "#d62728", "C": "#2ca02c", "Q": "#9467bd"}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, metric, titulo in [
            (axes[0], "precision_at_m", "precision@m (ex-post)"),
            (axes[1], "ndcg_at_m", "NDCG@m (ranking)")]:
        for i, s in enumerate(orden):
            vals = [float(r[metric]) for r in mm if r["selector"] == s
                    and r[metric] not in ("", "nan")]
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                ax.annotate("sin ranking", (i, 0.01), ha="center", fontsize=7.5,
                            rotation=90, va="bottom")
                continue
            ax.bar(i, np.mean(vals), color=color[s], alpha=0.75,
                   edgecolor="black", linewidth=0.5)
            ax.scatter([i] * len(vals), vals, color="black", s=10, zorder=3,
                       alpha=0.6)
        ax.set_xticks(range(len(orden)))
        ax.set_xticklabels(orden)
        ax.set_title(titulo, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    ax = axes[2]
    for i, s in enumerate(orden):
        key = s if s in mi else s.replace(".", "_")
        r = mi.get(key)
        if r is None:
            continue
        v, null = float(r["mi_nats"]) * 1e4, float(r["mi_null_mean"]) * 1e4
        ax.bar(i, v, color=color[s], alpha=0.75, edgecolor="black",
               linewidth=0.5)
        ax.plot([i - 0.4, i + 0.4], [null, null], color="black", ls="--",
                lw=1.2)
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels(orden)
    ax.set_title("Información mutua (×10⁻⁴ nats)\nlínea discontinua = null de permutación",
                 fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Compuerta G1 — métricas desacopladas de la rotación: D no supera a R en ninguna",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig_g1_metricas.png", dpi=150)
    plt.close(fig)
    print(f"[OK] {FIG}/fig_g1_metricas.png")


if __name__ == "__main__":
    fig_tost()
    fig_metricas()
