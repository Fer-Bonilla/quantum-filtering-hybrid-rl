"""Figuras de las compuertas G2 y G3 (campaña v8, Tier 3).

Genera en outputs/figures/v8/:
  - fig_g2_topologias.png : cand_hit por topología y selector (D/C/R).
  - fig_g2_dispersion.png : dispersión RMS vs k por topología (régimen balístico).
  - fig_g3_soft.png       : Sharpe de la integración suave + EXP-7 vs R.

Uso::

    uv run python scripts/v8_g2g3_figures.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.paths import TABLES_DIR

FIG = Path("outputs/figures/v8")
TOPOS = ["dreg_aff", "dreg_uni", "cycle", "bipartite"]
NICE = {"dreg_aff": "3-regular\n(afinidad)", "dreg_uni": "3-regular\n(uniforme)",
        "cycle": "ciclo C₈", "bipartite": "bipartito"}
COL = {"D": "#d62728", "C": "#2ca02c", "R": "#ff7f0e"}


def _load(name):
    with (TABLES_DIR / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fig_topologias() -> None:
    rows = _load("regular_topologies_v8.csv")
    by = defaultdict(list)
    for r in rows:
        by[(r["topology"], r["model"])].append(float(r["candidate_hit_rate"]))

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    x = np.arange(len(TOPOS))
    w = 0.25
    for i, mdl in enumerate(["D", "C", "R"]):
        vals = [np.mean(by[(t, mdl)]) for t in TOPOS]
        err = [1.96 * np.std(by[(t, mdl)], ddof=1) / np.sqrt(len(by[(t, mdl)]))
               for t in TOPOS]
        ax.bar(x + (i - 1) * w, vals, w, yerr=err, capsize=3,
               label={"D": "D (DTQW)", "C": "C (clásica)", "R": "R (azar)"}[mdl],
               color=COL[mdl], alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[t] for t in TOPOS])
    ax.set_ylabel("candidate_hit_rate (media, n=10, IC95%)")
    ax.set_title("EXP-5 — topologías de regularidad controlada (M=8):\n"
                 "la DTQW NO supera al azar en ninguna; en ciclo y bipartito queda por debajo",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.30, 0.56)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_g2_topologias.png", dpi=150)
    plt.close(fig)
    print(f"[OK] {FIG}/fig_g2_topologias.png")


def fig_dispersion() -> None:
    # Valores medidos por analyze_v8_g2g3.dispersion_exponent (RMS medio k=1..6)
    series = {
        "bfs_irregular": ([1.00, 1.19, 1.44, 1.45, 1.47, 1.52], "#7f7f7f", "BFS irregular (base)"),
        "dreg_uni": ([1.00, 1.60, 1.88, 1.76, 1.66, 1.51], "#5b7fa6", "3-regular uniforme"),
        "cycle": ([1.00, 2.00, 3.00, 4.00, 3.00, 2.00], "#d62728", "ciclo C₈"),
        "bipartite": ([1.00, 1.73, 1.00, 0.00, 1.00, 1.73], "#9467bd", "bipartito"),
    }
    ks = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for _, (vals, color, label) in series.items():
        ax.plot(ks, vals, "-o", color=color, label=label, markersize=6)
    ax.plot(ks, ks, ":", color="black", alpha=0.6, label="balístico (RMS = k)")
    ax.plot(ks, np.sqrt(ks), "--", color="black", alpha=0.4,
            label="difusivo (RMS = √k)")
    ax.set_xlabel("pasos de caminata k")
    ax.set_ylabel("dispersión RMS de distancia al seed")
    ax.set_title("EXP-5 — verificación del régimen: en el ciclo la DTQW ES balística\n"
                 "(RMS=k hasta la mitad del anillo) y aun así no selecciona mejor que el azar",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_g2_dispersion.png", dpi=150)
    plt.close(fig)
    print(f"[OK] {FIG}/fig_g2_dispersion.png")


def fig_soft() -> None:
    soft = _load("soft_integration_v8.csv")
    hard_d = [float(r["sharpe_ratio"]) for r in _load("campaign_1_v3.csv")
              if r["model"] == "D"]
    inf = _load("informed_walkers_v8.csv")
    r_v6 = [float(r["sharpe_ratio"]) for r in _load("variant_R_v6.csv")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    modos = ["softD", "softC", "softR"]
    colores = ["#d62728", "#2ca02c", "#ff7f0e"]
    for i, (m, c) in enumerate(zip(modos, colores)):
        vals = [float(r["sharpe_ratio"]) for r in soft if r["model"] == m]
        ax1.bar(i, np.mean(vals), color=c, alpha=0.85, edgecolor="black",
                linewidth=0.5)
        ax1.scatter([i] * len(vals), vals, color="black", s=12, alpha=0.6,
                    zorder=3)
    ax1.axhline(np.mean(hard_d), color="#444444", ls="--", lw=1.2,
                label=f"D máscara dura ({np.mean(hard_d):+.3f})")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(["soft-D", "soft-C", "soft-R"])
    ax1.set_ylabel("Sharpe (por paso)")
    ax1.set_title("EXP-6 — integración suave (β*=0,5):\nsoftD ≈ softC > softR; "
                  "ninguna supera a la máscara dura", fontsize=10)
    ax1.legend(fontsize=8.5)
    ax1.grid(axis="y", alpha=0.3)

    walkers = ["softmax", "momentum_refresh", "thompson"]
    for i, wk in enumerate(walkers):
        vals = [float(r["candidate_hit_rate"]) for r in inf if r["model"] == wk]
        ax2.bar(i, np.mean(vals), color="#5b7fa6", alpha=0.85,
                edgecolor="black", linewidth=0.5)
        ax2.scatter([i] * len(vals), vals, color="black", s=12, alpha=0.6,
                    zorder=3)
    r_cand = np.mean([float(r["candidate_hit_rate"])
                      for r in _load("variant_R_v6.csv")])
    ax2.axhline(r_cand, color="#ff7f0e", ls="--", lw=1.4,
                label=f"R azar ({r_cand:.3f})")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["softmax(τ*)", "momentum\nrefresh", "thompson"])
    ax2.set_ylabel("candidate_hit_rate")
    ax2.set_title("EXP-7 — información+rotación:\nningún selector supera a R (FDR)",
                  fontsize=10)
    ax2.legend(fontsize=8.5)
    ax2.grid(axis="y", alpha=0.3)
    del r_v6
    fig.tight_layout()
    fig.savefig(FIG / "fig_g3_soft.png", dpi=150)
    plt.close(fig)
    print(f"[OK] {FIG}/fig_g3_soft.png")


if __name__ == "__main__":
    fig_topologias()
    fig_dispersion()
    fig_soft()
