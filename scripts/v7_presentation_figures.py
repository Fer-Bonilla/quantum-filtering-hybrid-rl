"""Figuras y tabla de presentación del experimento 1 v7 (dosis-respuesta).

Consume ``outputs/tables/rotation_sweep_v7.csv`` (6 niveles de ``rotation_p`` x
10 semillas) y produce material listo para la defensa, en el mismo estilo que
``v6_presentation_figures.py``.

FIGURAS (en outputs/figures/v7/):
  1. fig_dosis_respuesta.png  — la curva central: candidate_hit_rate (media +
     banda IC95% bootstrap inter-semilla) y topm_hit_rate frente a rotation_p,
     con la rotación REALIZADA en eje secundario (valida la perilla). Anota la
     pendiente pareada, su IC95% y Spearman. Marca p=1.0 ≡ Modelo R (v6).
  2. fig_dosis_panel.png      — 2x2: cand_hit y topm (responden, p<0.001) frente
     a sharpe y convergencia (no responden, n.s.). Muestra que el efecto es
     SELECTIVO de las métricas de alineación.

TABLA (en outputs/tables/v7/):
  - tabla_dosis_respuesta.md / .tex — media±sd por nivel + rotación realizada.

Uso::

    uv run python scripts/v7_presentation_figures.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from src.graph.subgraph_selector import Subgraph
from src.quantum.sticky_walker import StickyWalker

TABLES = Path("outputs/tables")
FIG_DIR = Path("outputs/figures/v7")
TAB_DIR = Path("outputs/tables/v7")
SWEEP = TABLES / "rotation_sweep_v7.csv"

C_CAND = "#1f77b4"   # azul — métrica primaria
C_TOPM = "#9467bd"   # morado — métrica secundaria
C_TURN = "#7f7f7f"   # gris — rotación realizada (perilla)
C_SHARPE = "#ff7f0e"  # naranja
C_CONV = "#d62728"   # rojo


def _f(s: str) -> float:
    try:
        v = float(s)
        return float("nan") if v == -1.0 else v
    except (TypeError, ValueError):
        return float("nan")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _by_p(rows: list[dict[str, str]]) -> dict[float, list[dict[str, str]]]:
    out: dict[float, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        out[round(float(r["rotation_p"]), 2)].append(r)
    return out


def _stats(rows: list[dict[str, str]], metric: str) -> tuple[float, float, float, float]:
    """mean, ci_lo, ci_hi (bootstrap inter-semilla 5000), sd."""
    vals = np.array([_f(r[metric]) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (float("nan"),) * 4
    if vals.size == 1:
        return float(vals[0]), float(vals[0]), float(vals[0]), 0.0
    rng = np.random.default_rng(2026)
    boot = np.array([rng.choice(vals, vals.size, replace=True).mean() for _ in range(5000)])
    return (float(vals.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)), float(vals.std(ddof=1)))


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    xx, yy = x[mask], y[mask]
    if np.var(xx) == 0.0:
        return float("nan")
    return float(np.cov(xx, yy, ddof=0)[0, 1] / np.var(xx))


def _slope_boot(rows: list[dict[str, str]], metric: str) -> tuple[float, float, float, float]:
    """Pendiente pareada por semilla + bootstrap: mean, ci_lo, ci_hi, p_two."""
    seeds = sorted({int(r["seed"]) for r in rows})
    slopes = []
    for s in seeds:
        sr = [r for r in rows if int(r["seed"]) == s]
        x = np.array([float(r["rotation_p"]) for r in sr])
        y = np.array([_f(r[metric]) for r in sr])
        slopes.append(_ols_slope(x, y))
    slopes = np.array([s for s in slopes if np.isfinite(s)])
    if slopes.size < 2:
        return (float("nan"),) * 4
    rng = np.random.default_rng(2026)
    boot = np.array([rng.choice(slopes, slopes.size, replace=True).mean() for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_pos = float((boot > 0).mean())
    return float(slopes.mean()), float(lo), float(hi), 2.0 * min(p_pos, 1.0 - p_pos)


def _realized_turnover(levels: list[float]) -> dict[float, float]:
    """Rotación realizada (Jaccard) por nivel sobre un subgrafo estable M=8,m=3."""
    M, m = 8, 3
    rng = np.random.default_rng(0)
    W = rng.random((M, M)); W = 0.5 * (W + W.T); np.fill_diagonal(W, 0.0)
    sub = Subgraph(W_local=W, global_node_ids=np.arange(M, dtype=np.int64), seed_idx_local=0)
    out: dict[float, float] = {}
    for p in levels:
        w = StickyWalker(seed=0, rotation_p=p)
        for _ in range(1000):
            w.candidate_set(sub, k=3, m=m)
        out[p] = w.mean_turnover
    return out


def _pfmt(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


# ---------------------------------------------------------------------------
# Figura 1: curva dosis-respuesta
# ---------------------------------------------------------------------------


def fig_dosis_respuesta(rows: list[dict[str, str]]) -> None:
    byp = _by_p(rows)
    levels = sorted(byp)
    cand = [_stats(byp[p], "candidate_hit_rate") for p in levels]
    topm = [_stats(byp[p], "topm_hit_rate") for p in levels]
    turn = _realized_turnover(levels)

    cm = [c[0] for c in cand]; clo = [c[1] for c in cand]; chi = [c[2] for c in cand]
    tm = [t[0] for t in topm]
    terr = [[t[0] - t[1] for t in topm], [t[2] - t[0] for t in topm]]
    tv = [turn[p] for p in levels]

    sl_m, sl_lo, sl_hi, sl_p = _slope_boot(rows, "candidate_hit_rate")
    pall = np.array([float(r["rotation_p"]) for r in rows])
    yall = np.array([_f(r["candidate_hit_rate"]) for r in rows])
    rho, rho_p = spearmanr(pall, yall)

    fig, ax1 = plt.subplots(figsize=(10, 6.5))
    ax1.fill_between(levels, clo, chi, color=C_CAND, alpha=0.15, zorder=1)
    ax1.plot(levels, cm, "o-", color=C_CAND, linewidth=2.5, markersize=8,
             markeredgecolor="black", label="candidate_hit_rate (media · IC95%)", zorder=3)
    for p, v in zip(levels, cm, strict=False):
        ax1.annotate(f"{v:.3f}", (p, v), xytext=(0, 11), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold", color=C_CAND)
    ax1.errorbar(levels, tm, yerr=terr, fmt="s--", color=C_TOPM, linewidth=1.8,
                 markersize=6, capsize=4, alpha=0.9, label="topm_hit_rate (media · IC95%)", zorder=2)

    # p=1.0 ≡ Modelo R
    ax1.annotate("p = 1.0  ≡  Modelo R (v6)", (1.0, cm[-1]), xytext=(-12, -26),
                 textcoords="offset points", ha="right", fontsize=9,
                 color="#333333", fontstyle="italic",
                 arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))

    ax2 = ax1.twinx()
    ax2.plot(levels, tv, ":", color=C_TURN, linewidth=1.6, marker="^", markersize=6,
             label="rotación realizada (Jaccard)")
    ax2.set_ylabel("rotación realizada — Jaccard (perilla)", color=C_TURN)
    ax2.tick_params(axis="y", labelcolor=C_TURN)
    ax2.set_ylim(0, 0.85)

    ax1.set_xlabel("rotation_p   (0 = máscara congelada  ·  1 = re-sorteo total ≡ Modelo R)")
    ax1.set_ylabel("hit rate (n = 10 semillas)")
    ax1.set_ylim(0.10, 0.50)
    ax1.set_xticks(levels)
    ax1.grid(axis="y", alpha=0.3)

    txt = (f"pendiente cand_hit = {sl_m:+.3f}\n"
           f"IC95% [{sl_lo:+.3f}, {sl_hi:+.3f}],  p {_pfmt(sl_p)}\n"
           f"Spearman ρ = {rho:+.2f}  (p {_pfmt(rho_p)})")
    ax1.text(0.04, 0.96, txt, transform=ax1.transAxes, fontsize=9.5, va="top",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#999999"))

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower right", fontsize=9, framealpha=0.9)

    ax1.set_title(
        "Dosis-respuesta: la ROTACIÓN de la máscara causa la alineación\n"
        "candidate_hit_rate sube de forma monótona y saturante con rotation_p "
        "(información fija en cero)",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_dosis_respuesta.png", dpi=150)
    plt.close(fig)
    print("[OK] fig_dosis_respuesta.png")


# ---------------------------------------------------------------------------
# Figura 2: panel 2x2 — efecto selectivo
# ---------------------------------------------------------------------------


def fig_dosis_panel(rows: list[dict[str, str]]) -> None:
    byp = _by_p(rows)
    levels = sorted(byp)
    specs = [
        ("candidate_hit_rate", "candidate_hit_rate", C_CAND, False),
        ("topm_hit_rate", "topm_hit_rate", C_TOPM, False),
        ("sharpe_ratio", "sharpe_ratio", C_SHARPE, False),
        ("episodes_to_convergence", "episodes_to_convergence", C_CONV, True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, label, color, lower_better) in zip(axes.flat, specs, strict=False):
        st = [_stats(byp[p], metric) for p in levels]
        mean = [s[0] for s in st]
        lo = [s[0] - s[1] for s in st]
        hi = [s[2] - s[0] for s in st]
        ax.errorbar(levels, mean, yerr=[lo, hi], fmt="o-", color=color, linewidth=2,
                    markersize=7, capsize=4, markeredgecolor="black")
        ax.fill_between(levels, [s[1] for s in st], [s[2] for s in st], color=color, alpha=0.12)
        sl_m, sl_lo, sl_hi, sl_p = _slope_boot(rows, metric)
        sig = "p < 0.001  ✓ responde" if sl_p < 0.001 else f"p {_pfmt(sl_p)}  ·  n.s."
        ax.set_title(f"{label}\npendiente {sl_m:+.3f} [{sl_lo:+.3f}, {sl_hi:+.3f}]  —  {sig}",
                     fontsize=10.5, fontweight="bold")
        ax.set_xlabel("rotation_p")
        ax.set_xticks(levels)
        ax.grid(axis="y", alpha=0.3)
        if lower_better:
            ax.set_ylabel("rollouts (menor = mejor)")
    fig.suptitle("El efecto de la rotación es SELECTIVO: mejora la alineación "
                 "(cand_hit, topm),\nno el retorno (Sharpe) ni la convergencia",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "fig_dosis_panel.png", dpi=150)
    plt.close(fig)
    print("[OK] fig_dosis_panel.png")


# ---------------------------------------------------------------------------
# Tabla
# ---------------------------------------------------------------------------


def _emit_table(headers: list[str], rows: list[list[str]], stem: str, caption: str) -> None:
    md = ["| " + " | ".join(headers) + " |",
          "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    (TAB_DIR / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    col = "l" + "r" * (len(headers) - 1)
    tex = [r"\begin{table}[H]", r"\centering", r"\small",
           rf"\caption{{{caption}}}", rf"\label{{tab:{stem}}}",
           rf"\begin{{tabular}}{{{col}}}", r"\toprule",
           " & ".join(h.replace("_", r"\_") for h in headers) + r" \\", r"\midrule"]
    for r in rows:
        tex.append(" & ".join(c.replace("_", r"\_").replace("±", r"$\pm$") for c in r) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / f"{stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"[OK] {stem}.md / .tex")


def tabla_dosis_respuesta(rows: list[dict[str, str]]) -> None:
    byp = _by_p(rows)
    levels = sorted(byp)
    turn = _realized_turnover(levels)
    headers = ["rotation_p", "n", "cand_hit", "topm_hit", "sharpe", "converg.", "rot. real."]
    out = []
    for p in levels:
        def ms(metric: str, fmt: str = "{:.3f}") -> str:
            mean, _, _, sd = _stats(byp[p], metric)
            return f"{fmt.format(mean)}±{fmt.format(sd)}"
        n = sum(np.isfinite(_f(r["candidate_hit_rate"])) for r in byp[p])
        out.append([f"{p:.2f}", str(int(n)), ms("candidate_hit_rate"),
                    ms("topm_hit_rate"), ms("sharpe_ratio"),
                    ms("episodes_to_convergence", "{:.1f}"), f"{turn[p]:.3f}"])
    _emit_table(headers, out, "tabla_dosis_respuesta",
                "Barrido dosis-respuesta de rotación (Modelo S, v7). Media±sd "
                "inter-semilla (n=10) por nivel de rotation\\_p; rot. real. = "
                "rotación realizada (Jaccard) de la perilla.")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    if not SWEEP.exists():
        raise SystemExit(f"No existe {SWEEP}; ejecuta run_rotation_sweep_v7.py primero.")
    rows = _read(SWEEP)
    print(f"Cargados {len(rows)} runs del barrido.")
    fig_dosis_respuesta(rows)
    fig_dosis_panel(rows)
    tabla_dosis_respuesta(rows)
    print(f"\n[OK] Figuras en {FIG_DIR}/  ·  Tabla en {TAB_DIR}/")


if __name__ == "__main__":
    main()
