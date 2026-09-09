"""Figuras y tablas comparativas de presentación (v6).

Consolida TODOS los selectores (A, B, R, C, D, Q) y los benchmarks
clásicos en un conjunto de figuras y tablas listas para presentar los
experimentos del feedback del director. Produce:

FIGURAS (en outputs/figures/v6/):
  1. fig_selectores_metricas.png   — barras agrupadas A/B/R/C/D/Q en
     las 4 métricas clave (cand_hit, sharpe, convergencia, coverage)
     con barras de error (IC95% bootstrap inter-semilla).
  2. fig_benchmarks_sharpe.png     — Sharpe de benchmarks clásicos +
     oráculo vs agentes RL (barh ordenado), con anotaciones.
  3. fig_forest_pareado.png        — forest plot de las comparaciones
     pareadas (R-C, R-D, Q-C, Q-D, Q-R) en candidate_hit_rate con
     IC95% y línea en 0.
  4. fig_gradiente_seleccion.png   — el hallazgo central: cand_hit vs
     "informatividad" del selector (R>D>C>Q) con línea de tendencia.
  5. fig_convergencia.png          — episodes_to_convergence por
     selector (R/A rápidos, C/D/Q lentos).
  6. fig_annealing_fidelidad.png   — compresión del gap y tasa de
     fallo del annealing vs M (doble eje).

TABLAS (en outputs/tables/v6/):
  - tabla_selectores.md / .tex     — resumen por selector.
  - tabla_benchmarks.md / .tex     — benchmarks vs RL.
  - tabla_pareado.md / .tex        — bootstrap pareado completo.

Uso::

    uv run python scripts/v6_presentation_figures.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TABLES = Path("outputs/tables")
FIG_DIR = Path("outputs/figures/v6")
TAB_DIR = Path("outputs/tables/v6")

# Orden por "informatividad" del selector (hallazgo central del Exp. 6)
SELECTOR_ORDER = ["A", "B", "R", "D", "C", "Q"]
SELECTOR_LABEL = {
    "A": "A\n(PPO puro)",
    "B": "B\n(relacional)",
    "R": "R\n(máscara azar)",
    "C": "C\n(caminata clás.)",
    "D": "D\n(DTQW)",
    "Q": "Q\n(QUBO/anneal)",
}
SELECTOR_COLOR = {
    "A": "#7f7f7f",  # gris
    "B": "#9ecae1",  # azul claro
    "R": "#ff7f0e",  # naranja (control clave)
    "C": "#2ca02c",  # verde
    "D": "#d62728",  # rojo (cuántico DTQW)
    "Q": "#9467bd",  # morado (annealing)
}

METRIC_LABEL = {
    "candidate_hit_rate": "candidate_hit_rate\n(métrica primaria)",
    "sharpe_ratio": "sharpe_ratio",
    "episodes_to_convergence": "episodes_to_convergence\n(menor = mejor)",
    "asset_coverage": "asset_coverage",
}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(s: str) -> float:
    try:
        v = float(s)
        return float("nan") if v == -1.0 else v
    except (TypeError, ValueError):
        return float("nan")


def load_selectors() -> dict[str, list[dict[str, str]]]:
    """Devuelve filas por selector A/B/C/D (de campaign_1_v3) + R + Q."""
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in _read(TABLES / "campaign_1_v3.csv"):
        if r["model"] in {"A", "B", "C", "D"}:
            out[r["model"]].append(r)
    for r in _read(TABLES / "variant_R_v6.csv"):
        out["R"].append(r)
    for r in _read(TABLES / "variant_Q_v6.csv"):
        out["Q"].append(r)
    return out


def _stats(rows: list[dict[str, str]], metric: str) -> tuple[float, float, float]:
    """mean, ci_lo, ci_hi por bootstrap inter-semilla (5000 iter)."""
    vals = np.array([_f(r[metric]) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    rng = np.random.default_rng(2026)
    boot = np.array([rng.choice(vals, vals.size, replace=True).mean() for _ in range(5000)])
    return float(vals.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------------------
# Figura 1: barras agrupadas por selector en 4 métricas
# ---------------------------------------------------------------------------


def fig_selectores_metricas(sel: dict[str, list[dict[str, str]]]) -> None:
    metrics = ["candidate_hit_rate", "sharpe_ratio", "episodes_to_convergence", "asset_coverage"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    models = [m for m in SELECTOR_ORDER if m in sel]
    for ax, metric in zip(axes.flat, metrics, strict=False):
        means, los, his, colors = [], [], [], []
        for m in models:
            mean, lo, hi = _stats(sel[m], metric)
            means.append(mean)
            los.append(mean - lo)
            his.append(hi - mean)
            colors.append(SELECTOR_COLOR[m])
        x = np.arange(len(models))
        ax.bar(x, means, yerr=[los, his], capsize=4, color=colors, alpha=0.85,
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([SELECTOR_LABEL[m] for m in models], fontsize=8)
        ax.set_title(METRIC_LABEL.get(metric, metric), fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        # Anotar valores
        for xi, mv in zip(x, means, strict=False):
            ax.annotate(f"{mv:.3f}" if abs(mv) < 10 else f"{mv:.1f}",
                        (xi, mv), ha="center",
                        va="bottom" if mv >= 0 else "top", fontsize=7)
    fig.suptitle("Comparación de selectores en las métricas clave (n=10 semillas, IC95% bootstrap)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR / "fig_selectores_metricas.png", dpi=150)
    plt.close(fig)
    print("[OK] fig_selectores_metricas.png")


# ---------------------------------------------------------------------------
# Figura 2: benchmarks vs RL en Sharpe (barh ordenado)
# ---------------------------------------------------------------------------


def fig_benchmarks_sharpe(sel: dict[str, list[dict[str, str]]]) -> None:
    bench = _read(TABLES / "benchmarks_v6.csv")
    by_strat: dict[str, list[float]] = defaultdict(list)
    for r in bench:
        by_strat[r["strategy"]].append(_f(r["sharpe_ratio"]))
    entries: list[tuple[str, float, str]] = []
    for strat, vals in by_strat.items():
        v = np.array([x for x in vals if np.isfinite(x)])
        entries.append((strat, float(v.mean()), "benchmark"))
    for m in ["A", "B", "R", "C", "D", "Q"]:
        if m in sel:
            mean, _, _ = _stats(sel[m], "sharpe_ratio")
            entries.append((f"Modelo {m}", mean, "rl"))
    entries.sort(key=lambda e: e[1])

    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    colors = ["#1f77b4" if e[2] == "benchmark" else "#d62728" for e in entries]
    # oráculo destacado
    for i, e in enumerate(entries):
        if e[0] == "oracle_expost":
            colors[i] = "#2ca02c"

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    for yi, v in zip(y, values, strict=False):
        ax.annotate(f"{v:+.3f}", (v, yi), va="center",
                    ha="left" if v >= 0 else "right", fontsize=8,
                    xytext=(3 if v >= 0 else -3, 0), textcoords="offset points")
    ax.set_xlabel("Sharpe ratio (por step)")
    ax.set_title("Benchmarks clásicos y solución óptima vs agentes RL\n"
                 "(verde: oráculo ex-post; azul: benchmark clásico; rojo: agente RL)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_benchmarks_sharpe.png", dpi=150)
    plt.close(fig)
    print("[OK] fig_benchmarks_sharpe.png")


# ---------------------------------------------------------------------------
# Figura 3: forest plot de comparaciones pareadas (cand_hit)
# ---------------------------------------------------------------------------


def fig_forest_pareado() -> None:
    rows = _read(TABLES / "paired_variants_v6.csv")
    # También las comparaciones D-C y R-A desde otros CSVs para contexto
    pares = [
        ("R-C", "candidate_hit_rate"),
        ("R-D", "candidate_hit_rate"),
        ("Q-C", "candidate_hit_rate"),
        ("Q-D", "candidate_hit_rate"),
        ("Q-R", "candidate_hit_rate"),
    ]
    idx = {(r["comparison"], r["metric"]): r for r in rows}
    # Añadir D-C de campaign_1_v3 (referencia)
    dc = {r["metric"]: r for r in _read(TABLES / "campaign_1_v3_paired_c_vs_d.csv")}

    labels, means, los, his, sig = [], [], [], [], []
    # D-C primero como referencia
    if "candidate_hit_rate" in dc:
        r = dc["candidate_hit_rate"]
        labels.append("D − C  (referencia DTQW)")
        means.append(float(r["mean_diff_d_minus_c"]))
        los.append(float(r["mean_diff_d_minus_c"]) - float(r["ci_lo"]))
        his.append(float(r["ci_hi"]) - float(r["mean_diff_d_minus_c"]))
        sig.append(float(r.get("p_bonferroni", 1)) < 0.05)
    for comp, metric in pares:
        r = idx.get((comp, metric))
        if not r:
            continue
        labels.append(comp.replace("-", " − "))
        md = float(r["mean_diff"])
        means.append(md)
        los.append(md - float(r["ci_lo"]))
        his.append(float(r["ci_hi"]) - md)
        sig.append(float(r["p_bonferroni_k11"]) < 0.05)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(labels))[::-1]
    colors = ["#2ca02c" if s else "#999999" for s in sig]
    ax.errorbar(means, y, xerr=[los, his], fmt="o", capsize=5, color="black",
                ecolor="gray", markersize=7, zorder=3)
    for yi, mv, c in zip(y, means, colors, strict=False):
        ax.plot(mv, yi, "o", color=c, markersize=9, zorder=4)
    ax.axvline(0, color="red", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Diferencia media en candidate_hit_rate (IC95% bootstrap)")
    ax.set_title("Comparaciones pareadas en la métrica primaria\n"
                 "(verde: sobrevive Bonferroni k=11; gris: no significativo)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_forest_pareado.png", dpi=150)
    plt.close(fig)
    print("[OK] fig_forest_pareado.png")


# ---------------------------------------------------------------------------
# Figura 4: gradiente de selección (el hallazgo central)
# ---------------------------------------------------------------------------


def fig_gradiente_seleccion(sel: dict[str, list[dict[str, str]]]) -> None:
    # Ordenar selectores con módulo de selección por cand_hit descendente
    sels = ["R", "D", "C", "Q"]
    info_label = ["R\nazar\n(máx. rotación)", "D\nDTQW", "C\ncaminata clás.", "Q\nQUBO óptimo\n(mín. rotación)"]
    means, los, his = [], [], []
    for m in sels:
        mean, lo, hi = _stats(sel[m], "candidate_hit_rate")
        means.append(mean)
        los.append(mean - lo)
        his.append(hi - mean)

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(sels))
    colors = [SELECTOR_COLOR[m] for m in sels]
    ax.errorbar(x, means, yerr=[los, his], fmt="none", ecolor="gray", capsize=5, zorder=2)
    ax.plot(x, means, "-", color="#333333", linewidth=1.5, alpha=0.5, zorder=1)
    for xi, mv, c in zip(x, means, colors, strict=False):
        ax.plot(xi, mv, "o", color=c, markersize=16, markeredgecolor="black", zorder=3)
        ax.annotate(f"{mv:.3f}", (xi, mv), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(info_label, fontsize=9)
    ax.set_ylabel("candidate_hit_rate (n=10)")
    ax.set_xlabel("← más rotación / menos información    ·    menos rotación / más información →")
    ax.set_title("Hallazgo central: el gradiente de alineación es INVERSO a la\n"
                 "información del selector — el mecanismo es la rotación de la máscara, no lo cuántico",
                 fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gradiente_seleccion.png", dpi=150)
    plt.close(fig)
    print("[OK] fig_gradiente_seleccion.png")


# ---------------------------------------------------------------------------
# Figura 5: convergencia por selector
# ---------------------------------------------------------------------------


def fig_convergencia(sel: dict[str, list[dict[str, str]]]) -> None:
    models = [m for m in SELECTOR_ORDER if m in sel]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = []
    for m in models:
        vals = np.array([_f(r["episodes_to_convergence"]) for r in sel[m]])
        data.append(vals[np.isfinite(vals)])
    bp = ax.boxplot(data, tick_labels=[SELECTOR_LABEL[m] for m in models],
                    patch_artist=True, showmeans=True)
    for patch, m in zip(bp["boxes"], models, strict=False):
        patch.set_facecolor(SELECTOR_COLOR[m])
        patch.set_alpha(0.7)
    ax.set_ylabel("episodes_to_convergence (rollouts)")
    ax.set_title("Eficiencia de convergencia: las caminatas informadas (C, D, Q)\n"
                 "RALENTIZAN el aprendizaje frente a A y a la máscara aleatoria R",
                 fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_convergencia.png", dpi=150)
    plt.close(fig)
    print("[OK] fig_convergencia.png")


# ---------------------------------------------------------------------------
# Figura 6: fidelidad del annealing
# ---------------------------------------------------------------------------


def fig_annealing_fidelidad() -> None:
    rows = _read(TABLES / "annealing_fidelity_v6.csv")
    M = [int(r["M"]) for r in rows]
    mismatch = [float(r["set_mismatch_rate"]) * 100 for r in rows]
    gap = [float(r["gap_compression_median"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    color1 = "#d62728"
    ax1.bar([m - 0.0 for m in M], mismatch, width=0.9, color=color1, alpha=0.4,
            edgecolor="black", label="Tasa de fallo del conjunto (%)")
    ax1.set_xlabel("Tamaño del subgrafo M")
    ax1.set_ylabel("Tasa de fallo del conjunto (%)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 110)

    ax2 = ax1.twinx()
    color2 = "#1f77b4"
    ax2.plot(M, gap, "o-", color=color2, linewidth=2, markersize=9,
             label="Compresión del gap (×)")
    ax2.set_ylabel("Compresión del gap del sector factible (×, log)", color=color2)
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor=color2)
    for m, g in zip(M, gap, strict=False):
        ax2.annotate(f"{g:.0f}×", (m, g), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, color=color2)

    ax1.set_title("Por qué el Quantum Annealing de tiempo finito no resuelve el QUBO:\n"
                  "la penalización de cardinalidad comprime el gap ~(M−m)²",
                  fontsize=11.5, fontweight="bold")
    ax1.set_xticks(M)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_annealing_fidelidad.png", dpi=150)
    plt.close(fig)
    print("[OK] fig_annealing_fidelidad.png")


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------


def _emit_table(headers: list[str], rows: list[list[str]], stem: str, caption: str) -> None:
    # Markdown
    md = ["| " + " | ".join(headers) + " |",
          "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    (TAB_DIR / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    # LaTeX
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


def tabla_selectores(sel: dict[str, list[dict[str, str]]]) -> None:
    headers = ["Selector", "Sharpe", "cand_hit", "topm_hit", "coverage", "converg.", "lat.(ms)"]
    rows = []
    names = {"A": "A (PPO puro)", "B": "B (relacional)", "R": "R (máscara azar)",
             "C": "C (caminata clás.)", "D": "D (DTQW)", "Q": "Q (QUBO/anneal)"}
    for m in SELECTOR_ORDER:
        if m not in sel:
            continue
        def ms(metric: str, fmt: str = "{:+.3f}") -> str:
            mean, lo, hi = _stats(sel[m], metric)
            return fmt.format(mean)
        rows.append([
            names[m],
            ms("sharpe_ratio"),
            ms("candidate_hit_rate", "{:.3f}"),
            ms("topm_hit_rate", "{:.3f}"),
            ms("asset_coverage", "{:.3f}"),
            ms("episodes_to_convergence", "{:.1f}"),
            ms("mean_latency_ms", "{:.2f}"),
        ])
    _emit_table(headers, rows, "tabla_selectores",
                "Resumen por selector (n=10 semillas, media). Modelos A-D de "
                "campaign\\_1\\_v3; R y Q de las campañas de variante v6.")


def tabla_benchmarks() -> None:
    bench = _read(TABLES / "benchmarks_v6.csv")
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    fam: dict[str, str] = {}
    for r in bench:
        s = r["strategy"]
        fam[s] = r["family"]
        for k in ["sharpe_ratio", "topm_hit_rate", "asset_coverage", "cumulative_return"]:
            by[s][k].append(_f(r[k]))
    order = ["oracle_expost", "momentum_20d", "equal_weight", "markowitz",
             "buyhold_best", "random"]
    headers = ["Estrategia", "Familia", "Sharpe", "topm_hit", "coverage"]
    rows = []
    for s in order:
        if s not in by:
            continue
        def m(k: str, fmt: str = "{:.3f}") -> str:
            v = np.array([x for x in by[s][k] if np.isfinite(x)])
            return fmt.format(v.mean()) if v.size else "—"
        rows.append([s, fam[s].replace("_", " "), m("sharpe_ratio", "{:+.3f}"),
                     m("topm_hit_rate"), m("asset_coverage")])
    _emit_table(headers, rows, "tabla_benchmarks",
                "Benchmarks clásicos y solución óptima (oracle). Sharpe por step "
                "comparable con los agentes RL.")


def tabla_pareado() -> None:
    rows_in = _read(TABLES / "paired_variants_v6.csv")
    headers = ["Comparación", "Métrica", "mean diff", "IC95%", "P(>0)", "p_Bonf", "Sig."]
    key_metrics = ["candidate_hit_rate", "sharpe_ratio", "topm_hit_rate",
                   "episodes_to_convergence", "asset_coverage"]
    rows = []
    for r in rows_in:
        if r["metric"] not in key_metrics:
            continue
        pb = float(r["p_bonferroni_k11"])
        rows.append([
            r["comparison"], r["metric"],
            f"{float(r['mean_diff']):+.3f}",
            f"[{float(r['ci_lo']):+.3f}, {float(r['ci_hi']):+.3f}]",
            f"{float(r['p_positive']):.3f}",
            f"{pb:.3f}",
            "Sí" if pb < 0.05 else "—",
        ])
    _emit_table(headers, rows, "tabla_pareado",
                "Bootstrap pareado de las variantes v6 (R, Q) contra C, D y entre sí "
                "(5000 iter, Bonferroni k=11).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    sel = load_selectors()
    print(f"Selectores cargados: {[(k, len(v)) for k, v in sorted(sel.items())]}")

    fig_selectores_metricas(sel)
    fig_benchmarks_sharpe(sel)
    fig_forest_pareado()
    fig_gradiente_seleccion(sel)
    fig_convergencia(sel)
    fig_annealing_fidelidad()

    tabla_selectores(sel)
    tabla_benchmarks()
    tabla_pareado()

    print(f"\n[OK] Figuras en {FIG_DIR}/  ·  Tablas en {TAB_DIR}/")


if __name__ == "__main__":
    main()
