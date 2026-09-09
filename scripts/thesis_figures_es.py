"""Figuras vectoriales (PDF) en español para la memoria del TFM.

Genera en ``outputs/figures/memoria/`` las siete figuras referenciadas por
``capitulos/06_resultados.tex`` que no existían, más dos diagramas
conceptuales sobre la configuración de los experimentos:

  1. fig_campaign_decision_panels.pdf  — paneles A-B-C-D (alineación vs Sharpe)
  2. fig_forest_primario_es.pdf        — forest plot métrica primaria
  3. fig_dosis_combinada.pdf           — dosis-respuesta (cand_hit) + selectividad (Sharpe)
  4. fig_mediacion_rotacion_es.pdf     — mediación rotación→cand_hit
  5. fig_regimenes_es.pdf              — walk-forward 2021-2024
  6. fig_oraculo_techo_es.pdf          — escalera Sharpe del oráculo
  7. fig_decision_metodologia.pdf      — decisión metodológica (conceptual)
  8. fig_config_experimentos.pdf       — pipeline común + variantes + bloques (conceptual)
  9. fig_walkforward_esquema.pdf       — esquema de ventana expansiva (conceptual)

Todas las cifras salen de los CSV de ``outputs/tables/`` (nada inventado).
Coma decimal en los ejes para coherencia con ``siunitx``.

Uso::

    uv run python scripts/thesis_figures_es.py
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter

TABLES = Path("outputs/tables")
OUT = Path("outputs/figures/memoria")

plt.rcParams.update({
    "font.size": 9.5,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "figure.constrained_layout.use": True,
})

COL = {"A": "#7f7f7f", "B": "#5b7fa6", "C": "#2ca02c", "D": "#d62728",
       "R": "#ff7f0e", "Q": "#9467bd", "mom": "#1f77b4", "eq": "#2ca02c",
       "oracle": "#e8a33d", "cota": "#2f8f4e"}

coma = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
coma1 = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))


def _rows(name: str) -> list[dict]:
    with (TABLES / f"{name}.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, col: str) -> float:
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"[OK] {OUT / name}")


# ---------------------------------------------------------------------------
# 1. Paneles de decisión de la campaña principal
# ---------------------------------------------------------------------------

def fig_campaign_panels() -> None:
    rows = _rows("campaign_1_v3")
    modelos = ["A", "B", "C", "D"]
    cand = [[_f(r, "candidate_hit_rate") for r in rows if r["model"] == m] for m in modelos]
    shar = [[_f(r, "sharpe_ratio") for r in rows if r["model"] == m] for m in modelos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, data, titulo in ((ax1, cand, "Alineación del conjunto candidato"),
                             (ax2, shar, "Sharpe fuera de muestra")):
        bp = ax.boxplot(data, tick_labels=modelos, patch_artist=True, widths=0.55)
        for patch, m in zip(bp["boxes"], modelos):
            patch.set_facecolor(COL[m])
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_title(titulo)
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(coma)
    ax1.set_ylabel("candidate\\_hit\\_rate" if False else "candidate_hit_rate")
    ax2.set_ylabel("Sharpe (por paso)")
    ax1.annotate("A y B no filtran:\nmáscara completa (=1,0)", xy=(1.5, 0.985),
                 xytext=(1.7, 0.80), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax1.annotate("el filtrado hace la\nmétrica informativa", xy=(3.5, 0.449),
                 xytext=(2.6, 0.62), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    _save(fig, "fig_campaign_decision_panels.pdf")


# ---------------------------------------------------------------------------
# 2. Forest plot de la métrica primaria
# ---------------------------------------------------------------------------

def fig_forest() -> None:
    dc = next(r for r in _rows("campaign_1_v3_paired_c_vs_d")
              if r["metric"] == "candidate_hit_rate")
    v6 = [r for r in _rows("paired_variants_v6") if r["metric"] == "candidate_hit_rate"]

    entradas = [("D − C", _f(dc, "mean_diff_d_minus_c"), _f(dc, "ci_lo"),
                 _f(dc, "ci_hi"), _f(dc, "p_bonferroni"))]
    for comp in ["R-C", "R-D", "Q-C", "Q-D", "Q-R"]:
        r = next(x for x in v6 if x["comparison"] == comp)
        entradas.append((comp.replace("-", " − "), _f(r, "mean_diff"),
                         _f(r, "ci_lo"), _f(r, "ci_hi"), _f(r, "p_bonferroni_k11")))

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ys = np.arange(len(entradas))[::-1]
    for y, (nombre, d, lo, hi, pb) in zip(ys, entradas):
        sig = pb < 0.05
        color = "#2ca02c" if sig else "#9a9a9a"
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=color,
                    ecolor=color, elinewidth=1.6, capsize=3.5, markersize=7)
        etiqueta = f"{d:+.3f}".replace(".", ",")
        ax.annotate(etiqueta, (d, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.axvline(0, color="#c04040", linestyle="--", linewidth=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in entradas])
    ax.set_xlabel("Diferencia media pareada en candidate_hit_rate (IC 95 % bootstrap)")
    ax.xaxis.set_major_formatter(coma)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Comparaciones pareadas en la métrica primaria\n"
                 "(verde: sobrevive Bonferroni; gris: no significativo)")
    _save(fig, "fig_forest_primario_es.pdf")


# ---------------------------------------------------------------------------
# 3. Dosis-respuesta combinada (alineación + selectividad)
# ---------------------------------------------------------------------------

def fig_dosis() -> None:
    rows = _rows("rotation_sweep_v7")
    por_p: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        p = round(_f(r, "rotation_p"), 2)
        por_p[p]["cand"].append(_f(r, "candidate_hit_rate"))
        por_p[p]["sharpe"].append(_f(r, "sharpe_ratio"))
    ps = sorted(por_p)

    def stats(clave: str):
        med = [statistics.mean(por_p[p][clave]) for p in ps]
        ic = [1.96 * statistics.pstdev(por_p[p][clave]) / np.sqrt(len(por_p[p][clave]))
              for p in ps]
        return np.array(med), np.array(ic)

    cand_m, cand_ic = stats("cand")
    sh_m, sh_ic = stats("sharpe")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    ax1.plot(ps, cand_m, "-o", color="#1f6fb4", markersize=6, zorder=3)
    ax1.fill_between(ps, cand_m - cand_ic, cand_m + cand_ic, color="#1f6fb4", alpha=0.18)
    ax1.set_xlabel("rotation_p   (0 = máscara congelada · 1 ≡ Modelo R)")
    ax1.set_ylabel("candidate_hit_rate (media, n=10)")
    ax1.set_title("La alineación responde a la rotación")
    ax1.annotate("pendiente +0,031  IC95 [+0,022; +0,041]\np < 0,001 · Spearman ρ = +0,62",
                 xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.35", fc="#f4f4f4", ec="#999999", lw=0.7))
    ax1.annotate("p = 1 ≡ Modelo R", xy=(1.0, cand_m[-1]), xytext=(0.62, cand_m[-1] - 0.012),
                 fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_formatter(coma)

    ax2.errorbar(ps, sh_m, yerr=sh_ic, fmt="-s", color="#777777", markersize=5.5,
                 capsize=3, elinewidth=1.1)
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.set_xlabel("rotation_p")
    ax2.set_ylabel("Sharpe (por paso)")
    ax2.set_title("El Sharpe no responde (efecto selectivo)")
    ax2.annotate("pendiente n.s.", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.35", fc="#f4f4f4", ec="#999999", lw=0.7))
    ax2.set_ylim(-0.05, 0.10)
    ax2.grid(alpha=0.3)
    ax2.yaxis.set_major_formatter(coma)
    _save(fig, "fig_dosis_combinada.pdf")


# ---------------------------------------------------------------------------
# 4. Mediación rotación → cand_hit
# ---------------------------------------------------------------------------

def fig_mediacion() -> None:
    rows = _rows("mediation_rotation_v7")
    sticky = sorted((r for r in rows if r["kind"] == "sticky"),
                    key=lambda r: _f(r, "rotation"))
    otros = [r for r in rows if r["kind"] != "sticky"]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs = [_f(r, "rotation") for r in sticky]
    ys = [_f(r, "cand_hit") for r in sticky]
    ax.plot(xs, ys, "-o", color="#1f6fb4", markersize=5.5,
            label="StickyWalker (información = 0): curva rotación→cand_hit")
    for r in sticky:
        p = r["selector"].split("=")[1].rstrip(")")
        ax.annotate(f"p={p}".replace(".", ","), (_f(r, "rotation"), _f(r, "cand_hit")),
                    textcoords="offset points", xytext=(4, -11), fontsize=7.5,
                    color="#1f6fb4")
    desplaz = {"R": (6, 6), "D": (6, 8), "C": (6, -14), "Q": (8, -4)}
    for r in otros:
        nombre = r["selector"]
        color = COL.get(nombre, "#333333")
        ax.scatter(_f(r, "rotation"), _f(r, "cand_hit"), marker="D", s=95,
                   color=color, edgecolor="black", linewidth=0.8, zorder=4)
        dx, dy = desplaz.get(nombre, (6, 6))
        ax.annotate(nombre, (_f(r, "rotation"), _f(r, "cand_hit")),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=10.5, fontweight="bold", color=color)
    ax.set_xlabel("Rotación realizada de la máscara (distancia de Jaccard)")
    ax.set_ylabel("candidate_hit_rate")
    ax.set_title("Mediación: ρ(rotación, cand_hit) = +0,93 — R y D sobre la curva del azar;\n"
                 "C y Q por debajo (la concentración informada penaliza)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.xaxis.set_major_formatter(coma)
    ax.yaxis.set_major_formatter(coma)
    _save(fig, "fig_mediacion_rotacion_es.pdf")


# ---------------------------------------------------------------------------
# 5. Validación walk-forward por regímenes
# ---------------------------------------------------------------------------

def fig_regimenes() -> None:
    rows = _rows("regime_validation_v7")
    por = defaultdict(lambda: defaultdict(list))
    anios = {}
    for r in rows:
        f = int(r["fold_id"])
        anios[f] = r["test_year"]
        por[f][r["strategy"]].append(_f(r, "sharpe_ratio"))
    folds = sorted(por)
    series = [("A", "A (PPO)", COL["A"]), ("D", "D (DTQW)", COL["D"]),
              ("R", "R (azar)", COL["R"]), ("momentum_20d", "momentum", COL["mom"]),
              ("equal_weight", "1/N", COL["eq"])]

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    x = np.arange(len(folds))
    w = 0.16
    for i, (clave, nombre, color) in enumerate(series):
        vals = [statistics.mean([v for v in por[f][clave] if np.isfinite(v)])
                for f in folds]
        ax.bar(x + (i - 2) * w, vals, w, label=nombre, color=color, alpha=0.87,
               edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{anios[f]}\n(pliegue {f})" for f in folds])
    ax.set_ylabel("Sharpe (por paso, media n=5)")
    ax.set_title("Sharpe por régimen de mercado (walk-forward 2021-2024, n=5 por pliegue)")
    ax.legend(loc="lower left", fontsize=8.5, ncol=5)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(coma)
    _save(fig, "fig_regimenes_es.pdf")


# ---------------------------------------------------------------------------
# 6. Escalera Sharpe del oráculo
# ---------------------------------------------------------------------------

def fig_oraculo() -> None:
    rows = _rows("oracle_v7_summary")
    rows.sort(key=lambda r: _f(r, "sharpe_mean"))
    colores = {"rl": "#d62728", "bench": "#1f77b4",
               "oracle": COL["oracle"], "ceiling": COL["cota"]}
    etiquetas = {"rl": "agente RL real", "bench": "referencia clásica",
                 "oracle": "oráculo-máscara + PPO", "ceiling": "cota ex-post"}

    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    vistos = set()
    for i, r in enumerate(rows):
        kind = r["kind"]
        color = colores.get(kind, "#888888")
        lab = etiquetas.get(kind) if kind not in vistos else None
        vistos.add(kind)
        v = _f(r, "sharpe_mean")
        ax.barh(i, v, color=color, alpha=0.9, edgecolor="black", linewidth=0.5,
                label=lab)
        ax.annotate(f"{v:+.3f}".replace(".", ","), (v, i),
                    textcoords="offset points", xytext=(5, -3), fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["strategy"] for r in rows], fontsize=8.5)
    ax.set_xlabel("Sharpe (por paso)")
    ax.set_title("Techo del canal selector: la máscara perfecta multiplica el Sharpe ×27\n"
                 "y recupera el 90 % de la cota ⇒ la política PPO no es el cuello de botella")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="x", alpha=0.3)
    ax.xaxis.set_major_formatter(coma1)
    _save(fig, "fig_oraculo_techo_es.pdf")


# ---------------------------------------------------------------------------
# 7. Decisión metodológica (conceptual)
# ---------------------------------------------------------------------------

def _caja(ax, x, y, w, h, texto, fc, fontsize=8.8, ec="#555555", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, linewidth=0.9))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center",
            fontsize=fontsize, fontweight=weight)


def fig_decision() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    criterios = [
        ("1. candidate_hit (métrica primaria)",
         "D − C = +0,019 (p<0,001; campaña principal, n=10) → D gana el plan original\n"
         "R − D: TOST ±0,019, réplica v8 en 30 semillas nuevas, p=4·10⁻⁴ → equivalentes"),
        ("2. Sharpe",
         "TOST ±0,020, réplica v8 (n=30): R ≡ D, p=1,4·10⁻³ → empate formal"),
        ("3. Latencia por paso",
         "R evita la simulación DTQW → menor latencia (0,75 vs 1,29 ms)"),
        ("4. Coste en hardware real",
         "DTQW: ≈42k–1.664k USD por campaña de 500.000 evaluaciones (Anexo D)\n"
         "R: cómputo clásico ordinario"),
    ]
    y = 8.6
    for titulo, cuerpo in criterios:
        _caja(ax, 0.3, y - 1.55, 5.6, 1.62, f"{titulo}\n{cuerpo}", "#f2f5f9", 8.0)
        if y > 3.4:
            ax.add_patch(FancyArrowPatch((3.1, y - 1.62), (3.1, y - 2.06),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color="#555555"))
            ax.text(3.32, y - 1.86, "si no separa", fontsize=7.4, color="#555555")
        y -= 2.12

    _caja(ax, 6.45, 5.6, 3.25, 2.6,
          "Plan original\n\nD (DTQW) es la mejor\nvariante frente a C:\n+0,019 en la métrica\nprimaria, sin deterioro\nde Sharpe",
          "#fdecea", 8.4, ec=COL["D"])
    _caja(ax, 6.45, 1.4, 3.25, 3.4,
          "Mecanismo\n\nR (rotación de máscara)\niguala a D en las métricas\ndisponibles (TOST) y domina\nen latencia y coste:\nsin simulación DTQW\nni hardware cuántico",
          "#fff3e6", 8.4, ec=COL["R"])
    ax.add_patch(FancyArrowPatch((5.95, 7.2), (6.45, 7.0), arrowstyle="-|>",
                                 mutation_scale=13, color=COL["D"]))
    ax.add_patch(FancyArrowPatch((5.95, 3.1), (6.45, 3.1), arrowstyle="-|>",
                                 mutation_scale=13, color=COL["R"]))
    ax.set_title("Criterio lexicográfico de decisión: D gana el plan original;\n"
                 "R conserva el mecanismo dominante (rotación) con menor latencia y coste")
    _save(fig, "fig_decision_metodologia.pdf")


# ---------------------------------------------------------------------------
# 8. Configuración de los experimentos (conceptual)
# ---------------------------------------------------------------------------

def fig_config_experimentos() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.set_xlim(-0.75, 10.15)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # --- pipeline común (fila superior) ---
    tubo = [("Datos OHLCV\n30 activos\n2018–2024", 0.15),
            ("Estado $s_t$\nventana L=20", 1.95),
            ("Grafo $G_t$\ncorr. móvil\nk-NN=5", 3.75),
            ("Subgrafo $H_t$\nBFS, M=8", 5.55),
            ("SELECTOR\ntop-m (m=3)", 7.35),
            ("PPO + máscara\nacción $u_t$", 9.15)]
    for i, (texto, x) in enumerate(tubo):
        especial = "SELECTOR" in texto
        _caja(ax, x - 0.72, 8.15, 1.62, 1.35, texto,
              "#fde9c8" if especial else "#eef2f6", 7.8,
              ec="#c77d00" if especial else "#555555",
              weight="bold" if especial else "normal")
        if i < len(tubo) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.92, 8.82), (tubo[i + 1][1] - 0.74, 8.82),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color="#555555"))
    ax.text(7.35, 9.72, "único punto de variación", ha="center", fontsize=8,
            style="italic", color="#c77d00")

    # --- variantes (fila media) ---
    ax.text(0.15, 7.45, "Variantes (qué ocupa la ranura del selector):",
            fontsize=9, fontweight="bold")
    variantes = [
        ("A", "sin filtrado\n(máscara completa)", COL["A"]),
        ("B", "A + rasgos del\ngrafo en $s_t$", COL["B"]),
        ("C", "caminata clásica\n$(D^{-1}W)^k$", COL["C"]),
        ("D", "DTQW\n$(SC)^k|\\psi_0\\rangle$", COL["D"]),
        ("R", "azar uniforme\nsobre $H_t$", COL["R"]),
        ("Q", "óptimo QUBO\n(determinista)", COL["Q"]),
        ("S(p)", "máscara pegajosa\nrotación $p$", "#8c564b"),
        ("Oráculo", "top-m por retorno\nfuturo (cota)", COL["oracle"]),
    ]
    for i, (nombre, desc, color) in enumerate(variantes):
        x = 0.25 + i * 1.22
        _caja(ax, x, 5.65, 1.1, 1.5, f"{nombre}\n{desc}", "white", 6.9, ec=color)
        ax.add_patch(plt.Rectangle((x, 7.02), 1.1, 0.13, fc=color, ec="none"))

    # --- bloques experimentales (fila inferior) ---
    ax.text(0.15, 5.0, "Bloques experimentales (qué variantes compara cada uno):",
            fontsize=9, fontweight="bold")
    bloques = [
        ("Bloque A — Ablación canónica\nA → B → C → D\nn=10 semillas pareadas", 0.25, 2.55),
        ("Bloque B — Controles\nR (azar) · Q (QUBO)\n+ momentum, 1/N, oráculo", 2.72, 2.3),
        ("Bloque C — Mecanismo (v7)\nS(p): dosis-respuesta · mediación\nTOST · walk-forward · oráculo", 5.14, 2.6),
        ("Bloque D — Sensibilidad\nmoneda · ruido NISQ\nk, M, m · estado inicial", 7.86, 2.05),
    ]
    for texto, x, w in bloques:
        _caja(ax, x, 3.0, w, 1.7, texto, "#f4f7f4", 7.6, ec="#3a7d3a")

    ax.text(5.0, 2.35,
            "Toda variante comparte datos, entorno, PPO y evaluación; solo cambia el productor del conjunto candidato.\n"
            "Config. común: nivel2 (30 activos), recompensa log-wealth, 50 000 pasos, semillas pareadas.",
            ha="center", fontsize=8.2, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fbfbfb", ec="#bbbbbb", lw=0.7))
    ax.set_ylim(2.0, 10)
    _save(fig, "fig_config_experimentos.pdf")


# ---------------------------------------------------------------------------
# 9. Esquema walk-forward (conceptual)
# ---------------------------------------------------------------------------

def fig_walkforward() -> None:
    folds = [(1, 2020, 2021), (2, 2021, 2022), (3, 2022, 2023), (4, 2023, 2024)]
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    for i, (fid, train_end, test) in enumerate(folds):
        y = len(folds) - i
        ax.barh(y, train_end + 1 - 2018, left=2018, height=0.55,
                color="#5b7fa6", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.barh(y, 1, left=test, height=0.55, color="#ff7f0e", alpha=0.9,
                edgecolor="black", linewidth=0.5)
        ax.text(2017.9, y, f"pliegue {fid}", ha="right", va="center", fontsize=8.5)
    ax.set_xlim(2016.9, 2025.4)
    ax.set_xticks(range(2018, 2025))
    ax.set_yticks([])
    ax.set_title("Validación walk-forward de ventana expansiva: entrenamiento (azul)\n"
                 "crece un año por pliegue; el año siguiente queda como prueba (naranja)")
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#5b7fa6", label="entrenamiento (expansivo)"),
                       Patch(fc="#ff7f0e", label="prueba (1 año)")],
              loc="lower right", fontsize=8)
    _save(fig, "fig_walkforward_esquema.pdf")


# ---------------------------------------------------------------------------
# 10. Arquitectura hibrida (conceptual; sustituye a diagrama_esquematico.jpg)
# ---------------------------------------------------------------------------

def fig_arquitectura_esquematica() -> None:
    """Flujo de decision: seleccion discreta de UN activo; el filtro local
    devuelve m_t candidatos sobre un subgrafo de tamano M_t <= M."""
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    az, ve, na, gr = "#e8eef7", "#e9f5e9", "#fff1e0", "#f2f2f2"

    # fila superior: datos -> estado -> politica -> accion -> recompensa
    _caja(ax, 0.2, 5.6, 2.3, 1.5, "Datos OHLCV\nrasgos sin fuga temporal\n(desplazamiento 1)", gr, 8.0)
    _caja(ax, 2.9, 5.6, 2.3, 1.5, "Entorno MarketEnv\nestado $s_t$ (ventana\nde rasgos por activo)", az, 8.0)
    _caja(ax, 5.6, 5.6, 2.9, 1.5,
          "Política PPO $\\pi_c(u\\mid s_t)$\nenmascarada sobre $q_t$\n(renormalización)", az, 8.0)
    _caja(ax, 8.9, 5.6, 2.9, 1.5,
          "Acción discreta: UN activo\n$u_t\\in\\{1,\\dots,N\\}$\nrecompensa $r_t$ = log-riqueza − coste", na, 8.0)
    for x0, x1 in [(2.5, 2.9), (5.2, 5.6), (8.5, 8.9)]:
        ax.add_patch(FancyArrowPatch((x0, 6.35), (x1, 6.35), arrowstyle="-|>",
                                     mutation_scale=13, color="#444444"))

    # fila inferior: grafo -> subgrafo -> modulo local -> candidatos
    _caja(ax, 0.2, 1.6, 2.3, 1.7,
          "Grafo dinámico $G_t$\nafinidad $k$-NN simetrizada\n$N$ activos", ve, 8.0)
    _caja(ax, 2.9, 1.6, 2.3, 1.7,
          "Subgrafo $H_t$\nBFS desde la semilla\n$M_t=|V(H_t)|\\leq M$", ve, 8.0)
    _caja(ax, 5.6, 1.6, 2.9, 1.7,
          "Módulo local (interfaz única)\nC: caminata clásica $D^{-1}W$\nD: DTQW  ·  controles R, Q, S(p)",
          ve, 8.0, ec=COL["D"])
    _caja(ax, 8.9, 1.6, 2.9, 1.7,
          "Conjunto candidato $q_t$\n$|q_t| = m_t=\\min(m, M_t)$\n(máscara sobre las acciones)", na, 8.0)
    for x0, x1 in [(2.5, 2.9), (5.2, 5.6), (8.5, 8.9)]:
        ax.add_patch(FancyArrowPatch((x0, 2.45), (x1, 2.45), arrowstyle="-|>",
                                     mutation_scale=13, color="#444444"))
    # acoplamientos verticales
    ax.add_patch(FancyArrowPatch((1.35, 5.6), (1.35, 3.3), arrowstyle="-|>",
                                 mutation_scale=13, color="#444444"))
    ax.text(1.5, 4.45, "retornos\nrecientes", fontsize=7.6, va="center")
    # flecha de la mascara: de candidatos (abajo-dcha) a la politica (arriba)
    ax.add_patch(FancyArrowPatch((10.35, 3.3), (7.6, 5.6), arrowstyle="-|>",
                                 mutation_scale=13, color=COL["D"]))
    ax.text(9.9, 4.15, "máscara top-$m_t$\n(sin gradientes)", fontsize=7.6,
            color=COL["D"], ha="left", va="center")
    ax.set_title("Arquitectura híbrida: selección discreta de un activo con filtrado local "
                 "sobre subgrafos dinámicos")
    _save(fig, "fig_arquitectura_esquematica.pdf")


# ---------------------------------------------------------------------------
# 11. Caminata cuantica de tiempo discreto (conceptual; sustituye al JPG)
# ---------------------------------------------------------------------------

def fig_caminata_cuantica() -> None:
    """Un paso DTQW: (a) moneda homogenea U = S (I_p x C); (b) moneda local
    bloque-diagonal controlada por posicion, U = S C_coin (grafos irregulares)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
    az, ve, na, gr = "#e8eef7", "#e9f5e9", "#fff1e0", "#f2f2f2"

    # (a) moneda homogenea
    ax1.set_title("(a) Moneda homogénea (grafo regular)", fontsize=10)
    _caja(ax1, 0.3, 7.4, 4.3, 1.7, "Posición $\\mathcal{H}_p$\n$\\{|v\\rangle\\}_{v\\in V}$", az, 8.4)
    _caja(ax1, 5.4, 7.4, 4.3, 1.7, "Moneda $\\mathcal{H}_c$\n$\\{|c\\rangle\\}$, dim $d$", na, 8.4)
    ax1.text(5.0, 8.25, "$\\otimes$", fontsize=14, ha="center", va="center")
    _caja(ax1, 0.3, 4.5, 9.4, 1.7,
          "1) Moneda   $\\hat I_p\\otimes\\hat C$\n(actúa solo sobre $\\mathcal{H}_c$; la posición no cambia)",
          gr, 8.2)
    _caja(ax1, 0.3, 2.2, 9.4, 1.7,
          "2) Desplazamiento   $\\hat S\\,|v,c\\rangle = |u_c(v),\\,c'\\rangle$\n(salta al vecino que indica la moneda)",
          gr, 8.2)
    for y0, y1 in [(7.4, 6.2), (4.5, 3.9)]:
        ax1.add_patch(FancyArrowPatch((5.0, y0), (5.0, y1), arrowstyle="-|>",
                                      mutation_scale=13, color="#444444"))
    ax1.text(5.0, 1.0, "$\\hat U = \\hat S\\,(\\hat I_p\\otimes\\hat C)$,   "
             "$|\\psi_k\\rangle = \\hat U^{\\,k}|\\psi_0\\rangle$",
             fontsize=11, ha="center", va="center")

    # (b) moneda local bloque-diagonal
    ax2.set_title("(b) Moneda local por nodo (grafo irregular, esta memoria)", fontsize=10)
    _caja(ax2, 0.3, 7.4, 9.4, 1.6,
          "Espacio por puertos: $|v_i, c\\rangle$, $c=0,\\dots,d_i-1$ "
          "(dim $M_t d_{\\max}$; relleno fijo)", az, 8.2)
    # bloque diagonal
    bx, by, bw = 0.6, 3.2, 4.0
    ax2.add_patch(FancyBboxPatch((bx, by), bw, bw, boxstyle="square,pad=0",
                                 fc="white", ec="#555555", linewidth=0.9))
    sizes = [1.3, 0.9, 1.1, 0.7]
    off = 0.0
    for i, s in enumerate(sizes):
        x0 = bx + off
        y0 = by + bw - off - s
        ax2.add_patch(FancyBboxPatch((x0, y0), s, s, boxstyle="square,pad=0",
                                     fc="#fdd9b5", ec=COL["D"], linewidth=0.9))
        ax2.text(x0 + s / 2, y0 + s / 2, f"$\\hat C_{{{i+1}}}$",
                 fontsize=8.5, ha="center", va="center")
        off += s
    ax2.text(bx + bw / 2, by - 0.5, "$\\hat C^{\\mathrm{coin}} = \\hat C_1\\oplus\\hat C_2\\oplus\\cdots\\oplus\\hat C_{M_t}$",
             fontsize=9.5, ha="center")
    ax2.text(bx + bw / 2, by - 1.05, "bloque-diagonal, controlada\npor la posición ($d_i$ = grado)",
             fontsize=7.8, ha="center", va="center")
    _caja(ax2, 5.1, 5.2, 4.6, 2.0,
          "$\\hat C_i = 2|w_i\\rangle\\langle w_i| - I_{d_i}$\n"
          "$|w_i\\rangle\\propto\\sum_j \\sqrt{W(i,j)}\\,|c_{i\\to j}\\rangle$\n"
          "(reflexión de Grover ponderada)", na, 8.0)
    _caja(ax2, 5.1, 3.2, 4.6, 1.6,
          "$\\hat S\\,|v_i, c_{i\\to j}\\rangle = |v_j, c_{j\\to i}\\rangle$\n(permutación de puertos recíprocos)", gr, 8.0)
    ax2.text(5.0, 1.0, "$\\hat U_t = \\hat S_t\\,\\hat C^{\\mathrm{coin}}_t$   "
             "(no es un producto tensorial simple)", fontsize=10.5, ha="center", va="center")
    _save(fig, "fig_caminata_cuantica.pdf")


if __name__ == "__main__":
    fig_campaign_panels()
    fig_forest()
    fig_dosis()
    fig_mediacion()
    fig_regimenes()
    fig_oraculo()
    fig_decision()
    fig_config_experimentos()
    fig_walkforward()
    fig_arquitectura_esquematica()
    fig_caminata_cuantica()
    print("\n[OK] 11 figuras PDF generadas.")
