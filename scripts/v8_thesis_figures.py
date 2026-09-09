"""Figuras v8 en formato memoria (PDF vectorial, coma decimal, español).

Generador único y trazable de las cinco figuras v8 incluidas por LaTeX:
  - fig_v8_tost.pdf                 : contrastes de equivalencia con IC90.
  - fig_v8_metricas.pdf             : precision@m / NDCG@m / MI (3 paneles).
  - fig_v8_topologias.pdf           : barras D/C/R por topología (EXP-5).
  - fig_v8_dispersion.pdf           : RMS medida vs k (dispersion_rms_v8.csv).
  - fig_v8_dispersion_seleccion.pdf : panel (a)+(b) del documento editado.

Todas las series provienen de tablas de resultados medidos en
``outputs/tables/`` (ninguna serie está codificada a mano). La salida es
determinista (SOURCE_DATE_EPOCH fijo), de modo que una regeneración limpia
reproduce byte a byte los PDF incluidos; el manifiesto SHA-256 se escribe en
``outputs/figures/memoria/manifest_v8_sha256.txt``.

Uso::

    uv run python scripts/v8_thesis_figures.py            # regenera + manifiesto
    uv run python scripts/v8_thesis_figures.py --verify   # compara sin escribir
"""

from __future__ import annotations

import csv
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1754265600")  # 2026-08-04 UTC

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy import stats

from src.utils.paths import TABLES_DIR

FIG = Path("outputs/figures/memoria")
MARGIN = 0.019

plt.rcParams.update({
    "font.size": 9.5, "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "figure.constrained_layout.use": True,
})
coma3 = FuncFormatter(lambda v, _: f"{v:.3f}".replace(".", ","))


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
    # H-v8.2 como replica independiente: solo las 30 semillas nuevas
    # (enmienda declarada; las 10 originales quedan como resultado nominal).
    seeds_old = {7, 42, 99, 123, 314, 456, 789, 1024, 1729, 65535}
    d_rep = {s: v for s, v in d_map.items() if s not in seeds_old}
    s_rep = {s: v for s, v in s_map.items() if s not in seeds_old}

    # H-v8.1 tambien se reporta como replica independiente (30 semillas
    # nuevas, disjuntas de los 10 pares previos que informaron el diseno);
    # el agregado n=40 se muestra como descriptivo.
    r_rep = {s: v for s, v in r_map.items() if s not in seeds_old}
    d_rep_r = {s: v for s, v in d_map.items() if s not in seeds_old}

    entradas = [
        ("H-v8.1$^{\\dagger}$  R$-$D  réplica indep. (n=30)",
         _ci90(_pairs(r_rep, d_rep_r)), MARGIN),
        ("H-v8.1  R$-$D  agregado (n=40, descriptivo)",
         _ci90(_pairs(r_map, d_map)), MARGIN),
        ("H-v8.2$^{\\dagger}$  D$-$S(p*)  réplica indep. (n=30)",
         _ci90(_pairs(d_rep, s_rep)), MARGIN),
    ]
    for metric, marg, lab in [
            ("topm_hit_rate", 0.010, "réplica  R$-$D topm (n=30)"),
            ("sharpe_ratio", 0.020, "réplica  R$-$D Sharpe (n=30)")]:
        rm, dm = {}, {}
        for r in _load("tost_extension_v8.csv"):
            (rm if r["model"] == "R" else dm)[int(r["seed"])] = float(r[metric])
        entradas.append((lab, _ci90(_pairs(rm, dm)), marg))

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ys = np.arange(len(entradas))[::-1]
    for y, (nombre, (mean, lo, hi, n), marg) in zip(ys, entradas):
        ax.fill_betweenx([y - 0.32, y + 0.32], -marg, marg,
                         color="#c04040", alpha=0.07)
        ax.plot([-marg, -marg], [y - 0.32, y + 0.32], color="#c04040", lw=1.1)
        ax.plot([marg, marg], [y - 0.32, y + 0.32], color="#c04040", lw=1.1)
        dentro = lo > -marg and hi < marg
        color = "#2ca02c" if dentro else "#9a9a9a"
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=color, ecolor=color, elinewidth=1.9, capsize=4,
                    markersize=7.5)
        ax.annotate(f"{mean:+.4f}".replace(".", ","), (mean, y),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8)
    ax.axvline(0, color="#444444", lw=0.8, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in entradas], fontsize=9)
    ax.set_xlabel("Diferencia media pareada (IC 90 %); banda = margen de equivalencia")
    ax.set_title("Campaña v8: contrastes de equivalencia\n"
                 "(IC 90 % dentro del margen $\\Rightarrow$ equivalencia)")
    ax.annotate("$^{\\dagger}$confirmatorio: réplica en 30 semillas nuevas "
                "(agregado n=40 descriptivo)", xy=(0.01, 0.02),
                xycoords="axes fraction", fontsize=7.5, color="#555555")
    ax.grid(axis="x", alpha=0.3)
    ax.xaxis.set_major_formatter(coma3)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_v8_tost.pdf")
    plt.close(fig)
    print(f"[OK] {FIG}/fig_v8_tost.pdf")


def fig_metricas() -> None:
    mm = _load("mask_metrics_v8.csv")
    mi = {r["selector"]: r for r in _load("mask_information_v8.csv")}
    orden = ["R", "D", "C", "Q"]
    color = {"R": "#ff7f0e", "D": "#d62728", "C": "#2ca02c", "Q": "#9467bd"}

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for ax, metric, titulo in [
            (axes[0], "precision_at_m", "precision@m (ex-post)"),
            (axes[1], "ndcg_at_m", "NDCG@m (ranking)")]:
        for i, s in enumerate(orden):
            vals = [float(r[metric]) for r in mm if r["selector"] == s
                    and r[metric] not in ("", "nan")]
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                ax.annotate("sin ranking\nintrínseco", (i, 0.005), ha="center",
                            fontsize=7, va="bottom")
                continue
            ax.bar(i, np.mean(vals), color=color[s], alpha=0.78,
                   edgecolor="black", linewidth=0.5)
            ax.scatter([i] * len(vals), vals, color="black", s=8, zorder=3,
                       alpha=0.55)
        ax.set_xticks(range(len(orden)))
        ax.set_xticklabels(orden)
        ax.set_title(titulo, fontsize=9.5)
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(coma3)
    ax = axes[2]
    for i, s in enumerate(orden):
        r = mi.get(s)
        if r is None:
            continue
        v, null = float(r["mi_nats"]) * 1e4, float(r["mi_null_mean"]) * 1e4
        ax.bar(i, v, color=color[s], alpha=0.78, edgecolor="black",
               linewidth=0.5)
        ax.plot([i - 0.4, i + 0.4], [null, null], color="black", ls="--",
                lw=1.1)
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels(orden)
    ax.set_title("Inf. mutua ($\\times 10^{-4}$ nats)\nlínea = null de permutación",
                 fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(coma3)
    fig.savefig(FIG / "fig_v8_metricas.pdf")
    plt.close(fig)
    print(f"[OK] {FIG}/fig_v8_metricas.pdf")


def fig_topologias() -> None:
    """EXP-5: cand_hit por topología y selector (formato memoria)."""
    from collections import defaultdict
    rows = _load("regular_topologies_v8.csv")
    by = defaultdict(list)
    for r in rows:
        by[(r["topology"], r["model"])].append(float(r["candidate_hit_rate"]))
    topos = ["dreg_aff", "dreg_uni", "cycle", "bipartite"]
    nice = {"dreg_aff": "3-regular\n(afinidad)", "dreg_uni": "3-regular\n(uniforme)",
            "cycle": "ciclo $C_8$", "bipartite": "bipartito"}
    col = {"D": "#d62728", "C": "#2ca02c", "R": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    x = np.arange(len(topos))
    w = 0.25
    for i, mdl in enumerate(["D", "C", "R"]):
        vals = [np.mean(by[(t, mdl)]) for t in topos]
        err = [1.96 * np.std(by[(t, mdl)], ddof=1) / np.sqrt(len(by[(t, mdl)]))
               for t in topos]
        ax.bar(x + (i - 1) * w, vals, w, yerr=err, capsize=3,
               label={"D": "D (DTQW)", "C": "C (clásica)", "R": "R (azar)"}[mdl],
               color=col[mdl], alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([nice[t] for t in topos])
    ax.set_ylabel("candidate_hit_rate (media, n=10, IC 95 %)")
    ax.set_title("EXP-5: topologías de regularidad controlada (M=8)")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.30, 0.56)
    ax.yaxis.set_major_formatter(coma3)
    fig.savefig(FIG / "fig_v8_topologias.pdf")
    plt.close(fig)
    print(f"[OK] {FIG}/fig_v8_topologias.pdf")


_DISP_STYLE = {
    "bfs_irregular": ("BFS irregular (base)", "#7f7f7f", "-o"),
    "dreg_uni": ("3-regular uniforme", "#5b7fa6", "-s"),
    "cycle": ("ciclo $C_8$", "#d62728", "-D"),
    "bipartite": ("bipartito", "#9467bd", "-^"),
}


def _load_dispersion():
    """Series RMS medidas (analyze_v8_g2g3.py::dispersion_exponent)."""
    series = {}
    for r in _load("dispersion_rms_v8.csv"):
        topo = r["topology"]
        if topo not in _DISP_STYLE:
            continue
        label, color, marker = _DISP_STYLE[topo]
        series.setdefault(label, ([], color, marker))[0].append(
            (int(r["k"]), float(r["rms_mean"])))
    return {lab: ([v for _, v in sorted(pts)], c, m)
            for lab, (pts, c, m) in series.items()}


def fig_dispersion() -> None:
    """EXP-5: verificación del régimen balístico (formato memoria)."""
    series = _load_dispersion()
    ks = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for label, (vals, color, marker) in series.items():
        ax.plot(ks, vals, marker, color=color, label=label, markersize=5.5)
    ax.plot(ks, ks, ":", color="black", alpha=0.65, label="balístico (RMS $=k$)")
    ax.plot(ks, np.sqrt(ks), "--", color="black", alpha=0.4,
            label="difusivo (RMS $=\\sqrt{k}$)")
    ax.set_xlabel("pasos de caminata $k$")
    ax.set_ylabel("dispersión RMS de distancia al nodo semilla")
    ax.set_title("EXP-5: la DTQW alcanza el régimen balístico en el ciclo\n"
                 "(RMS $=k$ hasta la mitad del anillo) y reavivamientos en el bipartito")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.savefig(FIG / "fig_v8_dispersion.pdf")
    plt.close(fig)
    print(f"[OK] {FIG}/fig_v8_dispersion.pdf")


def fig_dispersion_seleccion() -> None:
    """Panel combinado (Figura 5.11 del documento editado): verificación del
    régimen dispersivo + disociación con la calidad de selección."""
    from collections import defaultdict
    series = _load_dispersion()
    ks = np.arange(1, 7)
    rows = _load("regular_topologies_v8.csv")
    by = defaultdict(list)
    for r in rows:
        by[(r["topology"], r["model"])].append(float(r["candidate_hit_rate"]))
    topos = ["dreg_aff", "dreg_uni", "cycle", "bipartite"]
    nice = {"dreg_aff": "3-reg.\n(afin.)", "dreg_uni": "3-reg.\n(unif.)",
            "cycle": "ciclo\n$C_8$", "bipartite": "bipar-\ntito"}
    col = {"D": "#d62728", "R": "#ff7f0e"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    for label, (vals, color, marker) in series.items():
        ax1.plot(ks, vals, marker, color=color, label=label, markersize=5)
    ax1.plot(ks, ks, ":", color="black", alpha=0.65, label="balístico (RMS $=k$)")
    ax1.plot(ks, np.sqrt(ks), "--", color="black", alpha=0.4,
             label="difusivo (RMS $=\\sqrt{k}$)")
    ax1.set_xlabel("pasos de caminata $k$")
    ax1.set_ylabel("dispersión RMS de distancia al seed")
    ax1.set_title("(a) Verificación del régimen dispersivo", fontsize=10)
    ax1.legend(fontsize=7.2, loc="upper left")
    ax1.grid(alpha=0.3)

    x = np.arange(len(topos))
    w = 0.36
    for i, mdl in enumerate(["D", "R"]):
        vals = [np.mean(by[(t, mdl)]) for t in topos]
        err = [1.96 * np.std(by[(t, mdl)], ddof=1) / np.sqrt(len(by[(t, mdl)]))
               for t in topos]
        ax2.bar(x + (i - 0.5) * w, vals, w, yerr=err, capsize=3,
                label={"D": "D (DTQW)", "R": "R (azar)"}[mdl],
                color=col[mdl], alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([nice[t] for t in topos], fontsize=8.5)
    ax2.set_ylabel("candidate_hit_rate")
    ax2.set_title("(b) Calidad de selección: D no supera a R", fontsize=10)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0.30, 0.52)
    ax2.yaxis.set_major_formatter(coma3)
    fig.savefig(FIG / "fig_v8_dispersion_seleccion.pdf")
    plt.close(fig)
    print(f"[OK] {FIG}/fig_v8_dispersion_seleccion.pdf")


FIGS = ["fig_v8_tost.pdf", "fig_v8_metricas.pdf", "fig_v8_topologias.pdf",
        "fig_v8_dispersion.pdf", "fig_v8_dispersion_seleccion.pdf"]
MANIFEST = FIG / "manifest_v8_sha256.txt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest() -> None:
    lines = [f"{_sha256(FIG / n)}  {n}" for n in FIGS]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] manifiesto: {MANIFEST}")


def _verify() -> int:
    """Compara los SHA-256 actuales contra el manifiesto. 0 = OK."""
    if not MANIFEST.exists():
        print("[FAIL] no existe el manifiesto; regenere las figuras.")
        return 1
    esperado = dict(line.split(None, 1)[::-1] for line in
                    MANIFEST.read_text(encoding="utf-8").splitlines() if line)
    fallos = 0
    for n in FIGS:
        actual = _sha256(FIG / n) if (FIG / n).exists() else "AUSENTE"
        ok = esperado.get(n) == actual
        print(f"  {'OK  ' if ok else 'FAIL'} {n}")
        fallos += 0 if ok else 1
    return 1 if fallos else 0


if __name__ == "__main__":
    if "--verify" in sys.argv:
        raise SystemExit(_verify())
    fig_tost()
    fig_metricas()
    fig_topologias()
    fig_dispersion()
    fig_dispersion_seleccion()
    _write_manifest()
