"""Curvas de ablación del régimen *sweet spot* sobre la métrica de alineación.

Las figuras originales ``sweet_abl_*.png`` (generate_figures.py) trazaban
``cumulative_return`` bajo el régimen de recompensa penalizada (lambda=0,1,
``risk_penalty``), no comparable con el cuerpo. Este generador traza
``candidate_hit_rate`` (media ± desviación entre semillas) desde los mismos
CSV, que es la métrica informativa de ese bloque. Salida en
``outputs/figures/memoria/``:

  - sweet_abl_k_cand.png, sweet_abl_Msize_cand.png,
    sweet_abl_mtop_cand.png, sweet_abl_init_cand.png

Uso::

    uv run python scripts/sweet_ablation_alignment_figures.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.utils.paths import TABLES_DIR

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path("outputs/figures/memoria")
METRIC = "candidate_hit_rate"
ABLACIONES = [("sweet_abl_k", "pasos de caminata $k$"),
              ("sweet_abl_Msize", "tamaño del subgrafo $M$"),
              ("sweet_abl_mtop", "tamaño del conjunto candidato $m$"),
              ("sweet_abl_init", "estado inicial de la caminata")]

plt.rcParams.update({"font.size": 9.5, "font.family": "serif",
                     "figure.constrained_layout.use": True})
coma = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))


def _sort_key(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return float("inf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, etiqueta in ABLACIONES:
        with (TABLES_DIR / f"{name}.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        por: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            try:
                por[r["ablation_value"]].append(float(r[METRIC]))
            except (KeyError, ValueError):
                continue
        vals = sorted(por, key=_sort_key)
        med = np.array([np.mean(por[v]) for v in vals])
        sd = np.array([np.std(por[v], ddof=1) if len(por[v]) > 1 else 0.0 for v in vals])
        n = min(len(por[v]) for v in vals)

        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        x = np.arange(len(vals))
        ax.errorbar(x, med, yerr=sd, fmt="-o", color="#d62728", capsize=4,
                    markersize=6, linewidth=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(vals)
        ax.set_xlabel(etiqueta)
        ax.set_ylabel("candidate_hit_rate (media ± sd)")
        ax.set_title(f"Ablación (Modelo D, régimen M=16): {etiqueta}\n"
                     f"n={n} semillas por valor", fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(coma)
        out = OUT / f"{name}_cand.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[OK] {out}: " + ", ".join(f"{v}={m:.3f}" for v, m in zip(vals, med, strict=True)))


if __name__ == "__main__":
    main()
