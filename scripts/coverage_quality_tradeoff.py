"""Trade-off cobertura ↔ calidad de exploración (rev. v2 — Problema 1.6).

Computa y visualiza el trade-off entre cobertura del universo (qué
fracción de tickers visita el agente) y la calidad de las selecciones
(``topm_hit_rate``). El revisor argumentó que la baja cobertura del
Modelo D (0.59 vs 0.95 de C) contradice la narrativa de "exploración
estructurada".

Métricas reportadas:

* ``precision@m`` = ``topm_hit_rate`` (calidad media por activo elegido).
* ``coverage`` = ``asset_coverage`` (cobertura del universo).
* ``precision_per_visited`` = ``topm_hit_rate / max(asset_coverage, eps)``
  — densidad de calidad ajustada por cobertura. Mide cuántas veces se
  acierta por unidad de "exploración usada".
* ``recall_proxy`` = ``candidate_hit_rate × (m_top / |promising|)``.
  Si no se conoce el ``m_top``, se omite.

Salida:

* CSV con las cuatro métricas por (model, seed).
* Figura ``coverage_quality_tradeoff.png`` con scatter coverage vs
  topm_hit_rate (un punto por seed), separada por modelo.

Uso::

    uv run python scripts/coverage_quality_tradeoff.py \
        --campaign outputs/tables/campaign_1_v2.csv \
        --output outputs/tables/campaign_1_v2_tradeoff.csv \
        --figure docs/thesis/figures/coverage_quality_tradeoff.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.paths import TABLES_DIR, ensure_dir

_EPS = 1e-8

# Paleta consistente con el resto de las figuras del informe.
_MODEL_COLORS = {
    "A": "#377eb8",  # azul
    "B": "#ff7f00",  # naranja
    "C": "#4daf4a",  # verde
    "D": "#e41a1c",  # rojo
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def compute_tradeoff_metrics(
    rows: list[dict[str, str]],
    *,
    m_top: int | None = None,
    n_promising_total: int | None = None,
) -> list[dict[str, float | str]]:
    """Añadir columnas de trade-off por fila.

    Args:
        rows: filas del CSV con al menos ``model``, ``seed``,
            ``topm_hit_rate``, ``asset_coverage``, ``candidate_hit_rate``.
        m_top: tamaño del top-m (si se conoce). Si se proporciona junto a
            ``n_promising_total``, se computa ``recall_proxy``.
        n_promising_total: número de activos prometedores típicos
            (e.g. para top 20 % de N=30 → 6).

    Returns:
        Filas enriquecidas con ``precision_per_visited`` y opcionalmente
        ``recall_proxy``.
    """
    enriched: list[dict[str, float | str]] = []
    can_compute_recall = m_top is not None and n_promising_total is not None
    for row in rows:
        precision = _f(row["topm_hit_rate"])
        coverage = _f(row["asset_coverage"])
        cand = _f(row["candidate_hit_rate"])
        ppv = precision / max(coverage, _EPS) if np.isfinite(coverage) else float("nan")
        out: dict[str, float | str] = {
            "campaign_id": row.get("campaign_id", ""),
            "model": row["model"],
            "seed": row["seed"],
            "topm_hit_rate": precision,
            "asset_coverage": coverage,
            "candidate_hit_rate": cand,
            "precision_per_visited": ppv,
        }
        if can_compute_recall:
            recall = cand * (m_top / n_promising_total)
            out["recall_proxy"] = float(min(recall, 1.0))
        enriched.append(out)
    return enriched


def aggregate_by_model(
    enriched: list[dict[str, float | str]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Devuelve ``out[model][metric] = {mean, std, n}`` para las tres métricas trade-off."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    by_model: dict[str, list[dict[str, float | str]]] = {}
    for row in enriched:
        by_model.setdefault(str(row["model"]), []).append(row)
    metric_names = ["asset_coverage", "topm_hit_rate", "precision_per_visited"]
    if any("recall_proxy" in r for r in enriched):
        metric_names.append("recall_proxy")
    for model, model_rows in by_model.items():
        out[model] = {}
        for m in metric_names:
            vals = np.array(
                [float(r[m]) for r in model_rows if isinstance(r.get(m), int | float)],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                out[model][m] = {"mean": float("nan"), "std": float("nan"), "n": 0.0}
                continue
            out[model][m] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "n": float(vals.size),
            }
    return out


def write_csv(enriched: list[dict[str, float | str]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({k for row in enriched for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)


def plot_tradeoff(
    enriched: list[dict[str, float | str]],
    summary: dict[str, dict[str, dict[str, float]]],
    output: Path,
) -> None:
    """Genera el scatter cobertura ↔ calidad con un punto por (model, seed)."""
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=120)
    by_model: dict[str, list[dict[str, float | str]]] = {}
    for row in enriched:
        by_model.setdefault(str(row["model"]), []).append(row)

    for model, rows in sorted(by_model.items()):
        xs = np.array([float(r["asset_coverage"]) for r in rows], dtype=np.float64)
        ys = np.array([float(r["topm_hit_rate"]) for r in rows], dtype=np.float64)
        color = _MODEL_COLORS.get(model, "#888888")
        ax.scatter(xs, ys, color=color, s=70, alpha=0.55, label=f"Modelo {model}")
        # Centroide
        cx = summary[model]["asset_coverage"]["mean"]
        cy = summary[model]["topm_hit_rate"]["mean"]
        ax.scatter(
            [cx], [cy], color=color, s=160, edgecolor="black", linewidth=1.4, marker="X"
        )

    # Curvas de iso-densidad precision_per_visited
    cov_grid = np.linspace(0.05, 1.0, 200)
    for ratio in [0.10, 0.20, 0.30, 0.40]:
        prec_grid = ratio * cov_grid
        ax.plot(
            cov_grid,
            prec_grid,
            "--",
            color="#999999",
            linewidth=0.7,
            alpha=0.6,
        )
        # Etiqueta sobre el eje y para cada curva
        ax.text(
            1.0,
            ratio * 1.0,
            f"  ppv={ratio:.2f}",
            color="#666666",
            fontsize=8,
            va="center",
        )

    ax.set_xlabel("Cobertura de activos (asset_coverage)")
    ax.set_ylabel("Calidad por elección (topm_hit_rate)")
    ax.set_title(
        "Trade-off cobertura ↔ calidad (rev. v2)\n"
        "Las líneas grises indican precision_per_visited constante (ppv=topm/cov)."
    )
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, max(0.5, max((float(r["topm_hit_rate"]) for r in enriched), default=0.3)) + 0.05)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    ensure_dir(output.parent)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True, help="CSV de la campaña.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV enriquecido (default: outputs/tables/<campaign>_tradeoff.csv).",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="Ruta de la figura (default: docs/thesis/figures/<campaign>_tradeoff.png).",
    )
    parser.add_argument("--m-top", type=int, default=3, help="Tamaño del top-m del modelo D.")
    parser.add_argument(
        "--n-promising",
        type=int,
        default=6,
        help="Activos prometedores típicos (top 20 % de N=30 → 6).",
    )
    args = parser.parse_args()

    rows = _read_csv(args.campaign)
    enriched = compute_tradeoff_metrics(
        rows, m_top=args.m_top, n_promising_total=args.n_promising
    )
    summary = aggregate_by_model(enriched)

    output_csv = args.output or TABLES_DIR / f"{args.campaign.stem}_tradeoff.csv"
    write_csv(enriched, output_csv)
    print(f"[OK] CSV enriquecido escrito en {output_csv}")

    figure = args.figure or Path("docs/thesis/figures") / f"{args.campaign.stem}_tradeoff.png"
    plot_tradeoff(enriched, summary, figure)
    print(f"[OK] Figura escrita en {figure}")

    # Resumen por modelo
    print("\nResumen trade-off (mean ± std por modelo):")
    print(f"  {'Modelo':<8}{'coverage':>12}{'topm_hit':>12}{'ppv':>12}{'recall':>12}")
    for model, metrics in sorted(summary.items()):
        cov = metrics["asset_coverage"]
        topm = metrics["topm_hit_rate"]
        ppv = metrics["precision_per_visited"]
        recall = metrics.get("recall_proxy", {"mean": float("nan")})
        print(
            f"  {model:<8}"
            f"{cov['mean']:>7.3f} ±{cov['std']:>4.2f}"
            f"{topm['mean']:>7.3f} ±{topm['std']:>4.2f}"
            f"{ppv['mean']:>7.3f} ±{ppv['std']:>4.2f}"
            f"{recall['mean']:>7.3f}"
        )


if __name__ == "__main__":
    main()
