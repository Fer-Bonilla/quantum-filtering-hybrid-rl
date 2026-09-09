"""Generación de figuras finales para la tesis.

Lee CSVs producidos por ``run_campaign.py`` y ``run_ablation.py`` y emite
PNGs en ``outputs/figures/``.

Figuras producidas:
- ``<stem>_financial.png``: boxplot de retorno/Sharpe/drawdown por modelo.
- ``<stem>_exploration.png``: boxplot de top-m hit-rate, candidate hit-rate,
  cobertura.
- ``<stem>_latency.png``: latencia media por step (ms) por modelo.
- ``ablation_<param>.png``: media + IC95% del retorno acumulado en función
  del valor barrido.

Uso::

    uv run python scripts/generate_figures.py --campaign outputs/tables/mini.csv
    uv run python scripts/generate_figures.py --ablation outputs/tables/ablation_k_mini.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin GUI
import matplotlib.pyplot as plt
import numpy as np
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import FIGURES_DIR, ensure_dir

_log = get_logger(__name__)

FINANCIAL_METRICS = ["cumulative_return", "sharpe_ratio", "max_drawdown"]
EXPLORATION_METRICS = ["topm_hit_rate", "candidate_hit_rate", "asset_coverage"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _to_float_filter(rows: list[dict[str, str]], key: str) -> np.ndarray:
    vals = []
    for r in rows:
        try:
            v = float(r[key])
            if np.isfinite(v) and v != -1.0:
                vals.append(v)
        except (KeyError, ValueError):
            continue
    return np.array(vals)


def _boxplot_per_model(
    rows: list[dict[str, str]],
    metrics: list[str],
    out_path: Path,
    title: str = "",
) -> None:
    """Boxplot agrupado por modelo para una lista de métricas."""
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics, strict=False):
        data = [_to_float_filter([r for r in rows if r["model"] == m], metric) for m in models]
        ax.boxplot(data, tick_labels=models)
        ax.set_title(metric)
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    _log.info("Figura escrita: %s", out_path)


def _bar_per_model_single(
    rows: list[dict[str, str]],
    metric: str,
    out_path: Path,
    title: str = "",
) -> None:
    """Boxplot para una sola métrica (e.g. latencia)."""
    models = sorted({r["model"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [_to_float_filter([r for r in rows if r["model"] == m], metric) for m in models]
    ax.boxplot(data, tick_labels=models)
    ax.set_title(title or metric)
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    _log.info("Figura escrita: %s", out_path)


def _ablation_curve(rows: list[dict[str, str]], out_path: Path) -> None:
    """Para cada ablation_value, media + std de cumulative_return."""
    by_val: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        try:
            val = float(r["cumulative_return"])
        except (KeyError, ValueError):
            continue
        if not np.isfinite(val):
            continue
        by_val[r.get("ablation_value", "?")].append(val)

    if not by_val:
        _log.warning("Sin datos para curva de ablación.")
        return

    # Ordenar valores numéricamente cuando sea posible
    def _sort_key(k: str) -> float:
        try:
            return float(k.split("=")[-1])
        except ValueError:
            return 0.0

    keys = sorted(by_val.keys(), key=_sort_key)
    means = [float(np.mean(by_val[k])) for k in keys]
    stds = [float(np.std(by_val[k], ddof=1)) if len(by_val[k]) > 1 else 0.0 for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(keys, means, yerr=stds, marker="o", capsize=4, linestyle="-")
    ablation_kind = rows[0].get("ablation_kind", "ablation")
    ax.set_title(f"Ablación: {ablation_kind}")
    ax.set_xlabel(ablation_kind)
    ax.set_ylabel("cumulative_return (mean ± std)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    _log.info("Figura ablación escrita: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=None,
        help="CSV de campaña (run_campaign.py).",
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        nargs="*",
        default=[],
        help="Uno o más CSV de ablación (run_ablation.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/figures/.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    out_dir = args.output_dir if args.output_dir is not None else FIGURES_DIR
    ensure_dir(out_dir)

    if args.campaign is not None:
        if not args.campaign.is_file():
            raise SystemExit(f"No existe: {args.campaign}")
        rows = _read_csv(args.campaign)
        stem = args.campaign.stem
        _boxplot_per_model(
            rows, FINANCIAL_METRICS, out_dir / f"{stem}_financial.png", "Métricas financieras"
        )
        _boxplot_per_model(
            rows,
            EXPLORATION_METRICS,
            out_dir / f"{stem}_exploration.png",
            "Métricas de exploración",
        )
        _bar_per_model_single(
            rows, "mean_latency_ms", out_dir / f"{stem}_latency.png", "Latencia por step (ms)"
        )

    for abl_csv in args.ablation:
        if not abl_csv.is_file():
            _log.warning("No existe: %s; saltando.", abl_csv)
            continue
        abl_rows = _read_csv(abl_csv)
        _ablation_curve(abl_rows, out_dir / f"{abl_csv.stem}.png")

    print(f"[OK] Figuras escritas en {out_dir}")


if __name__ == "__main__":
    main()
