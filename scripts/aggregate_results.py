"""Análisis agregado de una campaña experimental.

Lee el CSV producido por ``scripts/run_campaign.py`` y emite:
- Resumen por modelo (media, desviación, IC95 por bootstrap).
- Comparación pareada C vs D sobre las mismas semillas (Sec. 8.18).
- Tabla Markdown y CSV listas para el documento de tesis.

Uso::

    uv run python scripts/aggregate_results.py --campaign outputs/tables/campaign_1.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from src.utils.logging import get_logger, setup_logging
from src.utils.paired_stats import paired_bootstrap
from src.utils.paths import TABLES_DIR, ensure_dir

_log = get_logger(__name__)

METRICS_FINANCIAL = ["cumulative_return", "sharpe_ratio", "max_drawdown", "mean_reward"]
METRICS_LEARNING = ["episodes_to_convergence", "train_reward_last"]
METRICS_EXPLORATION = ["topm_hit_rate", "candidate_hit_rate", "asset_coverage"]
METRICS_LATENCY = ["mean_latency_ms", "duration_seconds"]
ALL_METRICS = METRICS_FINANCIAL + METRICS_LEARNING + METRICS_EXPLORATION + METRICS_LATENCY


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _to_float(value: str) -> float:
    try:
        v = float(value)
        if v == -1.0:  # convención: -1 indica "no convergió" / "no aplicable"
            return float("nan")
        return v
    except ValueError:
        return float("nan")


def _per_model_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, float]]]:
    """Estadísticas por modelo y métrica: ``out[model][metric] = {mean, std, ci_lo, ci_hi, n}``."""
    rng = np.random.default_rng(2026)
    n_boot = 1000
    out: dict[str, dict[str, dict[str, float]]] = {}
    by_model: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for model, model_rows in by_model.items():
        out[model] = {}
        for m in ALL_METRICS:
            values = np.array([_to_float(r[m]) for r in model_rows], dtype=np.float64)
            mask = np.isfinite(values)
            valid = values[mask]
            if valid.size == 0:
                out[model][m] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "n": 0.0,
                }
                continue
            mean = float(valid.mean())
            std = float(valid.std(ddof=1)) if valid.size > 1 else 0.0
            # Bootstrap CI95
            boot_means = np.array(
                [rng.choice(valid, size=valid.size, replace=True).mean() for _ in range(n_boot)]
            )
            ci_lo, ci_hi = (
                float(np.percentile(boot_means, 2.5)),
                float(np.percentile(boot_means, 97.5)),
            )
            out[model][m] = {
                "mean": mean,
                "std": std,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n": float(valid.size),
            }
    return out


def _paired_bootstrap_c_vs_d(
    rows: list[dict[str, str]],
    *,
    n_boot: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, float]]:
    """Bootstrap pareado D - C sobre semillas comunes (Sec. 8.18).

    Para cada métrica, calcula la diferencia D - C semilla a semilla y
    bootstrap el IC95% sobre la media de diferencias.
    """
    if rng is None:
        rng = np.random.default_rng(2026)
    by_seed_model: dict[int, dict[str, dict[str, str]]] = {}
    for r in rows:
        seed = int(r["seed"])
        by_seed_model.setdefault(seed, {})[r["model"]] = r

    common_seeds = [s for s, mm in by_seed_model.items() if "C" in mm and "D" in mm]
    out: dict[str, dict[str, float]] = {}
    if not common_seeds:
        return out

    for metric in ALL_METRICS:
        diffs: list[float] = []
        for s in common_seeds:
            c_val = _to_float(by_seed_model[s]["C"][metric])
            d_val = _to_float(by_seed_model[s]["D"][metric])
            if np.isfinite(c_val) and np.isfinite(d_val):
                diffs.append(d_val - c_val)
        diffs_arr = np.array(diffs, dtype=np.float64)
        if diffs_arr.size == 0:
            out[metric] = {
                "mean_diff": float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "n_pairs": 0.0,
                "p_better": float("nan"),
            }
            continue
        if diffs_arr.size < 2:
            out[metric] = {
                "mean_diff": float(diffs_arr.mean()),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_pairs": float(diffs_arr.size),
                "p_better": float("nan"), "p_two_sided": float("nan"),
            }
            continue
        # Especificacion unica del proyecto (Anexo B): IC percentil 2,5/97,5;
        # p2 = min(1, 2*min(P(m<=0), P(m>=0))); caso degenerado => sin p.
        bs = paired_bootstrap(diffs_arr, n_boot=n_boot, rng=rng)
        out[metric] = {
            "mean_diff": bs.mean,
            "ci_lo": bs.ci_lo,
            "ci_hi": bs.ci_hi,
            "n_pairs": float(bs.n),
            # Probabilidad bootstrap de que D > C en esa métrica (unilateral).
            "p_better": bs.p_positive,
            # P-value bilateral (sin corregir); NaN si degenerado. Las
            # correcciones múltiples se calculan más abajo y omiten los NaN.
            "p_two_sided": bs.p_two,
            "degenerate": float(bs.degenerate),
        }
    # Aplicar correcciones por múltiples comparaciones (Bonferroni, Holm, FDR)
    _apply_multiple_testing_correction(out)
    return out


def _apply_multiple_testing_correction(
    bootstrap_out: dict[str, dict[str, float]],
) -> None:
    """Añade p-values corregidos al diccionario de resultados in-place.

    Tres métodos (rev. v2 — Problema 1.2 del revisor):
    - Bonferroni: ``p_adj = min(p * k, 1)`` (conservador, simple).
    - Holm-Bonferroni step-down: ``p_adj_(i) = (k - i + 1) * p_(i)`` con
      corrección monotónica (menos conservador que Bonferroni puro).
    - Benjamini-Hochberg FDR: ``p_adj_(i) = (k / i) * p_(i)`` con corrección
      monotónica (controla tasa de falsa descobertura).

    Las claves nuevas añadidas a cada métrica:
        p_bonferroni, p_holm, p_fdr
    """
    # Métricas con p_two_sided válido (no NaN)
    items = [(k, v) for k, v in bootstrap_out.items()
             if "p_two_sided" in v and not np.isnan(v["p_two_sided"])]
    k_total = len(items)
    if k_total == 0:
        return

    # --- Bonferroni ---
    for _, v in items:
        v["p_bonferroni"] = min(v["p_two_sided"] * k_total, 1.0)

    # --- Holm-Bonferroni (step-down) ---
    sorted_items = sorted(items, key=lambda kv: kv[1]["p_two_sided"])
    max_so_far = 0.0
    for i, (_, v) in enumerate(sorted_items, start=1):
        raw = v["p_two_sided"]
        adj = min((k_total - i + 1) * raw, 1.0)
        # Monotónica no-decreciente
        max_so_far = max(max_so_far, adj)
        v["p_holm"] = max_so_far

    # --- Benjamini-Hochberg FDR ---
    sorted_items = sorted(items, key=lambda kv: kv[1]["p_two_sided"])
    fdr_adj = [(k_total / i) * v["p_two_sided"] for i, (_, v) in enumerate(sorted_items, start=1)]
    # Monotónica no-decreciente desde el más alto (step-up)
    for i in range(len(fdr_adj) - 2, -1, -1):
        fdr_adj[i] = min(fdr_adj[i], fdr_adj[i + 1])
    for (_, v), adj in zip(sorted_items, fdr_adj, strict=True):
        v["p_fdr"] = min(adj, 1.0)

    # NaN para métricas sin p_two_sided (no fueron testeadas)
    for v in bootstrap_out.values():
        if "p_bonferroni" not in v:
            v["p_bonferroni"] = float("nan")
            v["p_holm"] = float("nan")
            v["p_fdr"] = float("nan")


def _write_summary_csv(
    stats: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
) -> None:
    rows = []
    for model, metrics in stats.items():
        for metric, s in metrics.items():
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": s["mean"],
                    "std": s["std"],
                    "ci_lo": s["ci_lo"],
                    "ci_hi": s["ci_hi"],
                    "n": int(s["n"]),
                }
            )
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_paired_csv(
    paired: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    rows = []
    for metric, s in paired.items():
        rows.append(
            {
                "metric": metric,
                "mean_diff_d_minus_c": s["mean_diff"],
                "ci_lo": s["ci_lo"],
                "ci_hi": s["ci_hi"],
                "n_pairs": int(s["n_pairs"]),
                "p_d_better_than_c": s["p_better"],
                # P-values bilaterales y correcciones por múltiples
                # comparaciones (rev. v2 — Problema 1.2 del revisor).
                "p_two_sided": s.get("p_two_sided", float("nan")),
                "p_bonferroni": s.get("p_bonferroni", float("nan")),
                "p_holm": s.get("p_holm", float("nan")),
                "p_fdr": s.get("p_fdr", float("nan")),
            }
        )
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_markdown_summary(
    stats: dict[str, dict[str, dict[str, float]]],
    paired: dict[str, dict[str, float]],
) -> str:
    lines: list[str] = []
    lines.append("# Resumen de la campaña experimental\n")
    lines.append("## Métricas por modelo (media ± std · IC95% bootstrap)\n")
    headers = ["metric", *sorted(stats.keys())]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for metric in ALL_METRICS:
        row = [metric]
        for model in sorted(stats.keys()):
            s = stats[model].get(metric)
            if s is None or s["n"] == 0:
                row.append("—")
            else:
                row.append(
                    f"{s['mean']:.4f} ± {s['std']:.4f}  [{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]"
                )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if paired:
        lines.append("## Comparación pareada D vs C (Sec. 8.18)\n")
        lines.append("Diferencia ``D - C`` por semilla; bootstrap pareado 5000 iter.\n")
        lines.append("| metric | mean(D-C) | IC95% | n_pairs | P(D > C) |")
        lines.append("|---|---|---|---|---|")
        for metric in ALL_METRICS:
            s = paired.get(metric)
            if s is None or s["n_pairs"] == 0:
                continue
            lines.append(
                f"| {metric} | {s['mean_diff']:.4f} | "
                f"[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}] | "
                f"{int(s['n_pairs'])} | {s['p_better']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", type=Path, required=True, help="CSV producido por run_campaign.py"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio destino (default: outputs/tables/).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    if not args.campaign.is_file():
        raise SystemExit(f"No encontrado: {args.campaign}")

    rows = _read_csv(args.campaign)
    if not rows:
        raise SystemExit(f"CSV vacío: {args.campaign}")

    stats = _per_model_stats(rows)
    paired = _paired_bootstrap_c_vs_d(rows)

    out_dir = args.output_dir if args.output_dir is not None else TABLES_DIR
    ensure_dir(out_dir)
    stem = args.campaign.stem
    summary_csv = out_dir / f"{stem}_summary.csv"
    paired_csv = out_dir / f"{stem}_paired_c_vs_d.csv"
    md_path = out_dir / f"{stem}_report.md"

    _write_summary_csv(stats, summary_csv)
    if paired:
        _write_paired_csv(paired, paired_csv)
    md = _format_markdown_summary(stats, paired)
    md_path.write_text(md, encoding="utf-8")

    print(f"[OK] Resumen escrito en {summary_csv}")
    if paired:
        print(f"[OK] Comparación pareada en {paired_csv}")
    print(f"[OK] Reporte markdown en {md_path}")
    print("\n" + md)


if __name__ == "__main__":
    main()
