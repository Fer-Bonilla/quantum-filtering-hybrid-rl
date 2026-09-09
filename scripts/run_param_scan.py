"""Escaneo factorial reducido de parámetros para encontrar el sweet spot.

Diseño 2³ sobre (M, k, β) más combinaciones extremas, sólo para los
modelos C y D (los únicos que usan el LocalModule). Cada variante se
ejecuta con N seeds × M steps reducidos para identificar rápidamente
qué configuración maximiza el gap `mean(D − C)` en Sharpe.

Resultado: ``outputs/tables/param_scan.csv`` con columna ``variant``.

Uso::

    uv run python scripts/run_param_scan.py --universe nivel2 \
        --seeds 42 123 456 --steps 5000

Para análisis post-scan, usar ``scripts/aggregate_results.py`` con flag
``--paired-by-variant`` (a implementar) o agrupar manualmente.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys as _sys
from dataclasses import asdict
from pathlib import Path

from src.utils.logging import get_logger, setup_logging
from src.utils.paths import TABLES_DIR, ensure_dir

# Importar run_campaign manteniendo módulo en sys.modules
_path = Path(__file__).resolve().parent / "run_campaign.py"
_spec = importlib.util.spec_from_file_location("run_campaign", _path)
assert _spec is not None and _spec.loader is not None
run_campaign = importlib.util.module_from_spec(_spec)
_sys.modules["run_campaign"] = run_campaign
_spec.loader.exec_module(run_campaign)

_log = get_logger(__name__)


# Diseño factorial reducido (8 variantes)
VARIANTS = {
    "baseline": dict(M=8, k=3, beta=0.0, alpha=1.0, m=3),
    "s1_beta": dict(M=8, k=3, beta=0.5, alpha=0.5, m=3),
    "s2_M16": dict(M=16, k=3, beta=0.0, alpha=1.0, m=5),
    "s3_M16_beta": dict(M=16, k=3, beta=0.5, alpha=0.5, m=5),
    "s4_k5": dict(M=8, k=5, beta=0.0, alpha=1.0, m=3),
    "s5_M16_k5": dict(M=16, k=5, beta=0.0, alpha=1.0, m=5),
    "s6_full": dict(M=16, k=5, beta=0.5, alpha=0.5, m=5),  # ⭐ candidato fuerte
    "s7_M24": dict(M=24, k=5, beta=0.5, alpha=0.5, m=5),
}


def _run_variant(
    variant_name: str,
    variant_cfg: dict,
    model: str,
    seed: int,
    universe: str,
    steps: int,
    max_steps: int | None,
    window_length: int | None,
) -> dict:
    """Ejecutar 1 run con override de los parámetros de la variante."""
    original_load = run_campaign.load_config

    def patched_load(cfg_path):
        cfg = original_load(cfg_path)
        if cfg.graph is not None:
            cfg.graph.subgraph_max_size = variant_cfg["M"]
            cfg.graph.alpha = variant_cfg["alpha"]
            cfg.graph.beta = variant_cfg["beta"]
            # k_neighbors crece con M
            cfg.graph.k_neighbors = max(5, variant_cfg["M"] // 3)
        if cfg.quantum is not None:
            cfg.quantum.k_steps = variant_cfg["k"]
            cfg.quantum.m_top = variant_cfg["m"]
        cfg.data.cache_subdir = universe
        return cfg

    run_campaign.load_config = patched_load
    try:
        row = run_campaign._train_and_evaluate_one(
            model=model,
            seed=seed,
            universe=universe,
            steps=steps,
            campaign_id=f"scan_{variant_name}",
            max_steps_override=max_steps,
            window_override=window_length,
        )
    finally:
        run_campaign.load_config = original_load

    d = asdict(row)
    d["variant"] = variant_name
    d["variant_M"] = variant_cfg["M"]
    d["variant_k"] = variant_cfg["k"]
    d["variant_beta"] = variant_cfg["beta"]
    d["variant_m"] = variant_cfg["m"]
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Subset de variantes a ejecutar (default: todas).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["A", "B", "C", "D"],
        default=["C", "D"],
        help="Modelos a ejecutar. Default C y D (los únicos con LocalModule). "
        "Para validar sweet spot end-to-end con baselines, usar A B C D.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV destino (default: outputs/tables/param_scan.csv).",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_campaign._maybe_download(args.universe)

    variants_to_run = args.variants if args.variants else list(VARIANTS.keys())
    out_path = args.output if args.output else TABLES_DIR / "param_scan.csv"
    ensure_dir(out_path.parent)

    rows: list[dict] = []
    total = len(variants_to_run) * len(args.models) * len(args.seeds)
    idx = 0
    for variant_name in variants_to_run:
        if variant_name not in VARIANTS:
            _log.warning("Variante desconocida: %s; saltando.", variant_name)
            continue
        variant_cfg = VARIANTS[variant_name]
        for model in args.models:
            for seed in args.seeds:
                idx += 1
                print(
                    f"[{idx}/{total}] variant={variant_name} (M={variant_cfg['M']}, "
                    f"k={variant_cfg['k']}, beta={variant_cfg['beta']}) model={model} seed={seed}"
                )
                row = _run_variant(
                    variant_name=variant_name,
                    variant_cfg=variant_cfg,
                    model=model,
                    seed=seed,
                    universe=args.universe,
                    steps=args.steps,
                    max_steps=args.max_steps,
                    window_length=args.window_length,
                )
                rows.append(row)
                print(
                    f"        return={row['cumulative_return']:.4f} "
                    f"sharpe={row['sharpe_ratio']:.4f} "
                    f"latency={row['mean_latency_ms']:.2f}ms "
                    f"duration={row['duration_seconds']:.1f}s"
                )

    if not rows:
        raise SystemExit("Sin filas para escribir.")
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] {len(rows)} runs escritos en {out_path}")


if __name__ == "__main__":
    main()
