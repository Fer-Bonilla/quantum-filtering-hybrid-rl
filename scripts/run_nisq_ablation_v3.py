"""Ablación NISQ del Modelo D bajo reward v2 (rev. v3 — Punto 2.3 del revisor).

Ejecuta el Modelo D con DIFERENTES configuraciones de ruido cuántico
(ideal / depolarizing / dephasing) usando:
  - env v2 (lambda=0, log_wealth)
  - graph.subgraph_max_size=8 (sweet_spot v1 era 16; bajamos para tractabilidad)
  - backend pennylane (default.mixed) cuando hay ruido
  - --steps reducidos para mantener el coste manejable

Uso::

    uv run python scripts/run_nisq_ablation_v3.py \\
        --universe nivel2 --seeds 42 123 456 \\
        --steps 5000 --max-steps 50 --window-length 10

Salida: ``outputs/tables/abl_nisq_v3.csv`` con columnas
``ablation_kind``, ``ablation_value`` (e.g. "ideal", "depol=0.050",
"dephas=0.050") además de las métricas estándar.
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

_path = Path(__file__).resolve().parent / "run_campaign.py"
_spec = importlib.util.spec_from_file_location("run_campaign", _path)
assert _spec is not None and _spec.loader is not None
run_campaign = importlib.util.module_from_spec(_spec)
_sys.modules["run_campaign"] = run_campaign
_spec.loader.exec_module(run_campaign)

_log = get_logger(__name__)


def _run_one(
    *,
    seed: int,
    universe: str,
    steps: int,
    max_steps: int | None,
    window_length: int | None,
    label: str,
    depol_prob: float,
    dephas_prob: float,
    subgraph_M: int,
    model: str = "D",
) -> dict:
    """Ejecuta un run con un perfil específico de ruido + M reducido."""
    original_load = run_campaign.load_config

    def patched_load(cfg_path):
        cfg = original_load(cfg_path)
        # Asegurar v2 reward
        cfg.env.lambda_risk = 0.0
        cfg.env.reward_type = "log_wealth"
        # Bajar tamano de subgrafo
        if cfg.graph is not None:
            cfg.graph.subgraph_max_size = subgraph_M
        # Configurar ruido (solo aplicable a D; C usa classical walker)
        if cfg.quantum is not None:
            noisy = (depol_prob + dephas_prob) > 0.0
            cfg.quantum.noise = type(cfg.quantum.noise)(
                enabled=noisy,
                depolarizing_prob=depol_prob,
                dephasing_prob=dephas_prob,
            )
            cfg.quantum.backend = "pennylane" if noisy else "matrix"
        return cfg

    run_campaign.load_config = patched_load
    try:
        row = run_campaign._train_and_evaluate_one(
            model=model,
            seed=seed,
            universe=universe,
            steps=steps,
            campaign_id=f"abl_nisq_v3_{label}_{model}",
            max_steps_override=max_steps,
            window_override=window_length,
            config_suffix="v2",
        )
    finally:
        run_campaign.load_config = original_load

    d = asdict(row)
    d["ablation_kind"] = "nisq_v3"
    d["ablation_value"] = label
    d["depol_prob"] = depol_prob
    d["dephas_prob"] = dephas_prob
    d["subgraph_M"] = subgraph_M
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--window-length", type=int, default=10)
    parser.add_argument("--subgraph-M", type=int, default=8)
    parser.add_argument(
        "--model",
        choices=["C", "D"],
        default="D",
        help="Modelo a ejecutar (default D). C ignora perfiles de "
             "ruido pero usa el mismo M y configuración base.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_campaign._maybe_download(args.universe)
    if args.output is None:
        args.output = TABLES_DIR / "abl_nisq_v3.csv"
        ensure_dir(args.output.parent)

    # Perfiles de ruido a evaluar
    profiles = [
        ("ideal", 0.0, 0.0),
        ("depol=0.05", 0.05, 0.0),
        ("depol=0.10", 0.10, 0.0),
        ("dephas=0.05", 0.0, 0.05),
    ]

    rows: list[dict] = []
    total = len(profiles) * len(args.seeds)
    idx = 0
    for label, depol, dephas in profiles:
        for seed in args.seeds:
            idx += 1
            print(
                f"[{idx}/{total}] profile={label} (depol={depol}, dephas={dephas}) "
                f"seed={seed} M={args.subgraph_M}",
                flush=True,
            )
            row = _run_one(
                seed=seed,
                universe=args.universe,
                steps=args.steps,
                max_steps=args.max_steps,
                window_length=args.window_length,
                label=label,
                depol_prob=depol,
                dephas_prob=dephas,
                subgraph_M=args.subgraph_M,
                model=args.model,
            )
            rows.append(row)
            print(
                f"  return={row['cumulative_return']:.4f} "
                f"sharpe={row['sharpe_ratio']:.4f} "
                f"latency={row['mean_latency_ms']:.2f}ms "
                f"duration={row['duration_seconds']:.1f}s",
                flush=True,
            )

    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] {len(rows)} runs en {args.output}", flush=True)


if __name__ == "__main__":
    main()
