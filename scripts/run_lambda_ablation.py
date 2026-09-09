"""Ablación de la penalización por riesgo lambda (rev. v2 — Problema 1.3).

Ejecuta el Modelo A bajo distintos valores de ``lambda_risk`` y dos
esquemas de recompensa (``risk_penalty`` y ``log_wealth``) para
identificar configuraciones que producen Sharpe positivo. Si alguna
configuración converge a política rentable, esa se promueve como
base para los Modelos C y D.

Uso::

    uv run python scripts/run_lambda_ablation.py \
        --universe nivel2 --seeds 42 123 456 \
        --lambdas 0.0 0.01 0.05 0.1 \
        --reward-types risk_penalty log_wealth \
        --steps 50000
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
    lam: float,
    reward_type: str,
    seed: int,
    universe: str,
    steps: int,
    max_steps: int | None,
    window_length: int | None,
) -> dict:
    """Ejecuta un run de Modelo A con override de lambda y reward_type."""
    original_load = run_campaign.load_config

    def patched_load(cfg_path):
        cfg = original_load(cfg_path)
        cfg.data.cache_subdir = universe
        cfg.env.lambda_risk = lam
        cfg.env.reward_type = reward_type
        return cfg

    run_campaign.load_config = patched_load
    try:
        row = run_campaign._train_and_evaluate_one(
            model="A",
            seed=seed,
            universe=universe,
            steps=steps,
            campaign_id=f"abl_lambda_{reward_type}_{lam:.3f}",
            max_steps_override=max_steps,
            window_override=window_length,
        )
    finally:
        run_campaign.load_config = original_load

    d = asdict(row)
    d["lambda_risk"] = lam
    d["reward_type"] = reward_type
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1])
    parser.add_argument(
        "--reward-types",
        nargs="+",
        choices=["risk_penalty", "log_wealth"],
        default=["risk_penalty", "log_wealth"],
    )
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_campaign._maybe_download(args.universe)
    if args.output is None:
        args.output = TABLES_DIR / "abl_lambda.csv"
        ensure_dir(args.output.parent)

    rows: list[dict] = []
    total = len(args.lambdas) * len(args.reward_types) * len(args.seeds)
    idx = 0
    for reward_type in args.reward_types:
        for lam in args.lambdas:
            for seed in args.seeds:
                idx += 1
                print(
                    f"[{idx}/{total}] reward={reward_type} lambda={lam:.3f} seed={seed}",
                    flush=True,
                )
                row = _run_one(
                    lam=lam,
                    reward_type=reward_type,
                    seed=seed,
                    universe=args.universe,
                    steps=args.steps,
                    max_steps=args.max_steps,
                    window_length=args.window_length,
                )
                rows.append(row)
                print(
                    f"  return={row['cumulative_return']:.4f} "
                    f"sharpe={row['sharpe_ratio']:.4f} "
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
