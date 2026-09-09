"""Validacion walk-forward temporal (rev. v3 - Punto 2.4 del revisor).

Implementa expanding-window walk-forward CV con 4 folds:

  Fold 1: train 2018-01..2020-12 (3 anos), test 2021 (1 ano)
  Fold 2: train 2018-01..2021-12 (4 anos), test 2022 (1 ano)
  Fold 3: train 2018-01..2022-12 (5 anos), test 2023 (1 ano)
  Fold 4: train 2018-01..2023-12 (6 anos), test 2024 (1 ano)

Para cada fold se ejecutan los modelos C y D con n=3 semillas (12 runs total)
usando el sweet_spot v2 (M=16, beta=0.5, k=3). Salida CSV con columna
``fold_id`` y los rangos temporales explicitos.

Uso::

    uv run python scripts/run_walk_forward.py --universe nivel2 \\
        --seeds 42 123 456 --steps 50000 --max-steps 50
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


# (train_end_year, test_year) -> filter_dates, train_frac
# Cada fold cubre [2018-01-02, test_year-12-31] y usa el ultimo
# ~1 ano como test.
FOLDS = [
    {"fold_id": 1, "train_end": 2020, "test_year": 2021},
    {"fold_id": 2, "train_end": 2021, "test_year": 2022},
    {"fold_id": 3, "train_end": 2022, "test_year": 2023},
    {"fold_id": 4, "train_end": 2023, "test_year": 2024},
]


def _filter_for_fold(train_end: int, test_year: int) -> tuple[str, str]:
    start = "2018-01-02"
    end = f"{test_year}-12-31"
    return start, end


def _train_frac_for_fold(train_end: int, test_year: int) -> float:
    """train_frac para colocar el ultimo ano como test."""
    n_years_total = test_year - 2018 + 1
    return (train_end - 2018 + 1) / n_years_total


def _run_one(
    *,
    model: str,
    seed: int,
    universe: str,
    steps: int,
    max_steps: int | None,
    window_length: int | None,
    fold: dict,
) -> dict:
    """Ejecuta un run con filtro temporal de fold y train_frac ajustado."""
    train_end = fold["train_end"]
    test_year = fold["test_year"]
    start, end = _filter_for_fold(train_end, test_year)
    train_frac = _train_frac_for_fold(train_end, test_year)
    val_frac = 0.01  # minimal val; el split interno hace 60/20/20 por defecto
    # Forzaremos: train, val, test segun train_frac / val_frac

    original_load = run_campaign.load_config

    def patched_load(cfg_path):
        cfg = original_load(cfg_path)
        # Sweet spot v2
        if cfg.graph is not None:
            cfg.graph.subgraph_max_size = 16
            cfg.graph.alpha = 0.5
            cfg.graph.beta = 0.5
        if cfg.quantum is not None:
            cfg.quantum.k_steps = 3
            cfg.quantum.m_top = 5
        # Override fracciones del split interno
        cfg.env.split.train_frac = train_frac
        cfg.env.split.val_frac = val_frac
        return cfg

    run_campaign.load_config = patched_load
    try:
        row = run_campaign._train_and_evaluate_one(
            model=model,
            seed=seed,
            universe=universe,
            steps=steps,
            campaign_id=f"walk_fwd_fold{fold['fold_id']}",
            max_steps_override=max_steps,
            window_override=window_length,
            config_suffix="v2",
            filter_dates=(start, end),
        )
    finally:
        run_campaign.load_config = original_load

    d = asdict(row)
    d["fold_id"] = fold["fold_id"]
    d["train_end_year"] = train_end
    d["test_year"] = test_year
    d["filter_start"] = start
    d["filter_end"] = end
    d["train_frac"] = train_frac
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--window-length", type=int, default=20)
    parser.add_argument(
        "--models", nargs="+", choices=["C", "D"], default=["C", "D"],
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_campaign._maybe_download(args.universe)
    if args.output is None:
        args.output = TABLES_DIR / "walk_forward_v3.csv"
        ensure_dir(args.output.parent)

    rows: list[dict] = []
    total = len(FOLDS) * len(args.models) * len(args.seeds)
    idx = 0
    for fold in FOLDS:
        for model in args.models:
            for seed in args.seeds:
                idx += 1
                print(
                    f"[{idx}/{total}] fold={fold['fold_id']} model={model} "
                    f"seed={seed} train_end={fold['train_end']} "
                    f"test={fold['test_year']}",
                    flush=True,
                )
                row = _run_one(
                    model=model,
                    seed=seed,
                    universe=args.universe,
                    steps=args.steps,
                    max_steps=args.max_steps,
                    window_length=args.window_length,
                    fold=fold,
                )
                rows.append(row)
                print(
                    f"  return={row['cumulative_return']:.4f} "
                    f"sharpe={row['sharpe_ratio']:.4f} "
                    f"cand_hit={row['candidate_hit_rate']:.3f} "
                    f"topm_hit={row['topm_hit_rate']:.3f} "
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
