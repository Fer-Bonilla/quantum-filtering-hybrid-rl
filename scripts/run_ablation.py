"""Ablaciones obligatorias del Cap. 8.20.

Ablaciones soportadas:
- ``init``: uniform vs seed_centered (Sec. 8.20.4).
- ``noise``: ideal vs depolarizing (Sec. 8.20.5).
- ``M``: barrido de tamaño máximo del subgrafo (4, 6, 8).
- ``k``: barrido del número de pasos DTQW (2..6).
- ``m``: barrido del tamaño top-m (1..M).

Cada ablación reutiliza el config base ``model_d.yaml`` y sobreescribe los
campos correspondientes. Cada run se registra en MLflow con tags
``parent_campaign`` y ``ablation_kind`` para enlazarlo a la campaña principal.

Uso::

    uv run python scripts/run_ablation.py init --universe nivel1 --seeds 42 123 456
    uv run python scripts/run_ablation.py noise --universe nivel1 --seeds 42 123 \
        --depolarizing 0.0 0.05 0.1
    uv run python scripts/run_ablation.py k --universe nivel1 --seeds 42 \
        --k-values 2 3 4 5 6
"""

from __future__ import annotations

import argparse
import csv

# Reutilizamos la maquinaria de run_campaign.py
import importlib.util
import sys as _sys
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from src.utils.logging import get_logger, setup_logging
from src.utils.paths import TABLES_DIR, ensure_dir

_path = Path(__file__).resolve().parent / "run_campaign.py"
_spec = importlib.util.spec_from_file_location("run_campaign", _path)
assert _spec is not None and _spec.loader is not None
run_campaign = importlib.util.module_from_spec(_spec)
# Registrar el módulo ANTES de ejecutarlo: las dataclasses lo necesitan en sys.modules.
_sys.modules["run_campaign"] = run_campaign
_spec.loader.exec_module(run_campaign)

_log = get_logger(__name__)


def _row_to_dict(row, *, ablation_kind: str, ablation_value: str) -> dict:
    """Convertir un ``RunRow`` (frozen) a dict + metadatos de ablación."""
    d = asdict(row)
    d["ablation_kind"] = ablation_kind
    d["ablation_value"] = ablation_value
    return d


def _write_csv(rows: Iterable[dict], output_path: Path) -> None:
    rows = list(rows)
    if not rows:
        _log.warning("Sin filas para escribir en %s", output_path)
        return
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_with_overrides(
    *,
    seed: int,
    universe: str,
    steps: int,
    campaign_id: str,
    max_steps: int | None,
    window_length: int | None,
    overrides: dict,
    ablation_kind: str,
    ablation_value: str,
) -> dict:
    """Ejecutar un run de modelo D aplicando overrides al config cargado."""
    original_load = run_campaign.load_config

    def patched_load(cfg_path):
        cfg = original_load(cfg_path)
        for path, value in overrides.items():
            obj = cfg
            keys = path.split(".")
            for k in keys[:-1]:
                obj = getattr(obj, k)
            setattr(obj, keys[-1], value)
        return cfg

    run_campaign.load_config = patched_load
    try:
        row = run_campaign._train_and_evaluate_one(
            model="D",
            seed=seed,
            universe=universe,
            steps=steps,
            campaign_id=campaign_id,
            max_steps_override=max_steps,
            window_override=window_length,
        )
    finally:
        run_campaign.load_config = original_load

    return _row_to_dict(row, ablation_kind=ablation_kind, ablation_value=ablation_value)


def cmd_init(args: argparse.Namespace) -> None:
    """Ablación init_mode: uniform vs seed_centered."""
    rows: list[dict] = []
    for init_mode in ("uniform", "seed_centered"):
        for seed in args.seeds:
            print(f"[init={init_mode} seed={seed}] ...")
            row = _run_with_overrides(
                seed=seed,
                universe=args.universe,
                steps=args.steps,
                campaign_id="abl_init",
                max_steps=args.max_steps,
                window_length=args.window_length,
                overrides={"quantum.init_mode": init_mode},
                ablation_kind="init_mode",
                ablation_value=init_mode,
            )
            rows.append(row)
            print(
                f"  return={row['cumulative_return']:.4f} sharpe={row['sharpe_ratio']:.4f} "
                f"topm_hit={row['topm_hit_rate']:.3f}"
            )
    _write_csv(rows, args.output)


def cmd_noise(args: argparse.Namespace) -> None:
    """Ablación noise: ideal vs depolarizing con barrido de probabilidades."""
    rows: list[dict] = []
    for p in args.depolarizing:
        for seed in args.seeds:
            print(f"[depol={p:.3f} seed={seed}] ...")
            row = _run_with_overrides_noise(
                seed=seed,
                universe=args.universe,
                steps=args.steps,
                campaign_id="abl_noise",
                max_steps=args.max_steps,
                window_length=args.window_length,
                depol_prob=p,
                ablation_value=f"depol={p:.3f}",
            )
            rows.append(row)
            print(f"  return={row['cumulative_return']:.4f} latency={row['mean_latency_ms']:.2f}ms")
    _write_csv(rows, args.output)


def _run_with_overrides_noise(
    *,
    seed: int,
    universe: str,
    steps: int,
    campaign_id: str,
    max_steps: int | None,
    window_length: int | None,
    depol_prob: float,
    ablation_value: str,
) -> dict:
    """Run con override del campo noise del config quantum."""
    original_load = run_campaign.load_config

    def patched_load(cfg_path):
        cfg = original_load(cfg_path)
        if cfg.quantum is not None:
            cfg.quantum.noise = type(cfg.quantum.noise)(
                enabled=depol_prob > 0.0,
                depolarizing_prob=depol_prob,
                dephasing_prob=0.0,
            )
            cfg.quantum.backend = "pennylane" if depol_prob > 0.0 else "matrix"
        return cfg

    run_campaign.load_config = patched_load
    try:
        row = run_campaign._train_and_evaluate_one(
            model="D",
            seed=seed,
            universe=universe,
            steps=steps,
            campaign_id=campaign_id,
            max_steps_override=max_steps,
            window_override=window_length,
        )
    finally:
        run_campaign.load_config = original_load

    return _row_to_dict(row, ablation_kind="noise_depolarizing", ablation_value=ablation_value)


def cmd_param_sweep(args: argparse.Namespace, param: str) -> None:
    """Barrido genérico de un parámetro del config quantum o graph."""
    rows: list[dict] = []
    overrides_path = {
        "M": "graph.subgraph_max_size",
        "k": "quantum.k_steps",
        "m": "quantum.m_top",
    }[param]
    for value in args.values:
        for seed in args.seeds:
            print(f"[{param}={value} seed={seed}] ...")
            row = _run_with_overrides(
                seed=seed,
                universe=args.universe,
                steps=args.steps,
                campaign_id=f"abl_{param}",
                max_steps=args.max_steps,
                window_length=args.window_length,
                overrides={overrides_path: value},
                ablation_kind=param,
                ablation_value=str(value),
            )
            rows.append(row)
            print(
                f"  return={row['cumulative_return']:.4f} sharpe={row['sharpe_ratio']:.4f} "
                f"latency={row['mean_latency_ms']:.2f}ms"
            )
    _write_csv(rows, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)

    def _common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--universe", required=True)
        sp.add_argument("--seeds", type=int, nargs="+", required=True)
        sp.add_argument("--steps", type=int, default=5000)
        sp.add_argument("--max-steps", type=int, default=None)
        sp.add_argument("--window-length", type=int, default=None)
        sp.add_argument("--output", type=Path, default=None)
        sp.add_argument("--log-level", default="WARNING")

    p_init = sub.add_parser("init", help="Ablación init_mode")
    _common(p_init)

    p_noise = sub.add_parser("noise", help="Ablación de ruido")
    _common(p_noise)
    p_noise.add_argument(
        "--depolarizing",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1],
        help="Probabilidades depolarizing a probar.",
    )

    p_M = sub.add_parser("M", help="Barrido del tamaño máximo del subgrafo")
    _common(p_M)
    p_M.add_argument("--values", type=int, nargs="+", default=[4, 6, 8])

    p_k = sub.add_parser("k", help="Barrido de pasos DTQW")
    _common(p_k)
    p_k.add_argument("--values", type=int, nargs="+", default=[2, 3, 4, 5, 6])

    p_m = sub.add_parser("m", help="Barrido del tamaño top-m")
    _common(p_m)
    p_m.add_argument("--values", type=int, nargs="+", default=[1, 2, 3, 4])

    args = parser.parse_args()
    setup_logging(args.log_level)
    run_campaign._maybe_download(args.universe)
    if args.output is None:
        args.output = TABLES_DIR / f"ablation_{args.kind}.csv"
        ensure_dir(args.output.parent)

    if args.kind == "init":
        cmd_init(args)
    elif args.kind == "noise":
        cmd_noise(args)
    elif args.kind in {"M", "k", "m"}:
        cmd_param_sweep(args, args.kind)
    else:
        raise SystemExit(f"Ablación desconocida: {args.kind}")

    print(f"\n[OK] Ablación '{args.kind}' guardada en {args.output}")


if __name__ == "__main__":
    main()
