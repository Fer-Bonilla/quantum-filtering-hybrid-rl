"""Ablaciones del Cap. 8.20 sobre la configuración `sweet_spot`.

Configuración base: M=16, k=3, beta=0.5, alpha=0.5, m=5, Nivel 2.
Cada subcomando varía UN parámetro fijando el resto al sweet_spot.

Sub-comandos:
- ``noise``: depolarizing prob (Sec. 8.20.5).
- ``k``: pasos DTQW.
- ``M``: tamaño del subgrafo.
- ``init``: uniform vs seed_centered (Sec. 8.20.4).
- ``m``: tamaño top-m.

Cada ablación produce ``outputs/tables/sweet_abl_<kind>.csv`` con columnas
``ablation_kind`` y ``ablation_value``.

Uso::

    uv run python scripts/run_sweet_ablations.py noise --seeds 42 123 456 \\
        --depolarizing 0.0 0.05 --steps 20000
    uv run python scripts/run_sweet_ablations.py k --seeds 42 123 456 \\
        --values 2 3 5 --steps 20000
    uv run python scripts/run_sweet_ablations.py M --seeds 42 123 456 \\
        --values 8 16 24 --steps 20000
    uv run python scripts/run_sweet_ablations.py init --seeds 42 123 456 \\
        --steps 20000
    uv run python scripts/run_sweet_ablations.py m --seeds 42 123 456 \\
        --values 3 5 7 --steps 20000
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

# Configuración sweet_spot que se mantiene fija en TODAS las ablaciones.
SWEET_BASE = dict(
    universe="nivel2",
    M=16,
    k=3,
    alpha=0.5,
    beta=0.5,
    m=5,
    k_neighbors=10,
    init_mode="uniform",
    noise_depol=0.0,
)


def _apply_sweet_overrides(cfg, *, vary: dict | None = None):
    """Aplicar sweet_spot a un config + overrides puntuales de la ablación."""
    base = dict(SWEET_BASE)
    if vary is not None:
        base.update(vary)

    cfg.data.cache_subdir = base["universe"]
    if cfg.graph is not None:
        cfg.graph.subgraph_max_size = base["M"]
        cfg.graph.alpha = base["alpha"]
        cfg.graph.beta = base["beta"]
        cfg.graph.k_neighbors = base["k_neighbors"]
    if cfg.quantum is not None:
        cfg.quantum.k_steps = base["k"]
        cfg.quantum.m_top = base["m"]
        cfg.quantum.init_mode = base["init_mode"]
        # Configurar ruido (sólo si depol_prob > 0)
        depol = base["noise_depol"]
        cfg.quantum.noise = type(cfg.quantum.noise)(
            enabled=depol > 0.0,
            depolarizing_prob=depol,
            dephasing_prob=0.0,
        )
        cfg.quantum.backend = "pennylane" if depol > 0.0 else "matrix"
    return cfg


def _run_single(
    *,
    ablation_kind: str,
    ablation_value: str,
    vary: dict,
    seed: int,
    steps: int,
    max_steps: int | None,
    window_length: int | None,
    model: str = "D",
) -> dict:
    """Ejecutar un run con sweet_spot + overrides específicos."""
    original_load = run_campaign.load_config

    def patched_load(cfg_path):
        cfg = original_load(cfg_path)
        return _apply_sweet_overrides(cfg, vary=vary)

    run_campaign.load_config = patched_load
    try:
        row = run_campaign._train_and_evaluate_one(
            model=model,
            seed=seed,
            universe=SWEET_BASE["universe"],
            steps=steps,
            campaign_id=f"sweet_abl_{ablation_kind}",
            max_steps_override=max_steps,
            window_override=window_length,
        )
    finally:
        run_campaign.load_config = original_load

    d = asdict(row)
    d["ablation_kind"] = ablation_kind
    d["ablation_value"] = ablation_value
    return d


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        _log.warning("Sin filas para %s", path)
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cmd_noise(args) -> None:
    rows: list[dict] = []
    for depol in args.depolarizing:
        for seed in args.seeds:
            print(f"[noise depol={depol:.3f} seed={seed}] ...", flush=True)
            row = _run_single(
                ablation_kind="noise_depol",
                ablation_value=f"{depol:.3f}",
                vary={"noise_depol": depol},
                seed=seed,
                steps=args.steps,
                max_steps=args.max_steps,
                window_length=args.window_length,
            )
            rows.append(row)
            print(
                f"  return={row['cumulative_return']:.4f} sharpe={row['sharpe_ratio']:.4f} "
                f"topm_hit={row['topm_hit_rate']:.3f} latency={row['mean_latency_ms']:.2f}ms "
                f"duration={row['duration_seconds']:.1f}s",
                flush=True,
            )
    _write_csv(rows, args.output)


def cmd_axis(args, axis: str) -> None:
    """Ablación genérica de un eje (k, M, m)."""
    rows: list[dict] = []
    for value in args.values:
        for seed in args.seeds:
            print(f"[{axis}={value} seed={seed}] ...", flush=True)
            row = _run_single(
                ablation_kind=axis,
                ablation_value=str(value),
                vary={axis: value},
                seed=seed,
                steps=args.steps,
                max_steps=args.max_steps,
                window_length=args.window_length,
            )
            rows.append(row)
            print(
                f"  return={row['cumulative_return']:.4f} sharpe={row['sharpe_ratio']:.4f} "
                f"topm_hit={row['topm_hit_rate']:.3f} duration={row['duration_seconds']:.1f}s",
                flush=True,
            )
    _write_csv(rows, args.output)


def cmd_init(args) -> None:
    rows: list[dict] = []
    for mode in ("uniform", "seed_centered"):
        for seed in args.seeds:
            print(f"[init_mode={mode} seed={seed}] ...", flush=True)
            row = _run_single(
                ablation_kind="init_mode",
                ablation_value=mode,
                vary={"init_mode": mode},
                seed=seed,
                steps=args.steps,
                max_steps=args.max_steps,
                window_length=args.window_length,
            )
            rows.append(row)
            print(
                f"  return={row['cumulative_return']:.4f} sharpe={row['sharpe_ratio']:.4f} "
                f"topm_hit={row['topm_hit_rate']:.3f} duration={row['duration_seconds']:.1f}s",
                flush=True,
            )
    _write_csv(rows, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)

    def _common(sp):
        sp.add_argument("--seeds", type=int, nargs="+", required=True)
        sp.add_argument("--steps", type=int, default=20000)
        sp.add_argument("--max-steps", type=int, default=None)
        sp.add_argument("--window-length", type=int, default=None)
        sp.add_argument("--output", type=Path, default=None)
        sp.add_argument("--log-level", default="WARNING")

    p_noise = sub.add_parser("noise", help="Ablación Sec. 8.20.5 (depolarizing)")
    _common(p_noise)
    p_noise.add_argument("--depolarizing", type=float, nargs="+", default=[0.0, 0.05])

    p_k = sub.add_parser("k", help="Barrido pasos DTQW")
    _common(p_k)
    p_k.add_argument("--values", type=int, nargs="+", default=[2, 3, 5])

    p_M = sub.add_parser("M", help="Barrido tamaño subgrafo")
    _common(p_M)
    p_M.add_argument("--values", type=int, nargs="+", default=[8, 16, 24])

    p_init = sub.add_parser("init", help="Ablación init_mode (Sec. 8.20.4)")
    _common(p_init)

    p_m = sub.add_parser("m", help="Barrido top-m")
    _common(p_m)
    p_m.add_argument("--values", type=int, nargs="+", default=[3, 5, 7])

    args = parser.parse_args()
    setup_logging(args.log_level)
    run_campaign._maybe_download(SWEET_BASE["universe"])
    if args.output is None:
        args.output = TABLES_DIR / f"sweet_abl_{args.kind}.csv"
        ensure_dir(args.output.parent)

    if args.kind == "noise":
        cmd_noise(args)
    elif args.kind == "k":
        cmd_axis(args, "k")
    elif args.kind == "M":
        cmd_axis(args, "M")
        # Renombrar output a Msize para evitar colisión case-insensitive en Windows.
        new_path = args.output.with_name(args.output.name.replace("sweet_abl_M", "sweet_abl_Msize"))
        if args.output.exists() and args.output != new_path:
            args.output.rename(new_path)
            args.output = new_path
    elif args.kind == "m":
        cmd_axis(args, "m")
        # Análogamente: mtop para diferenciar de Msize.
        new_path = args.output.with_name(args.output.name.replace("sweet_abl_m", "sweet_abl_mtop"))
        if args.output.exists() and args.output != new_path:
            args.output.rename(new_path)
            args.output = new_path
    elif args.kind == "init":
        cmd_init(args)
    else:
        raise SystemExit(f"Ablación desconocida: {args.kind}")

    print(f"\n[OK] Ablación '{args.kind}' escrita en {args.output}", flush=True)


if __name__ == "__main__":
    main()
