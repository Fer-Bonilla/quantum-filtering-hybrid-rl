"""Campañas de variantes del módulo local (v6 — Puntos 2 y 3 del director).

Ejecuta variantes del Modelo D sustituyendo el ``LocalModule``:

* ``--variant R``  →  ``RandomWalker``: máscara aleatoria de tamaño m.
  Control para descomponer el aporte (¿la ventaja viene de la caminata
  informada o solo de restringir el espacio de acción?).
* ``--variant Q``  →  ``AnnealingWalker``: selección por QUBO + evolución
  adiabática simulada (Quantum Annealing, propuesta del director).

Todo lo demás (PPO, grafo, subgrafo, reward v2) es idéntico a las
campañas C/D, garantizando comparabilidad pareada por semilla.

Uso::

    uv run python scripts/run_variant_campaign_v6.py --variant R \\
        --universe nivel2 --seeds 42 123 456 789 1024 7 99 314 1729 65535 \\
        --steps 50000
    uv run python scripts/run_variant_campaign_v6.py --variant Q \\
        --universe nivel2 --seeds 42 123 456 --steps 50000 \\
        --update-frequency 10
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


def _make_local_module(variant: str, seed: int, args):
    if variant == "R":
        from src.quantum.random_walker import RandomWalker

        return RandomWalker(seed=seed)
    if variant == "Q":
        from src.quantum.annealing_walker import AnnealingWalker

        return AnnealingWalker(
            alpha=args.qubo_alpha,
            beta=args.qubo_beta,
            trotter_per_k=args.trotter_per_k,
            total_time=args.total_time,
        )
    raise ValueError(f"variant desconocida: {variant}")


def _run_one(
    *,
    variant: str,
    seed: int,
    universe: str,
    steps: int,
    max_steps: int | None,
    window_length: int | None,
    subgraph_M: int | None,
    update_frequency: int | None,
    args,
) -> dict:
    """Un run del modelo variante con el hook híbrido parcheado."""
    original_hook = run_campaign._build_hybrid_hook
    original_load = run_campaign.load_config

    def patched_load(cfg_path):
        cfg = original_load(cfg_path)
        if subgraph_M is not None and cfg.graph is not None:
            cfg.graph.subgraph_max_size = subgraph_M
        if update_frequency is not None and cfg.graph is not None:
            cfg.graph.update_frequency = update_frequency
        return cfg

    def patched_hook(cfg, env, features, classical, model):
        from src.agents.hybrid_agent import (
            HybridAgent,
            HybridSpec,
            build_returns_panel,
            build_ticker_index,
        )
        from src.data.sector_map import build_sector_map
        from src.graph.graph_builder import GraphSpec

        if cfg.graph is None or cfg.quantum is None:
            raise ValueError("La variante requiere graph y quantum configs.")
        local = _make_local_module(variant, seed, args)
        hybrid_spec = HybridSpec(
            graph_spec=GraphSpec(
                alpha=cfg.graph.alpha,
                beta=cfg.graph.beta,
                eps=cfg.graph.eps,
                k_neighbors=cfg.graph.k_neighbors,
                sym_mode=cfg.graph.sym_mode,
            ),
            subgraph_max_size=cfg.graph.subgraph_max_size,
            seed_score_window=cfg.graph.seed_score_window,
            graph_lookback=cfg.graph.lookback_window,
            k_steps=cfg.quantum.k_steps,
            m_top=cfg.quantum.m_top,
            update_frequency=cfg.graph.update_frequency,
        )
        sectors_map = build_sector_map(env.tickers, allow_online=False)
        sectors = (
            [sectors_map[t] for t in env.tickers] if cfg.graph.beta > 0 else None
        )
        hybrid = HybridAgent(
            classical_agent=classical,
            local_module=local,
            hybrid_spec=hybrid_spec,
            returns_panel=build_returns_panel(features, env.tickers),
            ticker_to_idx=build_ticker_index(env.tickers),
            sectors=sectors,
        )
        return hybrid.make_mask

    run_campaign._build_hybrid_hook = patched_hook
    run_campaign.load_config = patched_load
    try:
        # Config base del Modelo D (graph + quantum presentes); el
        # LocalModule se sustituye en el hook parcheado.
        row = run_campaign._train_and_evaluate_one(
            model="D",
            seed=seed,
            universe=universe,
            steps=steps,
            campaign_id=f"variant_{variant}_v6",
            max_steps_override=max_steps,
            window_override=window_length,
            config_suffix="v2",
        )
    finally:
        run_campaign._build_hybrid_hook = original_hook
        run_campaign.load_config = original_load

    d = asdict(row)
    d["model"] = variant  # renombrar para el análisis pareado
    d["variant"] = variant
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["R", "Q"])
    parser.add_argument("--universe", default="nivel2")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument("--subgraph-M", type=int, default=None)
    parser.add_argument(
        "--update-frequency", type=int, default=None,
        help="Recomputar máscara cada N steps (clave para Q con M=16).",
    )
    parser.add_argument("--qubo-alpha", type=float, default=1.0)
    parser.add_argument("--qubo-beta", type=float, default=2.0)
    parser.add_argument("--trotter-per-k", type=int, default=10)
    parser.add_argument("--total-time", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_campaign._maybe_download(args.universe)
    output = args.output or TABLES_DIR / f"variant_{args.variant}_v6.csv"
    ensure_dir(output.parent)

    rows: list[dict] = []
    for i, seed in enumerate(args.seeds, 1):
        print(f"[{i}/{len(args.seeds)}] variant={args.variant} seed={seed}", flush=True)
        row = _run_one(
            variant=args.variant,
            seed=seed,
            universe=args.universe,
            steps=args.steps,
            max_steps=args.max_steps,
            window_length=args.window_length,
            subgraph_M=args.subgraph_M,
            update_frequency=args.update_frequency,
            args=args,
        )
        rows.append(row)
        print(
            f"  sharpe={row['sharpe_ratio']:+.4f} "
            f"cand_hit={row['candidate_hit_rate']:.3f} "
            f"topm={row['topm_hit_rate']:.3f} "
            f"cov={row['asset_coverage']:.3f} "
            f"lat={row['mean_latency_ms']:.2f}ms "
            f"dur={row['duration_seconds']:.1f}s",
            flush=True,
        )

    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] {len(rows)} runs en {output}", flush=True)


if __name__ == "__main__":
    main()
