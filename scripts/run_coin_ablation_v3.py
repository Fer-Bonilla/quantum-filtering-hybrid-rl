"""Ablacion empirica de la moneda DTQW (rev. v3 - Punto 3.2 del revisor).

Ejecuta el Modelo D bajo tres elecciones de moneda:
  - weighted_householder (default, sweet_spot): C_{i,t} = 2|w_i><w_i| - I
    con |w_i> ponderado por afinidades del grafo.
  - grover: C_{i,t} = 2|u><u| - I con |u> uniforme (ignora pesos).
    Aisla el efecto de la ponderacion por afinidad.
  - fourier: C_{i,t} = F_{d_i} (DFT normalizada). Difusion uniforme
    entre puertos sin preferencia direccional ni reflexion.

Sweet_spot v2 (M=16, beta=0.5, k=3, m=5), reward log_wealth, lambda=0.

Uso::

    uv run python scripts/run_coin_ablation_v3.py --universe nivel2 \\
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


def _run_one(
    *,
    coin_type: str,
    seed: int,
    universe: str,
    steps: int,
    max_steps: int | None,
    window_length: int | None,
) -> dict:
    """Ejecuta un run Modelo D con un coin_type especifico."""
    # Monkey-patch al builder de QuantumWalker dentro de run_campaign
    # para inyectar coin_type sin tocar las configs (que no exponen el campo).
    from src.quantum import quantum_walker as qw

    original_init = qw.QuantumWalker.__init__

    def patched_init(self, *args, **kwargs):
        # Forzar coin_type del ablation; los demas argumentos pasan tal cual.
        kwargs.setdefault("coin_type", coin_type)
        kwargs["coin_type"] = coin_type
        original_init(self, *args, **kwargs)

    # Necesitamos un dataclass mutable workaround: usamos object.__setattr__
    # Pero QuantumWalker es frozen. La forma es modificar la fabrica.
    # Solucion: monkey-patch del helper _build_hybrid_hook dentro de
    # run_campaign para que pase coin_type a QuantumWalker.
    original_hook = run_campaign._build_hybrid_hook

    def patched_hook(cfg, env, features, classical, model):
        from src.agents.hybrid_agent import (
            HybridAgent,
            HybridSpec,
            build_returns_panel,
            build_ticker_index,
        )
        from src.data.sector_map import build_sector_map
        from src.graph.graph_builder import GraphSpec
        from src.quantum.classical_walker import ClassicalWalker
        from src.quantum.noise import NoiseSpec
        from src.quantum.quantum_walker import QuantumWalker

        if model in {"A", "B"}:
            return None
        if cfg.graph is None or cfg.quantum is None:
            raise ValueError(f"Modelo {model} requiere graph y quantum configs.")
        if model == "C":
            local = ClassicalWalker()
        else:
            noise_spec = (
                NoiseSpec(
                    depolarizing_prob=cfg.quantum.noise.depolarizing_prob,
                    dephasing_prob=cfg.quantum.noise.dephasing_prob,
                )
                if cfg.quantum.noise.enabled
                else NoiseSpec()
            )
            local = QuantumWalker(
                init_mode=cfg.quantum.init_mode,
                renormalize_threshold=cfg.quantum.renormalize_threshold,
                backend=cfg.quantum.backend,
                noise=noise_spec,
                coin_type=coin_type,  # <-- la clave
            )
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

    # Override config para sweet_spot v2
    original_load = run_campaign.load_config

    def patched_load(cfg_path):
        cfg = original_load(cfg_path)
        if cfg.graph is not None:
            cfg.graph.subgraph_max_size = 16
            cfg.graph.alpha = 0.5
            cfg.graph.beta = 0.5
        if cfg.quantum is not None:
            cfg.quantum.k_steps = 3
            cfg.quantum.m_top = 5
        return cfg

    run_campaign.load_config = patched_load
    try:
        row = run_campaign._train_and_evaluate_one(
            model="D",
            seed=seed,
            universe=universe,
            steps=steps,
            campaign_id=f"abl_coin_{coin_type}",
            max_steps_override=max_steps,
            window_override=window_length,
            config_suffix="v2",
        )
    finally:
        run_campaign.load_config = original_load
        run_campaign._build_hybrid_hook = original_hook

    d = asdict(row)
    d["ablation_kind"] = "coin"
    d["ablation_value"] = coin_type
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--window-length", type=int, default=20)
    parser.add_argument(
        "--coins",
        nargs="+",
        choices=["weighted_householder", "grover", "fourier"],
        default=["weighted_householder", "grover", "fourier"],
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_campaign._maybe_download(args.universe)
    if args.output is None:
        args.output = TABLES_DIR / "abl_coin_v3.csv"
        ensure_dir(args.output.parent)

    rows: list[dict] = []
    total = len(args.coins) * len(args.seeds)
    idx = 0
    for coin in args.coins:
        for seed in args.seeds:
            idx += 1
            print(f"[{idx}/{total}] coin={coin} seed={seed}", flush=True)
            row = _run_one(
                coin_type=coin,
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
