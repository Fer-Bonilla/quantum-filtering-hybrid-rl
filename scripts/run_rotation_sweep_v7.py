"""Barrido dosis-respuesta de rotación de máscara (v7 — experimento 1).

Manipula causalmente el mecanismo identificado en v6 (Punto 2). El selector es
``StickyWalker``: máscara aleatoria pegajosa que re-sortea cada ranura con
probabilidad ``rotation_p``. La INFORMACIÓN se mantiene en cero (elecciones
uniformes) en todo el barrido; solo varía la ROTACIÓN:

    rotation_p = 0.0  →  máscara congelada (rotación mínima; análogo limpio del
                          determinismo del Modelo Q, sin su información).
    rotation_p = 1.0  →  re-sorteo total ≡ Modelo R (máscara aleatoria).

Si ``candidate_hit_rate`` crece monótonamente con ``rotation_p``, el mecanismo
"rotación de la máscara" queda demostrado de forma causal (curva dosis-respuesta)
y no solo inferido del gradiente R>D>C>Q.

Todo lo demás (PPO, grafo, subgrafo, reward v2, M=8, m=3, update_frequency=1) es
idéntico a la campaña del Modelo R, garantizando comparabilidad pareada por
semilla. El nivel ``p=1.0`` debe reproducir (modulo RNG) los resultados de R.

Uso::

    uv run python scripts/run_rotation_sweep_v7.py \\
        --universe nivel2 --steps 50000 \\
        --p-values 0.0 0.1 0.25 0.5 0.75 1.0 \\
        --seeds 42 123 456 789 1024 7 99 314 1729 65535
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
    rotation_p: float,
    seed: int,
    universe: str,
    steps: int,
) -> dict:
    """Un run del Modelo S(p) con el hook híbrido parcheado."""
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
        from src.quantum.sticky_walker import StickyWalker

        if cfg.graph is None or cfg.quantum is None:
            raise ValueError("El barrido requiere graph y quantum configs.")
        local = StickyWalker(seed=seed, rotation_p=rotation_p)
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
    try:
        # Config base del Modelo D (graph + quantum presentes); el
        # LocalModule se sustituye en el hook parcheado.
        row = run_campaign._train_and_evaluate_one(
            model="D",
            seed=seed,
            universe=universe,
            steps=steps,
            campaign_id=f"rotation_p{rotation_p:.2f}_v7",
            config_suffix="v2",
        )
    finally:
        run_campaign._build_hybrid_hook = original_hook

    d = asdict(row)
    d["model"] = f"S_p{rotation_p:.2f}"  # nivel distinto por p para el análisis
    d["rotation_p"] = rotation_p
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="nivel2")
    parser.add_argument(
        "--p-values", type=float, nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[42, 123, 456, 789, 1024, 7, 99, 314, 1729, 65535],
    )
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_campaign._maybe_download(args.universe)
    output = args.output or TABLES_DIR / "rotation_sweep_v7.csv"
    ensure_dir(output.parent)

    # Reanudación: si el CSV ya existe, conservar los runs hechos y saltarlos.
    # Cada (seed, p) es independiente (re-siembra el RNG global), así que el
    # barrido es idempotente y robusto a interrupciones (p. ej. suspensión).
    rows: list[dict] = []
    done: set[tuple[int, str]] = set()
    if output.exists():
        existing = list(csv.DictReader(output.open("r", encoding="utf-8")))
        rows.extend(existing)
        for r in existing:
            done.add((int(r["seed"]), f"{float(r['rotation_p']):.2f}"))
        print(f"[resume] {len(existing)} runs ya presentes en {output}", flush=True)

    total = len(args.p_values) * len(args.seeds)
    i = 0
    # Bucle semilla-externo, p-interno: cada semilla completa su curva
    # dosis-respuesta entera antes de pasar a la siguiente, de modo que los
    # resultados parciales (escritura incremental) ya contienen curvas
    # completas para las semillas terminadas. El orden no afecta a ningún run
    # individual (cada run re-siembra el RNG global en _train_and_evaluate_one).
    for seed in args.seeds:
        for p in args.p_values:
            i += 1
            if (seed, f"{p:.2f}") in done:
                print(f"[{i}/{total}] seed={seed} rotation_p={p:.2f} (ya hecho, skip)", flush=True)
                continue
            print(f"[{i}/{total}] seed={seed} rotation_p={p:.2f}", flush=True)
            row = _run_one(
                rotation_p=p,
                seed=seed,
                universe=args.universe,
                steps=args.steps,
            )
            rows.append(row)
            print(
                f"  sharpe={row['sharpe_ratio']:+.4f} "
                f"cand_hit={row['candidate_hit_rate']:.3f} "
                f"topm={row['topm_hit_rate']:.3f} "
                f"cov={row['asset_coverage']:.3f} "
                f"conv={row['episodes_to_convergence']:.1f} "
                f"dur={row['duration_seconds']:.1f}s",
                flush=True,
            )
            # Escritura incremental para no perder progreso en barridos largos.
            with output.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    print(f"\n[OK] {len(rows)} runs en {output}", flush=True)


if __name__ == "__main__":
    main()
