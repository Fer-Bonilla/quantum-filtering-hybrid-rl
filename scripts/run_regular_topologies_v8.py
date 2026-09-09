"""EXP-5 v8 — Subgrafos de regularidad controlada (H-v8.4, compuerta G2).

Entrena {C, D, R} sobre cuatro topologías recableadas en los mismos top-M
nodos del BFS (d-regular con pesos de afinidad, d-regular uniforme, ciclo C_M
y bipartito completo K_{a,b}), con las 10 semillas pareadas y configuración
Tabla 5.4 (M=8, d=3). El recableo se hace vía ``TopologyWrapper`` sin tocar
entorno, grafo, política ni evaluación (regla 2 del pre-registro v8).

Reanudación idempotente por (topología, selector, semilla).

Uso::

    uv run python scripts/run_regular_topologies_v8.py
"""

from __future__ import annotations

import csv
import importlib.util
import sys as _sys
from dataclasses import asdict
from pathlib import Path

from src.utils.logging import setup_logging
from src.utils.paths import TABLES_DIR, ensure_dir

_path = Path(__file__).resolve().parent / "run_campaign.py"
_spec = importlib.util.spec_from_file_location("run_campaign", _path)
assert _spec is not None and _spec.loader is not None
run_campaign = importlib.util.module_from_spec(_spec)
_sys.modules["run_campaign"] = run_campaign
_spec.loader.exec_module(run_campaign)

SEEDS = [42, 123, 456, 789, 1024, 7, 99, 314, 1729, 65535]
TOPOLOGIES = ["dreg_aff", "dreg_uni", "cycle", "bipartite"]
SELECTORS = ["C", "D", "R"]
UNIVERSE = "nivel2"
STEPS = 50_000
D_REG = 3  # M=8 (Tabla 5.4)
OUTPUT = TABLES_DIR / "regular_topologies_v8.csv"


def _make_inner(selector: str, seed: int, cfg):
    if selector == "C":
        from src.quantum.classical_walker import ClassicalWalker
        return ClassicalWalker()
    if selector == "D":
        from src.quantum.noise import NoiseSpec
        from src.quantum.quantum_walker import QuantumWalker
        return QuantumWalker(
            init_mode=cfg.quantum.init_mode,
            renormalize_threshold=cfg.quantum.renormalize_threshold,
            backend=cfg.quantum.backend, noise=NoiseSpec())
    if selector == "R":
        from src.quantum.random_walker import RandomWalker
        return RandomWalker(seed=seed)
    raise ValueError(selector)


def _run_one(topology: str, selector: str, seed: int) -> dict:
    original_hook = run_campaign._build_hybrid_hook

    def patched_hook(cfg, env, features, classical, model):
        from src.agents.hybrid_agent import (
            HybridAgent, HybridSpec, build_returns_panel, build_ticker_index,
        )
        from src.data.sector_map import build_sector_map
        from src.graph.graph_builder import GraphSpec
        from src.graph.regular_subgraphs import TopologyWrapper

        sectors_map = build_sector_map(env.tickers, allow_online=False)
        sectors_all = [sectors_map[t] for t in env.tickers]
        inner = _make_inner(selector, seed, cfg)
        local = TopologyWrapper(inner=inner, kind=topology, d=D_REG,
                                rng_seed=seed, sectors=sectors_all)
        spec = HybridSpec(
            graph_spec=GraphSpec(
                alpha=cfg.graph.alpha, beta=cfg.graph.beta, eps=cfg.graph.eps,
                k_neighbors=cfg.graph.k_neighbors, sym_mode=cfg.graph.sym_mode),
            subgraph_max_size=cfg.graph.subgraph_max_size,
            seed_score_window=cfg.graph.seed_score_window,
            graph_lookback=cfg.graph.lookback_window,
            k_steps=cfg.quantum.k_steps, m_top=cfg.quantum.m_top,
            update_frequency=cfg.graph.update_frequency)
        sectors_cfg = sectors_all if cfg.graph.beta > 0 else None
        return HybridAgent(
            classical_agent=classical, local_module=local, hybrid_spec=spec,
            returns_panel=build_returns_panel(features, env.tickers),
            ticker_to_idx=build_ticker_index(env.tickers),
            sectors=sectors_cfg).make_mask

    run_campaign._build_hybrid_hook = patched_hook
    try:
        row = run_campaign._train_and_evaluate_one(
            model="D", seed=seed, universe=UNIVERSE, steps=STEPS,
            campaign_id=f"v8_exp5_{topology}", config_suffix="v2")
    finally:
        run_campaign._build_hybrid_hook = original_hook
    d = asdict(row)
    d["model"] = selector
    d["topology"] = topology
    return d


def main() -> None:
    setup_logging("WARNING")
    run_campaign._maybe_download(UNIVERSE)
    ensure_dir(OUTPUT.parent)

    rows: list[dict] = []
    done: set[tuple] = set()
    if OUTPUT.exists():
        rows = list(csv.DictReader(OUTPUT.open(encoding="utf-8")))
        done = {(r["topology"], r["model"], str(r["seed"])) for r in rows}
        print(f"[resume] {len(rows)} filas ya presentes.", flush=True)

    def save() -> None:
        with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    total = len(TOPOLOGIES) * len(SELECTORS) * len(SEEDS)
    i = len(rows)
    for seed in SEEDS:                      # semilla-exterior: pareo robusto
        for topology in TOPOLOGIES:
            for selector in SELECTORS:
                if (topology, selector, str(seed)) in done:
                    continue
                i += 1
                print(f"[{i}/{total}] {topology}/{selector} seed={seed} ...",
                      flush=True)
                d = _run_one(topology, selector, seed)
                rows.append({k: str(v) for k, v in d.items()})
                print(f"  sharpe={float(d['sharpe_ratio']):+.4f} "
                      f"cand={float(d['candidate_hit_rate']):.4f} "
                      f"dur={float(d['duration_seconds']):.0f}s", flush=True)
                save()
    print(f"\n[OK] {len(rows)} filas en {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
