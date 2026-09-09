"""Comprobaciones de la 3ª revisión académica (v8).

Fase 1 — Condición inicial emparejada (hallazgo 1): Modelo D con
``init_mode=seed_centered`` (misma condición inicial que la caminata clásica),
10 semillas pareadas. Permite el contraste C–D con dinámica como ÚNICO factor.

Fase 2 — Sensibilidad al coste de transacción (hallazgo 3): {C, D, R} con
penalización efectiva mu_c*kappa en {5e-4, 1e-3} (5 y 10 puntos básicos por
cambio, frente a los 5e-7 de la configuración base), 10 semillas pareadas.

Reanudable por (variant, cost_level, seed). Salida: review3_checks_v8.csv.

Uso::

    uv run python scripts/run_review3_checks_v8.py
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
UNIVERSE = "nivel2"
STEPS = 50_000
COST_LEVELS = {"base": None, "5bp": 0.5, "10bp": 1.0}  # transaction_cost => mu_c*k
OUTPUT = TABLES_DIR / "review3_checks_v8.csv"


def _run_one(variant: str, seed: int, *, init_mode: str | None = None,
             transaction_cost: float | None = None, tag: str = "") -> dict:
    original_hook = run_campaign._build_hybrid_hook
    original_load = run_campaign.load_config

    def patched_load(cfg_path):
        cfg = original_load(cfg_path)
        if init_mode is not None and cfg.quantum is not None:
            cfg.quantum.init_mode = init_mode
        if transaction_cost is not None:
            cfg.env.transaction_cost = transaction_cost
        return cfg

    def patched_hook(cfg, env, features, classical, model):
        if variant != "R":
            return original_hook(cfg, env, features, classical, model)
        from src.agents.hybrid_agent import (
            HybridAgent, HybridSpec, build_returns_panel, build_ticker_index,
        )
        from src.data.sector_map import build_sector_map
        from src.graph.graph_builder import GraphSpec
        from src.quantum.random_walker import RandomWalker

        spec = HybridSpec(
            graph_spec=GraphSpec(
                alpha=cfg.graph.alpha, beta=cfg.graph.beta, eps=cfg.graph.eps,
                k_neighbors=cfg.graph.k_neighbors, sym_mode=cfg.graph.sym_mode),
            subgraph_max_size=cfg.graph.subgraph_max_size,
            seed_score_window=cfg.graph.seed_score_window,
            graph_lookback=cfg.graph.lookback_window,
            k_steps=cfg.quantum.k_steps, m_top=cfg.quantum.m_top,
            update_frequency=cfg.graph.update_frequency)
        sectors_map = build_sector_map(env.tickers, allow_online=False)
        sectors = [sectors_map[t] for t in env.tickers] if cfg.graph.beta > 0 else None
        return HybridAgent(
            classical_agent=classical, local_module=RandomWalker(seed=seed),
            hybrid_spec=spec, returns_panel=build_returns_panel(features, env.tickers),
            ticker_to_idx=build_ticker_index(env.tickers), sectors=sectors).make_mask

    base_model = "D" if variant == "R" else variant
    run_campaign._build_hybrid_hook = patched_hook
    run_campaign.load_config = patched_load
    try:
        row = run_campaign._train_and_evaluate_one(
            model=base_model, seed=seed, universe=UNIVERSE, steps=STEPS,
            campaign_id=f"v8_review3_{tag}", config_suffix="v2")
    finally:
        run_campaign._build_hybrid_hook = original_hook
        run_campaign.load_config = original_load
    d = asdict(row)
    d["model"] = variant
    d["tag"] = tag
    return d


def main() -> None:
    setup_logging("WARNING")
    run_campaign._maybe_download(UNIVERSE)
    ensure_dir(OUTPUT.parent)
    rows: list[dict] = []
    done: set[tuple] = set()
    if OUTPUT.exists():
        rows = list(csv.DictReader(OUTPUT.open(encoding="utf-8")))
        done = {(r["tag"], str(r["seed"])) for r in rows}
        print(f"[resume] {len(rows)} filas.", flush=True)

    def save() -> None:
        with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    trabajos: list[tuple[str, dict]] = []
    # Fase 1: D con inicializacion centrada en la semilla
    for seed in SEEDS:
        trabajos.append((f"Dcentered", dict(variant="D", seed=seed,
                                            init_mode="seed_centered")))
    # Fase 2: sensibilidad al coste (C, D, R) x {5bp, 10bp}
    for level, tc in COST_LEVELS.items():
        if tc is None:
            continue
        for variant in ("C", "D", "R"):
            for seed in SEEDS:
                trabajos.append((f"{variant}_cost{level}",
                                 dict(variant=variant, seed=seed,
                                      transaction_cost=tc)))

    total = len(trabajos)
    for i, (tag, kw) in enumerate(trabajos, 1):
        if (tag, str(kw["seed"])) in done:
            continue
        print(f"[{i}/{total}] {tag} seed={kw['seed']} ...", flush=True)
        d = _run_one(tag=tag, **kw)
        rows.append({k: str(v) for k, v in d.items()})
        print(f"  sharpe={float(d['sharpe_ratio']):+.4f} "
              f"cand={float(d['candidate_hit_rate']):.4f} "
              f"dur={float(d['duration_seconds']):.0f}s", flush=True)
        save()
    print(f"\n[OK] {len(rows)} filas en {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
