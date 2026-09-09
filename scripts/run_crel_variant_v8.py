"""Variante Crel (v8, hallazgo 1 de la 2ª revisión académica).

Crel = estado relacional del Modelo B (rasgos de G_t concatenados al estado)
+ filtrado local del Modelo C (caminata clásica sobre H_t). Con ella, la
cadena de ablación gana dos contrastes de UN solo factor:

  - B vs Crel  : aísla el filtrado local (estado idéntico).
  - Crel vs C  : aísla los rasgos relacionales del estado (filtrado idéntico).

Configuración: idéntica a la Tabla 5.4 (nivel2, 50k pasos, 10 semillas
pareadas de campaign_1_v3). Reanudable.

Uso::

    uv run python scripts/run_crel_variant_v8.py
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
OUTPUT = TABLES_DIR / "crel_variant_v8.csv"


def _run_one(seed: int) -> dict:
    original_hook = run_campaign._build_hybrid_hook
    original_load = run_campaign.load_config

    def patched_load(cfg_path):
        cfg = original_load(cfg_path)
        # Modelo B v2 no trae bloque quantum: inyectar k/m de la Tabla 5.4.
        if cfg.quantum is None:
            from src.utils.config import QuantumConfig
            cfg.quantum = QuantumConfig()
        return cfg

    def patched_hook(cfg, env, features, classical, model):
        # Para el modelo B, run_campaign no crea hook; Crel añade el de C.
        from src.agents.hybrid_agent import (
            HybridAgent, HybridSpec, build_returns_panel, build_ticker_index,
        )
        from src.data.sector_map import build_sector_map
        from src.graph.graph_builder import GraphSpec
        from src.quantum.classical_walker import ClassicalWalker

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
            classical_agent=classical, local_module=ClassicalWalker(),
            hybrid_spec=spec,
            returns_panel=build_returns_panel(features, env.tickers),
            ticker_to_idx=build_ticker_index(env.tickers),
            sectors=sectors).make_mask

    run_campaign._build_hybrid_hook = patched_hook
    run_campaign.load_config = patched_load
    try:
        row = run_campaign._train_and_evaluate_one(
            model="B", seed=seed, universe=UNIVERSE, steps=STEPS,
            campaign_id="v8_crel", config_suffix="v2")
    finally:
        run_campaign._build_hybrid_hook = original_hook
        run_campaign.load_config = original_load
    d = asdict(row)
    d["model"] = "Crel"
    return d


def main() -> None:
    setup_logging("WARNING")
    run_campaign._maybe_download(UNIVERSE)
    ensure_dir(OUTPUT.parent)
    rows: list[dict] = []
    done: set[str] = set()
    if OUTPUT.exists():
        rows = list(csv.DictReader(OUTPUT.open(encoding="utf-8")))
        done = {str(r["seed"]) for r in rows}
        print(f"[resume] {len(rows)} filas.", flush=True)

    def save() -> None:
        with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    for i, seed in enumerate(SEEDS, 1):
        if str(seed) in done:
            continue
        print(f"[{i}/{len(SEEDS)}] Crel seed={seed} ...", flush=True)
        d = _run_one(seed)
        rows.append({k: str(v) for k, v in d.items()})
        print(f"  sharpe={float(d['sharpe_ratio']):+.4f} "
              f"cand={float(d['candidate_hit_rate']):.4f} "
              f"dur={float(d['duration_seconds']):.0f}s", flush=True)
        save()
    print(f"\n[OK] {len(rows)} filas en {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
