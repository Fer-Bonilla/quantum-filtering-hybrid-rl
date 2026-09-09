"""EXP-3 v8 — Extensión de semillas para el TOST R–D (H-v8.1).

Ejecuta las variantes R (RandomWalker vía hook parcheado) y D (DTQW) con las
30 semillas nuevas del pre-registro (docs/preregistro_v8.md, commit 0e3ee5e),
configuración idéntica a la Tabla 5.4 (nivel2, M=8, k=3, m=3, log_wealth,
50k pasos). Los 10 pares existentes (variant_R_v6 + campaign_1_v3) se
reutilizan en el análisis, como declara el pre-registro.

Reanudación idempotente por (model, seed); escritura incremental.

Uso::

    uv run python scripts/run_tost_extension_v8.py
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

# Semillas nuevas del pre-registro v8 (manifiesto para Tabla C.1).
NEW_SEEDS = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
             71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139]
UNIVERSE = "nivel2"
STEPS = 50_000
OUTPUT = TABLES_DIR / "tost_extension_v8.csv"


def _run_r(seed: int) -> dict:
    """Variante R: modelo D con el LocalModule sustituido por RandomWalker."""
    original_hook = run_campaign._build_hybrid_hook

    def patched_hook(cfg, env, features, classical, model):
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

    run_campaign._build_hybrid_hook = patched_hook
    try:
        row = run_campaign._train_and_evaluate_one(
            model="D", seed=seed, universe=UNIVERSE, steps=STEPS,
            campaign_id="v8_exp3", config_suffix="v2")
    finally:
        run_campaign._build_hybrid_hook = original_hook
    d = asdict(row)
    d["model"] = "R"
    return d


def _run_d(seed: int) -> dict:
    row = run_campaign._train_and_evaluate_one(
        model="D", seed=seed, universe=UNIVERSE, steps=STEPS,
        campaign_id="v8_exp3", config_suffix="v2")
    return asdict(row)


def main() -> None:
    setup_logging("WARNING")
    run_campaign._maybe_download(UNIVERSE)
    ensure_dir(OUTPUT.parent)

    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if OUTPUT.exists():
        rows = list(csv.DictReader(OUTPUT.open(encoding="utf-8")))
        done = {(r["model"], str(r["seed"])) for r in rows}
        print(f"[resume] {len(rows)} filas ya presentes.", flush=True)

    def save() -> None:
        with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    total = len(NEW_SEEDS) * 2
    i = len(rows)
    for seed in NEW_SEEDS:
        for model, runner in (("R", _run_r), ("D", _run_d)):
            if (model, str(seed)) in done:
                continue
            i += 1
            print(f"[{i}/{total}] {model} seed={seed} ...", flush=True)
            d = runner(seed)
            d["model"] = model
            rows.append({k: str(v) for k, v in d.items()})
            print(f"  sharpe={float(d['sharpe_ratio']):+.4f} "
                  f"cand={float(d['candidate_hit_rate']):.4f} "
                  f"dur={float(d['duration_seconds']):.0f}s", flush=True)
            save()
    print(f"\n[OK] {len(rows)} filas en {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
