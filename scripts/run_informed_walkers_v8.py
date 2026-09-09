"""EXP-7 v8 — Selectores informados-con-rotación (compuerta G3, exploratorio).

Fase 1 (tuning): SoftmaxWalker con tau en {0.5, 1, 2}, 3 semillas, ventana
2018-2022 (train hasta 2021, prueba 2022) — tau* por candidate_hit.
Fase 2 (principal): {SoftmaxWalker(tau*), MomentumRefreshWalker,
ThompsonWalker} x 10 semillas pareadas, configuración Tabla 5.4 completa.
Comparador R: corridas existentes de ``variant_R_v6`` (mismas semillas).

Los walkers que necesitan t lo reciben por atributo, actualizado por un
wrapper del hook; la interfaz LocalModule queda intacta.

Uso::

    uv run python scripts/run_informed_walkers_v8.py
"""

from __future__ import annotations

import csv
import importlib.util
import sys as _sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.utils.logging import setup_logging
from src.utils.paths import TABLES_DIR, ensure_dir

_path = Path(__file__).resolve().parent / "run_campaign.py"
_spec = importlib.util.spec_from_file_location("run_campaign", _path)
assert _spec is not None and _spec.loader is not None
run_campaign = importlib.util.module_from_spec(_spec)
_sys.modules["run_campaign"] = run_campaign
_spec.loader.exec_module(run_campaign)

SEEDS = [42, 123, 456, 789, 1024, 7, 99, 314, 1729, 65535]
TUNE_SEEDS = [42, 123, 456]
TAUS = [0.5, 1.0, 2.0]
UNIVERSE = "nivel2"
STEPS = 50_000
OUT_TUNE = TABLES_DIR / "informed_walkers_v8_tuning.csv"
OUT_MAIN = TABLES_DIR / "informed_walkers_v8.csv"


def _make_walker(name: str, seed: int, tau: float, returns_panel: np.ndarray):
    if name == "softmax":
        from src.quantum.informed_walkers import SoftmaxWalker
        return SoftmaxWalker(seed=seed, tau=tau)
    if name == "momentum_refresh":
        from src.quantum.informed_walkers import MomentumRefreshWalker
        return MomentumRefreshWalker(returns_panel=returns_panel)
    if name == "thompson":
        from src.quantum.informed_walkers import ThompsonWalker
        return ThompsonWalker(returns_panel=returns_panel, seed=seed)
    raise ValueError(name)


def _run_one(walker_name: str, seed: int, tau: float, *,
             campaign: str, filter_dates=None, train_frac=None) -> dict:
    original_hook = run_campaign._build_hybrid_hook
    original_load = run_campaign.load_config

    def patched_load(cfg_path):
        cfg = original_load(cfg_path)
        if train_frac is not None:
            cfg.env.split.train_frac = train_frac
            cfg.env.split.val_frac = 0.01
        return cfg

    def patched_hook(cfg, env, features, classical, model):
        from src.agents.hybrid_agent import (
            HybridAgent, HybridSpec, build_returns_panel, build_ticker_index,
        )
        from src.data.sector_map import build_sector_map
        from src.graph.graph_builder import GraphSpec

        panel = build_returns_panel(features, env.tickers)
        walker = _make_walker(walker_name, seed, tau, panel)
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
        hybrid = HybridAgent(
            classical_agent=classical, local_module=walker, hybrid_spec=spec,
            returns_panel=panel, ticker_to_idx=build_ticker_index(env.tickers),
            sectors=sectors)
        inner_make = hybrid.make_mask

        def hook(env_, obs_, info_):
            if hasattr(walker, "t"):
                walker.t = int(env_.current_t)
            return inner_make(env_, obs_, info_)

        return hook

    run_campaign._build_hybrid_hook = patched_hook
    run_campaign.load_config = patched_load
    try:
        row = run_campaign._train_and_evaluate_one(
            model="D", seed=seed, universe=UNIVERSE, steps=STEPS,
            campaign_id=campaign, config_suffix="v2",
            filter_dates=filter_dates)
    finally:
        run_campaign._build_hybrid_hook = original_hook
        run_campaign.load_config = original_load
    d = asdict(row)
    d["model"] = walker_name
    d["tau"] = tau
    return d


def _load_done(path: Path, keys: tuple[str, ...]):
    rows: list[dict] = []
    done: set[tuple] = set()
    if path.exists():
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        done = {tuple(r[k] for k in keys) for r in rows}
    return rows, done


def _save(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    setup_logging("WARNING")
    run_campaign._maybe_download(UNIVERSE)
    ensure_dir(OUT_MAIN.parent)

    # --- Fase 1: tuning de tau (2018-2022; train hasta 2021, test 2022) ---
    rows_t, done_t = _load_done(OUT_TUNE, ("model", "tau", "seed"))
    for tau in TAUS:
        for seed in TUNE_SEEDS:
            if ("softmax", str(tau), str(seed)) in done_t:
                continue
            print(f"[tuning] softmax tau={tau} seed={seed} ...", flush=True)
            d = _run_one("softmax", seed, tau, campaign="v8_exp7_tuning",
                         filter_dates=("2018-01-02", "2022-12-31"),
                         train_frac=0.75)
            rows_t.append({k: str(v) for k, v in d.items()})
            print(f"  cand={float(d['candidate_hit_rate']):.4f}", flush=True)
            _save(OUT_TUNE, rows_t)
    por_tau = {}
    for r in rows_t:
        por_tau.setdefault(float(r["tau"]), []).append(
            float(r["candidate_hit_rate"]))
    tau_star = max(por_tau, key=lambda t: np.mean(por_tau[t]))
    print(f"\n[tau*] {tau_star} (cand_hit medio "
          f"{np.mean(por_tau[tau_star]):.4f})\n", flush=True)

    # --- Fase 2: principal (config completa, 10 semillas pareadas) ---
    rows_m, done_m = _load_done(OUT_MAIN, ("model", "seed"))
    walkers = [("softmax", tau_star), ("momentum_refresh", 1.0),
               ("thompson", 1.0)]
    total = len(walkers) * len(SEEDS)
    i = len(rows_m)
    for seed in SEEDS:
        for name, tau in walkers:
            if (name, str(seed)) in done_m:
                continue
            i += 1
            print(f"[{i}/{total}] {name} seed={seed} ...", flush=True)
            d = _run_one(name, seed, tau, campaign=f"v8_exp7_{name}")
            rows_m.append({k: str(v) for k, v in d.items()})
            print(f"  sharpe={float(d['sharpe_ratio']):+.4f} "
                  f"cand={float(d['candidate_hit_rate']):.4f} "
                  f"dur={float(d['duration_seconds']):.0f}s", flush=True)
            _save(OUT_MAIN, rows_m)
    print(f"\n[OK] tuning={len(rows_t)} main={len(rows_m)}", flush=True)


if __name__ == "__main__":
    main()
