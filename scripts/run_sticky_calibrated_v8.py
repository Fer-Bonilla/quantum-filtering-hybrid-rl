"""EXP-4 v8 — Control de rotación calibrada S(p*) (H-v8.2).

1. Lee la rotación realizada de D por semilla (``mask_metrics_v8.csv``, EXP-1).
2. Invierte la curva p→rotación del StickyWalker medida en v7 sobre la misma
   secuencia de evaluación (``mediation_rotation_v7.csv``) para obtener p*.
   Regla pre-registrada: p* global si sd_semillas(rot_D) <= 0,05.
3. Entrena S(p*) con las 10 semillas pareadas de ``campaign_1_v3``
   (configuración Tabla 5.4), reanudable, salida ``sticky_calibrated_v8.csv``.

Uso::

    uv run python scripts/run_sticky_calibrated_v8.py
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

# Ampliación post-G1 (declarada): las 10 semillas pareadas originales del
# pre-registro + las 30 nuevas del EXP-3, cuyos brazos D ya existen en
# tost_extension_v8.csv => n=40 pares para H-v8.2 (cierra la potencia al
# umbral corregido; el pre-registro fijaba n=10).
SEEDS = [42, 123, 456, 789, 1024, 7, 99, 314, 1729, 65535,
         11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
         71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139]
UNIVERSE = "nivel2"
STEPS = 50_000
OUTPUT = TABLES_DIR / "sticky_calibrated_v8.csv"


def _p_star() -> tuple[float, float, float]:
    """(p*, rot_D_media, sd_semillas) según el pre-registro."""
    with (TABLES_DIR / "mask_metrics_v8.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rot_d = np.array([float(r["rotation_realized"]) for r in rows
                      if r["selector"] == "D"])
    sd = float(np.std(rot_d, ddof=1))
    target = float(np.mean(rot_d))

    with (TABLES_DIR / "mediation_rotation_v7.csv").open(encoding="utf-8") as fh:
        med = [r for r in csv.DictReader(fh) if r["kind"] == "sticky"]
    ps = np.array([float(r["selector"].split("=")[1].rstrip(")")) for r in med])
    rots = np.array([float(r["rotation"]) for r in med])
    order = np.argsort(rots)
    p_star = float(np.interp(target, rots[order], ps[order]))
    return p_star, target, sd


def _run_sticky(seed: int, p_star: float) -> dict:
    original_hook = run_campaign._build_hybrid_hook

    def patched_hook(cfg, env, features, classical, model):
        from src.agents.hybrid_agent import (
            HybridAgent, HybridSpec, build_returns_panel, build_ticker_index,
        )
        from src.data.sector_map import build_sector_map
        from src.graph.graph_builder import GraphSpec
        from src.quantum.sticky_walker import StickyWalker

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
            classical_agent=classical,
            local_module=StickyWalker(seed=seed, rotation_p=p_star),
            hybrid_spec=spec,
            returns_panel=build_returns_panel(features, env.tickers),
            ticker_to_idx=build_ticker_index(env.tickers), sectors=sectors).make_mask

    run_campaign._build_hybrid_hook = patched_hook
    try:
        row = run_campaign._train_and_evaluate_one(
            model="D", seed=seed, universe=UNIVERSE, steps=STEPS,
            campaign_id="v8_exp4", config_suffix="v2")
    finally:
        run_campaign._build_hybrid_hook = original_hook
    d = asdict(row)
    d["model"] = "S_star"
    d["p_star"] = p_star
    return d


def main() -> None:
    setup_logging("WARNING")
    p_star, rot_d, sd = _p_star()
    regla = "GLOBAL" if sd <= 0.05 else "POR SEMILLA (sd > 0.05!)"
    print(f"Calibracion: rot(D)={rot_d:.4f} (sd={sd:.4f}) -> p*={p_star:.4f} "
          f"[regla: {regla}]", flush=True)
    if sd > 0.05:
        print("[AVISO] sd > 0.05: el pre-registro exige calibracion por semilla."
              " Este runner usa p* global; escalar al humano.", flush=True)

    run_campaign._maybe_download(UNIVERSE)
    ensure_dir(OUTPUT.parent)
    rows: list[dict] = []
    done: set[str] = set()
    if OUTPUT.exists():
        rows = list(csv.DictReader(OUTPUT.open(encoding="utf-8")))
        done = {str(r["seed"]) for r in rows}
        print(f"[resume] {len(rows)} filas ya presentes.", flush=True)

    def save() -> None:
        with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    for i, seed in enumerate(SEEDS, 1):
        if str(seed) in done:
            continue
        print(f"[{i}/{len(SEEDS)}] S(p*={p_star:.3f}) seed={seed} ...", flush=True)
        d = _run_sticky(seed, p_star)
        rows.append({k: str(v) for k, v in d.items()})
        print(f"  sharpe={float(d['sharpe_ratio']):+.4f} "
              f"cand={float(d['candidate_hit_rate']):.4f} "
              f"dur={float(d['duration_seconds']):.0f}s", flush=True)
        save()
    print(f"\n[OK] {len(rows)} filas en {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
