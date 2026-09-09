"""Validez externa por ventanas temporales (v7 — experimento 7).

Replica las conclusiones clave de v6/v7 en 4 folds walk-forward de ventana
expansiva (test = 2021, 2022, 2023, 2024) para ver si pasan de "caso" a
"patrón". Por fold ejecuta, con la MISMA config que los experimentos v7
(default M=8, m=3, beta=0; NO el sweet_spot), los modelos RL {A, D, R} con
n semillas y computa los benchmarks {momentum_20d, equal_weight, oracle_expost,
random} sobre la MISMA ventana de test.

Contrasta dos claims a través de regímenes:
  - R ≈ D   (lo cuántico no supera al azar)            — claim central v7.
  - momentum ≥ RL en Sharpe                            — claim punto 1 v6.

El split interno se ajusta (train_frac) para colocar el último año como test,
igual que ``run_walk_forward.py``. Reanudación idempotente por (fold, strategy,
seed).

Uso::

    uv run python scripts/run_regime_validation_v7.py --universe nivel2 \\
        --seeds 42 123 456 789 1024 --steps 50000 --max-steps 50
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys as _sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.data.cleaning import CleaningPolicy, clean
from src.data.download import load_ohlcv
from src.data.features import FeatureSpec, compute_features
from src.data.splits import SplitSpec, chronological_split
from src.env.market_env import MarketEnv, MarketEnvSpec
from src.env.reward import RewardCoefficients
from src.training.evaluate import PromisingSpec, compute_promising_matrix
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import TABLES_DIR, ensure_dir


def _import(name: str):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


run_campaign = _import("run_campaign")
bench = _import("run_classical_benchmarks")
_log = get_logger(__name__)

FOLDS = [
    {"fold_id": 1, "train_end": 2020, "test_year": 2021},
    {"fold_id": 2, "train_end": 2021, "test_year": 2022},
    {"fold_id": 3, "train_end": 2022, "test_year": 2023},
    {"fold_id": 4, "train_end": 2023, "test_year": 2024},
]
RL_MODELS = ["A", "D", "R"]


def _fold_filter(test_year: int) -> tuple[str, str]:
    return "2018-01-02", f"{test_year}-12-31"


def _fold_train_frac(train_end: int, test_year: int) -> float:
    return (train_end - 2018 + 1) / (test_year - 2018 + 1)


# ---------------------------------------------------------------------------
# RL: A, D, R con config v7 default (M=8), filtrado al fold
# ---------------------------------------------------------------------------


def _run_rl(model: str, seed: int, fold: dict, universe: str, steps: int,
            max_steps: int, window: int) -> dict:
    start, end = _fold_filter(fold["test_year"])
    train_frac = _fold_train_frac(fold["train_end"], fold["test_year"])

    original_load = run_campaign.load_config
    original_hook = run_campaign._build_hybrid_hook

    def patched_load(cfg_path):
        cfg = original_load(cfg_path)
        cfg.env.split.train_frac = train_frac
        cfg.env.split.val_frac = 0.01
        return cfg

    run_campaign.load_config = patched_load
    base_model = model
    if model == "R":
        base_model = "D"  # base con graph/quantum; el hook se sustituye

        def patched_hook(cfg, env, features, classical, mdl):
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
            model=base_model, seed=seed, universe=universe, steps=steps,
            campaign_id=f"regime_fold{fold['fold_id']}_{model}",
            max_steps_override=max_steps, window_override=window,
            config_suffix="v2", filter_dates=(start, end))
    finally:
        run_campaign.load_config = original_load
        run_campaign._build_hybrid_hook = original_hook

    d = asdict(row)
    return {"fold_id": fold["fold_id"], "test_year": fold["test_year"],
            "strategy": model, "seed": seed,
            "sharpe_ratio": d["sharpe_ratio"],
            "candidate_hit_rate": d["candidate_hit_rate"],
            "topm_hit_rate": d["topm_hit_rate"],
            "cumulative_return": d["cumulative_return"]}


# ---------------------------------------------------------------------------
# Benchmarks por fold (momentum, equal_weight, oracle, random)
# ---------------------------------------------------------------------------


def _run_benchmarks(fold: dict, universe: str, seeds: list[int],
                    max_steps: int, window: int) -> list[dict]:
    start, end = _fold_filter(fold["test_year"])
    train_frac = _fold_train_frac(fold["train_end"], fold["test_year"])
    raw = load_ohlcv(universe=universe)
    features = compute_features(clean(raw, CleaningPolicy()), FeatureSpec(return_type="log"))
    features = features.loc[start:end]
    splits = chronological_split(features, SplitSpec(train_frac=train_frac, val_frac=0.01))
    env_spec = MarketEnvSpec(
        window_length=window, max_steps=max_steps,
        reward_coefs=RewardCoefficients(lambda_risk=0.0, mu_cost=0.001,
                                        transaction_cost=0.0005, reward_type="log_wealth"),
        episode_sampling="random")
    env = MarketEnv(splits["test"], env_spec)
    n = env.n_tickers
    cols = [(t, "return") for t in env.tickers]
    test_returns = splits["test"].loc[:, cols].to_numpy(dtype=np.float64)
    promising = compute_promising_matrix(test_returns, PromisingSpec())
    rmat = env._return_matrix

    def momentum(t, rng):
        s = max(0, t - bench.MOMENTUM_WINDOW + 1)
        return int(np.argmax(rmat[s:t + 1].mean(axis=0)))

    def oracle(t, rng):
        tn = min(t + 1, rmat.shape[0] - 1)
        return int(np.argmax(rmat[tn]))

    out: list[dict] = []

    def emit(strategy, seed, metrics):
        out.append({"fold_id": fold["fold_id"], "test_year": fold["test_year"],
                    "strategy": strategy, "seed": seed,
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "candidate_hit_rate": metrics.get("candidate_hit_rate", float("nan")),
                    "topm_hit_rate": metrics["topm_hit_rate"],
                    "cumulative_return": metrics["cumulative_return"]})

    for seed in seeds:
        emit("momentum_20d", seed,
             bench._run_discrete_policy(env, momentum, promising, seed=seed))
        emit("random", seed,
             bench._run_discrete_policy(env, lambda t, rng: int(rng.integers(0, n)),
                                        promising, seed=seed))
        emit("oracle_expost", seed,
             bench._run_discrete_policy(env, oracle, promising, seed=seed))
    emit("equal_weight", "static",
         bench._portfolio_metrics(test_returns, np.full(n, 1.0 / n)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", default="nivel2")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--window-length", type=int, default=20)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()

    setup_logging(args.log_level)
    run_campaign._maybe_download(args.universe)
    output = args.output or TABLES_DIR / "regime_validation_v7.csv"
    ensure_dir(output.parent)

    rows: list[dict] = []
    done: set[tuple] = set()
    if output.exists():
        rows = list(csv.DictReader(output.open(encoding="utf-8")))
        done = {(r["fold_id"], r["strategy"], str(r["seed"])) for r in rows}
        print(f"[resume] {len(rows)} filas ya presentes.", flush=True)

    def save():
        with output.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    for fold in FOLDS:
        fid = str(fold["fold_id"])
        # Benchmarks (baratos): solo si falta alguno de este fold.
        if not any(r["fold_id"] == fid and r["strategy"] == "momentum_20d" for r in rows):
            print(f"[fold {fid}] benchmarks (test={fold['test_year']}) ...", flush=True)
            for b in _run_benchmarks(fold, args.universe, args.seeds,
                                     args.max_steps, args.window_length):
                rows.append({k: str(v) for k, v in b.items()})
            save()
        # RL
        for model in RL_MODELS:
            for seed in args.seeds:
                if (fid, model, str(seed)) in done:
                    continue
                print(f"[fold {fid}] {model} seed={seed} (test={fold['test_year']}) ...",
                      flush=True)
                r = _run_rl(model, seed, fold, args.universe, args.steps,
                            args.max_steps, args.window_length)
                rows.append({k: str(v) for k, v in r.items()})
                print(f"  sharpe={r['sharpe_ratio']:+.4f} "
                      f"cand_hit={r['candidate_hit_rate']:.3f} "
                      f"topm={r['topm_hit_rate']:.3f}", flush=True)
                save()

    print(f"\n[OK] {len(rows)} filas en {output}", flush=True)


if __name__ == "__main__":
    main()
