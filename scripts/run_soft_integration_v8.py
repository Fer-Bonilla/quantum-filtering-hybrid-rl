"""EXP-6 v8 — Integración suave de la señal de priorización (H-v8.5, G3).

Sesgo de logits z' = z + beta*log(P+eps) sobre el soporte de H_t (NEG_INF
fuera), con P = P_k (soft-D), p_k clásica (soft-C) o uniforme sobre H_t
(soft-R, control de subgrafo sin ranking). El sesgo se almacena en el búfer
para consistencia del cociente PPO (SoftPPOTrainer/SoftRolloutBuffer).

Fase 1 (tuning): soft-D con beta en {0.5, 1, 2}, 3 semillas, ventana
2018-2022 -> beta* por Sharpe. Fase 2 (contraste): soft-D / soft-C / soft-R
x 10 semillas pareadas, configuración Tabla 5.4 completa.

Uso::

    uv run python scripts/run_soft_integration_v8.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from src.agents.soft_hybrid import (
    BiasedClassicalAgent,
    SoftHybridAgent,
    SoftPPOTrainer,
    evaluate_soft_agent,
)
from src.agents.hybrid_agent import (
    HybridSpec,
    build_returns_panel,
    build_ticker_index,
)
from src.data.cleaning import CleaningPolicy, clean
from src.data.download import load_ohlcv
from src.data.features import FeatureSpec, compute_features
from src.data.sector_map import build_sector_map
from src.data.splits import SplitSpec, chronological_split
from src.env.market_env import MarketEnv, MarketEnvSpec
from src.env.reward import RewardCoefficients
from src.graph.graph_builder import GraphSpec
from src.training.evaluate import (
    PromisingSpec,
    compute_promising_matrix,
    episodes_to_convergence,
)
from src.training.seed_utils import set_global_seed
from src.training.trainer import PPOHyperparams
from src.utils.config import load_config
from src.utils.paths import CONFIGS_DIR, TABLES_DIR, ensure_dir

SEEDS = [42, 123, 456, 789, 1024, 7, 99, 314, 1729, 65535]
TUNE_SEEDS = [42, 123, 456]
BETAS = [0.5, 1.0, 2.0]
STEPS = 50_000
OUT_TUNE = TABLES_DIR / "soft_integration_v8_tuning.csv"
OUT_MAIN = TABLES_DIR / "soft_integration_v8.csv"


def _prepare(cfg, filter_dates=None, train_frac=None):
    raw = load_ohlcv(universe="nivel2")
    features = compute_features(clean(raw, CleaningPolicy()), FeatureSpec(
        return_type=cfg.env.feature.return_type,
        volatility_window=cfg.env.feature.volatility_window,
        volume_window=cfg.env.feature.volume_window,
        indicators=tuple(cfg.env.feature.indicators)))
    if filter_dates is not None:
        features = features.loc[filter_dates[0]: filter_dates[1]]
    tf = train_frac if train_frac is not None else cfg.env.split.train_frac
    vf = 0.01 if train_frac is not None else cfg.env.split.val_frac
    splits = chronological_split(features, SplitSpec(train_frac=tf, val_frac=vf))
    env_spec = MarketEnvSpec(
        window_length=cfg.env.feature.window_length, max_steps=cfg.env.max_steps,
        reward_coefs=RewardCoefficients(
            lambda_risk=cfg.env.lambda_risk, mu_cost=cfg.env.mu_cost,
            transaction_cost=cfg.env.transaction_cost,
            reward_type=cfg.env.reward_type),
        episode_sampling=cfg.env.episode_sampling)
    return features, splits, env_spec


def _make_soft_hook(cfg, env, features, mode: str, beta: float):
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
    return SoftHybridAgent(
        classical_agent=None, local_module=None, hybrid_spec=spec,
        returns_panel=build_returns_panel(features, env.tickers),
        ticker_to_idx=build_ticker_index(env.tickers), sectors=sectors,
        mode=mode, beta_bias=beta,
        init_mode=cfg.quantum.init_mode,
        renorm=cfg.quantum.renormalize_threshold).make_mask


def _run_one(mode: str, beta: float, seed: int, *,
             filter_dates=None, train_frac=None) -> dict:
    cfg = load_config(CONFIGS_DIR / "experiment" / "model_d_v2.yaml")
    cfg.data.cache_subdir = "nivel2"
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    features, splits, env_spec = _prepare(cfg, filter_dates, train_frac)
    env = MarketEnv(splits["train"], env_spec)
    agent = BiasedClassicalAgent(
        obs_dim=env.observation_space.shape[0], n_actions=env.n_tickers,
        hidden_sizes=cfg.agent.ppo.hidden_sizes)
    hp = PPOHyperparams(
        learning_rate=cfg.agent.ppo.learning_rate,
        n_epochs=cfg.agent.ppo.n_epochs, batch_size=cfg.agent.ppo.batch_size,
        rollout_steps=cfg.agent.ppo.rollout_steps,
        clip_coef=cfg.agent.ppo.clip_coef, gamma=cfg.agent.ppo.gamma,
        gae_lambda=cfg.agent.ppo.gae_lambda,
        entropy_coef=cfg.agent.ppo.entropy_coef,
        value_coef=cfg.agent.ppo.value_coef,
        max_grad_norm=cfg.agent.ppo.max_grad_norm)

    hook_train = _make_soft_hook(cfg, env, features, mode, beta)
    t0 = time.perf_counter()
    trainer = SoftPPOTrainer(env, agent, hp, rng=rng, logger=None,
                             step_hook=hook_train)
    result = trainer.train(total_steps=STEPS)

    test_env = MarketEnv(splits["test"], env_spec)
    hook_test = _make_soft_hook(cfg, test_env, features, mode, beta)
    cols = [(t, "return") for t in test_env.tickers]
    returns_test = splits["test"].loc[:, cols].to_numpy(dtype=float)
    promising = compute_promising_matrix(returns_test, PromisingSpec())
    ev = evaluate_soft_agent(test_env, agent, n_episodes=5, seed=seed,
                             step_hook=hook_test, promising_matrix=promising)
    conv = episodes_to_convergence(result.train_curve)
    return {
        "campaign_id": "v8_exp6", "model": mode, "beta_bias": beta,
        "seed": seed, "total_steps": STEPS,
        "cumulative_return": ev.cumulative_return,
        "sharpe_ratio": ev.sharpe_ratio, "max_drawdown": ev.max_drawdown,
        "mean_reward": ev.mean_reward, "asset_coverage": ev.asset_coverage,
        "topm_hit_rate": ev.topm_hit_rate,
        "candidate_hit_rate": ev.candidate_hit_rate,
        "mean_latency_ms": ev.mean_latency_ms,
        "episodes_to_convergence": conv,
        "duration_seconds": time.perf_counter() - t0,
    }


def _load_done(path: Path, keys):
    rows: list[dict] = []
    done: set[tuple] = set()
    if path.exists():
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        done = {tuple(r[k] for k in keys) for r in rows}
    return rows, done


def _save(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ensure_dir(OUT_MAIN.parent)

    # --- Fase 1: tuning de beta (soft-D, 2018-2022) ---
    rows_t, done_t = _load_done(OUT_TUNE, ("model", "beta_bias", "seed"))
    for beta in BETAS:
        for seed in TUNE_SEEDS:
            if ("softD", str(beta), str(seed)) in done_t:
                continue
            print(f"[tuning] softD beta={beta} seed={seed} ...", flush=True)
            d = _run_one("softD", beta, seed,
                         filter_dates=("2018-01-02", "2022-12-31"),
                         train_frac=0.75)
            rows_t.append({k: str(v) for k, v in d.items()})
            print(f"  sharpe={float(d['sharpe_ratio']):+.4f}", flush=True)
            _save(OUT_TUNE, rows_t)
    por_beta = {}
    for r in rows_t:
        por_beta.setdefault(float(r["beta_bias"]), []).append(
            float(r["sharpe_ratio"]))
    beta_star = max(por_beta, key=lambda b: np.mean(por_beta[b]))
    print(f"\n[beta*] {beta_star} (sharpe medio "
          f"{np.mean(por_beta[beta_star]):+.4f})\n", flush=True)

    # --- Fase 2: contraste soft-D / soft-C / soft-R ---
    rows_m, done_m = _load_done(OUT_MAIN, ("model", "seed"))
    total = 3 * len(SEEDS)
    i = len(rows_m)
    for seed in SEEDS:
        for mode in ("softD", "softC", "softR"):
            if (mode, str(seed)) in done_m:
                continue
            i += 1
            print(f"[{i}/{total}] {mode} (beta*={beta_star}) seed={seed} ...",
                  flush=True)
            d = _run_one(mode, beta_star, seed)
            rows_m.append({k: str(v) for k, v in d.items()})
            print(f"  sharpe={float(d['sharpe_ratio']):+.4f} "
                  f"cand={float(d['candidate_hit_rate']):.4f} "
                  f"dur={float(d['duration_seconds']):.0f}s", flush=True)
            _save(OUT_MAIN, rows_m)
    print(f"\n[OK] tuning={len(rows_t)} main={len(rows_m)}", flush=True)


if __name__ == "__main__":
    main()
