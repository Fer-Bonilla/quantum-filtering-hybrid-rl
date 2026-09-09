"""Tests smoke del trainer PPO sobre el entorno sintético.

No optimizamos por desempeño; verificamos que el loop entrena sin crash,
los losses son finitos, y los pesos se actualizan.
"""

from __future__ import annotations

import pandas as pd
import torch
from src.agents.classical_agent import ClassicalAgent
from src.data.features import FeatureSpec, compute_features
from src.env.market_env import MarketEnv, MarketEnvSpec
from src.training.seed_utils import set_global_seed
from src.training.trainer import PPOHyperparams, PPOTrainer


def test_trainer_runs_without_crash(synthetic_clean_df: pd.DataFrame) -> None:
    features = compute_features(
        synthetic_clean_df, FeatureSpec(volatility_window=5, volume_window=5)
    )
    env = MarketEnv(features, MarketEnvSpec(window_length=5, max_steps=20))
    rng = set_global_seed(0)
    obs_dim = env.observation_space.shape[0]
    agent = ClassicalAgent(obs_dim=obs_dim, n_actions=env.n_tickers)
    hp = PPOHyperparams(
        learning_rate=3e-4,
        n_epochs=2,
        batch_size=16,
        rollout_steps=64,
        normalize_advantages=True,
    )
    trainer = PPOTrainer(env, agent, hp, rng=rng)
    result = trainer.train(total_steps=128)
    assert len(result.train_curve) == 2
    assert all(p == p for p in result.losses["policy_loss"])  # no NaN
    assert all(v == v for v in result.losses["value_loss"])
    assert all(e == e for e in result.losses["entropy"])


def test_trainer_updates_weights(synthetic_clean_df: pd.DataFrame) -> None:
    features = compute_features(
        synthetic_clean_df, FeatureSpec(volatility_window=5, volume_window=5)
    )
    env = MarketEnv(features, MarketEnvSpec(window_length=5, max_steps=20))
    rng = set_global_seed(0)
    obs_dim = env.observation_space.shape[0]
    agent = ClassicalAgent(obs_dim=obs_dim, n_actions=env.n_tickers)
    before = {n: p.detach().clone() for n, p in agent.network.named_parameters()}
    hp = PPOHyperparams(rollout_steps=64, batch_size=16, n_epochs=2)
    trainer = PPOTrainer(env, agent, hp, rng=rng)
    trainer.train(total_steps=64)
    after = dict(agent.network.named_parameters())
    diff = sum(torch.norm(after[n].detach() - before[n]).item() for n in before)
    assert diff > 0.0, "Los pesos no se actualizaron tras una corrida PPO."
