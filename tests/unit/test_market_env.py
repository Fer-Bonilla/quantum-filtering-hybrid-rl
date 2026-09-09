"""Tests de src/env/market_env.py."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env
from src.data.features import FeatureSpec, compute_features
from src.env.market_env import MarketEnv, MarketEnvSpec, RecordingMarketEnv


@pytest.fixture
def env_features(synthetic_clean_df: pd.DataFrame) -> pd.DataFrame:
    return compute_features(synthetic_clean_df, FeatureSpec(volatility_window=5, volume_window=5))


def test_env_satisfies_gym_api(env_features: pd.DataFrame) -> None:
    env = MarketEnv(env_features, MarketEnvSpec(window_length=5, max_steps=50))
    # check_env corre múltiples reset/step y valida espacios, tipos, info.
    check_env(env, skip_render_check=True)


def test_env_action_space_size(env_features: pd.DataFrame) -> None:
    env = MarketEnv(env_features, MarketEnvSpec(window_length=5, max_steps=50))
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 4  # 4 tickers en synthetic_clean_df


def test_env_random_agent_no_crash(env_features: pd.DataFrame) -> None:
    env = MarketEnv(env_features, MarketEnvSpec(window_length=5, max_steps=100))
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert "candidate_mask" in info
    assert info["candidate_mask"].dtype == bool

    rng = np.random.default_rng(42)
    total_steps = 0
    for _ in range(20):  # múltiples episodios
        obs, info = env.reset(seed=int(rng.integers(0, 10_000)))
        for _ in range(env._spec.max_steps + 5):
            action = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            assert np.isfinite(reward)
            if terminated or truncated:
                break
    assert total_steps >= 200  # cubrimos al menos varios episodios


def test_env_reset_is_deterministic_with_seed(env_features: pd.DataFrame) -> None:
    env = MarketEnv(env_features, MarketEnvSpec(window_length=5, max_steps=50))
    obs1, _ = env.reset(seed=99)
    obs2, _ = env.reset(seed=99)
    np.testing.assert_array_equal(obs1, obs2)


def test_env_info_has_required_fields(env_features: pd.DataFrame) -> None:
    env = MarketEnv(env_features, MarketEnvSpec(window_length=5, max_steps=20))
    _, info = env.reset(seed=0)
    assert set(info) >= {"candidate_mask", "asset_ids", "t", "episode_step"}
    assert info["asset_ids"].shape == (4,)
    assert info["episode_step"] == 0


def test_env_truncates_at_max_steps(env_features: pd.DataFrame) -> None:
    max_steps = 10
    env = MarketEnv(env_features, MarketEnvSpec(window_length=5, max_steps=max_steps))
    env.reset(seed=0)
    truncated = False
    for i in range(max_steps + 2):
        _, _, terminated, truncated, _ = env.step(0)
        if truncated:
            assert i + 1 >= max_steps - 1
            break
        assert not terminated
    assert truncated


def test_env_no_future_access(env_features: pd.DataFrame) -> None:
    """CRÍTICO: el env nunca debe acceder a data[t'] con t' > t durante step(t).

    Implementación: ``RecordingMarketEnv`` instrumenta el acceso a la
    matriz de retornos. Permitimos acceso a t+1 (necesario para R_{u_t, t+1})
    pero NO a t+2 o posterior.
    """
    env = RecordingMarketEnv(env_features, MarketEnvSpec(window_length=5, max_steps=30))
    env.reset(seed=0)
    for _ in range(20):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    # Cualquier índice accedido por encima de t+1 es fuga
    excessive = [i for i in env.accessed_future_indices if i > env.current_t + 1]
    assert not excessive, f"Acceso a información futura detectado en índices: {excessive}"


def test_env_rejects_invalid_action(env_features: pd.DataFrame) -> None:
    env = MarketEnv(env_features, MarketEnvSpec(window_length=5, max_steps=50))
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(99)


def test_env_requires_multiindex(synthetic_clean_df: pd.DataFrame) -> None:
    flat = synthetic_clean_df.copy()
    flat.columns = [f"{a}_{b}" for a, b in flat.columns]
    with pytest.raises(TypeError):
        MarketEnv(flat, MarketEnvSpec(window_length=5, max_steps=50))
