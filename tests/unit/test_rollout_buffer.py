"""Tests de src/agents/rollout_buffer.py."""

from __future__ import annotations

import numpy as np
import pytest
from src.agents.rollout_buffer import RolloutBuffer


def _populate(buf: RolloutBuffer, rng: np.random.Generator) -> None:
    for _ in range(buf.n_steps):
        buf.add(
            obs=rng.standard_normal(buf.obs_dim).astype(np.float32),
            action=int(rng.integers(0, buf.n_actions)),
            log_prob=float(rng.standard_normal()),
            value=float(rng.standard_normal()),
            reward=float(rng.standard_normal()),
            done=bool(rng.integers(0, 2)),
            mask=np.ones(buf.n_actions, dtype=bool),
        )


def test_buffer_fills_and_resets() -> None:
    rng = np.random.default_rng(0)
    buf = RolloutBuffer(n_steps=8, obs_dim=4, n_actions=3)
    _populate(buf, rng)
    assert buf.is_full
    assert buf.size == 8
    buf.reset()
    assert buf.size == 0
    assert not buf.is_full


def test_buffer_overflow_raises() -> None:
    rng = np.random.default_rng(0)
    buf = RolloutBuffer(n_steps=2, obs_dim=2, n_actions=2)
    _populate(buf, rng)
    with pytest.raises(RuntimeError):
        buf.add(
            obs=np.zeros(2, dtype=np.float32),
            action=0,
            log_prob=0.0,
            value=0.0,
            reward=0.0,
            done=False,
            mask=np.ones(2, dtype=bool),
        )


def test_gae_returns_match_advantages_plus_values() -> None:
    rng = np.random.default_rng(42)
    buf = RolloutBuffer(n_steps=16, obs_dim=4, n_actions=3, gamma=0.99, gae_lambda=0.95)
    _populate(buf, rng)
    buf.compute_gae(last_value=0.5, last_done=False)
    expected_returns = buf.advantages + buf.values
    np.testing.assert_allclose(buf.returns, expected_returns, atol=1e-6)


def test_gae_zero_reward_zero_value_gives_zero_advantage() -> None:
    np.random.default_rng(7)
    buf = RolloutBuffer(n_steps=5, obs_dim=2, n_actions=2)
    for _ in range(buf.n_steps):
        buf.add(
            obs=np.zeros(2, dtype=np.float32),
            action=0,
            log_prob=0.0,
            value=0.0,
            reward=0.0,
            done=False,
            mask=np.ones(2, dtype=bool),
        )
    buf.compute_gae(last_value=0.0, last_done=False)
    np.testing.assert_allclose(buf.advantages, 0.0, atol=1e-7)


def test_minibatch_iterator_covers_all_steps() -> None:
    rng = np.random.default_rng(0)
    buf = RolloutBuffer(n_steps=10, obs_dim=3, n_actions=2)
    _populate(buf, rng)
    buf.compute_gae(last_value=0.0, last_done=False)
    seen = 0
    for batch in buf.iter_minibatches(batch_size=4, rng=np.random.default_rng(1)):
        seen += batch["obs"].shape[0]
    assert seen == buf.n_steps


def test_mask_action_consistency() -> None:
    """Tras GAE, mask[a_t] debe ser True para todos los samples (sanity)."""
    rng = np.random.default_rng(0)
    buf = RolloutBuffer(n_steps=8, obs_dim=3, n_actions=4)
    for _ in range(buf.n_steps):
        # Máscara aleatoria con al menos un True; la acción debe estar permitida.
        mask = rng.random(buf.n_actions) > 0.5
        mask[0] = True  # garantizar no-empty
        allowed = np.flatnonzero(mask)
        action = int(rng.choice(allowed))
        buf.add(
            obs=np.zeros(buf.obs_dim, dtype=np.float32),
            action=action,
            log_prob=0.0,
            value=0.0,
            reward=0.0,
            done=False,
            mask=mask,
        )
    for t in range(buf.n_steps):
        assert buf.masks[t, int(buf.actions[t])], f"mask[a_t] = False en t={t}"
