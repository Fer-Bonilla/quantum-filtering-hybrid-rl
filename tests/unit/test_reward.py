"""Tests de src/env/reward.py."""

from __future__ import annotations

from src.env.reward import RewardCoefficients, compute_reward, transaction_cost_for


def test_reward_positive_for_good_return() -> None:
    coefs = RewardCoefficients(lambda_risk=0.1, mu_cost=0.001)
    r = compute_reward(realized_return=0.02, realized_volatility=0.01, cost=0.0, coefs=coefs)
    assert r > 0


def test_reward_penalizes_volatility() -> None:
    coefs = RewardCoefficients(lambda_risk=1.0, mu_cost=0.0)
    low_vol = compute_reward(realized_return=0.01, realized_volatility=0.01, coefs=coefs, cost=0.0)
    high_vol = compute_reward(realized_return=0.01, realized_volatility=0.05, coefs=coefs, cost=0.0)
    assert low_vol > high_vol


def test_reward_penalizes_cost() -> None:
    coefs = RewardCoefficients(lambda_risk=0.0, mu_cost=1.0)
    no_cost = compute_reward(0.01, 0.0, cost=0.0, coefs=coefs)
    with_cost = compute_reward(0.01, 0.0, cost=0.005, coefs=coefs)
    assert no_cost > with_cost


def test_transaction_cost_first_action_is_zero() -> None:
    assert transaction_cost_for(action=3, previous_action=None, base_cost=0.001) == 0.0


def test_transaction_cost_unchanged_action() -> None:
    assert transaction_cost_for(action=3, previous_action=3, base_cost=0.001) == 0.0


def test_transaction_cost_changed_action() -> None:
    assert transaction_cost_for(action=3, previous_action=2, base_cost=0.001) == 0.001
