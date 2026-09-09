"""Tests del esquema de recompensa ``log_wealth`` (rev. v2).

Verifica:
- Correctitud de ``compute_reward_log_wealth`` para retornos sintéticos.
- ``reward_type='log_wealth'`` enruta correctamente desde
  ``compute_reward``.
- Quiebra teórica (1+R ≤ 0) produce sanción acotada finita.
"""

from __future__ import annotations

import math

import pytest

from src.env.reward import (
    RewardCoefficients,
    compute_reward,
    compute_reward_log_wealth,
)


def test_log_wealth_zero_return_zero_reward_when_no_penalties() -> None:
    """log(1+0) - 0 - 0 = 0."""
    coefs = RewardCoefficients(lambda_risk=0.0, mu_cost=0.0, reward_type="log_wealth")
    r = compute_reward_log_wealth(0.0, 0.0, 0.0, coefs)
    assert abs(r) < 1e-12


def test_log_wealth_positive_return_positive_reward() -> None:
    """R=5% sin penalizaciones → r = log(1.05) ≈ 0.04879."""
    coefs = RewardCoefficients(lambda_risk=0.0, mu_cost=0.0, reward_type="log_wealth")
    r = compute_reward_log_wealth(0.05, 0.0, 0.0, coefs)
    assert r == pytest.approx(math.log(1.05), abs=1e-9)


def test_log_wealth_compresses_extreme_gains() -> None:
    """log(1+R) crece más despacio que R para R grandes (compresión)."""
    coefs = RewardCoefficients(lambda_risk=0.0, mu_cost=0.0, reward_type="log_wealth")
    r_lw = compute_reward_log_wealth(2.0, 0.0, 0.0, coefs)  # log(3) ≈ 1.099
    r_lin = compute_reward(2.0, 0.0, 0.0, RewardCoefficients(lambda_risk=0.0, mu_cost=0.0))
    assert r_lw < r_lin


def test_log_wealth_amplifies_small_losses() -> None:
    """log(1-0.1) < -0.1 (escala log amplifica pérdidas pequeñas)."""
    coefs = RewardCoefficients(lambda_risk=0.0, mu_cost=0.0, reward_type="log_wealth")
    r = compute_reward_log_wealth(-0.1, 0.0, 0.0, coefs)
    # log(0.9) ≈ -0.1054 < -0.1
    assert r < -0.1


def test_log_wealth_total_loss_returns_finite_penalty() -> None:
    """1 + R ≤ 0 (quiebra) debe retornar sanción finita, no -inf."""
    coefs = RewardCoefficients(lambda_risk=0.0, mu_cost=0.0, reward_type="log_wealth")
    r_at_minus_one = compute_reward_log_wealth(-1.0, 0.0, 0.0, coefs)
    r_below_minus_one = compute_reward_log_wealth(-1.5, 0.0, 0.0, coefs)
    assert math.isfinite(r_at_minus_one)
    assert math.isfinite(r_below_minus_one)
    assert r_at_minus_one == -1000.0


def test_log_wealth_includes_risk_penalty_term() -> None:
    """λσ debe restar también en el esquema log_wealth."""
    coefs_zero = RewardCoefficients(lambda_risk=0.0, mu_cost=0.0, reward_type="log_wealth")
    coefs_high = RewardCoefficients(lambda_risk=1.0, mu_cost=0.0, reward_type="log_wealth")
    r0 = compute_reward_log_wealth(0.05, 0.02, 0.0, coefs_zero)
    r1 = compute_reward_log_wealth(0.05, 0.02, 0.0, coefs_high)
    assert r0 - r1 == pytest.approx(0.02, abs=1e-9)


def test_compute_reward_routes_to_log_wealth() -> None:
    """compute_reward enruta correctamente cuando reward_type='log_wealth'."""
    coefs = RewardCoefficients(lambda_risk=0.0, mu_cost=0.0, reward_type="log_wealth")
    direct = compute_reward_log_wealth(0.03, 0.0, 0.0, coefs)
    via_router = compute_reward(0.03, 0.0, 0.0, coefs)
    assert direct == via_router


def test_default_reward_type_is_risk_penalty_for_backwards_compat() -> None:
    """RewardCoefficients() sin override usa el esquema original."""
    coefs = RewardCoefficients()
    assert coefs.reward_type == "risk_penalty"
    r = compute_reward(0.05, 0.02, 0.001, coefs)
    expected = 0.05 - coefs.lambda_risk * 0.02 - coefs.mu_cost * 0.001
    assert r == pytest.approx(expected, abs=1e-12)
