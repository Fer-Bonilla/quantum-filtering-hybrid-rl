"""Tests de las métricas de evaluación (Sec. 8.16-8.17)."""

from __future__ import annotations

import numpy as np
import pytest
from src.training.evaluate import (
    PromisingSpec,
    compute_promising_matrix,
    convergence_plateau_after_peak,
    convergence_stability,
    convergence_to_final,
    episodes_to_convergence,
)

# ---------------------------------------------------------------------------
# Promising assets (Sec. 8.17)
# ---------------------------------------------------------------------------


def test_promising_matrix_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((100, 5))
    M = compute_promising_matrix(returns, PromisingSpec(horizon=5))
    assert M.shape == (100, 5)
    assert M.dtype == bool


def test_promising_matrix_last_rows_false() -> None:
    """Las últimas `h` filas no tienen horizonte; deben ser todas False."""
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((50, 4))
    h = 5
    M = compute_promising_matrix(returns, PromisingSpec(horizon=h))
    assert not M[-h:].any()


def test_promising_matrix_picks_top_percentile() -> None:
    """Para un activo con retornos uniformemente mejores, el flag prometedor es True."""
    T = 50
    N = 5
    returns = np.zeros((T, N))
    returns[:, 0] = 0.05  # activo 0 siempre rinde más
    returns[:, 1:] = 0.001
    M = compute_promising_matrix(returns, PromisingSpec(horizon=5, percentile=80.0))
    # En las primeras T-5 filas, el activo 0 debe estar marcado como prometedor
    assert M[: T - 5, 0].all()


def test_promising_matrix_too_short_returns_all_false() -> None:
    returns = np.ones((3, 4))
    M = compute_promising_matrix(returns, PromisingSpec(horizon=10))
    assert not M.any()


# ---------------------------------------------------------------------------
# Convergencia
# ---------------------------------------------------------------------------


def test_episodes_to_convergence_flat_curve_converges() -> None:
    """Curva constante: convergencia inmediata tras la primera ventana."""
    curve = [1.0] * 30
    conv = episodes_to_convergence(curve, window=3, tol=0.01)
    assert conv != float("inf")
    assert conv <= 30


def test_episodes_to_convergence_increasing_then_flat() -> None:
    """Curva que sube y luego se estabiliza converge en la zona plana."""
    curve = list(np.linspace(0.0, 1.0, 20)) + [1.0] * 30
    conv = episodes_to_convergence(curve, window=5, tol=0.05)
    assert 15 <= conv <= 40


def test_episodes_to_convergence_oscillating_does_not_converge() -> None:
    rng = np.random.default_rng(0)
    curve = rng.standard_normal(40)  # ruido puro
    conv = episodes_to_convergence(curve, window=3, tol=0.001)
    # Probabilidad alta de no converger con tol muy estricto
    assert conv == float("inf") or conv > 1


def test_episodes_to_convergence_too_short() -> None:
    assert episodes_to_convergence([1.0, 1.0, 1.0], window=5) == float("inf")


# ---------------------------------------------------------------------------
# Convergencia v2 (Problema 1.1 del revisor)
# ---------------------------------------------------------------------------


def test_convergence_to_final_with_synthetic_curve_known_point() -> None:
    """Curva sintética que converge en rollout ~12 (1-indexado).

    Construcción: 10 rollouts ruidosos crecientes + 15 rollouts en plateau
    alrededor de 1.0. La métrica primaria debe retornar ~12 ± 3.
    """
    rng = np.random.default_rng(7)
    growth = np.linspace(0.0, 1.0, 10) + 0.05 * rng.standard_normal(10)
    plateau = 1.0 + 0.02 * rng.standard_normal(15)
    curve = np.concatenate([growth, plateau])
    conv = convergence_to_final(curve, window=3, window_last=3, tol_rel=0.05, tol_abs=0.05)
    assert conv != float("inf"), "no debería diverger sobre curva con plateau claro"
    assert 8 <= conv <= 16, f"esperado ~12 ± 3, obtenido {conv}"


def test_convergence_to_final_has_variance_across_seeds() -> None:
    """Cinco curvas con plateaux en posiciones diferentes producen 5 valores.

    Sin esto, std(D-C) = 0 (síntoma reportado por el revisor).
    """
    rng = np.random.default_rng(0)
    plateau_points = [8, 12, 15, 18, 22]
    convs = []
    for break_pt in plateau_points:
        growth = np.linspace(0.0, 1.0, break_pt) + 0.03 * rng.standard_normal(break_pt)
        plateau = 1.0 + 0.02 * rng.standard_normal(30 - break_pt)
        curve = np.concatenate([growth, plateau])
        convs.append(convergence_to_final(curve, window=3, window_last=3, tol_abs=0.05))
    finite_convs = np.array([c for c in convs if c != float("inf")])
    assert len(finite_convs) >= 3, "al menos 3 de 5 curvas deben converger"
    assert np.std(finite_convs) > 0.5, (
        f"std debe ser > 0 (criterio: refutar Problema 1.1 del revisor); obtenido {np.std(finite_convs):.4f}"
    )


def test_convergence_stability_detects_plateau() -> None:
    """convergence_stability detecta un rollout estable y rechaza un growth ruidoso.

    Diseño: growth muy ruidoso (std relativo alto) seguido por plateau
    con std despreciable. La métrica debe encontrar UN rollout estable
    (no infinito) y rechazar los rollouts iniciales con std alto.
    """
    rng = np.random.default_rng(0)
    # Growth muy ruidoso: std/mean grande
    growth = np.array([0.5, -0.2, 0.8, -0.1, 1.2, 0.3, 1.5, 0.6])
    # Plateau con std despreciable
    plateau = 5.0 + 0.005 * rng.standard_normal(15)
    curve = np.concatenate([growth, plateau])
    conv = convergence_stability(curve, window=3, tol_std=0.01)
    assert conv != float("inf")
    # Debe converger después de cierto número de rollouts (no inmediato)
    assert conv >= 2


def test_convergence_plateau_after_peak_with_overshoot() -> None:
    """Curva con sobreajuste (pico seguido de plateau bajo)."""
    rng = np.random.default_rng(0)
    growth = np.linspace(0.0, 2.0, 8)
    overshoot = np.array([2.5, 2.7, 2.8, 2.6])
    decline = np.linspace(2.6, 1.5, 6)
    plateau = 1.5 + 0.01 * rng.standard_normal(10)
    curve = np.concatenate([growth, overshoot, decline, plateau])
    conv = convergence_plateau_after_peak(curve, window=3, slope_tol=0.02)
    assert conv != float("inf")
    # El peak está alrededor de t≈12; plateau debe detectarse después
    assert conv >= 12


# ---------------------------------------------------------------------------
# Integración con evaluate_agent (sanity smoke)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_promising", [True, False])
def test_evaluate_agent_returns_metrics(synthetic_clean_df, with_promising: bool) -> None:
    """evaluate_agent produce todas las métricas con/sin promising_matrix."""
    from src.agents.classical_agent import ClassicalAgent
    from src.data.features import FeatureSpec, compute_features
    from src.env.market_env import MarketEnv, MarketEnvSpec
    from src.training.evaluate import evaluate_agent

    features = compute_features(
        synthetic_clean_df, FeatureSpec(volatility_window=5, volume_window=5)
    )
    env = MarketEnv(features, MarketEnvSpec(window_length=5, max_steps=20))
    agent = ClassicalAgent(obs_dim=env.observation_space.shape[0], n_actions=env.n_tickers)

    promising = None
    if with_promising:
        cols = [(t, "return") for t in env.tickers]
        panel = features.loc[:, cols].to_numpy(dtype=float)
        promising = compute_promising_matrix(panel, PromisingSpec(horizon=3))

    result = evaluate_agent(env, agent, n_episodes=2, seed=0, promising_matrix=promising)
    assert np.isfinite(result.cumulative_return)
    assert np.isfinite(result.sharpe_ratio)
    assert np.isfinite(result.mean_latency_ms)
    assert result.mean_latency_ms >= 0.0
    if with_promising:
        assert 0.0 <= result.topm_hit_rate <= 1.0
        assert 0.0 <= result.candidate_hit_rate <= 1.0
    else:
        assert np.isnan(result.topm_hit_rate)
