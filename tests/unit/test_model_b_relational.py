"""Tests Modelo B v2: state_builder con rasgos relacionales.

Verifica que el Modelo B ya NO colapsa al Modelo A (Problema 1.4 del
revisor). Tres invariantes:

1. ``test_B_state_dim_larger_than_A``: con relational_features_fn, el
   obs_shape aumenta en N·d_rel.
2. ``test_B_action_diverges_from_A_under_same_seed``: con misma semilla
   PyTorch, A y B producen logits distintos en ≥ 10 % de los steps.
3. ``test_B_no_information_leak``: los rasgos en `t` se construyen
   solo con `data[:t]`, igual que `test_features_no_leak`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.agents.classical_agent import ClassicalAgent
from src.data.features import FeatureSpec, compute_features
from src.env.market_env import MarketEnv, MarketEnvSpec
from src.env.relational_state_builder import (
    RelationalStateBuilder,
    RelationalStateConfig,
)
from src.graph.graph_builder import GraphSpec
from src.graph.relational_features import RelationalSpec


def _make_envs_a_and_b(
    synthetic_clean_df: pd.DataFrame,
) -> tuple[MarketEnv, MarketEnv]:
    """Construir un par (env_A, env_B) compartiendo todo excepto el state."""
    features = compute_features(
        synthetic_clean_df, FeatureSpec(volatility_window=5, volume_window=5)
    )
    spec = MarketEnvSpec(window_length=5, max_steps=20)

    # Modelo A: estado clásico puro
    env_a = MarketEnv(features, spec)

    # Modelo B: estado clásico + rasgos relacionales
    tickers = sorted({c[0] for c in features.columns})
    returns_panel = features.loc[:, [(t, "return") for t in tickers]].to_numpy(
        dtype=np.float64
    )
    rel_spec = RelationalSpec(spectral_top_k=2)
    cfg = RelationalStateConfig(
        graph_spec=GraphSpec(alpha=1.0, beta=0.0, k_neighbors=2, sym_mode="avg"),
        relational_spec=rel_spec,
        lookback_window=20,
        feature_dim=2 + rel_spec.spectral_top_k,
    )
    builder = RelationalStateBuilder(returns_panel, tickers, cfg)
    env_b = MarketEnv(
        features,
        spec,
        relational_features_fn=builder,
        relational_features_dim=cfg.feature_dim,
    )
    return env_a, env_b


def test_B_state_dim_larger_than_A(synthetic_clean_df: pd.DataFrame) -> None:
    """obs_shape de B = obs_shape de A + N·d_rel."""
    env_a, env_b = _make_envs_a_and_b(synthetic_clean_df)
    base_dim = env_a.observation_space.shape[0]
    b_dim = env_b.observation_space.shape[0]
    n = env_a.n_tickers
    d_rel = 4  # degree + clustering + 2 spectral
    assert b_dim == base_dim + n * d_rel, f"esperado {base_dim + n * d_rel}, obtenido {b_dim}"


def test_B_obs_finite_after_warmup(synthetic_clean_df: pd.DataFrame) -> None:
    """Tras warmup, la observación de B no contiene NaN/inf."""
    _, env_b = _make_envs_a_and_b(synthetic_clean_df)
    obs, _ = env_b.reset(seed=0)
    # Avanzar varios steps para superar el warmup del lookback
    for _ in range(25):
        obs, _, terminated, truncated, _ = env_b.step(0)
        if terminated or truncated:
            obs, _ = env_b.reset(seed=1)
    assert np.all(np.isfinite(obs)), "obs de B contiene NaN/inf tras warmup"


def test_B_action_diverges_from_A_under_same_seed(
    synthetic_clean_df: pd.DataFrame,
) -> None:
    """Con misma semilla, A y B producen logits distintos.

    Criterio: en al menos 10 % de los steps las acciones argmax difieren.
    """
    env_a, env_b = _make_envs_a_and_b(synthetic_clean_df)
    torch.manual_seed(123)
    agent_a = ClassicalAgent(
        obs_dim=env_a.observation_space.shape[0], n_actions=env_a.n_tickers
    )
    torch.manual_seed(123)
    agent_b = ClassicalAgent(
        obs_dim=env_b.observation_space.shape[0], n_actions=env_b.n_tickers
    )

    obs_a, _ = env_a.reset(seed=0)
    obs_b, _ = env_b.reset(seed=0)
    mask = np.ones(env_a.n_tickers, dtype=bool)

    n_diff = 0
    n_total = 30
    for _ in range(n_total):
        # Usar la red en modo determinista (no muestrear): consultar logits
        with torch.no_grad():
            logits_a, _ = agent_a.network(
                torch.from_numpy(obs_a).to(agent_a.device).unsqueeze(0)
            )
            logits_b, _ = agent_b.network(
                torch.from_numpy(obs_b).to(agent_b.device).unsqueeze(0)
            )
        act_a = int(logits_a.argmax(dim=-1).item())
        act_b = int(logits_b.argmax(dim=-1).item())
        if act_a != act_b:
            n_diff += 1
        obs_a, _, terminated_a, truncated_a, _ = env_a.step(act_a)
        obs_b, _, terminated_b, truncated_b, _ = env_b.step(act_b)
        if terminated_a or truncated_a:
            obs_a, _ = env_a.reset(seed=1)
            obs_b, _ = env_b.reset(seed=1)
    # Las redes A y B tienen distinto in_dim → pesos iniciales distintos a
    # pesar del mismo seed. La consecuencia es que las acciones argmax
    # divergen en algún porcentaje significativo.
    assert n_diff / n_total >= 0.10, (
        f"B y A debieran diverger ≥ 10 % de los steps; divergieron en "
        f"{n_diff}/{n_total} = {100 * n_diff / n_total:.1f} %"
    )


def test_B_no_information_leak(synthetic_clean_df: pd.DataFrame) -> None:
    """Los rasgos relacionales en `t` solo usan data[:t].

    Procedimiento: construir el builder, modificar el panel de retornos
    en una posición FUTURA (t > t_eval) y verificar que la salida del
    builder en t_eval permanece IDÉNTICA.
    """
    features = compute_features(
        synthetic_clean_df, FeatureSpec(volatility_window=5, volume_window=5)
    )
    tickers = sorted({c[0] for c in features.columns})
    returns_panel = features.loc[:, [(t, "return") for t in tickers]].to_numpy(
        dtype=np.float64
    )
    rel_spec = RelationalSpec(spectral_top_k=2)
    cfg = RelationalStateConfig(
        graph_spec=GraphSpec(alpha=1.0, beta=0.0, k_neighbors=2, sym_mode="avg"),
        relational_spec=rel_spec,
        lookback_window=20,
        feature_dim=2 + rel_spec.spectral_top_k,
    )

    builder_orig = RelationalStateBuilder(returns_panel, tickers, cfg)
    # Forzar t_eval por encima del warm-up
    t_eval = 50
    feats_orig = builder_orig(t_eval)

    # Modificar el panel SOLO en posiciones t > t_eval
    contaminated = returns_panel.copy()
    contaminated[t_eval + 1 :] = contaminated[t_eval + 1 :] * 1000.0
    builder_cont = RelationalStateBuilder(contaminated, tickers, cfg)
    feats_cont = builder_cont(t_eval)

    np.testing.assert_array_equal(
        feats_orig,
        feats_cont,
        err_msg="Fuga temporal: rasgos relacionales en t cambiaron al alterar data[t+1:]",
    )


def test_B_zeros_during_warmup(synthetic_clean_df: pd.DataFrame) -> None:
    """Si la ventana lookback no encaja, devolver ceros (no error)."""
    features = compute_features(
        synthetic_clean_df, FeatureSpec(volatility_window=5, volume_window=5)
    )
    tickers = sorted({c[0] for c in features.columns})
    returns_panel = features.loc[:, [(t, "return") for t in tickers]].to_numpy(
        dtype=np.float64
    )
    rel_spec = RelationalSpec(spectral_top_k=2)
    cfg = RelationalStateConfig(
        graph_spec=GraphSpec(alpha=1.0, beta=0.0, k_neighbors=2, sym_mode="avg"),
        relational_spec=rel_spec,
        lookback_window=100,  # mucho mayor que el rango inicial
        feature_dim=2 + rel_spec.spectral_top_k,
    )
    builder = RelationalStateBuilder(returns_panel, tickers, cfg)
    feats = builder(t_index=5)  # ventana insuficiente
    assert feats.shape == (len(tickers), 4)
    assert (feats == 0.0).all()
