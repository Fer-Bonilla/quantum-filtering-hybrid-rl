"""Tests del agente híbrido (Modelos C y D)."""

from __future__ import annotations

import pandas as pd
from src.agents.classical_agent import ClassicalAgent
from src.agents.hybrid_agent import (
    HybridAgent,
    HybridSpec,
    build_returns_panel,
    build_ticker_index,
)
from src.data.features import FeatureSpec, compute_features
from src.env.market_env import MarketEnv, MarketEnvSpec
from src.graph.graph_builder import GraphSpec
from src.quantum.classical_walker import ClassicalWalker
from src.quantum.quantum_walker import QuantumWalker


def _setup_env_and_agent(
    synthetic_clean_df: pd.DataFrame,
    local_module: ClassicalWalker | QuantumWalker,
) -> tuple[MarketEnv, HybridAgent]:
    features = compute_features(
        synthetic_clean_df, FeatureSpec(volatility_window=5, volume_window=5)
    )
    env = MarketEnv(features, MarketEnvSpec(window_length=5, max_steps=30))
    classical = ClassicalAgent(obs_dim=env.observation_space.shape[0], n_actions=env.n_tickers)
    spec = HybridSpec(
        graph_spec=GraphSpec(alpha=1.0, beta=0.0, k_neighbors=2, sym_mode="avg"),
        subgraph_max_size=3,
        seed_score_window=5,
        graph_lookback=10,
        k_steps=3,
        m_top=2,
    )
    returns_panel = build_returns_panel(features, env.tickers)
    hybrid = HybridAgent(
        classical_agent=classical,
        local_module=local_module,
        hybrid_spec=spec,
        returns_panel=returns_panel,
        ticker_to_idx=build_ticker_index(env.tickers),
    )
    return env, hybrid


def test_hybrid_classical_walker_produces_topm_mask(
    synthetic_clean_df: pd.DataFrame,
) -> None:
    env, hybrid = _setup_env_and_agent(synthetic_clean_df, ClassicalWalker())
    obs, info = env.reset(seed=0)
    # Avanzar varios steps para que haya histórico
    for _ in range(15):
        obs, _, _, _, info = env.step(0)
    mask = hybrid.make_mask(env, obs, info)
    assert mask.dtype == bool
    assert mask.shape == (env.n_tickers,)
    assert int(mask.sum()) <= 2  # m_top = 2


def test_hybrid_quantum_walker_produces_topm_mask(
    synthetic_clean_df: pd.DataFrame,
) -> None:
    env, hybrid = _setup_env_and_agent(synthetic_clean_df, QuantumWalker())
    obs, info = env.reset(seed=0)
    for _ in range(15):
        obs, _, _, _, info = env.step(0)
    mask = hybrid.make_mask(env, obs, info)
    assert mask.dtype == bool
    assert mask.shape == (env.n_tickers,)
    assert int(mask.sum()) <= 2


def test_hybrid_c_vs_d_swap_doesnt_break_pipeline(
    synthetic_clean_df: pd.DataFrame,
) -> None:
    """Test crítico estructural: cambiar LocalModule de C a D no rompe nada."""
    env_c, hybrid_c = _setup_env_and_agent(synthetic_clean_df, ClassicalWalker())
    env_d, hybrid_d = _setup_env_and_agent(synthetic_clean_df, QuantumWalker())
    obs_c, info_c = env_c.reset(seed=0)
    obs_d, info_d = env_d.reset(seed=0)
    for _ in range(15):
        obs_c, _, _, _, info_c = env_c.step(0)
        obs_d, _, _, _, info_d = env_d.step(0)
    mask_c = hybrid_c.make_mask(env_c, obs_c, info_c)
    mask_d = hybrid_d.make_mask(env_d, obs_d, info_d)
    # Misma estructura
    assert mask_c.shape == mask_d.shape
    assert mask_c.dtype == mask_d.dtype


def test_hybrid_mask_after_reset_is_valid(
    synthetic_clean_df: pd.DataFrame,
) -> None:
    """Tras reset, la máscara debe tener forma correcta y al menos una acción.

    El EpisodeSampler garantiza que el punto inicial tiene historia suficiente,
    así que el hybrid agent puede construir el grafo desde el primer step.
    """
    env, hybrid = _setup_env_and_agent(synthetic_clean_df, ClassicalWalker())
    obs, info = env.reset(seed=0)
    mask = hybrid.make_mask(env, obs, info)
    assert mask.shape == (env.n_tickers,)
    assert mask.dtype == bool
    assert mask.any(), "La máscara debe tener al menos una posición True."
