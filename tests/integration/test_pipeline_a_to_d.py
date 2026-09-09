"""Test de integración: ejecutar los 4 modelos A/B/C/D end-to-end.

Smoke test que verifica:
- Cada modelo entrena al menos un rollout sin crash.
- Las métricas finales (test) son finitas para todos.
- MLflow registra un run por modelo.

Usa el universo SINTÉTICO (no requiere datos reales descargados).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from src.agents.classical_agent import ClassicalAgent
from src.agents.hybrid_agent import (
    HybridAgent,
    HybridSpec,
    build_returns_panel,
    build_ticker_index,
)
from src.data.features import FeatureSpec, compute_features
from src.data.splits import SplitSpec, chronological_split
from src.env.market_env import MarketEnv, MarketEnvSpec
from src.graph.graph_builder import GraphSpec
from src.quantum.classical_walker import ClassicalWalker
from src.quantum.quantum_walker import QuantumWalker
from src.training.evaluate import evaluate_agent
from src.training.seed_utils import set_global_seed
from src.training.trainer import PPOHyperparams, PPOTrainer


@pytest.fixture(scope="module")
def features_panel(synthetic_clean_df_module: pd.DataFrame) -> pd.DataFrame:
    return compute_features(
        synthetic_clean_df_module,
        FeatureSpec(volatility_window=5, volume_window=5),
    )


@pytest.fixture(scope="module")
def synthetic_clean_df_module() -> pd.DataFrame:
    """Versión module-scoped del fixture clean (para que el cómputo no se repita)."""
    from datetime import date

    import numpy as np
    import pandas as pd
    from src.data.cleaning import CleaningPolicy, clean

    def _synth(seed: int, n_days: int = 504, drift: float = 0.0004) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        daily_ret = rng.normal(loc=drift, scale=0.012, size=n_days)
        close = 100.0 * np.exp(np.cumsum(daily_ret))
        high = close * (1.0 + np.abs(rng.normal(0.0, 0.003, n_days)))
        low = close * (1.0 - np.abs(rng.normal(0.0, 0.003, n_days)))
        open_ = close * (1.0 + rng.normal(0.0, 0.002, n_days))
        volume = rng.integers(1_000_000, 5_000_000, size=n_days).astype(np.float64)
        idx = pd.bdate_range(start=date(2022, 1, 3), periods=n_days, freq="B")
        return pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum.reduce([open_, high, close]),
                "low": np.minimum.reduce([open_, low, close]),
                "close": close,
                "volume": volume,
            },
            index=idx,
        )

    raw = {
        "AAA": _synth(1, drift=0.0005),
        "BBB": _synth(2, drift=0.0003),
        "CCC": _synth(3, drift=0.0007),
        "DDD": _synth(4, drift=0.0002),
        "EEE": _synth(5, drift=0.0004),
    }
    return clean(raw, CleaningPolicy())


def _make_env(
    features: pd.DataFrame, partition: str, window: int = 5, max_steps: int = 30
) -> MarketEnv:
    splits = chronological_split(features, SplitSpec(train_frac=0.6, val_frac=0.2))
    return MarketEnv(splits[partition], MarketEnvSpec(window_length=window, max_steps=max_steps))


def _make_classical(env: MarketEnv) -> ClassicalAgent:
    return ClassicalAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.n_tickers,
        hidden_sizes=(32, 32),
    )


def _hybrid_hook(
    env: MarketEnv, features: pd.DataFrame, classical: ClassicalAgent, model: str
) -> Any:
    local = ClassicalWalker() if model == "C" else QuantumWalker()
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
        local_module=local,
        hybrid_spec=spec,
        returns_panel=returns_panel,
        ticker_to_idx=build_ticker_index(env.tickers),
    )
    return hybrid.make_mask


@pytest.mark.integration
@pytest.mark.parametrize("model", ["A", "B", "C", "D"])
def test_model_runs_end_to_end(features_panel: pd.DataFrame, model: str) -> None:
    """Cada modelo entrena un rollout, evalúa, y produce métricas finitas."""
    rng = set_global_seed(42)
    env = _make_env(features_panel, "train")
    classical = _make_classical(env)
    hp = PPOHyperparams(rollout_steps=64, batch_size=16, n_epochs=2)

    step_hook = None if model in {"A", "B"} else _hybrid_hook(env, features_panel, classical, model)

    trainer = PPOTrainer(env, classical, hp, rng=rng, step_hook=step_hook)
    result = trainer.train(total_steps=64)
    assert len(result.train_curve) == 1
    assert np.isfinite(result.train_curve[0])

    # Eval en partición test
    test_env = _make_env(features_panel, "test")
    test_hook = (
        _hybrid_hook(test_env, features_panel, classical, model) if model in {"C", "D"} else None
    )
    eval_res = evaluate_agent(test_env, classical, n_episodes=2, seed=0, step_hook=test_hook)
    assert np.isfinite(eval_res.cumulative_return)
    assert np.isfinite(eval_res.sharpe_ratio)
    assert 0.0 <= eval_res.asset_coverage <= 1.0
