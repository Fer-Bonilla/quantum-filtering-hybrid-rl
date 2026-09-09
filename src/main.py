"""CLI principal del proyecto.

Comandos:
    python -m src.main download --config configs/data/nivel1.yaml
    python -m src.main train --config configs/experiment/model_a.yaml --seed 42
    python -m src.main train --config configs/experiment/model_d.yaml --seed 42 --steps 5000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from src.agents.classical_agent import ClassicalAgent
from src.agents.hybrid_agent import (
    HybridAgent,
    HybridSpec,
    LocalModule,
    build_returns_panel,
    build_ticker_index,
)
from src.data.cleaning import CleaningPolicy, clean
from src.data.download import fetch_ohlcv, load_ohlcv
from src.data.features import FeatureSpec, compute_features
from src.data.sector_map import build_sector_map
from src.data.splits import SplitSpec, chronological_split
from src.env.market_env import MarketEnv, MarketEnvSpec
from src.env.reward import RewardCoefficients
from src.graph.graph_builder import GraphSpec
from src.quantum.classical_walker import ClassicalWalker
from src.quantum.quantum_walker import QuantumWalker
from src.training.evaluate import (
    PromisingSpec,
    compute_promising_matrix,
    episodes_to_convergence,
    evaluate_agent,
)
from src.training.mlflow_logger import MLflowLogger
from src.training.seed_utils import set_global_seed
from src.training.trainer import PPOHyperparams, PPOTrainer
from src.utils.config import ExperimentConfig, load_config
from src.utils.logging import get_logger, setup_logging

_log = get_logger(__name__)

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("download")
def cmd_download(
    config: Path = typer.Option(..., "--config", help="YAML con sección 'data'."),
) -> None:
    """Descargar OHLCV según el config y escribir manifest."""
    setup_logging("INFO")
    cfg = load_config(config)
    fetch_ohlcv(
        cfg.data.tickers,
        cfg.data.start_date,
        cfg.data.end_date,
        universe=cfg.data.cache_subdir,
        interval=cfg.data.interval,
        auto_adjust=cfg.data.auto_adjust,
    )


@app.command("train")
def cmd_train(
    config: Path = typer.Option(..., "--config", help="YAML del experimento."),
    seed: int = typer.Option(42, "--seed"),
    steps: int | None = typer.Option(
        None,
        "--steps",
        help="Override total_steps (útil para smoke tests rápidos).",
    ),
) -> None:
    """Entrenar el modelo (A, B, C o D) descrito por el config."""
    setup_logging("INFO")
    cfg = load_config(config)
    if steps is not None:
        cfg.training.total_steps = steps

    rng = set_global_seed(seed)
    raw = load_ohlcv(universe=cfg.data.cache_subdir)
    clean_df = clean(raw, CleaningPolicy())
    feature_spec = FeatureSpec(
        return_type=cfg.env.feature.return_type,
        volatility_window=cfg.env.feature.volatility_window,
        volume_window=cfg.env.feature.volume_window,
        indicators=tuple(cfg.env.feature.indicators),
    )
    features = compute_features(clean_df, feature_spec)
    splits = chronological_split(
        features,
        SplitSpec(
            train_frac=cfg.env.split.train_frac,
            val_frac=cfg.env.split.val_frac,
        ),
    )

    env = MarketEnv(
        splits["train"],
        MarketEnvSpec(
            window_length=cfg.env.feature.window_length,
            max_steps=cfg.env.max_steps,
            reward_coefs=RewardCoefficients(
                lambda_risk=cfg.env.lambda_risk,
                mu_cost=cfg.env.mu_cost,
                transaction_cost=cfg.env.transaction_cost,
            ),
            episode_sampling=cfg.env.episode_sampling,
        ),
    )

    obs_dim = env.observation_space.shape[0]
    classical = ClassicalAgent(
        obs_dim=obs_dim,
        n_actions=env.n_tickers,
        hidden_sizes=cfg.agent.ppo.hidden_sizes,
    )
    hp = PPOHyperparams(
        learning_rate=cfg.agent.ppo.learning_rate,
        n_epochs=cfg.agent.ppo.n_epochs,
        batch_size=cfg.agent.ppo.batch_size,
        rollout_steps=cfg.agent.ppo.rollout_steps,
        clip_coef=cfg.agent.ppo.clip_coef,
        gamma=cfg.agent.ppo.gamma,
        gae_lambda=cfg.agent.ppo.gae_lambda,
        entropy_coef=cfg.agent.ppo.entropy_coef,
        value_coef=cfg.agent.ppo.value_coef,
        max_grad_norm=cfg.agent.ppo.max_grad_norm,
    )

    # Hook condicional según el modelo
    step_hook = _build_step_hook(cfg, env, features, classical)

    mlflow_logger = MLflowLogger(cfg.training.mlflow_experiment_name)
    with mlflow_logger.start_run(
        run_name=f"{cfg.agent.model}_seed{seed}",
        tags={"model": cfg.agent.model, "seed": str(seed)},
    ):
        mlflow_logger.log_params(
            {
                "model": cfg.agent.model,
                "seed": seed,
                "ppo": cfg.agent.ppo.model_dump(),
                "env": {
                    "window_length": cfg.env.feature.window_length,
                    "max_steps": cfg.env.max_steps,
                },
                "graph": cfg.graph.model_dump() if cfg.graph else None,
                "quantum": cfg.quantum.model_dump() if cfg.quantum else None,
            }
        )

        trainer = PPOTrainer(env, classical, hp, rng=rng, logger=mlflow_logger, step_hook=step_hook)
        result = trainer.train(total_steps=cfg.training.total_steps)

        # Evaluación en partición test
        test_env = MarketEnv(splits["test"], env._spec)
        test_hook = _build_step_hook(cfg, test_env, features, classical)
        # Construir promising matrix sobre el panel de retornos de TEST
        returns_panel_test = _returns_panel(splits["test"], test_env.tickers)
        promising = compute_promising_matrix(returns_panel_test, PromisingSpec())
        eval_result = evaluate_agent(
            test_env,
            classical,
            n_episodes=5,
            seed=seed,
            step_hook=test_hook,
            promising_matrix=promising,
        )
        convergence_idx = episodes_to_convergence(result.train_curve)
        mlflow_logger.log_metrics(
            {
                "test/cumulative_return": eval_result.cumulative_return,
                "test/sharpe": eval_result.sharpe_ratio,
                "test/max_drawdown": eval_result.max_drawdown,
                "test/mean_reward": eval_result.mean_reward,
                "test/coverage": eval_result.asset_coverage,
                "test/topm_hit_rate": eval_result.topm_hit_rate,
                "test/candidate_hit_rate": eval_result.candidate_hit_rate,
                "test/time_to_first_promising": eval_result.time_to_first_promising,
                "test/mean_latency_ms": eval_result.mean_latency_ms,
                "learning/episodes_to_convergence": convergence_idx
                if convergence_idx != float("inf")
                else -1.0,
            }
        )

    _log.info(
        "Entrenamiento %s finalizado: %d steps, last_reward=%.4f, "
        "test_return=%.4f, test_sharpe=%.4f",
        cfg.agent.model,
        result.total_steps,
        result.train_curve[-1] if result.train_curve else 0.0,
        eval_result.cumulative_return,
        eval_result.sharpe_ratio,
    )


def _returns_panel(features: Any, tickers: list[str]) -> Any:
    """Extraer matriz (T, N) de retornos para los tickers en orden."""
    cols = [(t, "return") for t in tickers]
    return features.loc[:, cols].to_numpy(dtype=float, copy=True)


def _build_step_hook(
    cfg: ExperimentConfig,
    env: MarketEnv,
    features: Any,
    classical: ClassicalAgent,
) -> Any | None:
    """Construir el step_hook según el modelo elegido."""
    if cfg.agent.model in {"A", "B"}:
        # Modelo A: máscara all-True por defecto del env.
        # Modelo B: rasgos relacionales se agregarían al state_builder en el
        # futuro; por ahora la máscara queda all-True y el grafo solo
        # influye vía estado (pendiente de Semana 5+).
        return None

    if cfg.graph is None or cfg.quantum is None:
        raise ValueError(f"Modelo {cfg.agent.model} requiere graph y quantum configs.")

    graph_spec = GraphSpec(
        alpha=cfg.graph.alpha,
        beta=cfg.graph.beta,
        eps=cfg.graph.eps,
        k_neighbors=cfg.graph.k_neighbors,
        sym_mode=cfg.graph.sym_mode,
    )
    local: LocalModule
    if cfg.agent.model == "C":
        local = ClassicalWalker()
    else:  # "D"
        from src.quantum.noise import NoiseSpec

        noise_spec = (
            NoiseSpec(
                depolarizing_prob=cfg.quantum.noise.depolarizing_prob,
                dephasing_prob=cfg.quantum.noise.dephasing_prob,
            )
            if cfg.quantum.noise.enabled
            else NoiseSpec()
        )
        local = QuantumWalker(
            init_mode=cfg.quantum.init_mode,
            renormalize_threshold=cfg.quantum.renormalize_threshold,
            backend=cfg.quantum.backend,
            noise=noise_spec,
        )

    returns_panel = build_returns_panel(features, env.tickers)
    sectors_map = build_sector_map(env.tickers, allow_online=False)
    sectors = [sectors_map[t] for t in env.tickers] if cfg.graph.beta > 0 else None
    hybrid_spec = HybridSpec(
        graph_spec=graph_spec,
        subgraph_max_size=cfg.graph.subgraph_max_size,
        seed_score_window=cfg.graph.seed_score_window,
        graph_lookback=cfg.graph.lookback_window,
        k_steps=cfg.quantum.k_steps,
        m_top=cfg.quantum.m_top,
        update_frequency=cfg.graph.update_frequency,
    )
    hybrid = HybridAgent(
        classical_agent=classical,
        local_module=local,
        hybrid_spec=hybrid_spec,
        returns_panel=returns_panel,
        ticker_to_idx=build_ticker_index(env.tickers),
        sectors=sectors,
    )
    return hybrid.make_mask


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
