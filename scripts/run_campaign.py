"""Campaña experimental mínima viable (Sec. 8.23 del documento).

Ejecuta los 4 modelos A, B, C, D con N semillas y registra:
- Cada run en MLflow (un experiment por modelo).
- Una fila por (model, seed) en ``outputs/tables/<campaign_id>.csv`` con
  las métricas finales financieras + de aprendizaje + de exploración.

Uso::

    uv run python scripts/run_campaign.py --universe nivel1 --seeds 42 123 456 --steps 5000
    uv run python scripts/run_campaign.py --universe smoke_train --seeds 42 --steps 512 --campaign-id smoke

Las dependencias (datos descargados) deben existir en ``data/raw/<universe>/``
antes de invocar; si no existen, el script intenta descargar con el config
``configs/data/<universe>.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from src.agents.classical_agent import ClassicalAgent
from src.data.cleaning import CleaningPolicy, clean
from src.data.download import fetch_ohlcv, load_ohlcv
from src.data.features import FeatureSpec, compute_features
from src.data.sector_map import build_sector_map
from src.data.splits import SplitSpec, chronological_split
from src.env.market_env import MarketEnv, MarketEnvSpec
from src.env.relational_state_builder import RelationalStateBuilder, RelationalStateConfig
from src.env.reward import RewardCoefficients
from src.graph.graph_builder import GraphSpec
from src.graph.relational_features import RelationalSpec
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
from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import CONFIGS_DIR, TABLES_DIR, ensure_dir

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RunRow:
    """Una fila del CSV agregado."""

    campaign_id: str
    model: str
    seed: int
    universe: str
    total_steps: int
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    mean_reward: float
    asset_coverage: float
    topm_hit_rate: float
    candidate_hit_rate: float
    time_to_first_promising: float
    mean_latency_ms: float
    episodes_to_convergence: float
    train_reward_last: float
    train_reward_mean: float
    n_train_steps: int
    timestamp_utc: str
    duration_seconds: float


def _build_relational_fn(
    cfg, panel_df, *, relational_spec: RelationalSpec | None = None
) -> tuple[RelationalStateBuilder | None, int]:
    """Construir el ``RelationalStateBuilder`` para el Modelo B v2.

    Args:
        cfg: ``ExperimentConfig`` (necesita ``cfg.graph`` y
            ``cfg.env.use_relational_features``).
        panel_df: DataFrame del split correspondiente (``train`` o ``test``).
            El builder se ata a ese panel para mantener consistencia con
            el ``MarketEnv`` que se construye sobre él.
        relational_spec: configuración de rasgos. Por defecto se usa
            ``RelationalSpec(spectral_top_k=2)`` → d_rel = 4.

    Returns:
        ``(builder, d_rel)``. Si ``use_relational_features`` está
        desactivado o el modelo no es B, devuelve ``(None, 0)``.
    """
    if not cfg.env.use_relational_features:
        return None, 0
    if cfg.graph is None:
        raise ValueError(
            "use_relational_features=True requiere cfg.graph (lookback_window, alpha, beta, ...)"
        )
    spec = relational_spec or RelationalSpec(spectral_top_k=2)
    d_rel = 2 + spec.spectral_top_k

    tickers = sorted({c[0] for c in panel_df.columns})
    cols = [(t, "return") for t in tickers]
    returns_panel = panel_df.loc[:, cols].to_numpy(dtype=np.float64, copy=True)

    graph_spec = GraphSpec(
        alpha=cfg.graph.alpha,
        beta=cfg.graph.beta,
        eps=cfg.graph.eps,
        k_neighbors=cfg.graph.k_neighbors,
        sym_mode=cfg.graph.sym_mode,
    )
    rel_cfg = RelationalStateConfig(
        graph_spec=graph_spec,
        relational_spec=spec,
        lookback_window=cfg.graph.lookback_window,
        feature_dim=d_rel,
    )
    sectors = None
    if cfg.graph.beta > 0:
        sectors_map = build_sector_map(tickers, allow_online=False)
        sectors = [sectors_map[t] for t in tickers]
    builder = RelationalStateBuilder(
        returns_panel=returns_panel,
        node_names=tickers,
        cfg=rel_cfg,
        sectors=sectors,
    )
    return builder, d_rel


def _build_hybrid_hook(cfg, env: MarketEnv, features, classical: ClassicalAgent, model: str):
    """Construir step_hook según el modelo (None para A/B)."""
    if model in {"A", "B"}:
        return None
    if cfg.graph is None or cfg.quantum is None:
        raise ValueError(f"Modelo {model} requiere graph y quantum configs.")

    from src.agents.hybrid_agent import (
        HybridAgent,
        HybridSpec,
        build_returns_panel,
        build_ticker_index,
    )

    if model == "C":
        local = ClassicalWalker()
    else:
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
    hybrid_spec = HybridSpec(
        graph_spec=GraphSpec(
            alpha=cfg.graph.alpha,
            beta=cfg.graph.beta,
            eps=cfg.graph.eps,
            k_neighbors=cfg.graph.k_neighbors,
            sym_mode=cfg.graph.sym_mode,
        ),
        subgraph_max_size=cfg.graph.subgraph_max_size,
        seed_score_window=cfg.graph.seed_score_window,
        graph_lookback=cfg.graph.lookback_window,
        k_steps=cfg.quantum.k_steps,
        m_top=cfg.quantum.m_top,
        update_frequency=cfg.graph.update_frequency,
    )
    sectors_map = build_sector_map(env.tickers, allow_online=False)
    sectors = [sectors_map[t] for t in env.tickers] if cfg.graph.beta > 0 else None
    hybrid = HybridAgent(
        classical_agent=classical,
        local_module=local,
        hybrid_spec=hybrid_spec,
        returns_panel=build_returns_panel(features, env.tickers),
        ticker_to_idx=build_ticker_index(env.tickers),
        sectors=sectors,
    )
    return hybrid.make_mask


def _train_and_evaluate_one(
    model: str,
    seed: int,
    universe: str,
    steps: int,
    campaign_id: str,
    *,
    max_steps_override: int | None = None,
    window_override: int | None = None,
    config_suffix: str = "",
    filter_dates: tuple[str, str] | None = None,
    sweet_spot_overrides: dict | None = None,
) -> RunRow:
    """Una corrida completa: train + eval + métricas.

    Args:
        config_suffix: si no vacío, carga ``model_{x}_{suffix}.yaml``
            en lugar de ``model_{x}.yaml``. Útil para campañas v2.
        filter_dates: si no None, ``(start, end)`` para restringir el
            panel de features a ese rango (inclusive). Usado para la
            validación hold-out 2023-24 (Anexo M).
        sweet_spot_overrides: si no None, dict con claves
            ``subgraph_max_size``, ``beta``, ``alpha``, ``k_steps``,
            ``m_top`` para fijar la configuración ``sweet_spot``.
    """
    suffix = f"_{config_suffix}" if config_suffix else ""
    cfg_path = CONFIGS_DIR / "experiment" / f"model_{model.lower()}{suffix}.yaml"
    cfg = load_config(cfg_path)
    # Ajustar al universo de la campaña
    cfg.data.cache_subdir = universe
    cfg.training.total_steps = steps
    cfg.training.mlflow_experiment_name = f"campaign_{campaign_id}_{model}"
    if max_steps_override is not None:
        cfg.env.max_steps = max_steps_override
    if window_override is not None:
        cfg.env.feature.window_length = window_override
    if sweet_spot_overrides is not None and cfg.graph is not None:
        if "subgraph_max_size" in sweet_spot_overrides:
            cfg.graph.subgraph_max_size = sweet_spot_overrides["subgraph_max_size"]
        if "beta" in sweet_spot_overrides:
            cfg.graph.beta = sweet_spot_overrides["beta"]
            cfg.graph.alpha = 1.0 - sweet_spot_overrides["beta"]
        if "k_steps" in sweet_spot_overrides and cfg.quantum is not None:
            cfg.quantum.k_steps = sweet_spot_overrides["k_steps"]
        if "m_top" in sweet_spot_overrides and cfg.quantum is not None:
            cfg.quantum.m_top = sweet_spot_overrides["m_top"]

    rng = set_global_seed(seed)
    raw = load_ohlcv(universe=universe)
    clean_df = clean(raw, CleaningPolicy())
    feature_spec = FeatureSpec(
        return_type=cfg.env.feature.return_type,
        volatility_window=cfg.env.feature.volatility_window,
        volume_window=cfg.env.feature.volume_window,
        indicators=tuple(cfg.env.feature.indicators),
    )
    features = compute_features(clean_df, feature_spec)

    # Hold-out temporal (rev. v2 — Anexo M): restringir el panel al
    # bloque deseado ANTES del split interno 60/20/20.
    if filter_dates is not None:
        start, end = filter_dates
        before = len(features)
        features = features.loc[start:end]
        _log.info(
            "Filtro temporal: %s..%s → %d/%d filas conservadas",
            start, end, len(features), before,
        )
        if len(features) < cfg.env.feature.window_length + cfg.env.max_steps + 10:
            raise ValueError(
                f"Panel filtrado a [{start},{end}] tiene {len(features)} filas; "
                f"insuficiente para window_length + max_steps."
            )

    splits = chronological_split(
        features, SplitSpec(train_frac=cfg.env.split.train_frac, val_frac=cfg.env.split.val_frac)
    )

    env_spec = MarketEnvSpec(
        window_length=cfg.env.feature.window_length,
        max_steps=cfg.env.max_steps,
        reward_coefs=RewardCoefficients(
            lambda_risk=cfg.env.lambda_risk,
            mu_cost=cfg.env.mu_cost,
            transaction_cost=cfg.env.transaction_cost,
            reward_type=cfg.env.reward_type,
        ),
        episode_sampling=cfg.env.episode_sampling,
    )

    # Modelo B v2: state_builder relacional (Problema 1.4 del revisor).
    if model == "B" and cfg.env.use_relational_features:
        train_rel_fn, train_rel_dim = _build_relational_fn(cfg, splits["train"])
    else:
        train_rel_fn, train_rel_dim = None, 0

    env = MarketEnv(
        splits["train"],
        env_spec,
        relational_features_fn=train_rel_fn,
        relational_features_dim=train_rel_dim,
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

    step_hook = _build_hybrid_hook(cfg, env, features, classical, model)

    logger = MLflowLogger(cfg.training.mlflow_experiment_name)
    start_t = time.perf_counter()
    with logger.start_run(
        run_name=f"{model}_seed{seed}",
        tags={"model": model, "seed": str(seed), "campaign_id": campaign_id, "universe": universe},
    ):
        logger.log_params(
            {
                "model": model,
                "seed": seed,
                "total_steps": steps,
                "universe": universe,
                "ppo": cfg.agent.ppo.model_dump(),
                "graph": cfg.graph.model_dump() if cfg.graph else None,
                "quantum": cfg.quantum.model_dump() if cfg.quantum else None,
            }
        )

        trainer = PPOTrainer(env, classical, hp, rng=rng, logger=logger, step_hook=step_hook)
        result = trainer.train(total_steps=steps)

        if model == "B" and cfg.env.use_relational_features:
            test_rel_fn, test_rel_dim = _build_relational_fn(cfg, splits["test"])
        else:
            test_rel_fn, test_rel_dim = None, 0
        test_env = MarketEnv(
            splits["test"],
            env_spec,
            relational_features_fn=test_rel_fn,
            relational_features_dim=test_rel_dim,
        )
        test_hook = _build_hybrid_hook(cfg, test_env, features, classical, model)
        cols = [(t, "return") for t in test_env.tickers]
        returns_panel_test = splits["test"].loc[:, cols].to_numpy(dtype=float)
        promising = compute_promising_matrix(returns_panel_test, PromisingSpec())
        eval_res = evaluate_agent(
            test_env,
            classical,
            n_episodes=5,
            seed=seed,
            step_hook=test_hook,
            promising_matrix=promising,
        )
        convergence = episodes_to_convergence(result.train_curve)
        logger.log_metrics(
            {
                "test/cumulative_return": eval_res.cumulative_return,
                "test/sharpe": eval_res.sharpe_ratio,
                "test/max_drawdown": eval_res.max_drawdown,
                "test/mean_reward": eval_res.mean_reward,
                "test/coverage": eval_res.asset_coverage,
                "test/topm_hit_rate": eval_res.topm_hit_rate,
                "test/candidate_hit_rate": eval_res.candidate_hit_rate,
                "test/time_to_first_promising": eval_res.time_to_first_promising
                if eval_res.time_to_first_promising != float("inf")
                else -1.0,
                "test/mean_latency_ms": eval_res.mean_latency_ms,
                "learning/episodes_to_convergence": convergence
                if convergence != float("inf")
                else -1.0,
            }
        )

    duration = time.perf_counter() - start_t
    train_curve = np.asarray(result.train_curve, dtype=np.float64)
    return RunRow(
        campaign_id=campaign_id,
        model=model,
        seed=seed,
        universe=universe,
        total_steps=steps,
        cumulative_return=eval_res.cumulative_return,
        sharpe_ratio=eval_res.sharpe_ratio,
        max_drawdown=eval_res.max_drawdown,
        mean_reward=eval_res.mean_reward,
        asset_coverage=eval_res.asset_coverage,
        topm_hit_rate=eval_res.topm_hit_rate,
        candidate_hit_rate=eval_res.candidate_hit_rate,
        time_to_first_promising=(
            eval_res.time_to_first_promising
            if eval_res.time_to_first_promising != float("inf")
            else -1.0
        ),
        mean_latency_ms=eval_res.mean_latency_ms,
        episodes_to_convergence=convergence if convergence != float("inf") else -1.0,
        train_reward_last=float(train_curve[-1]) if train_curve.size else 0.0,
        train_reward_mean=float(train_curve.mean()) if train_curve.size else 0.0,
        n_train_steps=int(result.total_steps),
        timestamp_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        duration_seconds=float(duration),
    )


def _maybe_download(universe: str) -> None:
    """Si los datos no existen, intentar descargarlos con configs/data/<universe>.yaml."""
    from src.utils.paths import DATA_RAW_DIR

    target = DATA_RAW_DIR / universe
    if target.is_dir() and any(target.glob("*.parquet")):
        return
    cfg_path = CONFIGS_DIR / "data" / f"{universe}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Datos no presentes en {target} y no se encontró config "
            f"{cfg_path}. Descargar manualmente con `uv run python -m src.main download`."
        )
    cfg = load_config(cfg_path)
    fetch_ohlcv(
        cfg.data.tickers,
        cfg.data.start_date,
        cfg.data.end_date,
        universe=universe,
        interval=cfg.data.interval,
        auto_adjust=cfg.data.auto_adjust,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True, help="Subdirectorio en data/raw/.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=5000, help="Pasos por run.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
    )
    parser.add_argument("--campaign-id", default="campaign_1")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta del CSV. Por defecto outputs/tables/<campaign_id>.csv.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps por episodio del entorno (útil para datasets pequeños).",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=None,
        help="Override window_length del estado.",
    )
    parser.add_argument(
        "--config-suffix",
        default="",
        help="Sufijo del config (e.g. 'v2' carga model_x_v2.yaml).",
    )
    parser.add_argument(
        "--filter-dates",
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Restringir el panel a [START, END] antes del split interno "
             "(formato YYYY-MM-DD). Usado para hold-out 2023-24 (Anexo M).",
    )
    parser.add_argument(
        "--sweet-spot",
        action="store_true",
        help="Aplicar overrides de la configuración sweet_spot "
             "(M=16, β=0.5, k=3, m=5).",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    _maybe_download(args.universe)

    output_path = args.output if args.output is not None else TABLES_DIR / f"{args.campaign_id}.csv"
    ensure_dir(output_path.parent)

    rows: list[RunRow] = []
    total = len(args.models) * len(args.seeds)
    idx = 0
    for model in args.models:
        for seed in args.seeds:
            idx += 1
            print(f"[{idx}/{total}] Run model={model} seed={seed} steps={args.steps} ...")
            row = _train_and_evaluate_one(
                model=model,
                seed=seed,
                universe=args.universe,
                steps=args.steps,
                campaign_id=args.campaign_id,
                max_steps_override=args.max_steps,
                window_override=args.window_length,
                config_suffix=args.config_suffix,
                filter_dates=tuple(args.filter_dates) if args.filter_dates else None,
                sweet_spot_overrides=(
                    {"subgraph_max_size": 16, "beta": 0.5, "k_steps": 3, "m_top": 5}
                    if args.sweet_spot else None
                ),
            )
            rows.append(row)
            print(
                f"        return={row.cumulative_return:.4f} sharpe={row.sharpe_ratio:.4f} "
                f"topm_hit={row.topm_hit_rate:.3f} cand_hit={row.candidate_hit_rate:.3f} "
                f"latency={row.mean_latency_ms:.2f}ms duration={row.duration_seconds:.1f}s"
            )

    # Escribir CSV agregado
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    print(f"\n[OK] {len(rows)} runs escritos en {output_path}")


if __name__ == "__main__":
    main()
