"""Benchmarks clásicos de selección de activos (v6 — Punto 1 del director).

Compara el agente RL híbrido contra estrategias estándar de optimización
de cartera y contra la solución óptima ex-post, sobre EL MISMO split de
test, con LA MISMA función de recompensa (log_wealth, lambda=0) y LAS
MISMAS métricas que ``evaluate_agent``.

Estrategias discretas (1 activo por step, vía MarketEnv idéntico al de
las campañas):

* ``random``           — política uniforme (n_seeds realizaciones).
* ``momentum_20d``     — argmax del retorno medio móvil de 20 días
                         (solo usa data[:t]; sin fuga).
* ``buyhold_best``     — activo con mejor Sharpe en TRAIN, mantenido
                         todo el test (decisión ex-ante; sin fuga).
* ``oracle_expost``    — argmax de R_{i,t+1} en cada step. FUGA
                         deliberada: es la SOLUCIÓN ÓPTIMA del problema
                         secuencial discreto (cota superior).

Estrategias vectoriales (cartera continua, calculadas directamente
sobre el panel de retornos del test; sin costes de transacción):

* ``equal_weight``     — cartera 1/N.
* ``markowitz``        — cartera tangente max-Sharpe long-only estimada
                         en TRAIN (covarianza shrinkage diagonal).

COMPARABILIDAD ENTRE FAMILIAS (columna ``family``):

* ``discrete_policy``  — harness idéntico a ``evaluate_agent`` (5 episodios
  de 252 días muestreados al azar del test). ``cumulative_return``,
  ``max_drawdown`` y ``mean_reward`` (suma por episodio) son directamente
  comparables con los agentes RL de ``campaign_1_v3``.
* ``static_portfolio`` — UNA pasada por todo el test (sin episodios, sin
  costes). Sus ``cumulative_return``/``max_drawdown`` NO están en la misma
  escala que la familia discreta; ``asset_coverage`` significa fracción de
  pesos activos (no tickers visitados).

Las ÚNICAS métricas comparables ENTRE familias son ``sharpe_ratio``
(media/std por step) y ``mean_reward_per_step``.

Uso::

    uv run python scripts/run_classical_benchmarks.py --universe nivel2 \\
        --seeds 42 123 456 789 1024 --output outputs/tables/benchmarks_v6.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from src.data.cleaning import CleaningPolicy, clean
from src.data.download import load_ohlcv
from src.data.features import FeatureSpec, compute_features
from src.data.splits import SplitSpec, chronological_split
from src.env.market_env import MarketEnv, MarketEnvSpec
from src.env.reward import RewardCoefficients
from src.training.evaluate import PromisingSpec, compute_promising_matrix
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import TABLES_DIR, ensure_dir

_log = get_logger(__name__)

MOMENTUM_WINDOW = 20


# ---------------------------------------------------------------------------
# Harness de políticas discretas (espejo de evaluate_agent, sin agente NN)
# ---------------------------------------------------------------------------


def _run_discrete_policy(
    env: MarketEnv,
    policy_fn,
    promising_matrix: np.ndarray,
    *,
    n_episodes: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Ejecuta una política determinista/estocástica sobre el env.

    ``policy_fn(t_index, rng) -> int`` devuelve la acción para el paso
    actual. El bucle replica ``evaluate_agent`` (mismos episodios
    aleatorios, mismas métricas) sin red neuronal.
    """
    rng = np.random.default_rng(seed)
    all_rewards: list[float] = []
    rewards_per_episode: list[float] = []
    visited: set[int] = set()
    n_steps = 0
    topm_hits = 0
    time_to_first = float("inf")
    latencies: list[float] = []

    for _ in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        ep_reward = 0.0
        while True:
            t_idx = env.current_t
            t0 = time.perf_counter()
            action = int(policy_fn(t_idx, rng))
            latencies.append((time.perf_counter() - t0) * 1000.0)
            visited.add(action)
            if t_idx < promising_matrix.shape[0] and promising_matrix[t_idx, action]:
                topm_hits += 1
                if np.isinf(time_to_first):
                    time_to_first = float(n_steps + 1)
            obs, reward, terminated, truncated, info = env.step(action)
            all_rewards.append(float(reward))
            ep_reward += float(reward)
            n_steps += 1
            if terminated or truncated:
                break
        rewards_per_episode.append(ep_reward)

    arr = np.asarray(all_rewards, dtype=np.float64)
    cum = float(arr.sum())
    sharpe = float(arr.mean() / (arr.std(ddof=1) + 1e-8)) if arr.size > 1 else 0.0
    cums = np.cumsum(arr)
    drawdown = float(np.max(np.maximum.accumulate(cums) - cums)) if arr.size else 0.0
    return {
        "family": "discrete_policy",
        "cumulative_return": cum,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown,
        "mean_reward": float(np.mean(rewards_per_episode)),
        "mean_reward_per_step": float(arr.mean()) if arr.size else 0.0,
        "asset_coverage": len(visited) / max(env.n_tickers, 1),
        "topm_hit_rate": topm_hits / max(n_steps, 1),
        "candidate_hit_rate": float("nan"),  # sin máscara: métrica no aplica
        "time_to_first_promising": time_to_first if np.isfinite(time_to_first) else -1.0,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# Estrategias vectoriales (cartera continua, sin env)
# ---------------------------------------------------------------------------


def _portfolio_metrics(returns_panel: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Métricas financieras de una cartera estática w sobre el panel de test.

    ``r_p(t) = log(1 + w · R_t)`` — misma escala log_wealth que el env,
    sin costes (cartera estática, sin rebalanceo).
    """
    port = returns_panel @ weights
    log_r = np.log1p(np.clip(port, -0.999999, None))
    sharpe = float(log_r.mean() / (log_r.std(ddof=1) + 1e-8)) if log_r.size > 1 else 0.0
    cums = np.cumsum(log_r)
    drawdown = float(np.max(np.maximum.accumulate(cums) - cums)) if log_r.size else 0.0
    return {
        "family": "static_portfolio",
        # cumulative_return / max_drawdown: UNA pasada por el test completo;
        # NO comparables con la familia discreta (5 episodios re-muestreados).
        "cumulative_return": float(log_r.sum()),
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown,
        # mean_reward (suma por episodio) no existe sin episodios: NaN.
        "mean_reward": float("nan"),
        "mean_reward_per_step": float(log_r.mean()),
        # Aquí coverage = fracción de pesos activos (no tickers visitados).
        "asset_coverage": float((weights > 1e-6).mean()),
        "topm_hit_rate": float("nan"),
        "candidate_hit_rate": float("nan"),
        "time_to_first_promising": -1.0,
        "mean_latency_ms": 0.0,
        "n_steps": int(log_r.size),
    }


def _markowitz_max_sharpe(train_returns: np.ndarray, *, shrinkage: float = 0.10) -> np.ndarray:
    """Cartera tangente long-only aproximada en el simplex.

    Covarianza con shrinkage diagonal (Ledoit-Wolf simplificado):
        Sigma_s = (1-s)·Sigma + s·diag(Sigma)
    Tangente sin restricción ``w ∝ Sigma_s^{-1} mu``; el long-only se
    aproxima recortando negativos y renormalizando (proyección simple,
    estándar como benchmark de referencia).
    """
    mu = train_returns.mean(axis=0)
    sigma = np.cov(train_returns, rowvar=False)
    sigma_s = (1.0 - shrinkage) * sigma + shrinkage * np.diag(np.diag(sigma))
    try:
        raw = np.linalg.solve(sigma_s, mu)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(sigma_s) @ mu
    w = np.clip(raw, 0.0, None)
    if w.sum() <= 1e-12:
        # mu mayormente negativo: degenerar a 1/N (mejor esfuerzo honesto)
        w = np.ones_like(w)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="nivel2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    output = args.output or TABLES_DIR / "benchmarks_v6.csv"
    ensure_dir(output.parent)

    # Pipeline de datos idéntico a run_campaign (configs env/v2)
    raw = load_ohlcv(universe=args.universe)
    clean_df = clean(raw, CleaningPolicy())
    features = compute_features(clean_df, FeatureSpec(return_type="log"))
    splits = chronological_split(features, SplitSpec(train_frac=0.6, val_frac=0.2))

    env_spec = MarketEnvSpec(
        window_length=20,
        max_steps=252,
        reward_coefs=RewardCoefficients(
            lambda_risk=0.0, mu_cost=0.001, transaction_cost=0.0005,
            reward_type="log_wealth",
        ),
        episode_sampling="random",
    )
    env = MarketEnv(splits["test"], env_spec)
    n_assets = env.n_tickers

    tickers = env.tickers
    cols = [(t, "return") for t in tickers]
    test_returns = splits["test"].loc[:, cols].to_numpy(dtype=np.float64)
    train_returns = splits["train"].loc[:, cols].to_numpy(dtype=np.float64)
    promising = compute_promising_matrix(test_returns, PromisingSpec())

    # Matriz interna del env para momentum/oracle (misma fuente que reward)
    return_matrix = env._return_matrix  # noqa: SLF001 — uso deliberado y documentado

    rows: list[dict] = []

    def _emit(strategy: str, seed, metrics: dict[str, float]) -> None:
        rows.append({"strategy": strategy, "seed": seed, **metrics})
        print(
            f"  {strategy:<16} seed={seed!s:<6} "
            f"sharpe={metrics['sharpe_ratio']:+.4f} "
            f"cum={metrics['cumulative_return']:+.3f} "
            f"topm={metrics['topm_hit_rate']:.3f} "
            f"cov={metrics['asset_coverage']:.3f}",
            flush=True,
        )

    # --- 1. random (n_seeds realizaciones) --------------------------------
    print("[1/6] random ...", flush=True)
    for seed in args.seeds:
        metrics = _run_discrete_policy(
            env, lambda t, rng: int(rng.integers(0, n_assets)), promising,
            n_episodes=args.n_episodes, seed=seed,
        )
        _emit("random", seed, metrics)

    # --- 2. momentum 20d ----------------------------------------------------
    print("[2/6] momentum_20d ...", flush=True)

    def momentum_policy(t: int, rng) -> int:
        start = max(0, t - MOMENTUM_WINDOW + 1)
        window = return_matrix[start : t + 1]  # solo data[:t] — sin fuga
        return int(np.argmax(window.mean(axis=0)))

    for seed in args.seeds:
        _emit("momentum_20d", seed,
              _run_discrete_policy(env, momentum_policy, promising,
                                   n_episodes=args.n_episodes, seed=seed))

    # --- 3. buy-and-hold del mejor activo en train -------------------------
    print("[3/6] buyhold_best ...", flush=True)
    train_sharpes = train_returns.mean(axis=0) / (train_returns.std(axis=0, ddof=1) + 1e-8)
    best_asset = int(np.argmax(train_sharpes))
    print(f"  (mejor activo en train: {tickers[best_asset]})", flush=True)
    for seed in args.seeds:
        _emit("buyhold_best", seed,
              _run_discrete_policy(env, lambda t, rng: best_asset, promising,
                                   n_episodes=args.n_episodes, seed=seed))

    # --- 4. oráculo ex-post (solución óptima discreta) ---------------------
    print("[4/6] oracle_expost ...", flush=True)

    def oracle_policy(t: int, rng) -> int:
        t_next = min(t + 1, return_matrix.shape[0] - 1)
        return int(np.argmax(return_matrix[t_next]))  # FUGA deliberada: cota superior

    for seed in args.seeds:
        _emit("oracle_expost", seed,
              _run_discrete_policy(env, oracle_policy, promising,
                                   n_episodes=args.n_episodes, seed=seed))

    # --- 5. equal weight ----------------------------------------------------
    print("[5/6] equal_weight ...", flush=True)
    _emit("equal_weight", "static",
          _portfolio_metrics(test_returns, np.full(n_assets, 1.0 / n_assets)))

    # --- 6. Markowitz max-Sharpe -------------------------------------------
    print("[6/6] markowitz ...", flush=True)
    w_mk = _markowitz_max_sharpe(train_returns)
    n_active = int((w_mk > 1e-6).sum())
    print(f"  (cartera tangente: {n_active} activos con peso > 0)", flush=True)
    _emit("markowitz", "static", _portfolio_metrics(test_returns, w_mk))

    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] {len(rows)} filas en {output}", flush=True)


if __name__ == "__main__":
    main()
