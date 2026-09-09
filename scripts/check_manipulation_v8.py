"""Control de manipulación EXP-4: rotación realizada de S(p*) vs D.

Pre-registro: |rot(D) − rot(S(p*))| < 0,03 sobre el replay de evaluación.

Uso::

    uv run python scripts/run_sticky_calibrated_v8.py  # primero (define p*)
    uv run python scripts/check_manipulation_v8.py
"""

from __future__ import annotations

import csv
import importlib.util
import sys as _sys
from pathlib import Path

import numpy as np

from src.utils.paths import TABLES_DIR

_path = Path(__file__).resolve().parent / "run_mask_metrics_v8.py"
_spec = importlib.util.spec_from_file_location("mask_metrics", _path)
assert _spec is not None and _spec.loader is not None
mm = importlib.util.module_from_spec(_spec)
_sys.modules["mask_metrics"] = mm
_spec.loader.exec_module(mm)

from src.data.cleaning import CleaningPolicy, clean  # noqa: E402
from src.data.download import load_ohlcv  # noqa: E402
from src.data.features import FeatureSpec, compute_features  # noqa: E402
from src.data.splits import SplitSpec, chronological_split  # noqa: E402
from src.env.market_env import MarketEnv, MarketEnvSpec  # noqa: E402
from src.env.reward import RewardCoefficients  # noqa: E402
from src.training.evaluate import PromisingSpec, compute_promising_matrix  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import CONFIGS_DIR  # noqa: E402


def main() -> None:
    with (TABLES_DIR / "mask_metrics_v8.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rot_d = float(np.mean([float(r["rotation_realized"]) for r in rows
                           if r["selector"] == "D"]))
    with (TABLES_DIR / "sticky_calibrated_v8.csv").open(encoding="utf-8") as fh:
        p_star = float(next(csv.DictReader(fh))["p_star"])

    cfg = load_config(CONFIGS_DIR / "experiment" / "model_d_v2.yaml")
    cfg.data.cache_subdir = "nivel2"
    raw = load_ohlcv(universe="nivel2")
    features = compute_features(clean(raw, CleaningPolicy()), FeatureSpec(
        return_type=cfg.env.feature.return_type,
        volatility_window=cfg.env.feature.volatility_window,
        volume_window=cfg.env.feature.volume_window,
        indicators=tuple(cfg.env.feature.indicators)))
    splits = chronological_split(features, SplitSpec(
        train_frac=cfg.env.split.train_frac, val_frac=cfg.env.split.val_frac))
    env_spec = MarketEnvSpec(
        window_length=cfg.env.feature.window_length, max_steps=cfg.env.max_steps,
        reward_coefs=RewardCoefficients(
            lambda_risk=cfg.env.lambda_risk, mu_cost=cfg.env.mu_cost,
            transaction_cost=cfg.env.transaction_cost,
            reward_type=cfg.env.reward_type),
        episode_sampling=cfg.env.episode_sampling)
    ref_env = MarketEnv(splits["test"], env_spec)
    cols = [(t, "return") for t in ref_env.tickers]
    returns_test = splits["test"].loc[:, cols].to_numpy(dtype=float)
    pspec = PromisingSpec()
    promising = compute_promising_matrix(returns_test, pspec)
    gmat = mm._g_score_matrix(returns_test, pspec)

    per_seed, _, _ = mm._replay_selector(
        f"S{p_star:.4f}", cfg, splits, env_spec, features, promising, gmat,
        cfg.graph.subgraph_max_size, cfg.quantum.m_top)
    rot_s = float(np.mean([r["rotation_realized"] for r in per_seed]))
    cand_s = float(np.mean([r["candidate_hit"] for r in per_seed]))
    delta = abs(rot_d - rot_s)
    print(f"rot(D)={rot_d:.4f}  rot(S(p*={p_star:.4f}))={rot_s:.4f}  "
          f"|delta|={delta:.4f}  cand(S(p*))_replay={cand_s:.4f}")
    print(f"CONTROL DE MANIPULACION: "
          f"{'OK (<0,03)' if delta < 0.03 else 'FALLA (>=0,03) — recalibrar'}")


if __name__ == "__main__":
    main()
