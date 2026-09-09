"""EXP-1 v8 — Métricas de alineación desacopladas de la rotación (Tier 1).

Replay (sin reentrenar, maquinaria validada en v7-mediación) de los selectores
C, D, R, Q y S(p) sobre la misma secuencia de evaluación, computando por
semilla las métricas pre-registradas (docs/preregistro_v8.md, commit 0e3ee5e):

  - precision@m  : fracción de la máscara en el quintil prometedor ex-post.
  - NDCG@m       : ranking inducido (D: P_k; C: p_k; R: permutación-null);
                   Q y S(p) excluidos (sin ranking intrínseco).
  - candidate_hit estratificado por tercil de rotación Jaccard (exploratorio).
  - rotación realizada por semilla (insumo de la calibración p* del EXP-4).

Guarda además las máscaras por paso en un NPZ para el EXP-2 (información
mutua). Análisis confirmatorio H-v8.3a/b: bootstrap pareado unilateral D−R
(y D−C como referencia), Bonferroni k=6.

Uso::

    uv run python scripts/run_mask_metrics_v8.py
"""

from __future__ import annotations

import csv

import numpy as np
from src.agents.hybrid_agent import (
    HybridAgent,
    HybridSpec,
    build_returns_panel,
    build_ticker_index,
)
from src.data.cleaning import CleaningPolicy, clean
from src.data.download import load_ohlcv
from src.data.features import FeatureSpec, compute_features
from src.data.sector_map import build_sector_map
from src.data.splits import SplitSpec, chronological_split
from src.env.market_env import MarketEnv, MarketEnvSpec
from src.env.reward import RewardCoefficients
from src.graph.classical_walk import random_walk_distribution
from src.graph.graph_builder import GraphSpec
from src.quantum.annealing_walker import AnnealingWalker
from src.quantum.classical_walker import ClassicalWalker
from src.quantum.dtqw import apply_dtqw
from src.quantum.noise import NoiseSpec
from src.quantum.quantum_walker import QuantumWalker
from src.quantum.random_walker import RandomWalker
from src.quantum.sticky_walker import StickyWalker
from src.training.evaluate import PromisingSpec, compute_promising_matrix
from src.utils.config import load_config
from src.utils.paired_stats import paired_bootstrap
from src.utils.paths import CONFIGS_DIR, TABLES_DIR

SEEDS = [42, 123, 456, 789, 1024, 7, 99, 314, 1729, 65535]
N_EPISODES = 5
SELECTORS = ["D", "C", "R", "Q", "S0.0", "S0.25", "S0.5", "S1.0"]
RANKED = {"D", "C", "R"}  # con fuente de ranking pre-registrada para NDCG
B_BOOT = 5000
K_BONF = 6
OUT_CSV = TABLES_DIR / "mask_metrics_v8.csv"
OUT_NPZ = TABLES_DIR / "mask_steps_v8.npz"
OUT_STRAT = TABLES_DIR / "mask_metrics_v8_strata.csv"


def _g_score_matrix(returns_panel: np.ndarray, spec: PromisingSpec) -> np.ndarray:
    """Score continuo g^(h) del Anexo B.4 (misma fórmula que compute_promising)."""
    T, N = returns_panel.shape
    g = np.full((T, N), np.nan)
    h = spec.horizon
    for t in range(T - h):
        window = returns_panel[t + 1 : t + 1 + h]
        mean = window.mean(axis=0)
        std = window.std(axis=0, ddof=1) if h > 1 else np.zeros(N)
        g[t] = mean / (std + spec.eps)
    return g


def _make_selector(name: str, seed: int, cfg):
    if name == "C":
        return ClassicalWalker()
    if name == "D":
        return QuantumWalker(
            init_mode=cfg.quantum.init_mode,
            renormalize_threshold=cfg.quantum.renormalize_threshold,
            backend=cfg.quantum.backend, noise=NoiseSpec())
    if name == "R":
        return RandomWalker(seed=seed)
    if name == "Q":
        return AnnealingWalker(alpha=1.0, beta=2.0, mode="exact")
    if name.startswith("S"):
        return StickyWalker(seed=seed, rotation_p=float(name[1:]))
    raise ValueError(name)


class Recorder:
    """Envuelve un LocalModule y captura (ids de H_t, ranking) por llamada."""

    def __init__(self, inner, kind: str, cfg, rank_rng: np.random.Generator):
        self._inner = inner
        self._kind = kind
        self._cfg = cfg
        self._rng = rank_rng
        self.last: tuple[np.ndarray, np.ndarray | None] | None = None

    @property
    def name(self) -> str:
        return getattr(self._inner, "name", self._kind)

    def candidate_set(self, subgraph, k: int, m: int) -> np.ndarray:
        idx = self._inner.candidate_set(subgraph, k, m)
        ranking = None
        if self._kind == "D":
            probs = apply_dtqw(
                subgraph.W_local, subgraph.seed_idx_local, k,
                init_mode=self._cfg.quantum.init_mode,
                renormalize_threshold=self._cfg.quantum.renormalize_threshold)
            order = np.argsort(-probs)
            ranking = subgraph.global_node_ids[order]
        elif self._kind == "C":
            probs = random_walk_distribution(subgraph.W_local, subgraph.seed_idx_local, k)
            order = np.argsort(-probs)
            ranking = subgraph.global_node_ids[order]
        elif self._kind == "R":
            perm = self._rng.permutation(len(subgraph.global_node_ids))
            ranking = subgraph.global_node_ids[perm]
        self.last = (subgraph.global_node_ids.copy(),
                     ranking.copy() if ranking is not None else None)
        return idx


def _build_hook(local, cfg, test_env, features):
    hybrid_spec = HybridSpec(
        graph_spec=GraphSpec(
            alpha=cfg.graph.alpha, beta=cfg.graph.beta, eps=cfg.graph.eps,
            k_neighbors=cfg.graph.k_neighbors, sym_mode=cfg.graph.sym_mode),
        subgraph_max_size=cfg.graph.subgraph_max_size,
        seed_score_window=cfg.graph.seed_score_window,
        graph_lookback=cfg.graph.lookback_window,
        k_steps=cfg.quantum.k_steps, m_top=cfg.quantum.m_top,
        update_frequency=cfg.graph.update_frequency)
    sectors_map = build_sector_map(test_env.tickers, allow_online=False)
    sectors = [sectors_map[t] for t in test_env.tickers] if cfg.graph.beta > 0 else None
    return HybridAgent(
        classical_agent=None, local_module=local, hybrid_spec=hybrid_spec,
        returns_panel=build_returns_panel(features, test_env.tickers),
        ticker_to_idx=build_ticker_index(test_env.tickers), sectors=sectors).make_mask


def _ndcg_at_m(ranking: np.ndarray, hood: np.ndarray, g_row: np.ndarray, m: int) -> float | None:
    rel_hood = np.maximum(g_row[hood], 0.0)
    if not np.isfinite(rel_hood).all() or rel_hood.sum() <= 0:
        return None
    ideal = np.sort(rel_hood)[::-1][:m]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    if idcg <= 0:
        return None
    top = ranking[:m]
    rel_top = np.maximum(g_row[top], 0.0)
    dcg = float(np.sum(rel_top / np.log2(np.arange(2, len(rel_top) + 2))))
    return dcg / idcg


def _replay_selector(name: str, cfg, splits, env_spec, features,
                     promising, gmat, M: int, m: int):
    """Replay por semilla → métricas agregadas + pasos crudos para MI."""
    per_seed = []
    steps_masks, steps_t, steps_seed = [], [], []
    strata_steps = []  # (jaccard, hit)
    for seed in SEEDS:
        test_env = MarketEnv(splits["test"], env_spec)
        inner = _make_selector(name, seed, cfg)
        rec = Recorder(inner, name[0] if name.startswith("S") else name, cfg,
                       np.random.default_rng(seed + 777))
        make_mask = _build_hook(rec, cfg, test_env, features)
        rng = np.random.default_rng(seed)
        prec_vals, ndcg_vals = [], []
        hits, evals = 0, 0
        jacc_sum, jacc_n = 0.0, 0
        for _ in range(N_EPISODES):
            obs, info = test_env.reset(seed=int(rng.integers(0, 1_000_000)))
            prev: set[int] | None = None
            while True:
                t_idx = int(test_env.current_t)
                rec.last = None
                mask = make_mask(test_env, obs, info)
                proper = mask.sum() <= M and rec.last is not None
                evaluable = t_idx < promising.shape[0] and np.isfinite(gmat[t_idx]).all()
                if proper and evaluable:
                    sel = np.flatnonzero(mask)
                    prec_vals.append(float(promising[t_idx, sel].mean()))
                    evals += 1
                    hit = bool(promising[t_idx, sel].any())
                    hits += int(hit)
                    if name in RANKED:
                        hood, ranking = rec.last
                        nd = _ndcg_at_m(ranking, hood, gmat[t_idx], m)
                        if nd is not None:
                            ndcg_vals.append(nd)
                    cur = set(sel.tolist())
                    if prev is not None:
                        union = len(prev | cur)
                        j = 1.0 - len(prev & cur) / union if union else 0.0
                        jacc_sum += j
                        jacc_n += 1
                        strata_steps.append((j, hit))
                    prev = cur
                    steps_masks.append(mask.astype(np.uint8))
                    steps_t.append(t_idx)
                    steps_seed.append(seed)
                obs, _, term, trunc, info = test_env.step(0)
                if term or trunc:
                    break
        per_seed.append({
            "selector": name, "seed": seed,
            "precision_at_m": float(np.mean(prec_vals)) if prec_vals else float("nan"),
            "ndcg_at_m": float(np.mean(ndcg_vals)) if ndcg_vals else float("nan"),
            "candidate_hit": hits / evals if evals else float("nan"),
            "rotation_realized": jacc_sum / jacc_n if jacc_n else float("nan"),
            "n_steps": evals,
        })
    raw = (np.array(steps_masks, dtype=np.uint8),
           np.array(steps_t, dtype=np.int64),
           np.array(steps_seed, dtype=np.int64))
    return per_seed, raw, strata_steps


def _paired_boot(rows, a: str, b: str, metric: str, one_sided: bool = True):
    da = {r["seed"]: r[metric] for r in rows if r["selector"] == a}
    db = {r["seed"]: r[metric] for r in rows if r["selector"] == b}
    common = sorted(set(da) & set(db))
    d = np.array([da[s] - db[s] for s in common], dtype=float)
    d = d[np.isfinite(d)]
    # Especificacion unica (src/utils/paired_stats.py)
    bs = paired_bootstrap(d, n_boot=B_BOOT, seed=20260705)
    p = bs.p_greater if one_sided else bs.p_two   # H1: a > b (unilateral)
    return {
        "diff": bs.mean, "n": bs.n,
        "ci_lo": bs.ci_lo, "ci_hi": bs.ci_hi,
        "p": p,
        "p_bonf6": min(1.0, p * K_BONF) if np.isfinite(p) else float("nan"),
    }


def main() -> None:
    cfg = load_config(CONFIGS_DIR / "experiment" / "model_d_v2.yaml")
    cfg.data.cache_subdir = "nivel2"
    M, m = cfg.graph.subgraph_max_size, cfg.quantum.m_top

    raw = load_ohlcv(universe="nivel2")
    clean_df = clean(raw, CleaningPolicy())
    fs = FeatureSpec(
        return_type=cfg.env.feature.return_type,
        volatility_window=cfg.env.feature.volatility_window,
        volume_window=cfg.env.feature.volume_window,
        indicators=tuple(cfg.env.feature.indicators))
    features = compute_features(clean_df, fs)
    splits = chronological_split(
        features, SplitSpec(train_frac=cfg.env.split.train_frac,
                            val_frac=cfg.env.split.val_frac))
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
    gmat = _g_score_matrix(returns_test, pspec)

    all_rows: list[dict] = []
    npz_payload: dict[str, np.ndarray] = {}
    strata_all: dict[str, list] = {}
    print(f"Replay v8 (M={M}, m={m}, {N_EPISODES} ep x {len(SEEDS)} semillas)\n")
    for name in SELECTORS:
        per_seed, (masks, ts, seeds), strata = _replay_selector(
            name, cfg, splits, env_spec, features, promising, gmat, M, m)
        all_rows.extend(per_seed)
        key = name.replace(".", "_")
        npz_payload[f"{key}__masks"] = masks
        npz_payload[f"{key}__t"] = ts
        npz_payload[f"{key}__seed"] = seeds
        strata_all[name] = strata
        agg = {k: float(np.nanmean([r[k] for r in per_seed]))
               for k in ["precision_at_m", "ndcg_at_m", "candidate_hit",
                         "rotation_realized"]}
        print(f"  {name:<6} prec@m={agg['precision_at_m']:.4f}  "
              f"ndcg@m={agg['ndcg_at_m']:.4f}  cand={agg['candidate_hit']:.4f}  "
              f"rot={agg['rotation_realized']:.3f}")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    np.savez_compressed(OUT_NPZ, **npz_payload,
                        promising=promising.astype(np.uint8))
    print(f"\n[OK] {OUT_CSV}\n[OK] {OUT_NPZ}")

    # --- rotación de D por semilla (insumo EXP-4) ---
    d_rots = [r["rotation_realized"] for r in all_rows if r["selector"] == "D"]
    print(f"\nRotacion D por semilla: media={np.mean(d_rots):.4f} "
          f"sd={np.std(d_rots, ddof=1):.4f} (regla p*: {'por semilla' if np.std(d_rots, ddof=1) > 0.05 else 'GLOBAL'})")

    # --- confirmatorio H-v8.3a/b + referencia D-C ---
    print("\n=== H-v8.3 (confirmatorio, unilateral D>R; Bonferroni k=6) ===")
    for metric, tag in [("precision_at_m", "H-v8.3a"), ("ndcg_at_m", "H-v8.3b")]:
        r = _paired_boot(all_rows, "D", "R", metric)
        print(f"  {tag} D-R {metric}: d={r['diff']:+.4f} "
              f"IC95[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
              f"p={r['p']:.4f} pBonf6={r['p_bonf6']:.4f} (n={r['n']})")
    print("--- referencia exploratoria D-C ---")
    for metric in ["precision_at_m", "ndcg_at_m"]:
        r = _paired_boot(all_rows, "D", "C", metric)
        print(f"  D-C {metric}: d={r['diff']:+.4f} "
              f"IC95[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p={r['p']:.4f}")

    # --- estratificación por terciles de rotación (exploratorio) ---
    strat_rows = []
    print("\n=== cand_hit por tercil de rotacion (exploratorio) ===")
    for name, steps in strata_all.items():
        if len(steps) < 30:
            continue
        js = np.array([s[0] for s in steps])
        hs = np.array([s[1] for s in steps], dtype=float)
        t1, t2 = np.percentile(js, [33.3, 66.7])
        lo, mid, hi = hs[js <= t1], hs[(js > t1) & (js <= t2)], hs[js > t2]
        strat_rows.append({"selector": name,
                           "tercil_bajo": float(lo.mean()) if len(lo) else float("nan"),
                           "tercil_medio": float(mid.mean()) if len(mid) else float("nan"),
                           "tercil_alto": float(hi.mean()) if len(hi) else float("nan"),
                           "n_steps": len(steps)})
        r = strat_rows[-1]
        print(f"  {name:<6} bajo={r['tercil_bajo']:.3f} medio={r['tercil_medio']:.3f} "
              f"alto={r['tercil_alto']:.3f}")
    with OUT_STRAT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(strat_rows[0].keys()))
        w.writeheader()
        w.writerows(strat_rows)
    print(f"\n[OK] {OUT_STRAT}")


if __name__ == "__main__":
    main()
