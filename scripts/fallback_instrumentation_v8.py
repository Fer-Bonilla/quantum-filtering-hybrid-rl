"""Instrumentación del respaldo con máscara completa (encargo §6-§7).

Replay determinista (sin reentrenar; maquinaria validada en v7) de los
selectores C, D y R sobre la secuencia de evaluación de la campaña principal,
registrando en cada paso si la máscara provino del respaldo ``q_t = U``
(máscara completa) y por qué motivo:

  - ``insufficient_history`` : t sin ventana de grafo/score completa.
  - ``no_eligible``          : eligible_mask completamente False.
  - ``no_positive_neighbors``: semilla sin vecinos con afinidad positiva
                               ("Subgrafo aislado").
  - ``other``                : cualquier otra ValueError de select_subgraph.

Límites documentados: las políticas PPO de la campaña no se conservaron
(sin checkpoints), por lo que ``selected_action`` y ``topm_hit`` no son
reproducibles sin reentrenar; se registran vacíos. ``candidate_hit`` es
selector-level y sí es exactamente reproducible.

Salidas:
  - outputs/tables/fallback_events.csv          (una fila por paso)
  - outputs/tables/fallback_summary.csv         (por brazo y semilla)
  - outputs/tables/fallback_paired_effects.csv  (D-C con y sin respaldos)

Uso::

    uv run python scripts/fallback_instrumentation_v8.py
"""

from __future__ import annotations

import csv
import sys

import numpy as np

import src.agents.hybrid_agent as ha
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
from src.graph.graph_builder import GraphSpec
from src.quantum.classical_walker import ClassicalWalker
from src.quantum.noise import NoiseSpec
from src.quantum.quantum_walker import QuantumWalker
from src.quantum.random_walker import RandomWalker
from src.training.evaluate import PromisingSpec, compute_promising_matrix
from src.utils.config import load_config
from src.utils.paths import CONFIGS_DIR, TABLES_DIR

sys.stdout.reconfigure(encoding="utf-8")

SEEDS = [42, 123, 456, 789, 1024, 7, 99, 314, 1729, 65535]
N_EPISODES = 5
ARMS = ["C", "D", "R"]
B_BOOT = 5000
RNG_BOOT = 20260705
EVENTS = TABLES_DIR / "fallback_events.csv"
SUMMARY = TABLES_DIR / "fallback_summary.csv"
PAIRED = TABLES_DIR / "fallback_paired_effects.csv"

_orig_select = ha.select_subgraph


class SelectProbe:
    """Captura el resultado del último select_subgraph (o su excepción)."""

    def __init__(self) -> None:
        self.reason: str | None = None
        self.subgraph = None

    def __call__(self, graph, scores, *, max_size, eligible_mask=None):
        self.reason = None
        self.subgraph = None
        try:
            sub = _orig_select(graph, scores, max_size=max_size,
                               eligible_mask=eligible_mask)
        except ValueError as exc:
            msg = str(exc)
            if "eligible_mask completamente False" in msg:
                self.reason = "no_eligible"
            elif "Subgrafo aislado" in msg:
                self.reason = "no_positive_neighbors"
            else:
                self.reason = "other"
            raise
        self.subgraph = sub
        return sub


def _make_selector(name: str, seed: int, cfg):
    if name == "C":
        return ClassicalWalker()
    if name == "D":
        return QuantumWalker(
            init_mode=cfg.quantum.init_mode,
            renormalize_threshold=cfg.quantum.renormalize_threshold,
            backend=cfg.quantum.backend, noise=NoiseSpec())
    return RandomWalker(seed=seed)


def main() -> None:
    cfg = load_config(CONFIGS_DIR / "experiment" / "model_d_v2.yaml")
    cfg.data.cache_subdir = "nivel2"
    M, m = cfg.graph.subgraph_max_size, cfg.quantum.m_top

    raw = load_ohlcv(universe="nivel2")
    features = compute_features(clean(raw, CleaningPolicy()), FeatureSpec(
        return_type=cfg.env.feature.return_type,
        volatility_window=cfg.env.feature.volatility_window,
        volume_window=cfg.env.feature.volume_window,
        indicators=tuple(cfg.env.feature.indicators)))
    splits = chronological_split(features, SplitSpec(
        train_frac=cfg.env.split.train_frac, val_frac=cfg.env.split.val_frac))
    test = splits["test"]
    dates = [d.date().isoformat() for d in test.index]
    env_spec = MarketEnvSpec(
        window_length=cfg.env.feature.window_length,
        max_steps=cfg.env.max_steps,
        reward_coefs=RewardCoefficients(
            lambda_risk=cfg.env.lambda_risk, mu_cost=cfg.env.mu_cost,
            transaction_cost=cfg.env.transaction_cost,
            reward_type=cfg.env.reward_type),
        episode_sampling=cfg.env.episode_sampling)
    ref_env = MarketEnv(test, env_spec)
    cols = [(t, "return") for t in ref_env.tickers]
    returns_test = test.loc[:, cols].to_numpy(dtype=float)
    promising = compute_promising_matrix(returns_test, PromisingSpec())
    N = ref_env.n_tickers

    probe = SelectProbe()
    ha.select_subgraph = probe  # instrumentación

    events: list[dict] = []
    summary: list[dict] = []
    per_seed_hits: dict[tuple[str, int], dict] = {}

    for arm in ARMS:
        for seed in SEEDS:
            test_env = MarketEnv(test, env_spec)
            local = _make_selector(arm, seed, cfg)
            spec = HybridSpec(
                graph_spec=GraphSpec(
                    alpha=cfg.graph.alpha, beta=cfg.graph.beta,
                    eps=cfg.graph.eps, k_neighbors=cfg.graph.k_neighbors,
                    sym_mode=cfg.graph.sym_mode),
                subgraph_max_size=M,
                seed_score_window=cfg.graph.seed_score_window,
                graph_lookback=cfg.graph.lookback_window,
                k_steps=cfg.quantum.k_steps, m_top=m,
                update_frequency=cfg.graph.update_frequency)
            sectors_map = build_sector_map(test_env.tickers, allow_online=False)
            sectors = ([sectors_map[t] for t in test_env.tickers]
                       if cfg.graph.beta > 0 else None)
            agent = HybridAgent(
                classical_agent=None, local_module=local, hybrid_spec=spec,
                returns_panel=build_returns_panel(features, test_env.tickers),
                ticker_to_idx=build_ticker_index(test_env.tickers),
                sectors=sectors)
            rng = np.random.default_rng(seed)
            counts = {"steps": 0, "fallback": 0,
                      "hit_all": 0, "eval_all": 0,
                      "hit_valid": 0, "eval_valid": 0,
                      "hit_fb": 0, "eval_fb": 0}
            last_reason = ""
            for ep in range(N_EPISODES):
                obs, info = test_env.reset(seed=int(rng.integers(0, 1_000_000)))
                while True:
                    t_idx = int(test_env.current_t)
                    probe.reason, probe.subgraph = None, None
                    g0 = max(0, t_idx - spec.graph_lookback + 1)
                    s0 = max(0, t_idx - spec.seed_score_window + 1)
                    mask = agent.make_mask(test_env, obs, info)
                    fb = bool(mask.sum() >= N)
                    recomputed = (probe.reason is not None
                                  or probe.subgraph is not None
                                  or g0 == t_idx or s0 == t_idx)
                    if recomputed:
                        if g0 == t_idx or s0 == t_idx:
                            last_reason = "insufficient_history"
                        elif probe.reason is not None:
                            last_reason = probe.reason
                        else:
                            last_reason = ""
                    reason = last_reason if fb else ""
                    sub = probe.subgraph
                    m_t_val = ("" if sub is None else
                               int(min(m, sub.size)))
                    counts["steps"] += 1
                    counts["fallback"] += int(fb)
                    evaluable = t_idx < promising.shape[0]
                    hit = ""
                    if evaluable:
                        sel = np.flatnonzero(mask)
                        hit_b = bool(promising[t_idx, sel].any())
                        hit = int(hit_b)
                        counts["eval_all"] += 1
                        counts["hit_all"] += int(hit_b)
                        if fb:
                            counts["eval_fb"] += 1
                            counts["hit_fb"] += int(hit_b)
                        else:
                            counts["eval_valid"] += 1
                            counts["hit_valid"] += int(hit_b)
                    events.append({
                        "campaign": "replay_v8_fallback", "arm": arm,
                        "seed": seed, "fold": "test_unica", "episode": ep,
                        "date": dates[t_idx], "t_idx": t_idx,
                        "fallback": int(fb), "fallback_reason": reason,
                        "M_t": "" if sub is None else int(sub.size),
                        "m_t": m_t_val,
                        "q_size": int(mask.sum()),
                        "candidate_hit": hit, "topm_hit": "",
                        "selected_action": ""})
                    obs, _, term, trunc, info = test_env.step(0)
                    if term or trunc:
                        break
            per_seed_hits[(arm, seed)] = counts
            fbr = counts["fallback"] / counts["steps"]
            summary.append({
                "arm": arm, "seed": seed,
                "n_steps": counts["steps"],
                "n_fallback": counts["fallback"],
                "fallback_rate": fbr,
                "candidate_hit_all": counts["hit_all"] / max(counts["eval_all"], 1),
                "candidate_hit_valid": counts["hit_valid"] / max(counts["eval_valid"], 1),
                "candidate_hit_fallback": (counts["hit_fb"] / counts["eval_fb"]
                                           if counts["eval_fb"] else ""),
                "topm_hit_all": "", "topm_hit_valid": "",
                "nota": "topm/acciones no reproducibles: politicas no conservadas"})
            print(f"{arm} seed={seed}: pasos={counts['steps']} "
                  f"respaldos={counts['fallback']} ({100*fbr:.2f}%) "
                  f"cand_all={summary[-1]['candidate_hit_all']:.4f} "
                  f"cand_valid={summary[-1]['candidate_hit_valid']:.4f}",
                  flush=True)

    ha.select_subgraph = _orig_select

    with EVENTS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(events[0].keys()))
        w.writeheader()
        w.writerows(events)
    with SUMMARY.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # --- comparacion C vs D: mismos pasos, fechas y motivos ---
    ev_c = [(e["seed"], e["episode"], e["t_idx"], e["fallback_reason"])
            for e in events if e["arm"] == "C" and e["fallback"]]
    ev_d = [(e["seed"], e["episode"], e["t_idx"], e["fallback_reason"])
            for e in events if e["arm"] == "D" and e["fallback"]]
    print(f"\nRespaldos C: {len(ev_c)}  D: {len(ev_d)}  "
          f"identicos (semilla, episodio, t, motivo): {set(ev_c) == set(ev_d)}")

    # --- D-C pareado con y sin respaldos (mismo bootstrap de la tesis) ---
    rows_paired = []
    rng_b = np.random.default_rng(RNG_BOOT)
    for metric_key, label in [("candidate_hit_all", "con_respaldos"),
                              ("candidate_hit_valid", "sin_respaldos")]:
        dd = {r["seed"]: r[metric_key] for r in summary if r["arm"] == "D"}
        cc = {r["seed"]: r[metric_key] for r in summary if r["arm"] == "C"}
        d = np.array([dd[s] - cc[s] for s in SEEDS], dtype=float)
        boots = np.array([rng_b.choice(d, size=len(d), replace=True).mean()
                          for _ in range(B_BOOT)])
        p2 = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
        rows_paired.append({
            "contraste": "D-C", "muestra": label, "n": len(d),
            "mean_diff": float(d.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "p_two_sided": p2, "B": B_BOOT, "rng_seed": RNG_BOOT})
        print(f"D-C {label}: {d.mean():+.4f} "
              f"IC95[{np.percentile(boots, 2.5):+.4f},"
              f"{np.percentile(boots, 97.5):+.4f}] p2={p2:.4f}")
    with PAIRED.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_paired[0].keys()))
        w.writeheader()
        w.writerows(rows_paired)

    # --- descomposicion ---
    print("\nDescomposicion cand_all = fbr*cand_fb + (1-fbr)*cand_valid:")
    for arm in ARMS:
        cs = [per_seed_hits[(arm, s)] for s in SEEDS]
        agg = {k: sum(c[k] for c in cs) for k in cs[0]}
        # tasa de respaldo sobre pasos evaluables (misma base que cand_hit)
        fbr = agg["eval_fb"] / max(agg["eval_all"], 1)
        c_all = agg["hit_all"] / max(agg["eval_all"], 1)
        c_val = agg["hit_valid"] / max(agg["eval_valid"], 1)
        c_fb = agg["hit_fb"] / agg["eval_fb"] if agg["eval_fb"] else 0.0
        recon = fbr * c_fb + (1 - fbr) * c_val
        print(f"  {arm}: fbr={fbr:.4f} all={c_all:.4f} valid={c_val:.4f} "
              f"fb={c_fb if agg['eval_fb'] else float('nan'):.4f} "
              f"reconstruido={recon:.4f}")
    print(f"\n[OK] {EVENTS}\n[OK] {SUMMARY}\n[OK] {PAIRED}")


if __name__ == "__main__":
    main()
