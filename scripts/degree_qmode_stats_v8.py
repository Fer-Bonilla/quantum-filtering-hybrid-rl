"""Diagnósticos de la 3ª revisión: grados tras simetrización OR y
concordancia de los dos modos del selector Q (exacto vs recocido).

Uso::

    uv run python scripts/degree_qmode_stats_v8.py
"""

from __future__ import annotations

import sys

import numpy as np

from src.data.cleaning import CleaningPolicy, clean
from src.data.download import load_ohlcv
from src.data.features import FeatureSpec, compute_features
from src.graph.graph_builder import GraphSpec, build_graph
from src.graph.subgraph_selector import seed_score, select_subgraph
from src.quantum.annealing_walker import AnnealingWalker
from src.utils.config import load_config
from src.utils.paths import CONFIGS_DIR

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    cfg = load_config(CONFIGS_DIR / "experiment" / "model_d_v2.yaml")
    cfg.data.cache_subdir = "nivel2"
    raw = load_ohlcv(universe="nivel2")
    features = compute_features(clean(raw, CleaningPolicy()), FeatureSpec(
        return_type=cfg.env.feature.return_type,
        volatility_window=cfg.env.feature.volatility_window,
        volume_window=cfg.env.feature.volume_window,
        indicators=tuple(cfg.env.feature.indicators)))
    cols = [c for c in features.columns if c[1] == "return"]
    panel = features.loc[:, cols].to_numpy(dtype=float)
    tickers = [c[0] for c in cols]
    T, N = panel.shape
    ts = np.linspace(int(T * 0.62), T - 10, 60, dtype=int)

    gspec = GraphSpec(alpha=cfg.graph.alpha, beta=cfg.graph.beta,
                      eps=cfg.graph.eps, k_neighbors=cfg.graph.k_neighbors,
                      sym_mode=cfg.graph.sym_mode)
    q_exact = AnnealingWalker(alpha=1.0, beta=2.0, mode="exact")
    q_anneal = AnnealingWalker(alpha=1.0, beta=2.0, trotter_per_k=10,
                               total_time=20.0)

    grados_all, dmax_g, dmax_h, msize = [], [], [], []
    identicas, jaccs = 0, []
    n_eval = 0
    for t in ts:
        g0 = max(0, t - cfg.graph.lookback_window + 1)
        s0 = max(0, t - cfg.graph.seed_score_window + 1)
        graph = build_graph(panel[g0: t + 1], tickers, None, gspec)
        deg = (graph.W > 0).sum(axis=1)
        grados_all.extend(deg.tolist())
        dmax_g.append(int(deg.max()))
        scores = seed_score(panel[s0: t + 1])
        try:
            sub = select_subgraph(graph, scores,
                                  max_size=cfg.graph.subgraph_max_size,
                                  eligible_mask=np.ones(N, bool))
        except ValueError:
            continue
        msize.append(sub.size)
        dmax_h.append(int((sub.W_local > 0).sum(axis=1).max()))
        a = set(q_exact.candidate_set(sub, cfg.quantum.k_steps,
                                      cfg.quantum.m_top).tolist())
        b = set(q_anneal.candidate_set(sub, cfg.quantum.k_steps,
                                       cfg.quantum.m_top).tolist())
        n_eval += 1
        identicas += int(a == b)
        jaccs.append(len(a & b) / len(a | b))

    grados = np.array(grados_all)
    print("=== Grados del grafo G_t tras simetrizacion OR "
          f"(k_NN={cfg.graph.k_neighbors}, {len(ts)} instantes) ===")
    print(f"  media={grados.mean():.2f}  mediana={np.median(grados):.0f}  "
          f"p95={np.percentile(grados, 95):.0f}  "
          f"max={grados.max()}  (N-1={N-1})")
    print(f"  d_max por instante: media={np.mean(dmax_g):.1f}  "
          f"max={max(dmax_g)}")
    print(f"  d_max dentro de H_t (M={cfg.graph.subgraph_max_size}): "
          f"media={np.mean(dmax_h):.1f}  max={max(dmax_h)}")
    print(f"  |H_t|: media={np.mean(msize):.2f}  min={min(msize)}  "
          f"max={max(msize)}")
    print()
    print(f"=== Concordancia Q exacto vs Q recocido ({n_eval} subgrafos) ===")
    print(f"  mascaras identicas: {identicas}/{n_eval} "
          f"({100*identicas/n_eval:.1f}%)")
    print(f"  Jaccard medio: {np.mean(jaccs):.4f}  min: {np.min(jaccs):.3f}")


if __name__ == "__main__":
    main()
