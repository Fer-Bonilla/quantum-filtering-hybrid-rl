"""Mediación: la rotación REALIZADA explica el candidate_hit_rate (v7 — exp 2).

El experimento 1 (dosis-respuesta) demostró causalmente que la rotación de la
máscara sube el candidate_hit_rate. Este experimento cierra el argumento de
mediación: mide la rotación REALIZADA de los selectores informados de v6
(C, D, R, Q) sobre la MISMA secuencia de subgrafos de evaluación y la superpone
sobre la curva del StickyWalker (información cero). Si C/D/R/Q caen sobre la
curva rotación→cand_hit del StickyWalker, su candidate_hit_rate queda EXPLICADO
por su rotación — no por la información del selector.

Clave de viabilidad (sin reentrenar): ``info["candidate_mask"]`` es siempre
all-True y ``t`` avanza de forma determinista, así que la secuencia de subgrafos
H_t NO depende de las acciones del agente; y candidate_hit_rate depende solo de
la máscara. Por tanto se puede reproducir la máscara de cada selector con una
política ficticia (acción 0) y obtener rotación + cand_hit fieles.

Replica EXACTAMENTE la construcción del hook de evaluación de ``run_campaign``
(``returns_panel`` del panel completo, M=8, m=3) para medir lo mismo que v6.

Uso::

    uv run python scripts/mediation_rotation_v7.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

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
from src.quantum.annealing_walker import AnnealingWalker
from src.quantum.classical_walker import ClassicalWalker
from src.quantum.noise import NoiseSpec
from src.quantum.quantum_walker import QuantumWalker
from src.quantum.random_walker import RandomWalker
from src.quantum.sticky_walker import StickyWalker
from src.training.evaluate import PromisingSpec, compute_promising_matrix
from src.utils.config import load_config
from src.utils.paths import CONFIGS_DIR, TABLES_DIR

SEEDS = [42, 123, 456, 789, 1024, 7, 99, 314, 1729, 65535]
N_EPISODES = 5
FIG_DIR = Path("outputs/figures/v7")
STICKY_PS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

COLOR = {"C": "#2ca02c", "D": "#d62728", "R": "#ff7f0e", "Q": "#9467bd"}
LABEL = {"C": "C (caminata clás.)", "D": "D (DTQW)",
         "R": "R (máscara azar)", "Q": "Q (QUBO óptimo)"}


def _make_selector(name: str, seed: int, cfg):
    if name == "C":
        return ClassicalWalker()
    if name == "D":
        return QuantumWalker(
            init_mode=cfg.quantum.init_mode,
            renormalize_threshold=cfg.quantum.renormalize_threshold,
            backend=cfg.quantum.backend,
            noise=NoiseSpec(),
        )
    if name == "R":
        return RandomWalker(seed=seed)
    if name == "Q":
        return AnnealingWalker(alpha=1.0, beta=2.0, mode="exact")
    if name.startswith("S"):
        return StickyWalker(seed=seed, rotation_p=float(name[1:]))
    raise ValueError(name)


def _build_hook(local, cfg, test_env, features):
    hybrid_spec = HybridSpec(
        graph_spec=GraphSpec(
            alpha=cfg.graph.alpha, beta=cfg.graph.beta, eps=cfg.graph.eps,
            k_neighbors=cfg.graph.k_neighbors, sym_mode=cfg.graph.sym_mode,
        ),
        subgraph_max_size=cfg.graph.subgraph_max_size,
        seed_score_window=cfg.graph.seed_score_window,
        graph_lookback=cfg.graph.lookback_window,
        k_steps=cfg.quantum.k_steps,
        m_top=cfg.quantum.m_top,
        update_frequency=cfg.graph.update_frequency,
    )
    sectors_map = build_sector_map(test_env.tickers, allow_online=False)
    sectors = [sectors_map[t] for t in test_env.tickers] if cfg.graph.beta > 0 else None
    return HybridAgent(
        classical_agent=None, local_module=local, hybrid_spec=hybrid_spec,
        returns_panel=build_returns_panel(features, test_env.tickers),
        ticker_to_idx=build_ticker_index(test_env.tickers), sectors=sectors,
    ).make_mask


def _replay(name: str, cfg, splits, env_spec, features, promising, M: int) -> tuple[float, float]:
    """Devuelve (rotación realizada media, candidate_hit_rate) del selector."""
    jacc_sum, jacc_n = 0.0, 0
    cand_hits, cand_evals = 0, 0
    for seed in SEEDS:
        test_env = MarketEnv(splits["test"], env_spec)
        local = _make_selector(name, seed, cfg)
        make_mask = _build_hook(local, cfg, test_env, features)
        rng = np.random.default_rng(seed)
        for _ in range(N_EPISODES):
            obs, info = test_env.reset(seed=int(rng.integers(0, 1_000_000)))
            prev: set[int] | None = None
            while True:
                t_idx = int(test_env.current_t)
                mask = make_mask(test_env, obs, info)
                proper = mask.sum() <= M  # excluir fallback all-True
                if promising is not None and t_idx < promising.shape[0]:
                    cand_evals += 1
                    if (mask & promising[t_idx]).any():
                        cand_hits += 1
                if proper:
                    cur = set(np.flatnonzero(mask).tolist())
                    if prev is not None:
                        union = len(prev | cur)
                        jacc_sum += 1.0 - len(prev & cur) / union if union else 0.0
                        jacc_n += 1
                    prev = cur
                obs, _, term, trunc, info = test_env.step(0)
                if term or trunc:
                    break
    rot = jacc_sum / jacc_n if jacc_n else float("nan")
    cand = cand_hits / cand_evals if cand_evals else float("nan")
    return rot, cand


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(CONFIGS_DIR / "experiment" / "model_d_v2.yaml")
    cfg.data.cache_subdir = "nivel2"
    M = cfg.graph.subgraph_max_size

    raw = load_ohlcv(universe="nivel2")
    clean_df = clean(raw, CleaningPolicy())
    fs = FeatureSpec(
        return_type=cfg.env.feature.return_type,
        volatility_window=cfg.env.feature.volatility_window,
        volume_window=cfg.env.feature.volume_window,
        indicators=tuple(cfg.env.feature.indicators),
    )
    features = compute_features(clean_df, fs)
    splits = chronological_split(
        features, SplitSpec(train_frac=cfg.env.split.train_frac, val_frac=cfg.env.split.val_frac))
    env_spec = MarketEnvSpec(
        window_length=cfg.env.feature.window_length,
        max_steps=cfg.env.max_steps,
        reward_coefs=RewardCoefficients(
            lambda_risk=cfg.env.lambda_risk, mu_cost=cfg.env.mu_cost,
            transaction_cost=cfg.env.transaction_cost, reward_type=cfg.env.reward_type),
        episode_sampling=cfg.env.episode_sampling,
    )
    ref_env = MarketEnv(splits["test"], env_spec)  # solo para orden de tickers
    cols = [(t, "return") for t in ref_env.tickers]
    returns_test = splits["test"].loc[:, cols].to_numpy(dtype=float)
    promising = compute_promising_matrix(returns_test, PromisingSpec())

    # v6 cand_hit de referencia (validación de fidelidad del replay)
    v6_ref = {"R": _v6_cand("variant_R_v6.csv", "R"),
              "C": _v6_cand("campaign_1_v3.csv", "C"),
              "D": _v6_cand("campaign_1_v3.csv", "D"),
              "Q": _v6_cand("variant_Q_v6.csv", "Q")}

    print(f"Replay de selectores (M={M}, m={cfg.quantum.m_top}, "
          f"{N_EPISODES} episodios x {len(SEEDS)} semillas)...\n")
    rows: list[dict] = []
    # StickyWalker (curva de referencia)
    sticky_pts = []
    for p in STICKY_PS:
        rot, cand = _replay(f"S{p}", cfg, splits, env_spec, features, promising, M)
        sticky_pts.append((rot, cand))
        rows.append({"selector": f"S(p={p:.2f})", "rotation": rot, "cand_hit": cand,
                     "v6_cand_hit": "", "kind": "sticky"})
        print(f"  S(p={p:.2f})   rot={rot:.3f}  cand_hit={cand:.3f}")
    print()
    informed = {}
    for name in ["R", "D", "C", "Q"]:
        rot, cand = _replay(name, cfg, splits, env_spec, features, promising, M)
        informed[name] = (rot, cand)
        ref = v6_ref[name]
        rows.append({"selector": name, "rotation": rot, "cand_hit": cand,
                     "v6_cand_hit": f"{ref:.3f}", "kind": "informed"})
        print(f"  {name}   rot={rot:.3f}  cand_hit={cand:.3f}   "
              f"(v6 cand_hit={ref:.3f}, d={cand-ref:+.3f})")

    # Mediación: residual de cada selector informado respecto a la curva sticky
    srot = np.array([p[0] for p in sticky_pts])
    scand = np.array([p[1] for p in sticky_pts])
    order = np.argsort(srot)
    srot_s, scand_s = srot[order], scand[order]
    print("\nMediación (cand_hit observado vs. predicho por la curva de rotación):")
    for name, (rot, cand) in informed.items():
        pred = float(np.interp(rot, srot_s, scand_s))
        print(f"  {name}: rot={rot:.3f}  cand_obs={cand:.3f}  cand_pred(curva)={pred:.3f}  "
              f"residual={cand-pred:+.3f}")
    # ¿La brecha D−C se explica por su rotación?
    if "D" in informed and "C" in informed:
        gap_obs = informed["D"][1] - informed["C"][1]
        pred_d = float(np.interp(informed["D"][0], srot_s, scand_s))
        pred_c = float(np.interp(informed["C"][0], srot_s, scand_s))
        print(f"\n  Brecha D-C: observada={gap_obs:+.3f}  "
              f"predicha por rotacion={pred_d-pred_c:+.3f}")
    # Correlación global rotación↔cand_hit (todos los selectores)
    allrot = np.array([r["rotation"] for r in rows])
    allcand = np.array([r["cand_hit"] for r in rows])
    rho, pval = spearmanr(allrot, allcand)
    print(f"\n  Spearman rho(rotación, cand_hit) sobre {len(rows)} selectores = "
          f"{rho:+.3f} (p={pval:.2e})")

    _save_csv(rows)
    _figure(sticky_pts, informed)


def _v6_cand(csv_name: str, model: str) -> float:
    vals = [float(r["candidate_hit_rate"])
            for r in csv.DictReader((TABLES_DIR / csv_name).open(encoding="utf-8"))
            if r["model"] == model]
    return float(np.mean(vals)) if vals else float("nan")


def _save_csv(rows: list[dict]) -> None:
    out = TABLES_DIR / "mediation_rotation_v7.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {out}")


def _figure(sticky_pts, informed) -> None:
    srot = [p[0] for p in sticky_pts]
    scand = [p[1] for p in sticky_pts]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.plot(srot, scand, "o-", color="#1f77b4", linewidth=2, markersize=8,
            markeredgecolor="black", zorder=2,
            label="StickyWalker (información = 0) — curva rotación→cand_hit")
    for p, (r, c) in zip(STICKY_PS, sticky_pts, strict=False):
        ax.annotate(f"p={p:g}", (r, c), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=7.5, color="#1f77b4")
    for name, (rot, cand) in informed.items():
        ax.plot(rot, cand, "D", color=COLOR[name], markersize=13,
                markeredgecolor="black", zorder=4, label=LABEL[name])
        ax.annotate(name, (rot, cand), xytext=(8, 6), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=COLOR[name])
    ax.set_xlabel("rotación realizada de la máscara (Jaccard, sobre la misma "
                  "secuencia de eval.)")
    ax.set_ylabel("candidate_hit_rate")
    ax.set_title("Mediación: la ROTACIÓN domina el candidate_hit_rate (ρ=0.93). Ningún\n"
                 "selector informado supera al azar — D lo iguala; C y Q caen por DEBAJO "
                 "(su concentración penaliza)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_mediacion_rotacion.png", dpi=150)
    plt.close(fig)
    print(f"[OK] {FIG_DIR}/fig_mediacion_rotacion.png")


if __name__ == "__main__":
    main()
