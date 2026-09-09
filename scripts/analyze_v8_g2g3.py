"""Análisis de las compuertas G2 (EXP-5) y G3 (EXP-6, EXP-7) — campaña v8.

Veredictos pre-registrados (commit 0e3ee5e, familia Bonferroni k=6):
  - H-v8.4: D−R > 0 en candidate_hit sobre subgrafos regulares (unilateral,
            confirmatoria en la AGREGADA de topologías; por-topología
            exploratorio).
  - H-v8.5: Sharpe(soft-D) − Sharpe(soft-R) > 0 (unilateral).
  - EXP-7 : exploratorio con corrección FDR (Benjamini–Hochberg).

Secundaria EXP-5: exponente de dispersión empírico de la DTQW (pendiente
log-log de la dispersión RMS de distancia al seed frente a k) por topología,
como verificación del régimen balístico.

Uso::

    uv run python scripts/analyze_v8_g2g3.py
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from src.utils.paths import TABLES_DIR

B_BOOT = 5000
K_BONF = 6
ALPHA_C = 0.05 / K_BONF
TOPOLOGIES = ["dreg_aff", "dreg_uni", "cycle", "bipartite"]


def _load(name: str) -> list[dict]:
    with (TABLES_DIR / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _boot_p(d: np.ndarray, one_sided: bool = True, seed: int = 20260705):
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, len(d), replace=True).mean()
                      for _ in range(B_BOOT)])
    p_pos = float((boots <= 0).mean())
    p_two = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {
        "diff": float(d.mean()), "n": len(d),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "p": p_pos if one_sided else p_two,
    }


def _fdr(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg."""
    m = len(pvals)
    orden = np.argsort(pvals)
    adj = np.empty(m)
    prev = 1.0
    for rank_desc, idx in enumerate(orden[::-1]):
        rank = m - rank_desc
        val = min(prev, pvals[idx] * m / rank)
        adj[idx] = val
        prev = val
    return adj.tolist()


# ---------------------------------------------------------------------------
# G2 — EXP-5
# ---------------------------------------------------------------------------


def analyze_g2() -> None:
    print("=" * 70)
    print("COMPUERTA G2 — EXP-5 subgrafos de regularidad controlada")
    print("=" * 70)
    rows = _load("regular_topologies_v8.csv")
    by: dict[tuple, dict] = {}
    for r in rows:
        by[(r["topology"], r["model"], int(r["seed"]))] = r

    def diffs(a: str, b: str, metric: str, topo: str) -> np.ndarray:
        seeds = sorted({k[2] for k in by if k[0] == topo and k[1] == a}
                       & {k[2] for k in by if k[0] == topo and k[1] == b})
        return np.array([float(by[(topo, a, s)][metric])
                         - float(by[(topo, b, s)][metric]) for s in seeds])

    # H-v8.4 confirmatoria: agregada sobre topologías (media por semilla).
    seeds = sorted({k[2] for k in by})
    agg = []
    for s in seeds:
        ds = [float(by[(t, "D", s)]["candidate_hit_rate"])
              - float(by[(t, "R", s)]["candidate_hit_rate"])
              for t in TOPOLOGIES if (t, "D", s) in by and (t, "R", s) in by]
        if len(ds) == len(TOPOLOGIES):
            agg.append(np.mean(ds))
    res = _boot_p(np.array(agg))
    print(f"\nH-v8.4  D-R cand_hit AGREGADA sobre {len(TOPOLOGIES)} topologias "
          f"(unilateral D>R)")
    print(f"  d={res['diff']:+.4f} IC95[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}] "
          f"p={res['p']:.4f} pBonf6={min(1, res['p']*K_BONF):.4f} (n={res['n']})")
    verd = ("RECHAZA H0 => regimen de valor cuantico (Escenario C)"
            if res["p"] * K_BONF < 0.05 else
            "NO significativa => D~R tambien en regulares (Escenario A ampliado)")
    print(f"  VEREDICTO: {verd}")

    print("\nPor topologia (exploratorio):")
    print(f"{'topologia':<12}{'D-R cand':>12}{'p':>9}{'D-C cand':>12}{'p':>9}"
          f"{'  medias D/C/R (cand)':>24}")
    for topo in TOPOLOGIES:
        dr = _boot_p(diffs("D", "R", "candidate_hit_rate", topo))
        dc = _boot_p(diffs("D", "C", "candidate_hit_rate", topo))
        medias = []
        for mdl in ["D", "C", "R"]:
            vals = [float(v["candidate_hit_rate"]) for k, v in by.items()
                    if k[0] == topo and k[1] == mdl]
            medias.append(np.mean(vals) if vals else float("nan"))
        print(f"{topo:<12}{dr['diff']:>+12.4f}{dr['p']:>9.4f}"
              f"{dc['diff']:>+12.4f}{dc['p']:>9.4f}"
              f"   {medias[0]:.3f}/{medias[1]:.3f}/{medias[2]:.3f}")

    print("\nSharpe por topologia (exploratorio, D-R):")
    for topo in TOPOLOGIES:
        dr = _boot_p(diffs("D", "R", "sharpe_ratio", topo), one_sided=False)
        print(f"  {topo:<12} d={dr['diff']:+.4f} p2={dr['p']:.4f}")


def dispersion_exponent() -> None:
    """Exponente de dispersión de la DTQW por topología (secundaria EXP-5)."""
    print("\n--- Exponente de dispersion (RMS distancia vs k; secundaria) ---")
    from src.data.cleaning import CleaningPolicy, clean
    from src.data.download import load_ohlcv
    from src.data.features import FeatureSpec, compute_features
    from src.data.splits import SplitSpec, chronological_split
    from src.graph.graph_builder import GraphSpec, build_graph
    from src.graph.regular_subgraphs import TopologyWrapper
    from src.graph.subgraph_selector import seed_score, select_subgraph
    from src.quantum.dtqw import apply_dtqw
    from src.utils.config import load_config
    from src.utils.paths import CONFIGS_DIR

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
    T = panel.shape[0]
    ts = np.linspace(int(T * 0.65), T - 10, 40, dtype=int)

    ks = np.arange(1, 7)
    resultados: dict[str, list] = defaultdict(list)
    for t in ts:
        g0 = max(0, t - cfg.graph.lookback_window + 1)
        s0 = max(0, t - cfg.graph.seed_score_window + 1)
        graph = build_graph(panel[g0: t + 1], tickers, None, GraphSpec(
            alpha=cfg.graph.alpha, beta=cfg.graph.beta, eps=cfg.graph.eps,
            k_neighbors=cfg.graph.k_neighbors, sym_mode=cfg.graph.sym_mode))
        scores = seed_score(panel[s0: t + 1])
        try:
            sub = select_subgraph(graph, scores,
                                  max_size=cfg.graph.subgraph_max_size,
                                  eligible_mask=np.ones(len(tickers), bool))
        except ValueError:
            continue
        variantes = {"bfs_irregular": sub}
        for kind in TOPOLOGIES:
            w = TopologyWrapper(inner=None, kind=kind, d=3, rng_seed=0,
                                sectors=None,
                                seed_scores=scores)
            try:
                variantes[kind] = w.rewire(sub)
            except ValueError:
                continue
        for nombre, s in variantes.items():
            # distancias BFS al seed sobre la topologia correspondiente
            M = s.size
            dist = np.full(M, np.inf)
            dist[s.seed_idx_local] = 0
            frontera = [s.seed_idx_local]
            while frontera:
                v = frontera.pop(0)
                for u in np.flatnonzero(s.W_local[v] > 0):
                    if dist[u] == np.inf:
                        dist[u] = dist[v] + 1
                        frontera.append(int(u))
            dist[~np.isfinite(dist)] = M  # componentes no alcanzadas
            rms = []
            for k in ks:
                p = apply_dtqw(s.W_local, s.seed_idx_local, int(k),
                               init_mode="seed_centered")
                rms.append(np.sqrt(float((p * dist ** 2).sum())))
            resultados[nombre].append(rms)

    print(f"{'topologia':<14}{'exponente (pendiente log-log)':>32}")
    import csv as _csv
    from src.utils.paths import TABLES_DIR as _TD
    out_csv = _TD / "dispersion_rms_v8.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["topology", "k", "rms_mean", "rms_sd", "n_instants",
                    "exponent"])
        for nombre, series in resultados.items():
            arr = np.array(series)          # (muestras, k)
            media = arr.mean(axis=0)
            sd = arr.std(axis=0, ddof=1)
            valid = media > 1e-9
            if valid.sum() < 3:
                continue
            pendiente = np.polyfit(np.log(ks[valid]),
                                   np.log(media[valid]), 1)[0]
            for k, m, s in zip(ks, media, sd):
                w.writerow([nombre, int(k), f"{m:.6f}", f"{s:.6f}",
                            arr.shape[0], f"{pendiente:.4f}"])
            print(f"{nombre:<14}{pendiente:>18.3f}   (RMS medio k=1..6: "
                  + ", ".join(f"{v:.2f}" for v in media) + ")")
    print(f"[OK] series RMS medidas guardadas en {out_csv}")
    print("  referencia: ~1.0 balistico, ~0.5 difusivo (saturacion por "
          "tamano finito M=8 reduce la pendiente para k grandes)")


# ---------------------------------------------------------------------------
# G3 — EXP-6 y EXP-7
# ---------------------------------------------------------------------------


def analyze_g3() -> None:
    print("\n" + "=" * 70)
    print("COMPUERTA G3 — EXP-6 (integracion suave) y EXP-7 (informados+rotacion)")
    print("=" * 70)

    # --- H-v8.5 ---
    try:
        soft = _load("soft_integration_v8.csv")
        by = {(r["model"], int(r["seed"])): r for r in soft}
        seeds = sorted({k[1] for k in by})
        beta_star = soft[0]["beta_bias"]

        def sdiff(a, b, metric):
            return np.array([float(by[(a, s)][metric]) - float(by[(b, s)][metric])
                             for s in seeds if (a, s) in by and (b, s) in by])

        res = _boot_p(sdiff("softD", "softR", "sharpe_ratio"))
        print(f"\nH-v8.5  Sharpe soft-D - soft-R (unilateral, beta*={beta_star})")
        print(f"  d={res['diff']:+.4f} IC95[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}] "
              f"p={res['p']:.4f} pBonf6={min(1, res['p']*K_BONF):.4f} (n={res['n']})")
        verd = ("RECHAZA H0 => senal cuantica con valor financiero al integrarse "
                "suave (Escenario C)" if res["p"] * K_BONF < 0.05 else
                "NO significativa")
        print(f"  VEREDICTO: {verd}")
        print("\n  Medias (Sharpe / cand_hit / topm):")
        for mode in ("softD", "softC", "softR"):
            vals = [r for r in soft if r["model"] == mode]
            sh = np.mean([float(r["sharpe_ratio"]) for r in vals])
            ca = np.mean([float(r["candidate_hit_rate"]) for r in vals])
            tp = np.mean([float(r["topm_hit_rate"]) for r in vals])
            print(f"    {mode:<7} {sh:+.4f} / {ca:.4f} / {tp:.4f}")
        # comparación con las duras (referencia)
        d_hard = np.mean([float(r["sharpe_ratio"])
                          for r in _load("campaign_1_v3.csv") if r["model"] == "D"])
        print(f"    (referencia dura D: Sharpe {d_hard:+.4f})")
        for a, b in [("softD", "softC")]:
            r2 = _boot_p(sdiff(a, b, "sharpe_ratio"), one_sided=False)
            print(f"  [exploratorio] {a}-{b} Sharpe d={r2['diff']:+.4f} p2={r2['p']:.4f}")
    except FileNotFoundError:
        print("\n[AVISO] soft_integration_v8.csv no existe aun.")

    # --- EXP-7 (FDR) ---
    try:
        inf = _load("informed_walkers_v8.csv")
        r_map = {int(r["seed"]): r for r in _load("variant_R_v6.csv")}
        by = defaultdict(dict)
        for r in inf:
            by[r["model"]][int(r["seed"])] = r
        print("\nEXP-7  selectores informados+rotacion vs R (exploratorio, FDR):")
        contrastes = []
        for walker in sorted(by):
            for metric in ("candidate_hit_rate", "sharpe_ratio"):
                seeds = sorted(set(by[walker]) & set(r_map))
                d = np.array([float(by[walker][s][metric])
                              - float(r_map[s][metric]) for s in seeds])
                res = _boot_p(d)  # unilateral walker > R
                contrastes.append((walker, metric, res))
        adj = _fdr([c[2]["p"] for c in contrastes])
        for (walker, metric, res), q in zip(contrastes, adj):
            marca = " <== supera a R (q<0.05)" if q < 0.05 else ""
            print(f"  {walker:<18} {metric:<20} d={res['diff']:+.4f} "
                  f"IC95[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}] "
                  f"p={res['p']:.4f} q_FDR={q:.4f}{marca}")
    except FileNotFoundError:
        print("\n[AVISO] informed_walkers_v8.csv no existe aun.")


if __name__ == "__main__":
    analyze_g2()
    dispersion_exponent()
    analyze_g3()
