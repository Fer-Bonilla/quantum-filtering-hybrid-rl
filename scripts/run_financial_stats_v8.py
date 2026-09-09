"""EXP-11 v8 — Rigor estadístico financiero sobre el walk-forward (Tier 4).

Sin reentrenar:
  (a) Deflated Sharpe Ratio (Bailey & López de Prado) con el número de
      configuraciones probadas declarado explícitamente.
  (b) IC de Sharpe por bootstrap estacionario por bloques (Politis–Romano,
      longitud media de bloque 20 días) para las series diarias
      reconstruibles (1/N y momentum 20d sobre 2021-2024).
  (c) Para los agentes RL (A, D, R): las series diarias por paso no quedaron
      registradas y no son reconstruibles sin reentrenar; se reporta el DSR
      bajo aproximación normal DECLARADA (asimetría 0, curtosis 3) sobre el
      Sharpe por paso medio del walk-forward (T=1260 pasos de evaluación),
      etiquetado como aproximación.

Uso::

    uv run python scripts/run_financial_stats_v8.py
"""

from __future__ import annotations

import csv

import numpy as np
from scipy import stats

from src.data.cleaning import CleaningPolicy, clean
from src.data.download import load_ohlcv
from src.data.features import FeatureSpec, compute_features
from src.utils.paths import TABLES_DIR

EULER = 0.5772156649015329
BLOCK_MEAN = 20
B_BOOT = 5000
# Configuraciones de selector/estrategia evaluadas sobre el mismo test a lo
# largo de la tesis (A, B, C, D, R, Q, S(p) x6, oraculo, soft x3, informados
# x3, topologias x12, benchmarks x4, NISQ x4, monedas x3...) — cota
# conservadora declarada:
N_TRIALS = 40
OUT = TABLES_DIR / "financial_stats_v8.csv"


def sharpe(r: np.ndarray) -> float:
    return float(r.mean() / (r.std(ddof=1) + 1e-12))


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """E[max SR] bajo n_trials pruebas (Bailey & López de Prado, 2014)."""
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER) * z1 + EULER * z2))


def dsr(sr_hat: float, T: int, skew: float, kurt: float,
        sr_star: float) -> float:
    """Deflated Sharpe Ratio = PSR(SR*) con momentos de la serie."""
    num = (sr_hat - sr_star) * np.sqrt(T - 1)
    den = np.sqrt(max(1 - skew * sr_hat + (kurt - 1) / 4.0 * sr_hat ** 2,
                      1e-12))
    return float(stats.norm.cdf(num / den))


def stationary_bootstrap_ci(r: np.ndarray, b: int = B_BOOT,
                            mean_block: int = BLOCK_MEAN,
                            seed: int = 20260705) -> tuple[float, float]:
    """IC95% del Sharpe por bootstrap estacionario (Politis–Romano)."""
    rng = np.random.default_rng(seed)
    T = len(r)
    p = 1.0 / mean_block
    out = np.empty(b)
    for i in range(b):
        idx = np.empty(T, dtype=np.int64)
        idx[0] = rng.integers(0, T)
        for t in range(1, T):
            idx[t] = rng.integers(0, T) if rng.random() < p else (idx[t - 1] + 1) % T
        out[i] = sharpe(r[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    raw = load_ohlcv(universe="nivel2")
    features = compute_features(clean(raw, CleaningPolicy()),
                                FeatureSpec(return_type="log"))
    cols = [c for c in features.columns if c[1] == "return"]
    fechas = features.index
    panel = features.loc[:, cols].to_numpy(dtype=float)
    wf = (fechas >= "2021-01-01") & (fechas <= "2024-12-31")
    idx = np.flatnonzero(wf)

    # Series diarias reconstruibles (mismas convenciones que los benchmarks)
    eq = panel[idx].mean(axis=1)                              # 1/N
    mom = np.empty(len(idx) - 1)
    for j, t in enumerate(idx[:-1]):
        s = max(0, t - 19)
        mom[j] = panel[t + 1, int(np.argmax(panel[s: t + 1].mean(axis=0)))]

    rows = []
    print(f"EXP-11 — DSR (N_trials declarado = {N_TRIALS}) + bootstrap "
          f"estacionario (bloque medio {BLOCK_MEAN}d, B={B_BOOT})\n")
    print(f"{'estrategia':<16}{'SR diario':>10}{'IC95 bloque':>22}"
          f"{'skew':>7}{'kurt':>7}{'SR*':>8}{'DSR':>8}")
    for nombre, serie in [("equal_weight", eq), ("momentum_20d", mom)]:
        sr = sharpe(serie)
        lo, hi = stationary_bootstrap_ci(serie)
        sk = float(stats.skew(serie))
        ku = float(stats.kurtosis(serie, fisher=False))
        var_sr = (1 - sk * sr + (ku - 1) / 4 * sr ** 2) / (len(serie) - 1)
        sr_star = expected_max_sr(N_TRIALS, var_sr)
        d = dsr(sr, len(serie), sk, ku, sr_star)
        rows.append({"estrategia": nombre, "tipo": "serie_diaria",
                     "sharpe": sr, "ci_lo": lo, "ci_hi": hi, "skew": sk,
                     "kurt": ku, "sr_star": sr_star, "dsr": d,
                     "T": len(serie), "nota": "reconstruida de datos"})
        print(f"{nombre:<16}{sr:>10.4f}   [{lo:+.4f},{hi:+.4f}]"
              f"{sk:>7.2f}{ku:>7.2f}{sr_star:>8.4f}{d:>8.4f}")

    # Agentes RL: Sharpe por paso medio del walk-forward (aprox. normal declarada)
    with (TABLES_DIR / "regime_validation_v7.csv").open(encoding="utf-8") as fh:
        reg = list(csv.DictReader(fh))
    T_eval = 1260  # 5 episodios x 252 pasos por fold
    print()
    for agente in ["A", "D", "R"]:
        srs = [float(r["sharpe_ratio"]) for r in reg
               if r["strategy"] == agente and np.isfinite(float(r["sharpe_ratio"]))]
        sr = float(np.mean(srs))
        var_sr = (1 + sr ** 2 / 2) / (T_eval - 1)
        sr_star = expected_max_sr(N_TRIALS, var_sr)
        d = dsr(sr, T_eval, 0.0, 3.0, sr_star)
        rows.append({"estrategia": agente, "tipo": "aprox_normal_declarada",
                     "sharpe": sr, "ci_lo": "", "ci_hi": "", "skew": 0.0,
                     "kurt": 3.0, "sr_star": sr_star, "dsr": d,
                     "T": T_eval, "nota": "serie diaria no registrada"})
        print(f"{agente:<16}{sr:>10.4f}   [aprox. normal declarada]"
              f"{0.0:>7.2f}{3.0:>7.2f}{sr_star:>8.4f}{d:>8.4f}")

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {OUT}")
    print("\nLectura: DSR < 0,95 => el Sharpe observado no es distinguible "
          "del mejor Sharpe esperado por azar tras N_trials configuraciones. "
          "Robustece el nulo financiero de la tesis.")


if __name__ == "__main__":
    main()
