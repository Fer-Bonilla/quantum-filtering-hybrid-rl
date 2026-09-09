"""Recalculo del DSR con varianza transversal entre ensayos (encargo §5).

Corrige el procedimiento anterior (etiquetado como no valido), que usaba la
varianza de MUESTREO de la serie individual en lugar de la varianza
TRANSVERSAL de los Sharpe de la familia de ensayos (Bailey & Lopez de Prado
2014). Supuesto nulo declarado: SR verdadero = 0 para todas las
configuraciones.

  SR* = sd_transversal * [(1-gammaE) * z_{1-1/K} + gammaE * z_{1-1/(K e)}]
  DSR = Phi( (SR_hat - SR*) * sqrt(T-1) /
             sqrt(1 - g3*SR_hat + (g4-1)/4 * SR_hat^2) )

K_eff: las series diarias alineadas de las configuraciones no se registraron
(solo agregados por corrida), por lo que la correlacion transversal y el
K_eff del apendice de bailey2014deflated NO son estimables con los artefactos
disponibles. Se reporta sensibilidad por valores asumidos K en
{10, 20, 39, 19}. A/D/R: sin serie diaria verificable => sin DSR.

Uso::

    uv run python scripts/recalculate_dsr_review.py
"""

from __future__ import annotations

import csv
import platform
import sys

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from src.data.cleaning import CleaningPolicy, clean
from src.data.download import load_ohlcv
from src.data.features import FeatureSpec, compute_features
from src.utils.paths import TABLES_DIR

sys.stdout.reconfigure(encoding="utf-8")

EULER = 0.5772156649015329
OUT = TABLES_DIR / "dsr_recalculated.csv"


def sharpe(r: np.ndarray) -> float:
    return float(r.mean() / (r.std(ddof=1) + 1e-12))


def sr_star(sd_cross: float, k: int) -> float:
    z1 = stats.norm.ppf(1 - 1.0 / k)
    z2 = stats.norm.ppf(1 - 1.0 / (k * np.e))
    return float(sd_cross * ((1 - EULER) * z1 + EULER * z2))


def dsr(sr_hat: float, T: int, g3: float, g4: float, star: float) -> float:
    num = (sr_hat - star) * np.sqrt(T - 1)
    den = np.sqrt(max(1 - g3 * sr_hat + (g4 - 1) / 4.0 * sr_hat ** 2, 1e-12))
    return float(stats.norm.cdf(num / den))


def main() -> None:
    fam = pd.read_csv(TABLES_DIR / "dsr_trial_sharpes.csv")
    cand = fam[fam["family"] != "diagnostico_no_candidato"]["daily_sharpe"]
    sel = fam[fam["family"] == "seleccion_original"]["daily_sharpe"]
    sd39 = float(cand.std(ddof=1))
    sd19 = float(sel.std(ddof=1))

    # Series diarias reconstruibles (identica construccion que EXP-11)
    raw = load_ohlcv(universe="nivel2")
    features = compute_features(clean(raw, CleaningPolicy()),
                                FeatureSpec(return_type="log"))
    cols = [c for c in features.columns if c[1] == "return"]
    fechas = features.index
    panel = features.loc[:, cols].to_numpy(dtype=float)
    wf = (fechas >= "2021-01-01") & (fechas <= "2024-12-31")
    idx = np.flatnonzero(wf)
    eq = panel[idx].mean(axis=1)
    mom = np.empty(len(idx) - 1)
    for j, t in enumerate(idx[:-1]):
        s = max(0, t - 19)
        mom[j] = panel[t + 1, int(np.argmax(panel[s: t + 1].mean(axis=0)))]

    escenarios = [
        ("K=39_sd_candidatas", 39, sd39,
         "familia candidata completa (sin oraculo); recomendado"),
        ("K=19_sd_seleccion", 19, sd19,
         "solo seleccion original sin oraculo (controles v8 separados)"),
        ("K=20_asumido", 20, sd39, "sensibilidad por valor asumido"),
        ("K=10_asumido", 10, sd39, "sensibilidad por valor asumido"),
    ]

    rows: list[dict] = []
    print(f"sd transversal: candidatas(39)={sd39:.4f}  seleccion(19)={sd19:.4f}")
    print(f"{'estrategia':<14}{'escenario':<22}{'SR':>8}{'SR*':>8}"
          f"{'DSR':>8}  veredicto")
    for nombre, serie in [("equal_weight", eq), ("momentum_20d", mom)]:
        sr = sharpe(serie)
        g3 = float(stats.skew(serie))
        g4 = float(stats.kurtosis(serie, fisher=False))
        T = len(serie)
        for esc, k, sd, nota in escenarios:
            star = sr_star(sd, k)
            d = dsr(sr, T, g3, g4, star)
            v = "supera 0,95" if d >= 0.95 else "no supera 0,95"
            rows.append({
                "estrategia": nombre, "escenario": esc, "K": k,
                "sd_transversal": round(sd, 6), "sharpe": round(sr, 6),
                "T": T, "skew": round(g3, 4), "kurt_pearson": round(g4, 4),
                "sr_star": round(star, 6), "dsr": round(d, 6),
                "veredicto_095": v, "estado": "recalculo_valido", "nota": nota})
            print(f"{nombre:<14}{esc:<22}{sr:>8.4f}{star:>8.4f}{d:>8.4f}  {v}")

    # Valores antiguos: procedimiento anterior no valido (varianza de
    # muestreo de la serie individual en lugar de la transversal)
    for nombre, d_old in [("momentum_20d", 0.594785), ("equal_weight", 0.293946)]:
        rows.append({
            "estrategia": nombre, "escenario": "publicado_v8_EXP11", "K": 40,
            "sd_transversal": "", "sharpe": "", "T": "", "skew": "",
            "kurt_pearson": "", "sr_star": "", "dsr": d_old,
            "veredicto_095": "no supera 0,95",
            "estado": "procedimiento_anterior_no_valido",
            "nota": "varianza de muestreo individual, no transversal"})

    # A/D/R: sin serie diaria verificable => sin DSR
    for agente in ["A", "D", "R"]:
        rows.append({
            "estrategia": agente, "escenario": "sin_DSR", "K": "",
            "sd_transversal": "", "sharpe": "", "T": "", "skew": "",
            "kurt_pearson": "", "sr_star": "", "dsr": "",
            "veredicto_095": "",
            "estado": "sin_serie_diaria_verificable",
            "nota": "sin registros por paso ni checkpoints; no reconstruible "
                    "sin reentrenar"})

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {OUT}")
    print(f"software: Python {platform.python_version()}, "
          f"numpy {np.__version__}, scipy {scipy.__version__}, "
          f"pandas {pd.__version__}; sin aleatoriedad (calculo cerrado)")


if __name__ == "__main__":
    main()
