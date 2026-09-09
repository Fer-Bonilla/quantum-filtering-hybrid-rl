"""Familia de ensayos del DSR (encargo §3-§4): un Sharpe trazable por
configuración, con familia de selección y estadísticas transversales.

Las 40 configuraciones declaradas (Anexo H) se separan en:
  - ``seleccion_original`` (20): evaluadas durante la selección del plan
    original y sus ablaciones (v3-v7).
  - ``control_posterior`` (20): ejecutadas en la campaña v8 como controles
    pre-registrados, posteriores a la selección.

Todos los Sharpe: por paso sobre la recompensa (retorno log neto diario del
activo elegido; lambda=0, coste ~0), frecuencia diaria, sin anualizar,
partición de prueba común 2023-08-11 a 2024-12-30 (349 fechas únicas; la
evaluación usa 5 episodios solapantes x 252 pasos, de modo que los 1260
pasos NO son fechas independientes: por eso ningún T de agentes se usa en el
DSR). Agregación: media entre semillas del Sharpe por corrida.

Dependencia transversal (K_eff): requiere las series diarias alineadas de
las 40 configuraciones; esas series no se registraron (solo agregados por
corrida), por lo que NO se calcula un K_eff: se reporta sensibilidad por
valores asumidos K en {10, 20, 40} (véase recalculate_dsr_review.py).

Uso::

    uv run python scripts/build_dsr_trial_family.py
"""

from __future__ import annotations

import csv
import sys

import numpy as np
import pandas as pd

from src.utils.paths import TABLES_DIR

sys.stdout.reconfigure(encoding="utf-8")

OUT = TABLES_DIR / "dsr_trial_sharpes.csv"
PERIOD = "test 2023-08-11..2024-12-30 (349 fechas unicas)"
T_NOTE = "1260 pasos eval (5 episodios solapantes x 252; no fechas unicas)"


def _mean_sharpe(df: pd.DataFrame) -> float:
    return float(df["sharpe_ratio"].astype(float).mean())


def main() -> None:
    rows: list[dict] = []

    def add(trial_id, family, config, source, df, method="media entre semillas"):
        rows.append({
            "trial_id": trial_id, "family": family,
            "configuration": config, "period": PERIOD,
            "T": 1260, "T_nota": T_NOTE,
            "daily_sharpe": round(_mean_sharpe(df), 6),
            "n_seeds": len(df),
            "source_file": source, "aggregation_method": method})

    c1 = pd.read_csv(TABLES_DIR / "campaign_1_v3.csv")
    c1 = c1[c1["model"].isin(["A", "B", "C", "D"])]
    for mdl in ["A", "B", "C", "D"]:
        add(f"{mdl}", "seleccion_original", f"Modelo {mdl} (campania principal)",
            "campaign_1_v3.csv", c1[c1["model"] == mdl])

    add("R", "seleccion_original", "RandomWalker (control azar)",
        "variant_R_v6.csv", pd.read_csv(TABLES_DIR / "variant_R_v6.csv"))
    add("Q", "seleccion_original", "AnnealingWalker (QUBO)",
        "variant_Q_v6.csv", pd.read_csv(TABLES_DIR / "variant_Q_v6.csv"))

    rot = pd.read_csv(TABLES_DIR / "rotation_sweep_v7.csv")
    for mdl in sorted(set(rot["model"])):
        add(mdl, "seleccion_original", f"StickyWalker {mdl}",
            "rotation_sweep_v7.csv", rot[rot["model"] == mdl])

    # La mascara-oraculo usa informacion futura: es una cota diagnostica del
    # canal selector, no una estrategia candidata en la carrera de seleccion.
    # Se excluye de la familia del DSR y se reporta aparte.
    add("ORACLE", "diagnostico_no_candidato", "mascara-oraculo ex-post + PPO",
        "oracle_v7.csv", pd.read_csv(TABLES_DIR / "oracle_v7.csv"))

    nisq = pd.read_csv(TABLES_DIR / "abl_nisq_v3.csv")
    for cid in sorted(set(nisq["campaign_id"])):
        tag = cid.replace("abl_nisq_v3_", "NISQ_")
        add(tag, "seleccion_original", f"Modelo D bajo ruido {cid}",
            "abl_nisq_v3.csv", nisq[nisq["campaign_id"] == cid])

    coin = pd.read_csv(TABLES_DIR / "abl_coin_v3.csv")
    for cid in sorted(set(coin["campaign_id"])):
        tag = cid.replace("abl_coin_", "coin_")
        add(tag, "seleccion_original", f"Modelo D con moneda {cid}",
            "abl_coin_v3.csv", coin[coin["campaign_id"] == cid])

    add("S_star", "control_posterior", "StickyWalker calibrado S(p*) (v8)",
        "sticky_calibrated_v8.csv",
        pd.read_csv(TABLES_DIR / "sticky_calibrated_v8.csv"))
    add("Crel", "control_posterior", "estado relacional B + filtrado C (v8)",
        "crel_variant_v8.csv", pd.read_csv(TABLES_DIR / "crel_variant_v8.csv"))

    topo = pd.read_csv(TABLES_DIR / "regular_topologies_v8.csv")
    for t in sorted(set(topo["topology"])):
        for mdl in sorted(set(topo["model"])):
            sub = topo[(topo["topology"] == t) & (topo["model"] == mdl)]
            if len(sub):
                add(f"topo_{t}_{mdl}", "control_posterior",
                    f"topologia {t}, selector {mdl} (v8 EXP-5)",
                    "regular_topologies_v8.csv", sub)

    soft = pd.read_csv(TABLES_DIR / "soft_integration_v8.csv")
    for mdl in sorted(set(soft["model"])):
        add(mdl, "control_posterior", f"integracion suave {mdl} (v8 EXP-6)",
            "soft_integration_v8.csv", soft[soft["model"] == mdl])

    inf = pd.read_csv(TABLES_DIR / "informed_walkers_v8.csv")
    for mdl in sorted(set(inf["model"])):
        add(f"inf_{mdl}", "control_posterior",
            f"selector informado {mdl} (v8 EXP-7)",
            "informed_walkers_v8.csv", inf[inf["model"] == mdl])

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sr_all = np.array([r["daily_sharpe"] for r in rows
                       if r["family"] != "diagnostico_no_candidato"])
    sr_sel = np.array([r["daily_sharpe"] for r in rows
                       if r["family"] == "seleccion_original"])
    print(f"[OK] {OUT}  ({len(rows)} configuraciones; "
          f"{len(sr_all)} candidatas tras excluir el oraculo diagnostico)")
    for name, sr in [("39 (candidatas)", sr_all),
                     ("19 (seleccion sin oraculo)", sr_sel)]:
        q1, q2, q3 = np.percentile(sr, [25, 50, 75])
        print(f"\nFamilia {name}: n={len(sr)}")
        print(f"  media={sr.mean():+.4f}  var(ddof=1)={sr.var(ddof=1):.6f}  "
              f"sd={sr.std(ddof=1):.4f}")
        print(f"  min={sr.min():+.4f}  Q1={q1:+.4f}  mediana={q2:+.4f}  "
              f"Q3={q3:+.4f}  max={sr.max():+.4f}")


if __name__ == "__main__":
    main()
