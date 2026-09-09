"""Inventario de artefactos para el seguimiento de revision (encargo §1).

Enumera los artefactos disponibles (tablas, codigo, configuraciones) y deja
constancia EXPLICITA de los ausentes que impiden reconstruir las series
diarias de A/D/R. Salida: outputs/tables/review_followup_inventory.csv.

Uso::

    uv run python scripts/review_followup_inventory.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

from src.utils.paths import TABLES_DIR

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OUT = TABLES_DIR / "review_followup_inventory.csv"
TEST_PERIOD = "2023-08-11..2024-12-30 (349 fechas)"
WF_PERIOD = "2021-01-01..2024-12-31 (por pliegue anual)"

CSVS = [
    ("campaign_1_v3.csv", "Sharpe/metricas agregadas por corrida A/B/C/D",
     TEST_PERIOD),
    ("variant_R_v6.csv", "corridas selector R", TEST_PERIOD),
    ("variant_Q_v6.csv", "corridas selector Q", TEST_PERIOD),
    ("rotation_sweep_v7.csv", "corridas S(p) x6", TEST_PERIOD),
    ("oracle_v7.csv", "corridas mascara-oraculo (diagnostico)", TEST_PERIOD),
    ("abl_nisq_v3.csv", "corridas D bajo ruido NISQ x4", TEST_PERIOD),
    ("abl_coin_v3.csv", "corridas D con monedas x3", TEST_PERIOD),
    ("sticky_calibrated_v8.csv", "corridas S(p*) (control posterior)",
     TEST_PERIOD),
    ("crel_variant_v8.csv", "corridas Crel (control posterior)", TEST_PERIOD),
    ("regular_topologies_v8.csv", "corridas topologia x selector x12",
     TEST_PERIOD),
    ("soft_integration_v8.csv", "corridas soft x3", TEST_PERIOD),
    ("informed_walkers_v8.csv", "corridas informados x3", TEST_PERIOD),
    ("tost_extension_v8.csv", "brazos R y D, 30 semillas nuevas", TEST_PERIOD),
    ("review3_checks_v8.csv", "D centrada + sensibilidad costes (70 corridas)",
     TEST_PERIOD),
    ("benchmarks_v6.csv", "benchmarks clasicos agregados", TEST_PERIOD),
    ("walk_forward_v4.csv", "walk-forward C/D agregado por pliegue", WF_PERIOD),
    ("regime_validation_v7.csv", "validacion por regimen (agregados)",
     WF_PERIOD),
    ("financial_stats_v8.csv", "EXP-11 original (procedimiento anterior)",
     WF_PERIOD),
    ("dsr_trial_sharpes.csv", "familia de ensayos DSR (nuevo)", TEST_PERIOD),
    ("dsr_recalculated.csv", "DSR recalculado (nuevo)", WF_PERIOD),
    ("fallback_events.csv", "eventos por paso del replay instrumentado (nuevo)",
     TEST_PERIOD),
    ("fallback_summary.csv", "resumen de respaldo por brazo/semilla (nuevo)",
     TEST_PERIOD),
    ("fallback_paired_effects.csv", "D-C con y sin respaldos (nuevo)",
     TEST_PERIOD),
    ("mask_steps_v8.npz", "mascaras por paso del replay v8 (uint8)",
     TEST_PERIOD),
]

CODE = [
    ("src/training/evaluate.py", "evaluacion: Sharpe por paso, cand/topm; "
     "NO persiste series por paso"),
    ("src/env/reward.py", "recompensa: retorno log, penalizaciones"),
    ("src/env/market_env.py", "entorno; coste de transaccion en el paso"),
    ("src/agents/hybrid_agent.py", "make_mask; respaldo q_t=U lineas 123-152"),
    ("src/graph/subgraph_selector.py", "select_subgraph; ValueError en "
     "aislado/no elegible"),
    ("scripts/run_campaign.py", "pipeline entrenar+evaluar (5 episodios)"),
    ("configs/experiment/model_d_v2.yaml", "configuracion exacta de D "
     "(A/C/R variantes del mismo pipeline)"),
    ("scripts/fallback_instrumentation_v8.py", "replay instrumentado (nuevo)"),
    ("scripts/build_dsr_trial_family.py", "familia de ensayos (nuevo)"),
    ("scripts/recalculate_dsr_review.py", "recalculo DSR (nuevo)"),
]

MISSING = [
    ("registros_por_paso_evaluacion", "fecha/accion/retorno por paso de las "
     "campanias: evaluate_agent solo devuelve agregados; nunca se escribieron"),
    ("checkpoints_politicas", "outputs/checkpoints esta vacio (.gitkeep): "
     "las politicas PPO entrenadas no se conservaron"),
    ("adr_daily_returns.csv", "NO PRODUCIBLE: requiere acciones ejecutadas "
     "por fecha, que exigirian reentrenar (fuera del alcance del encargo)"),
    ("adr_series_stats.csv", "NO PRODUCIBLE: depende del anterior"),
    ("series_diarias_alineadas_40_configs", "no existen series por fecha por "
     "configuracion => matriz de correlacion y K_eff no estimables"),
]


def main() -> None:
    rows: list[dict] = []
    for name, note, period in CSVS:
        p = TABLES_DIR / name
        n = ""
        if p.exists() and p.suffix == ".csv":
            try:
                n = len(pd.read_csv(p))
            except Exception:
                n = ""
        rows.append({"artifact": name, "path": f"outputs/tables/{name}",
                     "available": "si" if p.exists() else "NO",
                     "period": period, "frequency": "diaria (agregada)",
                     "rows": n, "notes": note})
    for path, note in CODE:
        rows.append({"artifact": Path(path).name, "path": path,
                     "available": "si" if (ROOT / path).exists() else "NO",
                     "period": "", "frequency": "", "rows": "",
                     "notes": note})
    for name, note in MISSING:
        rows.append({"artifact": name, "path": "", "available": "NO",
                     "period": "", "frequency": "", "rows": "",
                     "notes": note})

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    faltan = [r for r in rows if r["available"] == "NO"]
    print(f"[OK] {OUT}: {len(rows)} artefactos, {len(faltan)} ausentes")
    for r in faltan:
        print(f"  AUSENTE: {r['artifact']} - {r['notes']}")


if __name__ == "__main__":
    main()
