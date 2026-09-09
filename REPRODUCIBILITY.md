# Guía de reproducibilidad

Esta guía permite (a) **recomputar cada análisis y figura de la memoria a
partir de los resultados incluidos** en minutos, y (b) **reejecutar las
campañas de entrenamiento** desde cero si se desea regenerar esos resultados.

Entorno de referencia (el que produjo todos los números): Python 3.13.13,
numpy 2.4.4, scipy 1.17.1, pandas 2.3.3, torch 2.11.0 (CPU), pennylane
0.44.1, uv 0.11.8, Windows 11 x86-64. `uv sync` instala exactamente estas
versiones desde `uv.lock`.

Convenciones estadísticas comunes: bootstrap pareado por semilla con
B = 5 000 remuestreos, IC de percentiles 2,5/97,5, p₂ = 2·min(P(>0), P(<0)),
`numpy.random.default_rng(2026)` en la campaña principal y
`default_rng(20260705)` en v7/v8; TOST de Schuirmann con t de Student e IC90;
corrección Bonferroni sobre la familia declarada en cada campaña.

---

## 1. Datos

Los datos usados están en `data/raw/nivel1` (15 activos) y `data/raw/nivel2`
(30 activos; universo de todas las campañas reportadas), con
`manifest.json` que registra ticker, rango de fechas, filas y SHA-256 de
cada archivo (descarga: 2026-05-12, `auto_adjust=True`, 2018-01-02 a
2024-12-30). Para verificar la integridad:

```bash
uv run python -c "import json,hashlib,pathlib; m=json.load(open('data/raw/nivel2/manifest.json')); print(all(hashlib.sha256(open(pathlib.Path('data')/e['file_path'].replace('\\\\','/'),'rb').read()).hexdigest()==e['sha256'] for e in m['entries']))"
```

Para volver a descargar (los datos ajustados de Yahoo pueden cambiar
retroactivamente; por eso se incluye la copia usada):

```bash
uv run python -m src.data.download --config configs/data/nivel2.yaml
```

Particiones cronológicas (`train_frac=0,6`, `val_frac=0,2`): entrenamiento
2018-02-01…2022-03-23, validación 2022-03-24…2023-08-10, **prueba
2023-08-11…2024-12-30 (349 fechas)**. La evaluación usa 5 episodios de 252
pasos muestreados dentro de la prueba (1 260 pasos, con fechas solapadas).

## 2. Mapa memoria → script → resultado

### Campaña principal y ablación canónica (Iteraciones 1–2)

| Elemento de la memoria | Resultado incluido | Script de análisis | Script de campaña |
|---|---|---|---|
| Tabla maestra A–B–C–D, contrastes pareados D–C y B–A | `campaign_1_v3.csv`, `campaign_1_v3_paired_c_vs_d.csv`, `campaign_1_v3_paired_b_vs_a.csv` | `aggregate_results.py`, `paired_b_vs_a_v5.py` | `run_campaign.py --universe nivel2 --seeds 42 123 456 789 1024 7 99 314 1729 65535 --steps 50000 --config-suffix v2` |
| Ablación de λ / recompensa | `abl_lambda_mini.csv` | — | `run_lambda_ablation.py` |
| Ruido NISQ (4 perfiles) | `abl_nisq_v3.csv`, `abl_nisq_v5_*.csv` | — | `run_nisq_ablation_v3.py` |
| Moneda cuántica | `abl_coin_v3.csv` | — | `run_coin_ablation_v3.py` |
| Walk-forward por pliegues | `walk_forward_v3.csv`, `walk_forward_v4.csv` | — | `run_walk_forward.py` |
| Sweet spot (M=16) y hold-out | `sweet_spot*.csv`, `sweet_abl_*.csv` | `compare_tuning_vs_holdout.py`, `coverage_quality_tradeoff.py` | `run_param_scan.py`, `run_sweet_ablations.py` |

### Controles de mecanismo (Iteración 3, v6)

| Elemento | Resultado | Análisis | Campaña |
|---|---|---|---|
| Modelos R y Q, contrastes pareados | `variant_R_v6.csv`, `variant_Q_v6.csv`, `paired_variants_v6.csv` | `paired_variants_v6.py` | `run_variant_campaign_v6.py --variant R` / `--variant Q` |
| Benchmarks clásicos | `benchmarks_v6.csv` | — | `run_classical_benchmarks.py` |
| Fidelidad del recocido | `annealing_fidelity_v6.csv` | — | `annealing_fidelity_study.py` |
| Figuras v6 | `outputs/figures/v6/` | `v6_presentation_figures.py` | — |

### Demostración causal (Iteración 4, v7)

| Elemento | Resultado | Análisis | Campaña |
|---|---|---|---|
| Dosis-respuesta S(p) | `rotation_sweep_v7.csv` | `analyze_rotation_sweep_v7.py` | `run_rotation_sweep_v7.py` |
| Mediación por replay | `mediation_rotation_v7.csv` | `mediation_rotation_v7.py` (replay, sin reentrenar) | — |
| TOST R–D (n=10) | `tost_equivalence_v7.csv` | `tost_equivalence_v7.py` | — |
| Máscara-oráculo | `oracle_v7.csv` | `analyze_oracle_v7.py` | `run_oracle_campaign_v7.py` |
| Regímenes 2021–2024 | `regime_validation_v7.csv` | `analyze_regime_validation_v7.py` | `run_regime_validation_v7.py` |
| Figuras v7 | `outputs/figures/v7/` | `v7_presentation_figures.py` | — |

### Campaña pre-registrada v8 (Iteración 5)

Pre-registro: `docs/preregistro_v8.md` (hipótesis H-v8.1…H-v8.5, Bonferroni
k=6, potencia, 30 semillas nuevas: primos 11…139).

| Elemento | Resultado | Análisis | Campaña |
|---|---|---|---|
| EXP-1 precision@m, NDCG@m (replay) | `mask_metrics_v8.csv`, `mask_steps_v8.npz` | `run_mask_metrics_v8.py` | — |
| EXP-2 información mutua | `mask_information_v8.csv` | `run_mask_information_v8.py` | — |
| EXP-3 TOST R–D n=40 (30 semillas nuevas) | `tost_extension_v8.csv` | `analyze_v8_g1.py`, `tost_diagnostics_v8.py` | `run_tost_extension_v8.py` |
| EXP-4 rotación calibrada S(p\*) | `sticky_calibrated_v8.csv` | `analyze_v8_g1.py`, `check_manipulation_v8.py` | `run_sticky_calibrated_v8.py` |
| EXP-5 topologías regulares + dispersión | `regular_topologies_v8.csv`, `dispersion_rms_v8.csv` | `analyze_v8_g2g3.py` | `run_regular_topologies_v8.py` |
| EXP-6 integración suave | `soft_integration_v8.csv`, `*_tuning.csv` | `analyze_v8_g2g3.py` | `run_soft_integration_v8.py` |
| EXP-7 selectores informados | `informed_walkers_v8.csv` | `analyze_v8_g2g3.py` | `run_informed_walkers_v8.py` |
| EXP-11 DSR y bootstrap estacionario (original) | `financial_stats_v8.csv` | `run_financial_stats_v8.py` | — |
| Variante Crel | `crel_variant_v8.csv` | — | `run_crel_variant_v8.py` |
| Figuras G1–G3 | `outputs/figures/v8/` | `v8_g1_figures.py`, `v8_g2g3_figures.py` | — |

### Auditorías de reproducibilidad (Iteración 6)

| Elemento | Resultado | Script |
|---|---|---|
| Condición inicial emparejada y sensibilidad a costes (70 corridas) | `review3_checks_v8.csv` | `run_review3_checks_v8.py` (campaña), `analyze_review3_v8.py` |
| Grados tras simetrización y concordancia de modos Q | (consola) | `degree_qmode_stats_v8.py` |
| Diagnósticos TOST: normalidad, Wilcoxon, márgenes | (consola) | `tost_diagnostics_v8.py` |
| Familia de 40 ensayos del DSR | `dsr_trial_sharpes.csv` | `build_dsr_trial_family.py` |
| DSR con varianza transversal | `dsr_recalculated.csv` | `recalculate_dsr_review.py` |
| Sensibilidad del DSR en K = 2…39 | `dsr_k_sensitivity.csv` | `dsr_k_sensitivity_review.py` |
| Respaldo de máscara completa (37 800 pasos) | `fallback_events.csv`, `fallback_summary.csv`, `fallback_paired_effects.csv` | `fallback_instrumentation_v8.py` |
| Inventario de artefactos | `review_followup_inventory.csv` | `review_followup_inventory.py` |
| Informe de las auditorías | `docs/review_followup_calculations.md` | — |

### Figuras de la memoria

`outputs/figures/memoria/` contiene las figuras finales. Las cinco figuras
v8 son **deterministas** (`SOURCE_DATE_EPOCH` fijo): una regeneración limpia
reproduce los PDF byte a byte, y `manifest_v8_sha256.txt` lo certifica.

```bash
uv run python scripts/v8_thesis_figures.py            # regenera + manifiesto
uv run python scripts/v8_thesis_figures.py --verify   # 5/5 OK esperado
uv run python scripts/thesis_figures_es.py            # resto de figuras vectoriales
```

## 3. Reejecutar las campañas desde cero

Todos los scripts `run_*` son **reanudables**: escriben una fila por corrida
al terminar y omiten las ya presentes en el CSV de salida, de modo que una
campaña interrumpida se completa relanzando el mismo comando. Cada corrida
de entrenamiento (50 000 pasos PPO + evaluación) tarda 60–100 s en CPU.

| Campaña | Corridas | Tiempo CPU aprox. | Comando |
|---|---|---|---|
| Principal A–D, n=10 | 40 | ~1 h | `uv run python scripts/run_campaign.py --universe nivel2 --seeds 42 123 456 789 1024 7 99 314 1729 65535 --steps 50000 --config-suffix v2 --campaign-id campaign_1_v3` |
| R y Q (v6) | 20 | ~30 min | `uv run python scripts/run_variant_campaign_v6.py --variant R` y `--variant Q` |
| Dosis-respuesta S(p) (v7) | 60 | ~1,5 h | `uv run python scripts/run_rotation_sweep_v7.py` |
| Oráculo (v7) | 10 | ~15 min | `uv run python scripts/run_oracle_campaign_v7.py` |
| Regímenes (v7) | 60 + benchmarks | ~1,5 h | `uv run python scripts/run_regime_validation_v7.py` |
| EXP-3 v8 (30 semillas × R, D) | 60 | ~1,5 h | `uv run python scripts/run_tost_extension_v8.py` |
| EXP-4 v8 S(p\*) | 40 | ~1 h | `uv run python scripts/run_sticky_calibrated_v8.py` |
| EXP-5 v8 topologías | 120 | ~3 h | `uv run python scripts/run_regular_topologies_v8.py` |
| EXP-6 / EXP-7 v8 | 30 + 30 (+ tuning) | ~2 h | `uv run python scripts/run_soft_integration_v8.py`, `run_informed_walkers_v8.py` |
| Crel (v8) | 10 | ~15 min | `uv run python scripts/run_crel_variant_v8.py` |
| Revisión 3 (D centrada + costes) | 70 | ~2 h | `uv run python scripts/run_review3_checks_v8.py` |
| Replays sin reentrenar (EXP-1/2, mediación, respaldo) | — | 5–30 min | `run_mask_metrics_v8.py`, `run_mask_information_v8.py`, `mediation_rotation_v7.py`, `fallback_instrumentation_v8.py` |

Determinismo: cada corrida fija la semilla en NumPy, PyTorch y el entorno;
los selectores estocásticos (R, S(p)) reciben la semilla de la corrida. En
CPU los resultados de entrenamiento se reproducen exactamente con la misma
versión de PyTorch; con otra versión pueden diferir en el último decimal sin
cambiar ningún veredicto (todos los contrastes están lejos de sus umbrales
salvo donde la memoria lo declara explícitamente).

## 4. Qué no puede reproducirse y por qué

- **Series diarias de retornos de los agentes A/D/R**: la evaluación solo
  persistió agregados y no se conservaron *checkpoints* de las políticas;
  regenerarlas exige reentrenar. Por eso la memoria mantiene A/D/R sin
  Deflated Sharpe Ratio (`docs/review_followup_calculations.md`, §2).
- **K efectivo del DSR**: requiere series diarias alineadas por
  configuración, que no existen; se reporta sensibilidad en K.
- **`outputs/mlruns/`** (80 MB de registros MLflow) no se incluye; contiene
  las mismas métricas agregadas que los CSV.

## 5. Verificación rápida del paquete

```bash
uv sync
uv run pytest -q                                   # 313 passed
uv run python scripts/v8_thesis_figures.py --verify  # 5/5 OK
uv run python scripts/analyze_review3_v8.py        # reproduce el Anexo de la 3ª revisión
```
