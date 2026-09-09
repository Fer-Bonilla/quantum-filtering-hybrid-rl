# Resultados

Todos los resultados que sustentan la memoria, tal como se generaron. Cada
CSV de campaña tiene **una fila por corrida** (modelo/variante, semilla,
métricas financieras, de aprendizaje y de alineación, latencia, marca de
tiempo y duración). Las tablas `*_paired_*`, `*_summary` y `*_report.md`
son agregados derivados por `scripts/aggregate_results.py`.

## `tables/` — catálogo por iteración

| Iteración | Archivos |
|---|---|
| 1 Banco de pruebas (v1–v2) | `campaign_1*.csv`, `campaign_1_v2*.csv`, `scan_focused.csv`, `sweet_spot*.csv`, `sweet_abl_*.csv`, `abl_lambda_mini.csv`, `ablation_*_mini.csv`, `mini*.csv`, `smoke_*.csv`, `demo*.csv` |
| 2 Efecto y veredicto (v3–v5) | `campaign_1_v3*.csv` (campaña principal, n=10), `abl_nisq_v3.csv`, `abl_nisq_v5_*.csv`, `abl_coin_v3.csv`, `walk_forward_v3*.csv`, `walk_forward_v4.csv`, `sweet_spot_holdout_v2*.csv`, `sweet_spot_holdout_v3*.csv`, `sweet_spot_tuning_v2.csv` |
| 3 Mecanismo (v6) | `variant_R_v6.csv`, `variant_Q_v6.csv`, `paired_variants_v6.csv`, `benchmarks_v6.csv`, `annealing_fidelity_v6.csv`, `v6/` (tablas md/tex) |
| 4 Causalidad y techo (v7) | `rotation_sweep_v7*.csv`, `mediation_rotation_v7.csv`, `tost_equivalence_v7.csv`, `oracle_v7*.csv`, `regime_validation_v7.csv`, `v7/` |
| 5 Campaña pre-registrada (v8) | `tost_extension_v8.csv`, `sticky_calibrated_v8.csv`, `mask_metrics_v8*.csv`, `mask_steps_v8.npz`, `mask_information_v8.csv`, `regular_topologies_v8.csv`, `dispersion_rms_v8.csv`, `soft_integration_v8*.csv`, `informed_walkers_v8*.csv`, `financial_stats_v8.csv`, `crel_variant_v8.csv` |
| 6 Auditorías | `review3_checks_v8.csv`, `review_followup_inventory.csv`, `dsr_trial_sharpes.csv`, `dsr_recalculated.csv`, `dsr_k_sensitivity.csv`, `fallback_events.csv`, `fallback_summary.csv`, `fallback_paired_effects.csv` |

Columnas de las campañas de entrenamiento: `campaign_id, model, seed,
universe, total_steps, cumulative_return, sharpe_ratio, max_drawdown,
mean_reward, asset_coverage, topm_hit_rate, candidate_hit_rate,
time_to_first_promising, mean_latency_ms, episodes_to_convergence,
train_reward_last, train_reward_mean, n_train_steps, timestamp_utc,
duration_seconds` (más columnas específicas como `topology`, `beta_bias`,
`tag`). `sharpe_ratio` es el Sharpe por paso de la recompensa (diario, sin
anualizar); `candidate_hit_rate` y `topm_hit_rate` se definen en
`src/training/evaluate.py`.

## `figures/`

- `memoria/`: figuras de la memoria (PDF vectoriales y PNG). Las cinco
  `fig_v8_*.pdf` son deterministas y verificables con
  `scripts/v8_thesis_figures.py --verify` contra `manifest_v8_sha256.txt`.
- `v6/`, `v7/`, `v8/`: figuras de presentación de cada iteración.
- PNG de nivel superior: figuras de exploración de las campañas iniciales.

## `logs/`

Registros de ejecución de las campañas (progreso por corrida, avisos y
duración), útiles para auditar tiempos y semillas.
