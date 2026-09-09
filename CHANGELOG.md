# Changelog

Bitácora del proyecto siguiendo el plan de implementación de 9 semanas.
Cada entrada documenta los criterios de aceptación cumplidos.

## Semanas 1-7 — Pipeline MVP completo · COMPLETADAS (2026-05-11)

### Semana 1 — Setup + Capa de datos
- `pyproject.toml` con `uv` 0.11.8, Python 3.13.13, 16 deps runtime + 5 dev.
- `src/utils/`: paths, logging, hashing, config (pydantic con 5 grupos + ExperimentConfig).
- `src/data/`: download (yfinance + manifest SHA-256), cleaning, features (shift(1) defensivo), splits, sector_map.
- 9 configs YAML (data nivel1, env, agent, graph, quantum, experiment model_a-d).

### Semana 2 — Entorno Gymnasium + máscara
- `src/env/`: `MarketEnv` (Gymnasium), `reward`, `state_builder`, `episode_sampler`.
- `src/agents/masked_policy.py`: NEG_INF=-1e9 (estabilidad numérica) + entropía sin NaN.
- `RecordingMarketEnv` instrumentado para verificar ausencia de fuga futura.

### Semana 3 — PPO Modelo A end-to-end
- `src/agents/`: `policy_network` (ActorCritic con tronco compartido), `rollout_buffer` (con máscara almacenada para el update PPO), `classical_agent`.
- `src/training/`: `trainer` (PPO con GAE, máscara dinámica reaplicada en update), `evaluate` (Sharpe, drawdown, cobertura), `mlflow_logger`, `seed_utils`.

### Semana 4 — Grafo dinámico G_t
- `src/graph/`: `affinity`, `sparsify` (k-NN simetrizado en 3 modos), `graph_builder`, `subgraph_selector` (BFS ponderada desde el nodo semilla).

### Semana 5 — Modelos B y C
- `src/graph/relational_features.py`: degree, weighted clustering, Laplacian spectral embedding.
- `src/graph/classical_walk.py`: caminata aleatoria ponderada `D^-1 W`.
- `src/quantum/classical_walker.py`: fachada `LocalModule` para Modelo C.
- `src/agents/hybrid_agent.py`: orquesta grafo + subgrafo + LocalModule + máscara top-m.

### Semana 6 — Módulo cuántico DTQW
- `src/quantum/encoding.py`: codificación por puertos locales con padding controlado.
- `src/quantum/dtqw.py`: matrices densas `complex128` para C_t (Householder), S_t (permutación), U_t = S·C. Renormalización defensiva con threshold 1e-9.
- `src/quantum/measurement.py`: TopM con exclusión configurable.

### Semana 7 — Modelo D híbrido
- `src/quantum/quantum_walker.py`: fachada `LocalModule` para Modelo D (mismo protocolo que ClassicalWalker — C ↔ D intercambiables vía config).
- CLI completo en `src/main.py` con typer.
- Integración end-to-end de los 4 modelos validada.

## Validaciones acumuladas

| Categoría | Estado |
|---|---|
| **175 tests verdes** | ✓ unit (171) + integration (4) en < 5 s |
| `ruff check src tests scripts` | ✓ All checks passed |
| `ruff format` | ✓ 67 archivos formateados consistentemente |
| **`test_features_no_leak`** (P0) | ✓ 4 puntos de truncamiento, tol 1e-9 |
| **`test_env_no_future`** (P0) | ✓ con `RecordingMarketEnv` |
| **`test_U_unitary`** (P0) | ✓ M ∈ {2,3,4,6,8}, tol 1e-12 |
| **`test_probability_conservation_for_all_k`** (P0) | ✓ k ∈ [0,10], tol 1e-10 |
| **`test_c_vs_d_interface_equivalence`** (P0) | ✓ mismo protocolo `LocalModule` |
| `test_coin_squared_is_identity` | ✓ M ∈ {2,3,4,6,8}, tol 1e-12 |
| `test_shift_is_permutation` | ✓ M ∈ {2,3,4,6,8} |
| `test_quantum_vs_classical_distributions_differ` | ✓ L1 diff > 0.01 para k=5 |
| `test_dtqw_latency_M8_k6` | ✓ < 50 ms por llamada |
| **Integración A/B/C/D end-to-end** | ✓ entrenamiento + evaluación + métricas finitas |
| **Smoke test con datos reales del S&P 500** | ✓ 6 tickers, 4 modelos, 256 steps cada uno |

## Estructura final del repositorio

```
quantum_ai/
├── README.md, pyproject.toml, uv.lock, CHANGELOG.md, tasks.ps1, .gitignore
├── configs/
│   ├── data/nivel1.yaml
│   ├── env/default.yaml
│   ├── agent/ppo.yaml
│   ├── graph/default.yaml
│   ├── quantum/default.yaml
│   └── experiment/model_{a,b,c,d}.yaml
├── data/raw/
│   └── <universe>/*.parquet + manifest.json
├── outputs/
│   ├── mlruns/    (file-store local)
│   ├── checkpoints/
│   ├── figures/
│   └── tables/
├── src/
│   ├── data/      (download, cleaning, features, splits, sector_map)
│   ├── env/       (market_env, reward, state_builder, episode_sampler)
│   ├── graph/     (affinity, sparsify, graph_builder, subgraph_selector,
│   │              relational_features, classical_walk)
│   ├── agents/    (policy_network, masked_policy, classical_agent,
│   │              hybrid_agent, rollout_buffer)
│   ├── quantum/   (encoding, dtqw, measurement, classical_walker,
│   │              quantum_walker)
│   ├── training/  (trainer, evaluate, mlflow_logger, seed_utils)
│   ├── utils/     (paths, config, logging, hashing)
│   └── main.py    (typer CLI)
├── tests/
│   ├── conftest.py
│   ├── unit/      (24 archivos, 171 tests)
│   └── integration/ (1 archivo, 4 tests)
└── scripts/
    ├── smoke_data_pipeline.py
    └── smoke_train_all.py
```

## Semana 8 — Campaña experimental · COMPLETADA (2026-05-11)

### Entregables
- `src/training/evaluate.py` enriquecido:
  - `compute_promising_matrix` (Sec. 8.17: top 20% del score `g_i(h) = mean / (std+ε)`).
  - `episodes_to_convergence` (heurística ventana móvil con tolerancia relativa).
  - `EvalResult` ampliado: `mean_latency_ms`, `topm_hit_rate`, `candidate_hit_rate`, `time_to_first_promising`.
- `scripts/run_campaign.py`: orquesta 4 modelos × N semillas × train/eval con MLflow + CSV consolidado.
- `scripts/aggregate_results.py`: per-model bootstrap CI95 + **paired bootstrap D vs C** (Sec. 8.18) con `P(D > C)`.
- `notebooks/04_results_comparison.ipynb`: boxplots financieros/exploración/latencia + tabla de comparación pareada.
- `tasks.ps1`: nuevos comandos `campaign` y `aggregate`.

### Tests adicionales
- `tests/unit/test_evaluate_metrics.py` (10 tests): promising matrix, convergencia, integración con `evaluate_agent`.
- `tests/unit/test_aggregate_results.py` (5 tests): estadísticas por modelo, paired bootstrap, generación de markdown.

### Mini campaña ejecutada con éxito
- Universo `smoke_train` (6 tickers · 374 días).
- 4 modelos × 3 semillas × 1024 steps = **12 runs** registrados en MLflow.
- CSV agregado: `outputs/tables/mini.csv`, `mini_summary.csv`, `mini_paired_c_vs_d.csv`, `mini_report.md`.
- Pipeline experimental validado (los resultados absolutos son irrelevantes a este volumen).

### Criterios de aceptación
- [x] **190 tests verdes** (171 unit ya existentes + 15 nuevos + 4 integration) en < 9 segundos.
- [x] `ruff check src tests scripts` — All checks passed.
- [x] Pipeline end-to-end validado con `scripts/run_campaign.py` + `scripts/aggregate_results.py`.
- [x] Bootstrap pareado D vs C produce IC95% y `P(D>C)` para todas las métricas.

### Para la campaña final de la tesis
```powershell
# Universo Nivel 1 (15 tickers, datos en data/raw/nivel1/)
.\tasks.ps1 download-data
.\tasks.ps1 campaign -- --universe nivel1 --seeds 42 123 456 789 1024 --steps 50000 --campaign-id campaign_1
.\tasks.ps1 aggregate -- --campaign outputs\tables\campaign_1.csv
```

---

## Semana 9 — Ablaciones obligatorias (Sec. 8.20) · COMPLETADA (2026-05-11)

### Entregables

**Backend cuántico con ruido**:
- `src/quantum/noise.py`: `NoiseSpec` (depolarizing, dephasing) + `apply_noise_layer`.
- `src/quantum/pennylane_backend.py`: `apply_dtqw_pennylane` con padding a potencia de 2 más cercana. Cuando `noise.is_active()`, usa `default.mixed` con `qml.DepolarizingChannel`/`qml.PhaseDamping` después de cada paso.
- `QuantumWalker` ampliado con `backend` (`"matrix"` | `"pennylane"`) y `noise`. Cuando `noise.is_active()` se fuerza el backend PennyLane.
- Integración en `src/main.py` y `scripts/run_campaign.py` para propagar el ruido desde el YAML.

**Configs YAML de ablación**:
- `ablation_init_uniform.yaml`, `ablation_init_seed_centered.yaml`.
- `ablation_noise_ideal.yaml`, `ablation_noise_depolarizing.yaml`.

**Scripts de orquestación**:
- `scripts/run_ablation.py`: comando con sub-acciones `init`, `noise`, `M`, `k`, `m`. Cada ablación produce un CSV con columnas `ablation_kind` y `ablation_value`.
- `scripts/generate_figures.py`: emite PNGs en `outputs/figures/`: boxplots financieros/exploración/latencia por modelo + curvas de barrido para ablaciones.

**Tests** (23 nuevos, total **213 verdes** en < 9 s):
- `tests/unit/test_pennylane_backend.py` (20 tests):
  - Helpers internos (next power of two, padding).
  - **Consistencia matrix ≈ pennylane sin ruido**: 5 combinaciones (M, k), tol 1e-6.
  - Conservación de probabilidad bajo ruido.
  - **Saturación hacia uniforme con `depolarizing=0.3`** (varianza P_k disminuye).
  - QuantumWalker matrix vs pennylane intercambiables sin ruido.
- `tests/unit/test_run_ablation.py` (3 tests): metadata de ablación en CSV.

### Validaciones cumplidas
- [x] `apply_dtqw_pennylane` ≈ `apply_dtqw` matricial (sin ruido) a tol 1e-6.
- [x] **213 tests verdes** en < 9 s.
- [x] `ruff check src tests scripts` — All checks passed.
- [x] Mini ablación `k ∈ {2, 4, 6}` con 2 seeds (6 runs) completada exitosamente.
- [x] Mini ablación `noise ∈ {0.0, 0.05, 0.15}` con 1 seed (3 runs) completada: latencia sube de 0.95 ms (matrix) a ~7 ms (pennylane con ruido), demostrando el switch correcto de backend.
- [x] **5 figuras PNG generadas**: `mini_financial.png`, `mini_exploration.png`, `mini_latency.png`, `ablation_k_mini.png`, `ablation_noise_mini.png`.

### Para reproducir las ablaciones finales de la tesis
```powershell
# Inicialización: uniform vs seed_centered (5 seeds c/u = 10 runs)
.\tasks.ps1 ablation -- init --universe nivel1 --seeds 42 123 456 789 1024 --steps 50000

# Ruido: ideal vs depolarizing (3 niveles x 5 seeds = 15 runs)
.\tasks.ps1 ablation -- noise --universe nivel1 --seeds 42 123 456 789 1024 `
    --depolarizing 0.0 0.05 0.10 --steps 50000

# Barrido M ∈ {4,6,8} con 5 seeds (15 runs)
.\tasks.ps1 ablation -- M --universe nivel1 --seeds 42 123 456 789 1024 --steps 50000 --values 4 6 8

# Barrido k ∈ {2..6} con 5 seeds (25 runs)
.\tasks.ps1 ablation -- k --universe nivel1 --seeds 42 123 456 789 1024 --steps 50000 --values 2 3 4 5 6

# Figuras finales
.\tasks.ps1 figures -- --campaign outputs\tables\campaign_1.csv `
    --ablation outputs\tables\ablation_init.csv outputs\tables\ablation_noise.csv `
    outputs\tables\ablation_M.csv outputs\tables\ablation_k.csv
```

---

## Estado del proyecto

| Métrica | Valor |
|---|---|
| **Tests verdes** | 213 / 213 (< 9 s) |
| **Tests P0 críticos** | 9 / 9 |
| **Commits** | 4 |
| **Archivos en git** | ≥ 100 |
| **Semanas completadas** | 9 / 9 (100% del plan) |
| **Universo Nivel 1** | 15 tickers S&P 500 listos |
| **MLflow experiments** | model_a/b/c/d + ablation_* configurables |

El proyecto está **completo según el plan original** y listo para la campaña experimental final que sustentará la defensa de la tesis.
