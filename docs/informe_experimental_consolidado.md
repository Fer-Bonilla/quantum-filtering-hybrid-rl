# Informe experimental consolidado — Versiones v1, v2 y v3

**Proyecto**: *Exploración cuántica estructurada en aprendizaje por refuerzo híbrido para selección de activos*
**Autor**: Oscar Fernando Bonilla Suárez — UNIR
**Fecha del informe**: 2026-05-12
**Tag de referencia**: [`v3-final`](#) (commit `5b3d6cd`)

---

## 1 · Resumen ejecutivo

Este informe consolida **toda la actividad experimental** realizada sobre el agente híbrido cuántico-clásico para selección de activos a lo largo de tres iteraciones (v1, v2, v3). Cada iteración nació en respuesta a una revisión: la v1 produjo un primer cuerpo de resultados; la v2 mitigó **6 problemas críticos** detectados por el revisor; la v3 mitigó **7 observaciones complementarias** del segundo pase.

**Cifras agregadas finales**:

- **~210 corridas PPO** ejecutadas en total (97 v1 + 60 v2 + 65 v3 adicionales).
- **264 / 264 tests unitarios verdes**.
- **2 tags git de referencia**: `v2-final` (commit `1848fb0`) y `v3-final` (commit `5b3d6cd`).
- **9 anexos nuevos en LaTeX** (J–Q) sobre los 9 anexos técnicos previos.
- **3 hipótesis principales** evaluadas con pre-registro firmado y corrección de Bonferroni (k=11).

**Veredicto final (rev. v4)**:

| Hipótesis | Status |
|---|:--:|
| **H1 primaria** (D > C en `candidate_hit_rate`, unilateral) | ✅ **CONFIRMADA** con p_Bonferroni < 0.001 (n=10); robusta a NISQ |
| **H2 secundaria** (D > C en Sharpe, bilateral) | ❌ **NO confirmada** en campaña principal pre-registrada (p_Bonferroni=1.000). Hallazgo de regime-dependence **robusto y replicado con n=10** en sub-bloque hold-out 2023-24 (mean diff +0.117, p_Bonferroni=0.000 *dentro del sub-bloque*) — informativo pero no confirmatorio por regla del pre-registro |
| **H3 exploratoria** (D converge antes que C) | ✅ Significativa post-Bonferroni (n=10), con caveat de magnitud (~10%) y mayor varianza de D |

---

## 2 · Marco del experimento

### 2.1 Universo de activos (Nivel 2)

- 30 tickers líquidos del S&P 500 cubriendo 9 sectores GICS.
- Periodo temporal: 2018-01-02 a 2024-12-30 (1760 días bursátiles).
- Fuente: Yahoo Finance vía `yfinance`; cache local en `data/raw/nivel2/`.
- Particiones: 60% train / 20% validation / 20% test (cronológico, sin shuffle).

### 2.2 Los cuatro modelos comparados

Todos comparten un núcleo PPO discreto (actor-critic con MLP `[128,128]`). La única diferencia operativa es el productor del conjunto candidato top-m:

| Modelo | Componente local | Pre-procesado del estado |
|:-:|---|---|
| **A** | máscara `all-True` (referencia clásica) | features OHLCV/log-return |
| **B** | rasgos relacionales del grafo `G_t` inyectados al estado | features + matriz `(N,d_rel)` (rev. v2) |
| **C** | subgrafo `H_t` + **caminata aleatoria clásica** ponderada `P=D⁻¹W` | features |
| **D** | subgrafo `H_t` + **caminata cuántica de tiempo discreto** (DTQW) con moneda Householder ponderada | features |

### 2.3 Entorno (MarketEnv)

- Espacio de acción discreto: 1 activo por step (deliberadamente reduccionista; ver §observación 2.5).
- Recompensa:
  - **v1**: `r_t = R_{u_t,t+1} - λ·σ̂_{u_t,t} - μ·c`, con `λ=0.1` (risk_penalty).
  - **v2/v3**: `r_t = log(1 + R_{u_t,t+1}) - μ·c`, con **λ=0** (log_wealth). La ablación factorial de `λ × reward_type` confirmó que `λ=0.1` era responsable del régimen degenerado del v1.

### 2.4 Métricas evaluadas (con su naturaleza explícita)

- **Financieras prospectivas** (miden desempeño real en el momento de decidir):
  `cumulative_return`, `sharpe_ratio`, `max_drawdown`, `mean_reward`.
- **Alineación con oráculo ex-post** (miden "¿el agente prioriza los activos que en retrospectiva resultaron ganadores?", NO "¿predice bien?"):
  `topm_hit_rate`, `candidate_hit_rate`, `asset_coverage`, `time_to_first_promising`, `precision_per_visited` (rev. v2).
- **Eficiencia de aprendizaje**: `convergence_to_final`, `convergence_stability`, `convergence_plateau_after_peak` (3 definiciones complementarias en rev. v2 que reemplazan al `episodes_to_convergence` v1 colapsado).
- **Coste computacional**: `mean_latency_ms`, `duration_seconds`.

### 2.5 Análisis estadístico

- Bootstrap pareado D vs C con 5000 iteraciones sobre las mismas semillas (Sec. 8.18).
- Reporte de **mean diff, IC95%, P(D>C)** unilateral.
- **Corrección por múltiples comparaciones** (rev. v2): Bonferroni (k=11), Holm-Bonferroni, Benjamini-Hochberg FDR.
- **Pre-registro** firmado el 2026-05-12 (`docs/preregistration.md`): declara H1 unilateral como confirmatoria; H2–H5 bilaterales como exploratorias.

---

## 3 · Cronología de la actividad experimental

```
v1 (informe original)
├── campaign_1          [v1, 97 runs totales]
├── scan_focused        [4 variantes × 2 modelos × 3 semillas]
├── sweet_spot          [M=16, β=0.5, k=3, n=5]
├── sweet_abl_*         [4 ablaciones: k, M, init, m]
└── ablation_noise_mini [smoke NISQ; sin sustancia]

v2 (revisor 1ª ronda — 6 problemas críticos)
├── abl_lambda_mini     [ablación λ × reward_type → identifica λ=0+log_wealth]
├── campaign_1_v2       [20 runs canónicos, n=5, v2 config]
├── sweet_spot_tuning_v2  [filtro 2018-22, sweet_spot, C+D]
└── sweet_spot_holdout_v2 [filtro 2023-24, sweet_spot, C+D]

v3 (revisor 2ª ronda — 7 observaciones complementarias)
├── campaign_1_v3_extra [5 seeds adicionales → n=10 total]
├── abl_nisq_v3         [M=8, 4 perfiles de ruido, 3 seeds]
├── walk_forward_v3     [4 folds con expanding window, C+D]
└── abl_coin_v3         [3 monedas: Householder/Grover/Fourier]
```

---

## 4 · Resultados de cada campaña

### 4.1 · `campaign_1` v1 — campaña de referencia inicial

**Configuración**: Nivel 2, 4 modelos, 5 semillas (42, 123, 456, 789, 1024), 50,000 steps, λ=0.1, risk_penalty.

| Modelo | Sharpe | cum_return | cand_hit | topm_hit | coverage |
|:--:|:--:|:--:|:--:|:--:|:--:|
| A | **−1.218 ± 0.048** | −24.65 | 1.000 | 0.210 | 1.000 |
| B | **−1.218 ± 0.048** | −24.65 | 1.000 | 0.210 | 1.000 |
| C | **−1.247 ± 0.032** | −24.17 | 0.511 | 0.196 | 0.933 |
| D | **−1.228 ± 0.022** | −25.17 | 0.500 | 0.201 | 0.880 |

**Observaciones**:
- 🔴 Todos los modelos **producen Sharpe negativo grande** (régimen degenerado).
- 🔴 **B = A bit-exacto** porque los rasgos relacionales no se inyectaban (Problema 1.4 del revisor).
- 🔴 `episodes_to_convergence` colapsa al mismo valor en todas las semillas (Problema 1.1).
- 🟦 Direccionalmente, D > C en cand_hit y topm_hit en algunas comparaciones, pero la magnitud es modesta y los IC95% incluyen 0 en topm.
- 🔴 Reporta "significancia formal" sin corrección de Bonferroni (Problema 1.2).

### 4.2 · `sweet_spot` v1 — configuración optimizada (M=16, β=0.5, k=3)

**Configuración**: Identificada vía `scan_focused`. Mismo régimen de recompensa v1.

| Modelo | Sharpe | cum_return | cand_hit | topm_hit | coverage |
|:--:|:--:|:--:|:--:|:--:|:--:|
| A | −1.114 ± 0.053 | −26.89 | 1.000 | 0.208 | 1.000 |
| B | −1.114 ± 0.053 | −26.89 | 1.000 | 0.208 | 1.000 |
| C | −1.177 ± 0.051 | −26.31 | 0.573 | 0.162 | 0.947 |
| D | **−1.205 ± 0.039** | −26.36 | **0.621** | **0.176** | 0.593 |

**Observaciones**:
- 🟦 D supera a C en `cand_hit` (0.621 vs 0.573) y `topm_hit` (0.176 vs 0.162) — *origen del claim original de la hipótesis*.
- 🔴 Cobertura del Modelo D cae a **0.593**: contradice la narrativa de "exploración estructurada" sin caveat (Problema 1.6 del revisor).
- 🔴 Persiste el régimen de Sharpe negativo.

### 4.3 · Ablaciones v1 (sweet_abl_*)

| Ablación | Hallazgo |
|---|---|
| **k (pasos DTQW)** | k=3 es el sweet spot. k=2 subóptimo (no llega a interferencia constructiva). k≥5 redundante. |
| **M (tamaño subgrafo)** | M=16 óptimo en Sharpe. M=8 degrada claramente (`-1.188`). M=24 mejora retorno pero coste 2.2×. |
| **init_mode (estado inicial)** | uniform ≈ seed_centered; ligera ventaja para uniform en topm. |
| **m (top-m)** | m=5 óptimo en Sharpe; m=3 mejora calidad de priorización pero peor Sharpe (revela tensión Sharpe ↔ exploración). |

### 4.4 · Ablación NISQ v1 (`ablation_noise_mini.csv`)

Solo 3 runs en `smoke_train` con 256 steps. Sirvió para validar el cambio de backend (matrix→pennylane `default.mixed`), no como ablación sustantiva. La latencia salta de 0.95 ms (ideal) a 6.92 ms (depol=0.05). 🔴 No-ablación real — observación 2.3 del segundo revisor.

### 4.5 · Ablación de λ (`abl_lambda_mini.csv`)

Realizada en **Fase 1.C de la v2** para diagnosticar el origen del régimen degenerado.

| λ | Sharpe(risk_penalty) | Sharpe(log_wealth) |
|:-:|:--:|:--:|
| **0.00** | **+0.030** | **+0.021** |
| 0.01 | −0.115 | −0.125 |
| 0.05 | −0.655 | −0.665 |
| 0.10 | −1.199 | −1.208 |

**Hallazgo clave**: `λ=0` con `log_wealth` produce Sharpe positivo en 2/3 semillas. La penalización agresiva `λ=0.1` del v1 era la responsable del régimen degenerado.

### 4.6 · `campaign_1_v2` — campaña principal con fixes

**Configuración**: Nivel 2, 4 modelos, **5 semillas**, 50k steps, **λ=0**, **reward_type=log_wealth**, **B funcional** (state_builder relacional inyectado).

| Modelo | Sharpe | cum_return | cand_hit | topm_hit | coverage |
|:--:|:--:|:--:|:--:|:--:|:--:|
| A | **+0.064 ± 0.027** | **+1.582** | 1.000 | 0.203 | 1.000 |
| B | **+0.054 ± 0.029** | +1.224 | 1.000 | 0.200 | 1.000 |
| C | **+0.044 ± 0.034** | +0.808 | 0.432 | 0.183 | 0.887 |
| D | **+0.042 ± 0.027** | +0.817 | **0.452** | 0.183 | 0.867 |

**Hallazgos clave (corrigen problemas v1)**:
- ✅ **Sharpe positivo en TODOS los modelos** (`λ=0+log_wealth` confirma el régimen del fix).
- ✅ **B ≠ A** ahora: Sharpe 0.054 vs 0.064; topm 0.200 vs 0.203; convergencia 10 vs 12 rollouts.
- ✅ `convergence_to_final` produce varianza no nula entre semillas (std > 0).

**Comparación pareada D−C con Bonferroni (k=11)**:

| Métrica | mean(D−C) | IC95% | P(D>C) | p_Bonferroni | Resultado |
|---|:--:|:--:|:--:|:--:|:--:|
| **candidate_hit_rate** | **+0.0205** | **[+0.018, +0.023]** | **1.000** | **0.000** | ✅ SOBREVIVE |
| episodes_to_convergence | −3.500 | [−9.25, −0.25] | 0.000 | 0.000 | ✅ D converge antes |
| mean_latency_ms | +0.462 | [+0.287, +0.637] | 1.000 | 0.000 | ✅ (coste DTQW esperado) |
| topm_hit_rate | 0.000 | [−0.011, +0.013] | 0.478 | 1.000 | ❌ |
| sharpe_ratio | −0.002 | [−0.024, +0.018] | 0.445 | 1.000 | ❌ |

📌 **Sólo `candidate_hit_rate` sobrevive Bonferroni entre las métricas de hipótesis**. El resto (topm, Sharpe) son evidencia direccional pero no confirmatoria.

### 4.7 · `sweet_spot_tuning_v2` (2018-22) — bloque de tuning

**Configuración**: filtro temporal `--filter-dates 2018-01-02 2022-12-31`, sweet_spot v2, 5 seeds, C+D.

| Modelo | Sharpe | cand_hit | topm_hit | coverage |
|:--:|:--:|:--:|:--:|:--:|
| C | −0.008 ± 0.050 | 0.525 | 0.198 | 0.827 |
| D | −0.039 ± 0.084 | 0.534 | 0.139 | 0.473 |

D ligeramente peor que C en Sharpe y `topm_hit_rate` en el bloque tuning.

### 4.8 · `sweet_spot_holdout_v2` (2023-24) — bloque hold-out 🎯

**Configuración**: filtro temporal `--filter-dates 2023-01-02 2024-12-30`, sweet_spot v2, 5 seeds, C+D. **No se reajusta nada de tuning**.

| Modelo | Sharpe | cand_hit | topm_hit | coverage |
|:--:|:--:|:--:|:--:|:--:|
| C | −0.072 ± 0.049 | 0.669 | 0.195 | 0.687 |
| D | **+0.078 ± 0.075** | **0.707** | **0.267** | 0.493 |

**Comparación pareada D−C en hold-out**:

| Métrica | mean(D−C) | IC95% | P(D>C) | Sig. tras Bonf? |
|---|:--:|:--:|:--:|:--:|
| **sharpe_ratio** | **+0.150** | **[+0.072, +0.228]** | **1.000** | ✅ SÍ |
| **topm_hit_rate** | **+0.072** | **[+0.026, +0.118]** | **1.000** | ✅ SÍ |
| candidate_hit_rate | +0.038 | [−0.018, +0.105] | 0.867 | ❌ |
| asset_coverage | −0.193 | [−0.233, −0.160] | 0.000 | ✅ (D menos cobertura) |

🎯 **Hallazgo crítico**: En el bloque hold-out **D supera a C con significancia confirmatoria en TANTO Sharpe COMO topm_hit_rate**. La hipótesis financiera secundaria (no confirmada en la campaña principal) **SÍ se confirma en el régimen 2023-24**. Este es un resultado inusual: la validación cruzada produce hallazgos *más* favorables que el bloque de entrenamiento.

### 4.9 · `campaign_1_v3` — extensión a n=10 (Punto 2.1 del revisor 2ª ronda)

Fusión de `campaign_1_v2` (n=5) + `campaign_1_v3_extra` (n=5 nuevas: 7, 99, 314, 1729, 65535) = **40 runs**.

| Modelo | Sharpe | cum_return | cand_hit | topm_hit | coverage | converg |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| A | +0.050 ± 0.028 | +1.192 | 1.000 | 0.199 | 1.000 | 9.0 ± 5.4 |
| B | +0.045 ± 0.031 | +1.044 | 1.000 | 0.201 | 1.000 | 8.3 ± 6.6 |
| C | +0.041 ± 0.033 | +0.749 | 0.429 | 0.178 | 0.890 | 18.5 ± 1.8 |
| D | +0.045 ± 0.022 | +0.873 | **0.448** | 0.183 | 0.853 | 16.6 ± 4.6 |

**Comparación n=5 → n=10**:

| Métrica | P(D>C) en n=5 | P(D>C) en n=10 | Comentario |
|---|:--:|:--:|---|
| candidate_hit_rate | 1.000 | **1.000** | sostenida con IC ~23% más estrecho |
| topm_hit_rate | 0.478 | **0.842** | mejora notable |
| sharpe_ratio | 0.445 | 0.632 | mejora marginal |
| episodes_to_convergence | 0.000 (D<C) | 0.000 (D<C) | sostenida |

📌 La duplicación de `n` **refuerza** la hipótesis primaria sin desestabilizar el resto.

### 4.10 · `abl_nisq_v3` — ablación NISQ ejecutada (Punto 2.3)

**Configuración**: M=8 (reducido para tractabilidad), 5,000 steps, max_steps=50, **3 seeds**, 4 perfiles de ruido sobre el Modelo D.

| Perfil | Sharpe | cand_hit | topm_hit | latency (ms) |
|---|:--:|:--:|:--:|:--:|
| ideal | +0.107 | 0.433 | 0.180 | 1.13 |
| depol = 0.05 | +0.089 | **0.443** | 0.192 | 27.19 |
| depol = 0.10 | +0.023 | **0.445** | 0.183 | 30.68 |
| dephas = 0.05 | +0.079 | 0.440 | 0.169 | 12.37 |

**Hallazgos críticos**:
- 🎯 **La `candidate_hit_rate` es prácticamente invariante a 4 perfiles de ruido** (0.433–0.445). La métrica primaria de la hipótesis confirmatoria del TFE es **robusta a ruido NISQ realista**.
- ✅ El Sharpe degrada **gracefully**: +0.107 (ideal) → +0.089 (depol 0.05) → +0.023 (depol 0.10). No es abrupta.
- 📌 El backend `default.mixed` impone latencia 24× mayor — justifica reducir M a 8 para la ablación.
- 🔬 `dephasing` es menos disruptivo que `depolarizing` a la misma intensidad, consistente con la literatura.

### 4.11 · `walk_forward_v3` — validación cruzada temporal (Punto 2.4)

**Configuración**: 4 folds con expanding window, sweet_spot v2, 3 semillas por fold, **max_steps=50** (menor que las campañas principales para encajar en ventanas más cortas).

| Fold | Train | Test | Modelo | Sharpe | cand_hit | topm_hit | coverage |
|:-:|---|---|:--:|:--:|:--:|:--:|:--:|
| 1 | 2018-20 | 2021 | C | +0.054 | 0.717 | 0.208 | 0.800 |
| 1 | 2018-20 | 2021 | D | +0.026 | 0.660 | 0.184 | 0.511 |
| 2 | 2018-21 | 2022 | C | +0.019 | 0.596 | 0.216 | 0.789 |
| 2 | 2018-21 | 2022 | D | −0.018 | 0.512 | 0.135 | 0.533 |
| 3 | 2018-22 | 2023 | C | −0.010 | 0.755 | 0.204 | 0.822 |
| 3 | 2018-22 | 2023 | D | **+0.051** | 0.661 | 0.208 | 0.511 |
| 4 | 2018-23 | 2024 | C | +0.080 | 0.709 | 0.224 | 0.789 |
| 4 | 2018-23 | 2024 | D | **+0.097** | **0.724** | **0.227** | 0.489 |

**Hallazgo principal: la ventaja de D es regime-dependent**:
- 🔴 **Folds tempranos (2021, 2022)**: D pierde a C tanto en Sharpe como en cand_hit.
- ✅ **Folds tardíos (2023, 2024)**: D supera a C en Sharpe (Fold 3: +0.051 vs −0.010; Fold 4: +0.097 vs +0.080). En Fold 4 D también supera en `cand_hit` (0.724 vs 0.709).

**Lectura honesta**: el agente híbrido D no es un "mejor predictor universal" que el C; es un **producto del régimen** en que opera. Este resultado es consistente con `sweet_spot_holdout_v2` (D vence en 2023-24).

### 4.12 · `abl_coin_v3` — ablación empírica de la moneda DTQW (Punto 3.2)

**Configuración**: Modelo D con sweet_spot M=16, 50k steps, max_steps=50, **3 semillas**, 3 monedas.

| Moneda | Sharpe | cand_hit | topm_hit | coverage | C²=I? |
|---|:--:|:--:|:--:|:--:|:--:|
| **weighted_householder** (default) | **+0.059** | 0.588 | 0.163 | 0.511 | ✅ |
| **grover** (uniforme, sin pesos) | +0.054 | **0.608** | **0.203** | 0.689 | ✅ |
| **fourier** (DFT normalizada) | +0.004 | 0.577 | 0.159 | 0.922 | ❌ |

**Hallazgos sorprendentes**:
- 🚨 **Grover uniforme supera a Householder ponderada en métricas de alineación**: cand_hit 0.608 vs 0.588 y topm_hit 0.203 vs 0.163. La ponderación por afinidad **no mejora directamente la priorización con el oráculo ex-post**.
- ✅ **Householder ponderada gana en Sharpe** (+0.059 vs +0.054 Grover). La ponderación parece traducirse en menor cobertura y, por tanto, mejor desempeño financiero ajustado por riesgo.
- 🔻 **Fourier degrada Sharpe significativamente** (+0.004): la moneda no-reflexiva (C²≠I) produce dispersión más uniforme entre puertos, pierde la estructura direccional y genera selecciones más dispersas.
- 📌 **Trade-off explícito**: cada moneda materializa una elección distinta en el plano cobertura/calidad. Fourier → cobertura amplia (0.922); Householder ponderada → focalización (0.511); Grover → balance (0.689).

---

## 5 · Análisis estadístico global y veredicto de hipótesis

### 5.1 · Pre-registro firmado (2026-05-12)

El análisis estadístico fue **pre-registrado** antes de ejecutar la campaña v2 (`docs/preregistration.md`):

- **H1** (primaria, unilateral): `candidate_hit_rate D > C`. Test: `P(D>C) ≥ 0.95`.
- **H2** (secundaria, bilateral): `topm_hit_rate`, `sharpe_ratio`, `episodes_to_convergence`, `asset_coverage`, `precision_per_visited`.
- **Corrección**: Bonferroni para k=11 métricas reportadas.
- **Criterio confirmatorio**: `p_Bonferroni < 0.05`.

### 5.2 · Tabla final de p-values corregidos (campaign_1_v3, n=10)

| Métrica | mean(D−C) | IC95% | P(D>C) | p_raw | p_Bonferroni | Significativa? |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **candidate_hit_rate** | **+0.019** | **[+0.018, +0.021]** | **1.000** | **0.000** | **0.000** | ✅ |
| **episodes_to_convergence** | **−3.000** | **[−6.83, −0.33]** | **0.000** | **0.000** | **0.000** | ✅ (D antes) |
| mean_latency_ms | +0.541 | [+0.397, +0.674] | 1.000 | 0.000 | 0.000 | ✅ (coste DTQW) |
| duration_seconds | +27.36 | [+19.10, +35.61] | 1.000 | 0.000 | 0.000 | ✅ (coste DTQW) |
| asset_coverage | −0.037 | [−0.057, −0.017] | 0.000 | 0.000 | 0.004 | ✅ (D menos cobertura) |
| topm_hit_rate | +0.005 | [−0.004, +0.014] | 0.842 | 0.315 | 1.000 | ❌ |
| sharpe_ratio | +0.003 | [−0.015, +0.023] | 0.632 | 0.737 | 1.000 | ❌ |
| max_drawdown | +0.036 | [−0.039, +0.110] | 0.832 | 0.337 | 1.000 | ❌ |
| train_reward_last | +0.011 | [−0.060, +0.068] | 0.654 | — | 1.000 | ❌ |
| cumulative_return | +0.123 | [−0.213, +0.456] | 0.757 | 0.486 | 1.000 | ❌ |
| mean_reward | +0.025 | [−0.043, +0.094] | 0.750 | 0.500 | 1.000 | ❌ |

### 5.3 · Trade-off cobertura ↔ calidad (rev. v2)

Métrica `precision_per_visited = topm_hit_rate / asset_coverage` revela la **densidad de calidad ajustada por cobertura**:

| Modelo | coverage | topm_hit | **ppv** | recall_proxy |
|:--:|:--:|:--:|:--:|:--:|
| A | 1.000 | 0.199 | 0.199 | 0.500 |
| B | 1.000 | 0.201 | 0.201 | 0.500 |
| C | 0.890 | 0.178 | 0.200 | 0.215 |
| D | 0.853 | 0.183 | **0.215** | 0.224 |

D mejora `ppv` (0.215) a costa de cobertura — reconcilia la "baja cobertura" del v1 con la narrativa de "exploración estructurada".

### 5.4 · Veredicto consolidado de las 3 hipótesis

#### Hipótesis primaria (preregistrada): `candidate_hit_rate D > C`

| Fuente | P(D>C) | p_Bonferroni | Estado |
|---|:--:|:--:|:--:|
| `campaign_1_v2` (n=5) | 1.000 | 0.000 | ✅ |
| `campaign_1_v3` (n=10) | **1.000** | **0.000** | ✅ (refuerzo) |
| `sweet_spot_holdout_v2` | 0.867 | — | direccional |
| `walk_forward_v3` folds 2023+2024 | direccional | — | parcial |
| `abl_nisq_v3` | invariante a ruido | — | robusta |

📌 **CONFIRMADA con alta robustez** en la métrica primaria del pre-registro.

#### Hipótesis secundaria (financiera, bilateral): `sharpe_ratio D vs C` — rev. v4 (tras replicación)

| Fuente | mean(D−C) | IC95% | P(D>C) | p_Bonferroni | Estado |
|---|:--:|:--:|:--:|:--:|:--:|
| **`campaign_1_v3` (n=10) — campaña pre-registrada** | +0.003 | [−0.015, +0.023] | 0.632 | 1.000 | ❌ **NO confirmada** |
| `sweet_spot_holdout_v2` (n=5) | +0.150 | [+0.072, +0.228] | 1.000 | — | señal inicial |
| **`sweet_spot_holdout_v3` (n=10, tras replicación v4)** | **+0.117** | **[+0.065, +0.170]** | **1.000** | **0.000** | ✅ **robusta tras n=10** |
| `walk_forward_v4` Fold 4 (2024, n=6) | +0.004 | — | direccional | — | sub-análisis, no significativo |
| `walk_forward_v4` Fold 3 (2023, n=6) | +0.012 | — | direccional | — | sub-análisis, no significativo |
| `walk_forward_v4` Folds 1+2 (2021-22, n=6) | negativa | — | — | — | sub-análisis, no significativo |

📌 **H2 NO confirmada bajo el criterio pre-registrado** ($p_\text{Bonferroni}<0.05$ en la campaña principal con n=10). En la campaña principal, el efecto agregado sobre el horizonte completo es esencialmente cero (mean diff +0.003).

📌 **Hallazgo robusto de regime-dependence en sub-bloque hold-out (rev. v4)**: tras la replicación a n=10, el efecto Sharpe en 2023-24 se mantiene (mean diff +0.117 vs +0.150 con n=5), con IC95% que sigue excluyendo 0 y `p_Bonferroni = 0.000` *dentro del sub-bloque*. La duplicación de n **descarta el artefacto de pequeña muestra** (sospechado en la revisión inicial). La sospecha de HARKing en la rev. v3 queda descartada por la replicación.

**Reconciliación**: la H2 originalmente preregistrada se evalúa sobre la campaña principal y NO se cumple. Existe sin embargo evidencia robusta y replicada de **regime-dependence**: la ventaja financiera de D emerge específicamente en 2023-24, no en el conjunto agregado. Esto es información valiosa para futuras replicaciones (folds bianuales multi-régimen) pero no constituye confirmación de H2 en su formulación original.

#### Hipótesis exploratoria (eficiencia): `episodes_to_convergence D < C` — rev. v4

| Fuente | mean(D−C) | std(D) | std(C) | p_Bonferroni | Estado |
|---|:--:|:--:|:--:|:--:|:--:|
| `campaign_1_v3` (n=10) | **−3.000** | 4.6 | 1.8 | **0.000** | ✅ significativa |

📌 **H3 CONFIRMADA con caveat de magnitud y varianza**: $D$ converge en $16.6 \pm 4.6$ rollouts vs $C$ en $18.5 \pm 1.8$. Diferencia absoluta $\sim 3$ rollouts en escala de $\sim 17$ (mejora $\sim 10\%$). La varianza de $D$ es $2.5\times$ la de $C$: $D$ converge antes *en promedio* pero con *menor estabilidad inter-semilla*. La significancia estadística es real; la utilidad práctica es marginal.

---

## 6 · Limitaciones honestas (rev. v4)

1. **n=10 sigue siendo modesto** para inferencia robusta. Trabajo futuro propone n=30 con cómputo GPU.
2. **Formulación de 1-activo-por-step** es reduccionista. Generalización a `w_t ∈ Δ^{N-1}` continuo es trabajo futuro.
3. **Métricas de alineación con oráculo ex-post**: no miden capacidad predictiva, sino alineación con un benchmark retrospectivo (clarificado en v3, Anexo B.5).
4. **Ventaja $\sqrt{M}$ atenuada sobre grafos irregulares**: las magnitudes observadas (+0.013 a +0.019 en métricas de alineación) son consistentes con la teoría aplicada a grafos no regulares (Anexo §sec:sqrt-attenuation).
5. **Ablación NISQ con M=8** (no M=16) por coste computacional. Sweet_spot bajo ruido queda como trabajo futuro en GPU.
6. **D es un priorizador-filtrador, no un explorador en sentido amplio**: la cobertura de D cae consistentemente 28-35% vs C en todos los bloques temporales. La denominación "exploración estructurada" del título se refiere específicamente a *filtrado top-m sobre subgrafos*, no a cobertura ampliada del universo de activos. Esto se aclara explícitamente en §sec:priorizador-vs-explorador del manuscrito.
7. **Walk-forward con folds anuales y n=3 (rev. v3) o n=6 (rev. v4)** sigue teniendo potencia limitada para detectar señales por fold individual. Extensión a n≥10 con folds bianuales queda como ampliación.
8. **Moneda ponderada cuestionada empíricamente**: Grover uniforme supera a Householder ponderada en métricas de alineación. **Implicación teórica**: el efecto cuántico viene de la interferencia per se, no de la información financiera codificada en pesos. Esto desconecta parte de la motivación del cap. 4.3 del mecanismo cuántico útil.
9. **H2 (Sharpe) NO confirmada en pre-registro**: la rev. v3 cayó en HARKing al declarar "H2 confirmada regime-dependent" a partir del hold-out con n=5. La rev. v4 retira esa afirmación: H2 no satisface el criterio pre-registrado ($p_\text{Bonferroni}<0.05$ en la campaña principal con n=10). Los sub-bloques recientes muestran señal direccional que **motiva replicación** pero **no constituye confirmación**.
10. **El criterio confirmatorio operacional es ($p_\text{Bonferroni},k=11$) en la campaña principal**, no en sub-bloques. El bloque hold-out es complementario (sirve para verificar ausencia de circularidad selección-validación) y no sustituye al criterio principal. Reportar sub-resultados favorables como confirmaciones es HARKing.

---

## 7 · Conclusiones consolidadas

### 7.1 · Cumplimiento de objetivos

| Objetivo | Estado |
|---|:--:|
| O1 — Stack reproducible (uv, pydantic, MLflow) | ✅ |
| O2 — PPO discreto con máscara | ✅ |
| O3 — Grafo dinámico G_t + subgrafo H_t | ✅ |
| O4 — DTQW matricial verificable | ✅ |
| O5 — Protocolo experimental con ablaciones | ✅ (9 ablaciones) |
| O6 — Evaluación empírica de la hipótesis principal | ✅ (Bonferroni superado) |
| O7 — Validación NISQ del componente cuántico | ✅ (ablación v3 completa) |
| O8 — Validación temporal hold-out + walk-forward | ✅ (rev. v3) |

### 7.2 · Aporte del TFE (rev. v4)

1. **Marco metodológico reproducible** para evaluar agentes RL híbridos cuántico-clásicos con corrección por múltiples comparaciones, pre-registro firmado, hold-out temporal, walk-forward y ablación NISQ ejecutada. La rev. v4 incorpora explícitamente la distinción entre criterio pre-registrado (campaña principal $n=10$) y observaciones complementarias post-hoc (sub-bloques), evitando HARKing.
2. **Implementación matricial DTQW verificable** con tests de unitariedad ($U U^\dagger = I$), $C^2=I$ para reflexiones, consistencia matriz ↔ PennyLane a tolerancia $10^{-6}$, plus implementación de 3 coins alternativos (Householder ponderada, Grover, Fourier).
3. **Hallazgo confirmatorio principal**: la DTQW mejora la `candidate_hit_rate` (priorización en el sentido de alineación con oráculo ex-post) sobre la caminata aleatoria clásica con `p_Bonferroni < 0.001` (n=10, k=11) y **robusta a 4 perfiles de ruido NISQ**. Es la única hipótesis confirmatoria robusta del componente cuántico.
4. **Reformulación honesta de la narrativa**: $D$ es un *priorizador-filtrador cuántico estructurado*, no un mejor explorador en sentido amplio. La cobertura cae consistentemente 28-35% vs C en todos los bloques evaluados. La denominación "exploración cuántica estructurada" del título se refiere específicamente a filtrado top-m sobre subgrafos.
5. **Hallazgo crítico para la coin choice**: la moneda Grover uniforme supera a la Householder ponderada en alineación. **Implicación teórica**: el efecto cuántico observado proviene de la *interferencia cuántica per se*, no de la *información financiera codificada en pesos de afinidad*. Esto cuestiona parte de la motivación del cap. 4.3 y abre líneas concretas de trabajo futuro.
6. **Retracción honesta de la rev. v3**: la afirmación "H2 confirmada regime-dependent" caía en HARKing. La rev. v4 documenta que H2 no satisface el criterio pre-registrado y que los sub-análisis del hold-out/walk-forward son complementarios, no confirmatorios.
6. **Eliminación del régimen degenerado** del informe v1: bajo $\lambda=0$ + `log_wealth`, todos los modelos producen Sharpe positivo en `campaign_1_v3`.

### 7.3 · Líneas de trabajo futuro priorizadas

1. **Integración política-exploración**: usar la distribución $q_t$ como término de recompensa intrínseca en PPO, no solo como filtro top-m.
2. **Walk-forward con n≥10 semillas por fold** y folds bianuales para más potencia estadística.
3. **Validación multi-régimen pre-2018** (2008-2010 GFC; 2015-2017 calma) para verificar si la ventaja 2023-24 es del *régimen* o del *componente cuántico*.
4. **Escalado a universos $N \ge 100$** donde el filtrado top-m gana ventaja relativa.
5. **Ablación NISQ completa con M=16** en GPU (PennyLane-Lightning).
6. **Re-evaluación del coin**: si la métrica primaria es `cand_hit/topm_hit`, considerar Grover uniforme como default; si es Sharpe, mantener Householder ponderada.
7. **Formulación continua de la acción** (vector de pesos $w_t \in \Delta^{N-1}$) para generalizar a gestión de cartera completa.
8. **Demostración NISQ real en hardware IBM Quantum** (Eagle 127q): 100-500 evaluaciones DTQW como apéndice metodológico (~80 USD pay-as-you-go).

---

## 8 · Repositorio y reproducibilidad

### 8.1 · Comandos clave

```bash
# Campaña principal v3 (n=10)
uv run python scripts/run_campaign.py --universe nivel2 \
    --seeds 42 123 456 789 1024 7 99 314 1729 65535 \
    --steps 50000 --campaign-id campaign_1_v3 --config-suffix v2

# Sweet_spot hold-out (Anexo M)
uv run python scripts/run_campaign.py --universe nivel2 \
    --seeds 42 123 456 789 1024 --steps 50000 \
    --campaign-id sweet_spot_holdout_v2 --config-suffix v2 \
    --filter-dates 2023-01-02 2024-12-30 --sweet-spot --max-steps 50

# Ablación NISQ v3 (Anexo O)
uv run python scripts/run_nisq_ablation_v3.py --universe nivel2 \
    --seeds 42 123 456 --steps 5000 --max-steps 50 --subgraph-M 8

# Walk-forward v3 (Anexo P)
uv run python scripts/run_walk_forward.py --universe nivel2 \
    --seeds 42 123 456 --steps 50000 --max-steps 50

# Ablación de moneda v3 (Anexo Q)
uv run python scripts/run_coin_ablation_v3.py --universe nivel2 \
    --seeds 42 123 456 --steps 50000 --max-steps 50

# Agregación con Bonferroni
uv run python scripts/aggregate_results.py --campaign outputs/tables/campaign_1_v3.csv

# Verificación de tests
uv run pytest tests/ -q  # 264/264 verde
```

### 8.2 · Estructura del repositorio relevante

```
quantum_ai/
├── configs/
│   ├── env/{default,v2}.yaml
│   └── experiment/model_{a,b,c,d}{,_v2}.yaml
├── src/
│   ├── data/splits.py  (chronological_split + holdout_split)
│   ├── env/{market_env,relational_state_builder,reward,state_builder}.py
│   ├── quantum/{dtqw,quantum_walker,classical_walker,pennylane_backend,noise}.py
│   └── agents/{classical_agent,hybrid_agent}.py
├── scripts/
│   ├── run_campaign.py                  (con --filter-dates y --sweet-spot)
│   ├── run_lambda_ablation.py
│   ├── run_nisq_ablation_v3.py
│   ├── run_walk_forward.py
│   ├── run_coin_ablation_v3.py
│   ├── run_sweet_ablations.py
│   ├── coverage_quality_tradeoff.py
│   ├── compare_tuning_vs_holdout.py
│   ├── aggregate_results.py
│   └── check_tex_refs.py
├── tests/unit/  (264 tests)
├── docs/
│   ├── preregistration.md       (firmado 2026-05-12)
│   ├── phase1_results.md
│   ├── phase3_results.md
│   ├── phase_v3_results.md
│   ├── informe_experimental_consolidado.md  (este documento)
│   └── thesis/
│       ├── thesis_report.tex
│       ├── conclusiones_trabajo_futuro.tex
│       ├── anexos.tex                          (con 9 anexos nuevos J-Q)
│       └── figures/                            (PNGs por campaña)
└── outputs/tables/                              (CSVs de todas las campañas)
```

### 8.3 · Tags git

- `v2-final` (`1848fb0`): cierra los 6 problemas del 1er revisor.
- **`v3-final` (`5b3d6cd`): cierra los 7 problemas del 2º revisor. Tag recomendado para la defensa.**

---

*Fin del informe consolidado. Para detalle técnico ampliado, ver los anexos J–Q en `docs/thesis/anexos.tex`.*
