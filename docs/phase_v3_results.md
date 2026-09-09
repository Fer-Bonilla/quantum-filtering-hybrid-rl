# Fase v3 — Mitigación de las segundas observaciones del revisor

**Fecha**: 2026-05-12
**Autor**: Oscar Fernando Bonilla Suárez
**Proyecto**: *Exploración cuántica estructurada en aprendizaje por refuerzo híbrido para selección de activos*

---

## Resumen de las 7 observaciones del revisor

| # | Observación | Tipo | Estado |
|:-:|---|:--:|:--:|
| 2.1 | n=5 → n=10 (potencia estadística) | 🔬 Cómputo | ✅ Ejecutada |
| 2.2 | Métrica post-hoc: "alineación", no "predicción" | 📝 Doc | ✅ |
| 2.3 | Ablación NISQ con M=8 | 🔬 Cómputo | 🔄 En curso |
| 2.4 | Walk-forward CV 4 folds | 🛠️+🔬 | 🔄 En curso |
| 2.5 | PPO 1-activo-por-step explicitado en alcance | 📝 Doc | ✅ |
| 3.1 | √M atenuación en grafos irregulares | 📝 Doc | ✅ |
| 3.2 | Justificación + ablación de moneda Householder | 🛠️+🔬 | 🔄 En curso |

---

## 2.1 · Extensión a n=10 semillas

`campaign_1_v3` = `campaign_1_v2` (n=5, semillas 42,123,456,789,1024) ∪ `campaign_1_v3_extra` (n=5, semillas 7,99,314,1729,65535).

### Métricas por modelo (n=10)

| Modelo | Sharpe | candidate_hit | topm_hit | coverage | convergencia |
|:--:|:--:|:--:|:--:|:--:|:--:|
| A | 0.050 ± 0.029 | 1.000 ± 0.001 | 0.199 ± 0.010 | 1.000 | 9.0 ± 5.4 |
| B | 0.045 ± 0.031 | 1.000 ± 0.001 | 0.201 ± 0.011 | 1.000 | 8.3 ± 6.6 |
| C | 0.041 ± 0.033 | 0.429 ± 0.007 | 0.178 ± 0.015 | 0.890 ± 0.023 | 18.5 ± 1.8 |
| D | 0.045 ± 0.022 | **0.449 ± 0.009** | 0.183 ± 0.009 | 0.853 ± 0.036 | 16.6 ± 4.6 |

### Comparación pareada D−C (Bonferroni k=11)

| Métrica | mean(D−C) | IC95% | P(D>C) | p_Bonf | Significativa α=0.05? |
|---|:--:|:--:|:--:|:--:|:--:|
| **candidate_hit_rate** | **+0.019** | **[+0.018, +0.021]** | **1.000** | **0.000** | ✅ SÍ |
| episodes_to_convergence | −3.0 | [−6.83, −0.33] | 0.000 | 0.000 | ✅ (D converge antes) |
| mean_latency_ms | +0.541 | [+0.397, +0.674] | 1.000 | 0.000 | ✅ (esperado: DTQW más caro) |
| duration_seconds | +27.36 | [+19.10, +35.61] | 1.000 | 0.000 | ✅ (esperado) |
| asset_coverage | −0.037 | [−0.057, −0.017] | 0.000 | 0.000 | ✅ (D menos cobertura) |
| topm_hit_rate | +0.005 | [−0.004, +0.014] | **0.842** | 0.315 | ❌ |
| sharpe_ratio | +0.003 | [−0.015, +0.023] | **0.632** | 0.737 | ❌ |

### Comparación n=5 → n=10

| Métrica | P(D>C) en n=5 | P(D>C) en n=10 | Δ |
|---|:--:|:--:|:--:|
| candidate_hit_rate | 1.000 | 1.000 | sostenida |
| topm_hit_rate | 0.478 | **0.842** | mejorada |
| sharpe_ratio | 0.445 | 0.632 | mejorada |
| episodes_to_convergence | 0.000 (D<C) | 0.000 (D<C) | sostenida |

**Conclusión**: La duplicación de $n$ contrae IC95% y refuerza:
- La hipótesis primaria (candidate_hit_rate) sigue confirmatoria.
- La evidencia direccional en topm_hit_rate y sharpe_ratio se afianza (sin alcanzar significancia).
- La eficiencia de convergencia de D queda mejor sustentada.

---

## 2.2 · Aclaración de métrica post-hoc

Anexo B.5 actualizado con párrafo dedicado:

> "Las métricas como `topm_hit_rate` y `candidate_hit_rate` no miden la capacidad predictiva del agente, sino la **alineación entre las acciones del agente y un benchmark que solo existe en retrospectiva**. Responden a: '¿la acción $u_t$ del agente coincide con los activos que, vistos en retrospectiva, resultaron en la cohorte top?' — alineación con un oráculo ex-post."

Las métricas financieras prospectivas (Sharpe, retorno acumulado, drawdown) sí miden desempeño real en el momento.

---

## 2.3 · Ablación NISQ con M=8 (PENDING)

Configuración: M=8, 5000 steps, max_steps=50, 3 seeds, 4 perfiles (ideal, depol=0.05, depol=0.10, dephas=0.05).

| Perfil | Sharpe | cand_hit | topm_hit | latency (ms) |
|---|:--:|:--:|:--:|:--:|
| ideal | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ |
| depol=0.05 | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ |
| depol=0.10 | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ |
| dephas=0.05 | _pendiente_ | _pendiente_ | _pendiente_ | _pendiente_ |

Script: `scripts/run_nisq_ablation_v3.py`. Salida: `outputs/tables/abl_nisq_v3.csv`.

---

## 2.4 · Walk-forward CV 4 folds (PENDING)

Configuración: sweet_spot (M=16, β=0.5, k=3, m=5), 3 seeds, modelos C y D, 50k steps por fold.

| Fold | Train | Test | Sharpe_D | cand_hit_D | P(D>C) en cand_hit |
|---|---|---|:--:|:--:|:--:|
| 1 | 2018-2020 | 2021 | _pendiente_ | _pendiente_ | _pendiente_ |
| 2 | 2018-2021 | 2022 | _pendiente_ | _pendiente_ | _pendiente_ |
| 3 | 2018-2022 | 2023 | _pendiente_ | _pendiente_ | _pendiente_ |
| 4 | 2018-2023 | 2024 | _pendiente_ | _pendiente_ | _pendiente_ |

Script: `scripts/run_walk_forward.py`. Salida: `outputs/tables/walk_forward_v3.csv`.

---

## 2.5 · Reducción PPO 1-activo-por-step

Sección dedicada en §sec:modelos del informe:

> "La formulación 'seleccionar 1 activo por step' es deliberadamente reduccionista. Compatibilidad con la máscara (la construcción del conjunto candidato $q_t$ es discreta por naturaleza), reducción del problema a *stock-picking* ranurado, y delimitación clara del alcance del TFE. La generalización a vector de pesos $w_t \\in \\Delta^{N-1}$ es trabajo futuro."

---

## 3.1 · √M atenuación en grafos irregulares

Sección nueva §sec:sqrt-attenuation en el informe:

> "La ventaja $\\sqrt{t}$ de la DTQW está probada para grafos regulares (líneas, ciclos, lattices). Sobre **grafos irregulares** como los de afinidad financiera ($k$-NN simetrizado con pesos), la ventaja se atenúa. Esperar literalmente $\\sqrt{M}=4$ es teóricamente injustificado en este escenario. Las magnitudes observadas (+0.013 en topm_hit_rate) son consistentes con una ventaja atenuada sub-balística, no con la cota óptima $\\sqrt{M}$."

---

## 3.2 · Justificación + ablación de moneda

### Justificación teórica
Anexo C ahora explica:
1. La Householder ponderada es generalización de Grover.
2. Compatible con grafos irregulares (grados variables).
3. Conecta con la quantized random walk de Szegedy.

### Ablación empírica (PENDING)

| Moneda | Sharpe | cand_hit | topm_hit | C² = I? |
|---|:--:|:--:|:--:|:--:|
| weighted_householder | _pendiente_ | _pendiente_ | _pendiente_ | ✅ |
| grover (uniform) | _pendiente_ | _pendiente_ | _pendiente_ | ✅ |
| fourier | _pendiente_ | _pendiente_ | _pendiente_ | ❌ |

Script: `scripts/run_coin_ablation_v3.py`. Salida: `outputs/tables/abl_coin_v3.csv`. 9 tests unitarios verdes (`tests/unit/test_dtqw_coins.py`).

---

## Siguientes pasos

1. Esperar a que termine la cadena (NISQ → walk-forward → coin, ~50-60 min).
2. Inyectar números en Anexos N/O/P/Q.
3. Commit + tag `v3-final`.
