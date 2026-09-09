# Pre-registro del análisis estadístico (rev. v2)

> **Fecha de firma**: 2026-05-12 · *fijada antes de ejecutar la campaña v2.*
> **Autor**: Oscar Fernando Bonilla Suárez · UNIR · TFE en Computación Cuántica.
> **Contexto**: este documento se firma **antes** de las campañas v2 de Fase 3 del plan de mitigación, en respuesta a la observación 1.2 del revisor sobre selección adaptativa de métricas y test unilateral injustificado.

## 1. Pregunta de investigación

¿Mejora el módulo cuántico de exploración estructurada (DTQW sobre subgrafos `H_t`) la **calidad de la priorización** del conjunto candidato `q_t`, respecto a un mecanismo clásico equivalente (caminata aleatoria ponderada `D⁻¹W` sobre el mismo `H_t`)?

## 2. Hipótesis primaria (Sec. 8.4.1 del manuscrito)

**H1 (direccional, unilateral)**: el conjunto candidato `q_t` producido por la DTQW contiene activos ex-post prometedores con **mayor frecuencia** que el de la caminata clásica equivalente.

- **Métrica primaria**: `candidate_hit_rate`. Se define como la fracción de steps en que el conjunto `q_t` contiene **al menos un activo prometedor** según el criterio post-hoc del Cap. 8.17 (top 20% del score `g_i(h) = mean(R_{t+1:t+h}) / (std + ε)`).
- **Test**: unilateral pareado D − C con bootstrap de 5 000 iteraciones sobre seeds comunes; criterio `P(D > C) ≥ 0.95` complementado con IC95% del bootstrap que **excluya el 0**.
- **Justificación a priori del unilateral**:
  1. La DTQW está diseñada teóricamente para concentrar amplitud probabilística en nodos del grafo con interferencia constructiva (no en aleatorios). La hipótesis es DIRECCIONAL: predice "D supera a C", no "D difiere de C".
  2. La literatura (Yamagami et al., 2025; Childs & Goldstone, 2004) caracteriza ventaja cuántica unilateral sobre grafos regulares: `T_mix^quantum ≤ T_mix^classical / √M`.
  3. El pre-registro de esta direccionalidad se realiza **antes** de ver los resultados de la campaña v2 (los resultados v1 mostraron P(D>C)=1.000 en esta métrica, pero v2 usa hold-out temporal nuevo).

## 3. Hipótesis secundarias (exploratorias, bilaterales)

Test bilateral pareado D − C; sin pre-registro direccional:

| Hipótesis | Métrica | Interpretación |
|---|---|---|
| H2 | `topm_hit_rate` | Frecuencia con que la acción tomada es prometedora |
| H3 | `sharpe_ratio` | Eficiencia ajustada por riesgo |
| H4 | `cumulative_return` | Retorno acumulado total |
| H5 | `episodes_to_convergence` (v2) | Eficiencia de aprendizaje (métrica reescrita en Fase 1.A) |

## 4. Corrección por múltiples comparaciones

**Conjunto de métricas testeadas (k = 11)**: `cumulative_return`, `sharpe_ratio`, `max_drawdown`, `mean_reward`, `episodes_to_convergence`, `train_reward_last`, `topm_hit_rate`, `candidate_hit_rate`, `asset_coverage`, `mean_latency_ms`, `duration_seconds`.

**Métodos aplicados** (todos reportados):
- **Bonferroni**: `p_adj = min(p_raw · k, 1)` — criterio conservador, controla familywise error rate.
- **Holm-Bonferroni step-down**: menos conservador, mantiene FWER.
- **Benjamini-Hochberg FDR**: controla tasa de falsa descobertura (menos estricto).

**Umbral**: significancia tras corrección requiere `p_bonferroni < 0.05` para la métrica primaria H1. Las hipótesis secundarias se reportan sin pre-registro de umbral pero acompañadas de las tres correcciones.

## 5. Diseño experimental v2 (Fase 3)

**Datos**:
- Universo Nivel 2 (30 tickers S&P 500).
- **Bloque tuning** (2018-01-01 a 2022-12-31, 5 años): para identificar el `sweet_spot` paramétrico.
- **Bloque hold-out** (2023-01-01 a 2024-12-31, 2 años): para validar significancia. **No se evalúa hipótesis primaria sobre el bloque de tuning**.

**Modelos comparados**:
- **A**: PPO clásico baseline.
- **B** (corregido en Fase 1.B): PPO + state_builder relacional.
- **C**: PPO + subgrafo + caminata clásica.
- **D**: PPO + subgrafo + DTQW.

**Recompensa**: `reward_type = log_wealth`, `λ = 0.0` (decisión a priori basada en Fase 1.C, no en resultados del hold-out).

**Hiperparámetros del módulo cuántico** (`sweet_spot` derivado del scan en bloque tuning):
- `M = 16`, `β = 0.5`, `α = 0.5`, `k_steps = 3`, `m_top = 5`, `init_mode = uniform`.

**Semillas**: `{42, 123, 456, 789, 1024}` (n = 5) sobre el hold-out. Si los resultados son borderline (p_bonf ∈ [0.01, 0.10]), se ampliará a 10 seeds.

## 6. Criterios de detención y reporte

- Todas las métricas se reportan, **independientemente del signo**.
- Si la métrica primaria H1 NO alcanza significancia tras Bonferroni, se reportará explícitamente "hipótesis primaria no confirmada con significancia formal" y se discutirá si la evidencia direccional (`P(D>C) ≥ 0.95` sin corrección) merece mención como hallazgo preliminar.
- Si las hipótesis secundarias muestran resultados inesperados (e.g. C > D en alguna métrica), se reportarán como hallazgos en sí, sin reinterpretación post-hoc.

## 7. Diferencias respecto al análisis v1

| Aspecto | v1 (original) | v2 (este pre-registro) |
|---|---|---|
| Métrica primaria | `topm_hit_rate` y `candidate_hit_rate` (ambas reportadas como significativas) | **Solo `candidate_hit_rate`** (la única que sobrevive Bonferroni k=11) |
| Bloque de validación | Misma ventana del tuning (circularidad) | **Hold-out temporal 2023-2024** |
| Corrección por múltiples comparaciones | Ninguna | **Bonferroni + Holm + FDR** |
| `λ` para Modelo A | `0.1` (régimen donde nada es rentable) | **`0.0` + `log_wealth`** (régimen con Sharpe positivo accesible) |
| Modelo B | Idéntico a A (bug implementacional) | **state_builder relacional funcional** |
| Convergencia | Métrica devolvía 0 entre seeds (instrumentación rota) | **3 métricas complementarias instrumentadas** |

## 8. Compromiso de transparencia

Firmo (con fecha 2026-05-12) que:

1. No se ha ejecutado todavía la campaña v2 de Fase 3 sobre el bloque hold-out.
2. Los hiperparámetros listados en §5 fueron decididos a partir del scan en el bloque de tuning (2018-2022), no del bloque hold-out.
3. Cualquier desvío del protocolo aquí descrito será documentado explícitamente en el reporte final junto con su justificación.

— **Oscar Fernando Bonilla Suárez**, 2026-05-12.

---

## 9. Apéndice técnico — Definición operativa de p-values (firmado 2026-05-12, antes de v4)

> *Añadido en respuesta a la observación 3.2 del revisor sobre la rev. v3.* Esta sección aclara la fórmula exacta y se firma antes de la replicación v4.A.

### 9.1 Fórmula del p-value bilateral

Para cada métrica `m` y cada par de modelos (C, D), el procedimiento bootstrap pareado computa:

```
1. diffs = [m(D, seed) - m(C, seed) for seed in seeds_comunes]
2. boot  = [mean(random.choice(diffs, len(diffs), replace=True))
           for _ in range(B = 5000)]
3. p_better    = mean(boot > 0.0)            # probabilidad bootstrap unilateral
4. p_two_sided = 2 · min(p_better, 1 − p_better)
```

`p_two_sided` (también llamado `p_raw` en las tablas reportadas) es el **p-value bilateral estándar** del bootstrap pareado. La fórmula `2·min(p, 1−p)` es la convención clásica de Efron-Tibshirani para tests bilaterales construidos a partir de un estadístico unilateral del bootstrap.

### 9.2 Reconciliación con el test unilateral de H1

- **H1 es unilateral** (`P(D > C) ≥ 0.95`); este criterio se evalúa directamente sobre `p_better`, no sobre `p_two_sided`.
- **El criterio confirmatorio operacional** sobre H1 es `IC95% del bootstrap excluye el 0` y `p_bonferroni < 0.05` (donde `p_bonferroni` se calcula a partir de `p_two_sided` por consistencia con el resto de métricas).
- **Las hipótesis secundarias H2-H5** son bilaterales y se evalúan sobre `p_two_sided` y su corrección Bonferroni.

### 9.3 Conjuntos de corrección: k=11 (pre-registrado) vs k=5 (sólo hipótesis explícitas)

El conjunto pre-registrado de **k = 11 métricas reportadas** es el criterio operacional oficial. Las 11 son: `cumulative_return`, `sharpe_ratio`, `max_drawdown`, `mean_reward`, `episodes_to_convergence`, `train_reward_last`, `topm_hit_rate`, `candidate_hit_rate`, `asset_coverage`, `mean_latency_ms`, `duration_seconds`.

De éstas, **5 corresponden a las hipótesis explícitas** H1–H5 (`candidate_hit_rate` para H1; `topm_hit_rate`, `sharpe_ratio`, `cumulative_return`, `episodes_to_convergence` para H2–H5). Las 6 restantes (`max_drawdown`, `mean_reward`, `train_reward_last`, `asset_coverage`, `mean_latency_ms`, `duration_seconds`) son **descriptivas / instrumentales** no asociadas a hipótesis.

**Política pre-registrada de Bonferroni**:
- **Criterio oficial**: `p_bonferroni con k = 11` aplicado a *todas* las métricas reportadas. Una métrica se considera confirmatoria si `p_bonferroni < 0.05`.
- **Reporte informativo complementario**: en las tablas extendidas (Anexo K), se añade una columna `p_bonferroni_k5` que aplica corrección sólo sobre las 5 hipótesis explícitas. Esta columna **no sustituye** al criterio oficial; sirve para transparencia y para evidenciar que el criterio oficial es el más conservador.

**Justificación**: aplicar Bonferroni sobre 11 métricas (incluyendo 6 descriptivas) es más conservador que sobre 5. Si una métrica sobrevive `k=11`, también sobrevive `k=5`. La direccionalidad es: usar el más estricto como criterio confirmatorio, reportar el otro como información.

— **Oscar Fernando Bonilla Suárez**, addendum firmado 2026-05-12 antes de la Fase v4.A.

---

*Referencia cruzada: este documento es el Anexo J del manuscrito tras la reescritura de Fase 4.*
