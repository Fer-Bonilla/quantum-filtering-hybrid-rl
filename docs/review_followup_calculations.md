# Informe de seguimiento: DSR, series A/D/R y respaldo con máscara completa

Encargo del auditor académico (2026-08-05): recalcular el DSR, reconstruir
las series A/D/R y cuantificar el respaldo `q_t = U`. Regla observada: los
valores publicados y el manuscrito **no se modificaron**; este informe entrega
los cálculos reproducibles y el material LaTeX propuesto.

**Software**: Python 3.13.13, numpy 2.4.4, scipy 1.17.1, pandas 2.3.3
(entorno `uv` del repositorio).
**Semillas aleatorias**: bootstrap pareado `default_rng(20260705)`, B=5000;
semillas de replay/evaluación {42, 123, 456, 789, 1024, 7, 99, 314, 1729,
65535}; el recálculo del DSR es un cálculo cerrado sin aleatoriedad.

**Comandos exactos** (en la raíz del repositorio):

```bash
uv run python scripts/review_followup_inventory.py
uv run python scripts/build_dsr_trial_family.py
uv run python scripts/recalculate_dsr_review.py
uv run python scripts/fallback_instrumentation_v8.py
```

---

## Clasificación por punto del encargo

| # | Punto | Estado | Entregable |
|---|---|---|---|
| 1 | Inventario | **Resuelto** | `outputs/tables/review_followup_inventory.csv` (39 artefactos, 5 ausentes explícitos) |
| 2 | Reconstrucción A/D/R | **Imposible de reconstruir** con los artefactos disponibles (ver §2) | constancia en inventario + este informe |
| 3 | Familia de 40 ensayos | **Resuelto** (con separación de familias) | `outputs/tables/dsr_trial_sharpes.csv` |
| 4 | Dispersión transversal y K_eff | **Parcialmente resuelto**: dispersión transversal sí; K_eff **no estimable** (sin series alineadas); sensibilidad por K asumido | estadísticas en §4 |
| 5 | Recalculo del DSR | **Resuelto** | `scripts/recalculate_dsr_review.py`, `outputs/tables/dsr_recalculated.csv` |
| 6 | Instrumentación del respaldo | **Resuelto** (con límites declarados: topm/acción no reproducibles) | `outputs/tables/fallback_events.csv` (37 800 filas) |
| 7 | Efecto del respaldo | **Resuelto** | `fallback_summary.csv`, `fallback_paired_effects.csv` |
| 8 | Entregables | **Resuelto** | este informe + LaTeX en §7 |

---

## §2 — Por qué A/D/R no son reconstruibles

Tres artefactos imprescindibles no existen (verificado, no inferido):

1. **Registros por paso**: `src/training/evaluate.py::evaluate_agent`
   devuelve solo agregados (`EvalResult`); nunca escribió fecha, acción ni
   retorno por paso.
2. **Checkpoints de políticas**: `outputs/checkpoints/` contiene solo
   `.gitkeep`; `run_campaign.py` no guarda pesos. Sin la política no se
   pueden regenerar las acciones ejecutadas.
3. Por (1)+(2), `adr_daily_returns.csv` y `adr_series_stats.csv` **no se
   producen**: exigirían reentrenar, y el encargo prohíbe fabricar datos.

Consecuencia aplicada (criterio de aceptación): **A/D/R se mantienen sin
DSR**. Nota adicional: el T=1260 usado antes para su DSR aproximado provenía
de 5 episodios solapantes × 252 pasos sobre 349 fechas únicas
(2023-08-11..2024-12-30) — precisamente el tipo de T inflado que el encargo
prohíbe; el recálculo no lo utiliza.

## §3 — Familia de ensayos (`dsr_trial_sharpes.csv`)

40 configuraciones trazables, cada una con `trial_id`, `family`,
`configuration`, `period`, `T`, `daily_sharpe`, `source_file`,
`aggregation_method`. Homogeneidad verificada: todas evaluadas en la misma
partición de prueba (2023-08-11..2024-12-30), Sharpe por paso diario sin
anualizar sobre la recompensa (con λ=0 y coste ≈ 0, la recompensa es el
retorno log neto diario del activo elegido), agregación = media entre
semillas.

Separaciones aplicadas (como pide el encargo):

- **`seleccion_original`** (19): A, B, C, D, R, Q, S(p)×6, NISQ×4, monedas×3
  — participaron en la selección del plan original.
- **`control_posterior`** (20): S(p\*), Crel, topologías×12, soft×3,
  informados×3 — ejecutadas en v8 como controles pre-registrados, después de
  la selección.
- **`diagnostico_no_candidato`** (1): máscara-oráculo ex-post. Usa
  información futura: no es una estrategia seleccionable y su Sharpe (1,32)
  distorsionaría la varianza transversal (sd pasaría de 0,019 a 0,204).

## §4 — Dispersión transversal y K_eff

Sobre los `daily_sharpe` de la familia (varianza transversal, `ddof=1`):

| Familia | n | media | var | sd | mín | Q1 | mediana | Q3 | máx |
|---|---|---|---|---|---|---|---|---|---|
| Candidatas (sin oráculo) | 39 | +0,0430 | 0,000378 | 0,0194 | +0,0037 | +0,0334 | +0,0406 | +0,0471 | +0,1070 |
| Selección original (sin oráculo) | 19 | +0,0457 | 0,000582 | 0,0241 | +0,0037 | +0,0329 | +0,0413 | +0,0516 | +0,1070 |

**K_eff: no estimable.** El procedimiento del apéndice de bailey2014deflated
requiere la matriz de correlación de las series diarias alineadas de las
configuraciones; esas series no se registraron (solo agregados por corrida).
Conforme al encargo, **no se calcula un K_eff ficticio**: se reporta
sensibilidad por valores asumidos K ∈ {10, 20, 39} y la variante con la
familia de selección (K=19, sd=0,0241).

## §5 — DSR recalculado (`dsr_recalculated.csv`)

Fórmulas (nulo declarado SR=0; `scipy.stats.norm.cdf/ppf`):

- SR\* = sd_transversal · [(1−γ_E)·z_{1−1/K} + γ_E·z_{1−1/(K·e)}]
- DSR = Φ((SR̂ − SR\*)·√(T−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²))

| Estrategia | Escenario | SR̂ | T | γ₃ | γ₄ | SR\* | DSR | Veredicto (0,95) |
|---|---|---|---|---|---|---|---|---|
| 1/N | K=39, sd cand. (recomendado) | 0,0525 | 1004 | −0,290 | 5,25 | 0,0424 | **0,62** | no supera |
| 1/N | K=19, sd selección | 0,0525 | 1004 | −0,290 | 5,25 | 0,0453 | 0,59 | no supera |
| 1/N | K=20 asumido | 0,0525 | 1004 | −0,290 | 5,25 | 0,0370 | 0,69 | no supera |
| 1/N | K=10 asumido | 0,0525 | 1004 | −0,290 | 5,25 | 0,0306 | 0,75 | no supera |
| momentum 20d | K=39, sd cand. (recomendado) | 0,0771 | 1003 | −0,024 | 5,23 | 0,0424 | **0,86** | no supera |
| momentum 20d | K=19, sd selección | 0,0771 | 1003 | −0,024 | 5,23 | 0,0453 | 0,84 | no supera |
| momentum 20d | K=20 asumido | 0,0771 | 1003 | −0,024 | 5,23 | 0,0370 | 0,90 | no supera |
| momentum 20d | K=10 asumido | 0,0771 | 1003 | −0,024 | 5,23 | 0,0306 | 0,93 | no supera |
| A / D / R | — | — | — | — | — | — | **sin DSR** | sin serie diaria verificable |

Los valores publicados 0,59 (momentum) y 0,29 (1/N) quedan etiquetados en el
CSV como `procedimiento_anterior_no_valido` (usaban la varianza de muestreo
de la serie individual, no la transversal). **La conclusión sustantiva no
cambia: ninguna estrategia supera 0,95 en ningún escenario** — el nulo
financiero es robusto al procedimiento corregido, y de hecho los DSR
recalculados son mayores que los publicados sin alcanzar el umbral.

El bootstrap estacionario no cambia (era correcto): bloque medio 20 d,
p=1/20, B=5000, semilla 20260705, percentiles 2,5/97,5;
1/N [−0,0032, +0,1145] (incluye el cero), momentum [+0,0087, +0,1408].

## §6–§7 — Respaldo con máscara completa

Replay determinista instrumentado de C, D y R (10 semillas × 5 episodios ×
252 pasos = 12 600 pasos por brazo; 37 800 eventos en
`fallback_events.csv`), con `select_subgraph` sondado para clasificar motivos
(`insufficient_history`, `no_eligible`, `no_positive_neighbors`, `other`).

**Resultado: cero activaciones del respaldo** en los tres brazos, todas las
semillas y todos los pasos de la secuencia de evaluación. En consecuencia:

- `candidate_hit_all` = `candidate_hit_valid` exactamente, por brazo y
  semilla (C 0,4294; D 0,4485; R 0,4580 — reproducen los valores publicados
  por semilla, validación adicional del replay).
- La descomposición es trivial: fallback_rate = 0 ⇒ cand_all = cand_valid.
- D−C pareado (mismo bootstrap de la tesis, B=5000, rng 20260705):
  **con respaldos +0,0191, IC95 [+0,0177, +0,0208], p₂ < 4·10⁻⁴; sin
  respaldos idéntico** (las muestras coinciden al no existir eventos).
- C y D coinciden en número de respaldos (0), fechas (∅) y motivos (∅).

Límites declarados: `topm_hit` y `selected_action` dependen de la política
PPO entrenada, que no se conservó; esas columnas van vacías con nota. El
respaldo por historia insuficiente solo puede activarse en los primeros
pasos del panel completo (t < ventana), región que la partición de prueba
nunca visita; el respaldo existe como salvaguarda de código, no como
fenómeno operativo en la evaluación.

## §7bis — Material LaTeX propuesto (NO aplicado al manuscrito)

Párrafo para sustituir el DSR del Anexo H cuando el autor lo apruebe:

```latex
\textbf{DSR con varianza transversal (recalculo).} El umbral \(SR^{*}\) se
obtiene de la dispersion \emph{transversal} de los Sharpe de la familia de
ensayos (39 configuraciones candidatas; el oraculo ex-post se excluye por no
ser una estrategia seleccionable): \(\mathrm{sd}=0{,}0194\), y
\(SR^{*} = \mathrm{sd}\cdot[(1-\gamma_E)z_{1-1/K}+\gamma_E z_{1-1/(Ke)}]\).
Con \(K=39\): \(SR^{*}=0{,}0424\); momentum \(\mathrm{DSR}=0{,}86\) y 1/N
\(\mathrm{DSR}=0{,}62\), sin superar 0{,}95 en ninguna sensibilidad
(\(K\in\{10,20,39\}\) y familia de seleccion \(K=19\): DSR
\(\leq0{,}93\)). Los agentes A/D/R permanecen sin DSR: sus series diarias
no son reconstruibles (sin registros por paso ni \emph{checkpoints}) y su
\(T\) de evaluacion (episodios solapantes) no representa fechas
independientes. El K efectivo no es estimable sin series alineadas; la
sensibilidad en \(K\) acota su efecto.
```

```latex
\textbf{Respaldo con mascara completa.} Un replay instrumentado de la
secuencia de evaluacion (C, D y R; 10 semillas \(\times\) 5 episodios
\(\times\) 252 pasos; 37\,800 pasos) registra cero activaciones del respaldo
\(q_t=U\) en todos los brazos, semillas y pasos
(\codepath{fallback\_events.csv}); el contraste D--C es identico con y sin
pasos de respaldo (\(+0{,}0191\), IC95 \([+0{,}0177,+0{,}0208]\)). El
respaldo es una salvaguarda de codigo sin incidencia en ninguna metrica
publicada de evaluacion.
```

## Criterios de aceptación — verificación

- Ningún T multiplicando fechas repetidas: los T de agentes no se usan; 1/N
  y momentum usan sus fechas únicas (1004/1003). ✔
- Sharpe trazable por configuración: 40/40 con `source_file`. ✔
- Varianza del DSR transversal entre ensayos (`ddof=1`). ✔
- K_eff: método descrito; **datos insuficientes** → no se calcula; se
  reporta sensibilidad por K asumido (permitido por el encargo). ✔
- A/D/R sin series verificables → **sin DSR**. ✔
- Respaldo con contador por brazo y semilla (`fallback_summary.csv`). ✔
- D−C con y sin respaldos (`fallback_paired_effects.csv`). ✔
- Valores publicados intactos; el recálculo convive etiquetado en
  `dsr_recalculated.csv` hasta que el autor apruebe sustituirlos. ✔
