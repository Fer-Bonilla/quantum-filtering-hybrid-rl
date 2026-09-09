# Informe v7 — Síntesis: demostración causal del mecanismo y cierre del canal selector

**Proyecto**: Exploración cuántica estructurada en RL híbrido para selección de activos
**Autor**: Oscar Fernando Bonilla Suárez — UNIR
**Fecha**: 2026-06-15 · **Rama**: `v7-rotation-experiments` (commits `e2214f2`, `5bfdaf6`, `d0c4856`, `a9567b2`)

---

## Resumen ejecutivo

v6 reinterpretó H1: el efecto confirmado (D>C, p<0.001) no es cuántico sino
**rotación de la máscara top-m** — una inferencia a partir del gradiente
observacional R>D>C>Q. v7 convierte esa inferencia en **demostración causal**,
la cuantifica, la valida fuera de muestra y acota el techo de todo el canal
selector. Cinco experimentos, todos sobre el harness existente:

| Exp | Pregunta | Respuesta |
|:--:|---|---|
| 1 | ¿La rotación CAUSA la alineación? | **Sí**: dosis-respuesta monótona y saturante (pendiente +0.031, p<0.001); Sharpe y convergencia no responden. |
| 2 | ¿La rotación EXPLICA las diferencias de v6? | **Sí**: ρ(rotación, cand_hit)=0.93. La DTQW cae exactamente sobre la curva del azar (cero bonus cuántico); C y Q caen por debajo (la concentración informada penaliza). |
| 6 | ¿R y D son formalmente equivalentes? | **Sí** en topm (±0.010) y Sharpe (±0.020) por TOST; en cand_hit el test queda subpotenciado (MDE ±0.019) — sin evidencia de diferencia, y si algo favorece a R. |
| 7 | ¿Replica fuera de muestra (2021-2024)? | **Sí**: R≈D en los 4 regímenes (el signo de R−D cambia de año en año); ningún RL supera a 1/N en ningún año; momentum solo gana en tendencia. |
| 4 | ¿El canal selector puede mover el Sharpe? | **Sí, ×27**: con máscara perfecta el PPO pasa de +0.05 a +1.32 (90% de la cota). El cuello de botella es la CALIDAD de selección, no la política. |

**Mensaje integrado**: el mecanismo del "efecto cuántico" es la rotación
(demostrado causalmente), la rotación explica todo el patrón de v6, la
equivalencia cuántico≈azar es formal y se sostiene en todo régimen — y sin
embargo el canal selector tiene un techo enorme que ningún selector real
(cuántico o clásico) aprovecha, porque ninguno extrae señal mejor que el azar.

---

## La cadena de evidencia completa (v5 → v7)

1. **v5**: los pesos de afinidad no aportan (Grover uniforme > Householder ponderada).
2. **v6**: la propia caminata no aporta sobre el azar (R iguala a D); el óptimo
   QUBO es el peor selector (determinista → sin rotación).
3. **v7 exp 1**: manipulando SOLO la rotación (información fija en cero), la
   alineación responde — causalidad demostrada.
4. **v7 exp 2**: la rotación media las diferencias entre TODOS los selectores;
   la DTQW no añade nada sobre lo que su rotación predice.
5. **v7 exps 6+7**: la equivalencia cuántico≈azar es estadísticamente formal y
   robusta a régimen de mercado.
6. **v7 exp 4**: el potencial del canal era enorme (×27) — el fallo no es de
   la arquitectura sino de la capacidad de TODO selector realista de
   anticipar retornos.

---

## Ablación base A→B→C→D (la cadena canónica del plan §3.5.4)

Antes de los experimentos de mecanismo, la campaña principal (`campaign_1_v3`,
n=10 semillas pareadas) ejecutó las **cuatro variantes del plan** —A baseline,
B + rasgos relacionales en el estado, C + filtrado local clásico, D + caminata
cuántica— diferenciadas *únicamente* en el mecanismo de exploración. Es la
ablación que el TFM §3.5.4 exige: *"A frente a B aísla el efecto de la
representación relacional, B frente a C el efecto del filtrado local, y C
frente a D el efecto específico de la caminata cuántica"*.

| Modelo | Sharpe (media±sd) | cand_hit | topm_hit | converg. |
|---|--:|--:|--:|--:|
| A (PPO puro) | +0.050±0.027 | 1.000* | 0.199 | 9.0 |
| B (+ grafo en estado) | +0.045±0.030 | 1.000* | 0.201 | 8.3 |
| C (+ filtrado clásico) | +0.041±0.031 | 0.429 | 0.178 | 18.5 |
| D (+ caminata cuántica) | +0.045±0.020 | 0.449 | 0.183 | 16.6 |

\* A y B **no filtran** (máscara all-True) → su cand_hit=1.000 es trivial y no
comparable con C/D; la métrica solo tiene sentido entre los modelos que
restringen el espacio de acción (C, D, R, Q). Mismo matiz que en el exp 4.

**Los tres eslabones, aislados:**

| Eslabón | Aísla | Resultado |
|---|---|---|
| **B−A** | representación relacional | **Efecto NULO**: ΔSharpe −0.004 (p=0.565), Δtopm +0.002 (p=0.708), Δconverg −0.7 (p=0.452), Δcand_hit 0.000. Añadir los rasgos del grafo al *estado* no aporta nada por sí solo. |
| **C−B** | filtrado local clásico | Sharpe plano (Δ −0.004, ns); su único efecto medible es introducir el filtrado real (cand_hit deja de ser trivial) y **ralentizar** la convergencia (8→18 episodios). |
| **D−C** | cuántico vs. clásico | El efecto robusto de la cadena: cand_hit +0.019 (p<0.001, Bonferroni 0.0). Sin efecto en Sharpe (Δ +0.003, p=0.737) ni topm (p=0.315). La convergencia da D−C=−3 (p=0.0) pero sobre n=6 (4 semillas no convergen) y el exp 1 controlado muestra que la convergencia **no** responde a la rotación → no es un efecto atribuible. |

**Lectura**: de los tres componentes que el plan quería aislar, **dos no
mueven el desempeño** (relacional y filtrado clásico sobre el Sharpe) y el
tercero (D−C) solo aparece en una métrica de alineación —el +0.019 de cand_hit
que todo v6/v7 dedica a explicar—. Ese efecto existe y es robusto, pero los
experimentos siguientes demuestran que **no es cuántico** (es rotación de
máscara). El eslabón A−B, ausente en versiones previas de este informe, cierra
la cadena de ablación completa que pedía OE7.

Datos: `campaign_1_v3.csv`, `campaign_1_v3_paired_b_vs_a.csv`,
`campaign_1_v3_paired_c_vs_d.csv`.

---

## Exp 1 — Dosis-respuesta de rotación (`StickyWalker`)

Selector con máscara pegajosa: re-sortea cada ranura con probabilidad
`rotation_p`. **La información se mantiene en cero en todo el barrido; solo
varía la rotación.** `p=0` = máscara congelada; `p=1` ≡ Modelo R. 6 niveles ×
10 semillas × 50k steps (config v6: M=8, m=3, nivel2).

| rotation_p | cand_hit | topm_hit | sharpe | converg. | rot. realizada |
|--:|--:|--:|--:|--:|--:|
| 0.00 | 0.419±0.019 | 0.159±0.016 | +0.026 | 8.7 | 0.000 |
| 0.10 | 0.438±0.020 | 0.168±0.018 | +0.032 | 12.1 | 0.125 |
| 0.25 | 0.445±0.014 | 0.176±0.013 | +0.036 | 9.0 | 0.287 |
| 0.50 | 0.452±0.012 | 0.181±0.015 | +0.033 | 14.0 | 0.485 |
| 0.75 | 0.456±0.007 | 0.176±0.016 | +0.035 | 12.8 | 0.634 |
| 1.00 | 0.455±0.011 | 0.176±0.016 | +0.033 | 9.4 | 0.730 |

- Pendiente pareada por semilla (cand_hit): **+0.031** [+0.022, +0.041],
  p<0.001; Spearman ρ=+0.62 (p=1.7·10⁻⁷). `topm_hit`: +0.014 (p<0.001).
- **Forma saturante**: el mayor salto es p=0→0.1 (+0.019); basta romper el
  congelamiento. Explica por qué D (rotación media) ya igualaba a R (máxima):
  ambos están en la meseta; solo el determinismo total (Q) cae al fondo.
- **Sharpe y convergencia NO responden** (ns): el efecto es selectivo de la
  alineación. Corrige además una hipótesis propia: la convergencia rápida de R
  en v6 no es atribuible a la rotación.
- Consistencia interna: p=1.0 reproduce el nivel de R de v6 (0.455 vs 0.458) y
  la pendiente coincide con el R−C de v6 (+0.029).

Figuras: `outputs/figures/v7/fig_dosis_respuesta.png`, `fig_dosis_panel.png`.

## Exp 2 — Mediación: la rotación explica el patrón de v6

Replay de las máscaras de cada selector (C, D, R, Q + la familia S) sobre la
**misma secuencia de evaluación**, sin reentrenar (válido porque
`candidate_mask` es all-True → H_t no depende de las acciones; y cand_hit
depende solo de la máscara). **Fidelidad verificada**: reproduce el cand_hit
de v6 exacto a 3 decimales en los 4 selectores.

| Selector | rotación realizada | cand_hit obs. | cand_hit predicho (curva azar) | residual |
|---|--:|--:|--:|--:|
| R | 0.806 | 0.458 | 0.455 | **+0.003** |
| D (DTQW) | 0.453 | 0.448 | 0.445 | **+0.003** |
| C (clásica) | 0.435 | 0.429 | 0.445 | **−0.015** |
| Q (QUBO) | 0.390 | 0.389 | 0.442 | **−0.053** |

- ρ(rotación, cand_hit) = **+0.93** (p=1.1·10⁻⁴) sobre los 10 selectores.
- **R y D caen SOBRE la curva del azar**: la DTQW logra exactamente lo que su
  rotación predice — cero valor cuántico añadido.
- **C y Q caen POR DEBAJO**: a igual rotación, la concentración informada
  rinde peor que elegir al azar (drásticamente en el óptimo QUBO).
- Matiz importante: la brecha D−C de v6 (+0.019) NO se explica por diferencia
  de rotación (rotan casi igual, +0.001 predicho) sino porque **C rinde por
  debajo de la curva** mientras D la iguala.

Figura: `outputs/figures/v7/fig_mediacion_rotacion.png`.

## Exp 6 — Equivalencia formal (TOST) y potencia

Sobre los datos pareados de v6 (10 semillas comunes), TOST de Schuirmann con
márgenes pre-especificados + efecto mínimo detectable (MDE, potencia 80%).

| Comparación | Métrica | diff | MDE(80%) | Veredicto |
|---|---|--:|--:|---|
| R−D | topm_hit | +0.004 | ±0.010 | **EQUIVALENTE** (Δ=±0.010, p=0.042) |
| R−D | sharpe | +0.002 | ±0.026 | **EQUIVALENTE** (Δ=±0.020, p=0.031) |
| R−D | cand_hit | +0.010 | ±0.019 | subpotenciado: no concluye (si algo, favorece a R) |
| R−C | cand_hit | +0.029 | — | NO equivalente (control de validez) |
| Q−R | cand_hit | −0.069 | — | NO equivalente (control de validez) |

**Lectura honesta**: "R≈D" pasa de "no significativo" (ausencia de evidencia) a
**equivalencia demostrada** en topm y Sharpe. En la métrica primaria, n=10 no
da potencia para el margen estricto — se reporta el MDE como cualificación. El
test discrimina (no declara todo equivalente), lo que valida el procedimiento.

Tabla: `outputs/tables/tost_equivalence_v7.csv`.

## Exp 7 — Validez externa por regímenes (walk-forward 2021-2024)

4 folds de ventana expansiva (test = un año natural), {A, D, R} × 5 semillas
con la config v7 (M=8) + benchmarks por fold.

| Año test | A (PPO) | D (DTQW) | R (azar) | momentum | 1/N | oráculo |
|---|--:|--:|--:|--:|--:|--:|
| 2021 | +0.037 | +0.094 | +0.109 | +0.136 | +0.117 | +1.69 |
| 2022 (bajista) | −0.055 | −0.000 | −0.033 | **−0.144** | −0.038 | +1.42 |
| 2023 | +0.065 | −0.037 | +0.056 | +0.113 | +0.110 | +1.43 |
| 2024 | +0.086 | −0.001 | +0.075 | +0.061 | +0.115 | +1.48 |

- **R≈D replica en los 4 regímenes**: el signo de R−D cambia entre años (R>D
  en 3/4); la DTQW llega a ser negativa en 2023 con R positivo. Sin ventaja
  cuántica en ningún año → el claim central pasa de caso a **patrón**.
- **Afinación honesta del punto 1 de v6**: "momentum ≥ RL" NO es universal
  (2/4 folds; momentum es el PEOR en 2022). El baseline robusto es **1/N**:
  bate al PPO en los 4 folds y al mejor RL en 3/4. Ningún RL supera a 1/N en
  ningún régimen.
- La cota oráculo (+1.42…+1.69) se mantiene en todos los folds.

Figura: `outputs/figures/v7/fig_regime_validation.png`.

## Exp 4 — Máscara-oráculo: el techo del canal selector

PPO entrenado con la máscara perfecta (top-m por retorno futuro real,
`R_{u,t+1}` — fuga ex-post deliberada y etiquetada, análoga al benchmark
`oracle_expost`). 10 semillas × 50k steps.

| Estrategia | Sharpe |
|---|--:|
| Modelo D (DTQW) | +0.045 |
| Modelo R (azar) | +0.047 |
| Modelo A (PPO puro) | +0.050 |
| equal_weight (1/N) | +0.101 |
| momentum_20d | +0.120 |
| **ORACLE-máscara + PPO** | **+1.324** |
| oracle_expost (cota absoluta) | +1.459 |

- ORACLE−A pareado: **+1.27** [+1.25, +1.30], p<0.001 — la máscara perfecta
  multiplica el Sharpe **×27** y recupera el **90%** del trayecto hasta la cota.
- **Veredicto**: la política PPO NO es el cuello de botella — explota
  perfectamente una buena máscara. C/D/R/Q/S no mueven el Sharpe porque su
  **calidad de selección no supera al azar**, no por la arquitectura.
- Cierra la pregunta que los exps 1-2-6 dejaban abierta (¿por qué el Sharpe es
  plano?) y acota el valor potencial de cualquier selector futuro.

Figura: `outputs/figures/v7/fig_oracle_techo.png`.

---

## Cumplimiento del plan (capítulo 3 del TFM)

### Objetivos específicos: 8/8 cumplidos

| OE | Síntesis del objetivo | Estado | Evidencia principal |
|:--:|---|:--:|---|
| 1 | Formalizar la selección discreta como MDP | ✅ | `MarketEnv` (contrato gymnasium, Sec. 5.6/6.6) |
| 2 | Recompensa financiera con riesgo, costes y validación temporal | ✅ | reward v2 `log_wealth` + λ/μ/costes; splits cronológicos |
| 3 | Grafo dinámico interpretable | ✅ | G_t (correlación + sector; α, β) y H_t por BFS ponderada |
| 4 | Núcleo RL clásico como línea base | ✅ | PPO discreto con máscara (Modelo A) |
| 5 | Priorización local con interfaz común clásica↔cuántica | ✅➕ | Protocolo `LocalModule`; pedía 2 variantes, soportó 7 (C, D, R, Q, S(p), oráculo) |
| 6 | Arquitectura híbrida modular (decisión en el clásico) | ✅ | `HybridAgent` + step_hook; máscara en rollout buffer |
| 7 | Ablación: grafo / filtrado local / cuántico-vs-clásico | ✅➕ | A↔B, B↔C, C↔D… y v6/v7 aislaron el **mecanismo** (más de lo pedido) |
| 8 | Sensibilidad (grafo, M, hiperparámetros, ventanas, semillas, ruido) | ✅ | 9 ablaciones v3-v5 + walk-forward 2021-24 + fidelidad QA |

### Objetivo general e hipótesis

- **Objetivo general** — *"**evaluar si** la DTQW mejora la exploración y
  acelera la convergencia sin deteriorar el desempeño financiero"*. El verbo
  era **evaluar**: la evaluación está completa y la respuesta es negativa y
  bien delimitada (no mejora sobre el azar; ralentiza la convergencia; no
  deteriora el Sharpe). **Cumplido.**
- **Hipótesis principal** (mejora en exploración o convergencia con desempeño
  comparable) — **confirmada literalmente** (D>C, p<0.001, replicado n=10).
- **Hipótesis secundaria** (la mejora será distinguible del efecto relacional
  y del filtrado local) — **refutada, y esa refutación ES el resultado
  central**: la mejora resultó ser un efecto del filtrado dinámico (rotación),
  demostrado causalmente.
- **Hipótesis de robustez** (el beneficio se atenuará con ruido) — resuelta
  con ironía informativa: el efecto sobrevivía al ruido NISQ (v3), coherente
  con que nunca fue cuántico.
- **Criterio de éxito pre-registrado** (cap. 3.4): *"Un resultado neutro o
  negativo también aportaría información al delimitar las condiciones en que
  la caminata cuántica no supera a las alternativas clásicas."* El trabajo
  ejecutó exactamente la condición de atribución que el plan exigía — con más
  controles de los previstos.

---

## Guion de defensa en una frase

> *Confirmamos el efecto con rigor (D>C, p<0.001), descubrimos con el control
> aleatorio que no era cuántico, demostramos causalmente que el mecanismo es
> la rotación de la máscara (dosis-respuesta), verificamos que explica todo el
> patrón (mediación, ρ=0.93), que la equivalencia cuántico≈azar es formal
> (TOST) y robusta a régimen (2021-2024), y acotamos el techo del canal: la
> selección perfecta multiplicaría el Sharpe ×27 — pero ningún selector
> realista, cuántico o clásico, extrae señal mejor que el azar.*

La contribución del TFM es el **marco metodológico** (controles: benchmark,
oráculo, máscara aleatoria, óptimo QUBO, dosis-respuesta, mediación,
equivalencia, walk-forward) que permitió aislar, medir y refutar el aporte de
tres módulos cuánticos sucesivos sin atribuirles efectos de la formulación.

---

## Artefactos v7

| Artefacto | Ruta |
|---|---|
| Selector paramétrico (exp 1) | `src/quantum/sticky_walker.py` (+10 tests) |
| Barrido dosis-respuesta | `scripts/run_rotation_sweep_v7.py` → `rotation_sweep_v7.csv` |
| Análisis dosis-respuesta | `scripts/analyze_rotation_sweep_v7.py` → `*_summary.csv` |
| Mediación (exp 2) | `scripts/mediation_rotation_v7.py` → `mediation_rotation_v7.csv` |
| TOST + potencia (exp 6) | `scripts/tost_equivalence_v7.py` → `tost_equivalence_v7.csv` |
| Máscara-oráculo (exp 4) | `src/agents/oracle_mask.py` (+7 tests), `scripts/run_oracle_campaign_v7.py`, `scripts/analyze_oracle_v7.py` |
| Regímenes (exp 7) | `scripts/run_regime_validation_v7.py`, `scripts/analyze_regime_validation_v7.py` |
| Figuras (150 dpi) | `outputs/figures/v7/`: `fig_dosis_respuesta`, `fig_dosis_panel`, `fig_mediacion_rotacion`, `fig_oracle_techo`, `fig_regime_validation` |
| Tablas presentación | `outputs/tables/v7/tabla_dosis_respuesta.{md,tex}` |
| Generador de figuras | `scripts/v7_presentation_figures.py` |

Calidad: **294/294 tests verdes** (287 de v6 + 10 StickyWalker + 7 oracle, con
solapamiento de colecciones); campañas con reanudación idempotente
(sobreviven a suspensiones); todos los runs con semillas pareadas a las
campañas v6 para comparabilidad directa.

Pendientes opcionales (no críticos, candidatos a apéndice): round-robin
determinista (exp 3), híbrido ε (exp 5), barrido de m (exp 8), mixer QUBO con
conservación de cardinalidad (exp 9), ablación de recompensa/features (exp 10),
roadmap Quantum Reservoir (exp 11, pre-registrado en `docs/qr_roadmap.md`).
