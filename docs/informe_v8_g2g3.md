# Informe de compuertas G2 y G3 — Campaña v8 (Tier 3)

**Pre-registro**: `docs/preregistro_v8.md` (commit `0e3ee5e`). Familia
Bonferroni k=6 (α_c=0,00833). **Estado**: EXP-5, EXP-6 y EXP-7 completos.
**Cómputo**: 120 + 39 + 39 = 198 corridas (~5 h CPU con contención), dentro
del presupuesto. Análisis: `scripts/analyze_v8_g2g3.py`
(`outputs/v8_g2g3_analysis.txt`).

---

## Compuerta G2 — EXP-5: subgrafos de regularidad controlada

### H-v8.4 (confirmatoria): D − R > 0 en `candidate_hit`, agregada sobre topologías

| Resultado | Valor |
|---|---|
| Diferencia agregada D−R | **−0,0273** IC95 [−0,0385, −0,0175] |
| p (unilateral D>R) | 1,0000 → pBonf6 = 1,0000 |
| **Veredicto** | **NO significativa — y con signo INVERTIDO: D es peor que R en el agregado de topologías regulares** |

### Detalle por topología (exploratorio, n=10 pareado)

| Topología | media D | media C | media R | D−R (p) | D−C (p) |
|---|--:|--:|--:|--:|--:|
| 3-regular (afinidad) | 0,466 | 0,471 | 0,458 | +0,008 (0,064) | −0,005 (0,92) |
| 3-regular (uniforme) | 0,463 | 0,497 | 0,458 | +0,005 (0,26) | −0,034 (1,00) |
| ciclo C₈ | 0,406 | **0,527** | 0,458 | **−0,052** (1,00) | −0,121 (1,00) |
| bipartito K_{a,b} | 0,388 | 0,415 | 0,458 | **−0,070** (1,00) | −0,027 (1,00) |

### La pieza teórica clave: el régimen balístico SÍ aparece

El exponente de dispersión empírico (RMS de distancia al seed vs k, 40
subgrafos de la secuencia real) verifica que las topologías hacen su trabajo:

- **Ciclo C₈**: RMS = 1, 2, 3, 4 para k = 1..4 — **dispersión exactamente
  balística** (RMS=k) hasta la mitad del anillo; después revienta por el
  tamaño finito (wrap-around).
- **Bipartito**: patrón de reavivamientos perfecto (RMS = 1; 1,73; 1; **0**;
  1; 1,73) — el caminante se relocaliza por completo en k=4, la dinámica
  coherente de libro.
- **BFS irregular** (base de la tesis): exponente ≈ 0,24 — la atenuación por
  irregularidad que el Cap. 4 predijo.

**Lectura central de G2**: el experimento alcanzó el régimen donde la teoría
concede la ventaja dispersiva a la DTQW —verificado con la métrica secundaria—
y la ventaja de *selección* siguió sin aparecer: dispersarse más rápido por el
subgrafo es ortogonal a señalar activos prometedores. Más aún: justo donde la
DTQW es más coherente (ciclo, bipartito), su `candidate_hit` CAE por debajo
del azar, mientras la caminata clásica en el ciclo (que se queda difusivamente
cerca del seed) obtiene el mejor valor de toda la campaña (0,527) — otra
manifestación de que la geometría de la máscara, no la calidad de la
exploración, gobierna la métrica.

**Propuesta G2: Escenario A ampliado** — "régimen probado sin ventaja
detectable" (criterio pre-registrado), con el matiz de que el resultado
negativo es ahora más fuerte que el previsto: no solo D ≈ R, sino D < R en
las topologías más coherentes.

---

## Compuerta G3 — EXP-6 (integración suave) y EXP-7 (información+rotación)

### H-v8.5 (confirmatoria): Sharpe(soft-D) − Sharpe(soft-R) > 0

| Resultado | Valor |
|---|---|
| Diferencia (β*=0,5, n=10) | +0,0134 IC95 [+0,0019, +0,0285] |
| p (unilateral) | 0,0098 → **pBonf6 = 0,0588** |
| **Veredicto** | **No significativa al umbral corregido** (nominalmente sí, p<0,05) |

El desglose desactiva la lectura cuántica del efecto nominal:

| Variante | Sharpe | Lectura |
|---|--:|---|
| soft-D (P_k DTQW) | +0,0352 | ≈ soft-C: **el ranking cuántico no aporta sobre el clásico** (d=+0,0013, p₂=0,88) |
| soft-C (p_k clásica) | +0,0339 | |
| soft-R (uniforme en H_t) | +0,0219 | el beneficio nominal viene de *tener ranking*, no de que sea cuántico |
| **D máscara dura** (referencia) | **+0,0447** | **ninguna variante suave la supera**: el canal suave no desbloquea el techo del oráculo |

### EXP-7 (exploratorio, FDR): información+rotación vs R

| Selector | cand_hit vs R | Sharpe vs R | q (FDR) |
|---|--:|--:|--:|
| softmax(τ*=0,5) | +0,002 | −0,012 | ≥ 0,94 |
| momentum-refresh | **−0,048** | −0,010 | 1,00 |
| thompson | +0,010 (p=0,05) | **−0,032** | 0,30 / 1,00 |

**Ningún selector informado-con-rotación supera a R** tras FDR. El más
cercano (Thompson, +0,010 en cand_hit, q=0,30) paga su información con un
Sharpe significativamente peor. La forma fuerte de la conclusión sobrevive:
**información + rotación no supera a rotación sola** → el Paquete D (matiz
"la información añade Δ") **no se activa**.

---

## Figuras

- `outputs/figures/v8/fig_g2_topologias.png` — cand_hit por topología (D/C/R).
- `outputs/figures/v8/fig_g2_dispersion.png` — RMS vs k: el ciclo sigue la
  diagonal balística; el bipartito muestra reavivamientos; la base irregular
  queda atenuada.
- `outputs/figures/v8/fig_g3_soft.png` — Sharpe suave vs máscara dura + EXP-7.

## Artefactos

| Artefacto | Ruta |
|---|---|
| EXP-5 (120 runs) | `outputs/tables/regular_topologies_v8.csv` (MLflow `v8_exp5_*`) |
| EXP-6 (tuning + 30) | `outputs/tables/soft_integration_v8{_tuning,}.csv` |
| EXP-7 (tuning + 30) | `outputs/tables/informed_walkers_v8{_tuning,}.csv` |
| Módulos nuevos | `src/graph/regular_subgraphs.py`, `src/agents/soft_hybrid.py`, `src/quantum/informed_walkers.py` (+19 tests; suite 313 verdes) |
| Análisis | `scripts/analyze_v8_g2g3.py` → `outputs/v8_g2g3_analysis.txt` |

---

## Propuesta integrada de escenario (G2 + G3)

**Escenario A ampliado y reforzado**, sin activación de C ni D:

1. **G2**: la ventaja cuántica no aparece ni en el régimen que la teoría
   señala como favorable, con la verificación balística en mano. El párrafo
   de regularidad de §6.3 puede pasar de "régimen no probado" a "régimen
   probado sin ventaja detectable" (edición A4, que quedó pendiente en G1).
2. **G3**: el canal suave no supera a la máscara dura ni distingue cuántico
   de clásico; los selectores informados no superan al azar. Matiz de
   ingeniería valioso para §6.4: si se integra suavemente, lo que importa es
   tener *algún* ranking (cuántico o clásico por igual) — pero la máscara
   dura sigue siendo mejor.

### Qué sigue (requiere tu confirmación)

1. **Confirmar Escenario A ampliado** → aplico la Fase 2 de G2/G3: edición A4
   (§6.3 regularidad), síntesis G2/G3 en el cuerpo (ampliar la subsección v8
   del Cap. 5 y el Anexo H con H-v8.4/H-v8.5 y sus tablas), matiz de
   ingeniería en §6.4, y actualización de las filas pendientes del Anexo H.
2. O continuar antes con el **Tier 4** (EXP-8 universo N=100, EXP-11
   estadística financiera —barato, sin reentrenar—, EXP-10 ablaciones, demo
   hardware con aprobación de gasto).

Conforme al pre-registro, no se edita la memoria hasta tu confirmación.
