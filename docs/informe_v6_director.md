# Informe v6 — Respuesta a los 4 puntos de la reunión de dirección

**Proyecto**: Exploración cuántica estructurada en RL híbrido para selección de activos
**Autor**: Oscar Fernando Bonilla Suárez — UNIR
**Fecha**: 2026-06-09 · **Tag**: `v6-final`

---

## Resumen ejecutivo

Se implementaron y ejecutaron los experimentos de los puntos 1-3 y el roadmap del
punto 4. Los resultados son concluyentes y obligan a una reinterpretación honesta
del aporte del módulo cuántico:

| Punto | Pregunta | Respuesta |
|:--:|---|---|
| 1 | ¿El método compite con estrategias estándar? | **No en Sharpe**: momentum 20d (+0.120) supera a todos los agentes RL (+0.03–0.05). El oráculo ex-post (+1.459) marca la cota superior. |
| 2 | ¿La ventaja viene del módulo cuántico o de la formulación RL? | **De la formulación**: una máscara ALEATORIA de tamaño m (Modelo R) iguala o supera a la DTQW en la métrica primaria y converge más rápido. |
| 3 | ¿Quantum Annealing aporta? | **No**: incluso en el límite adiabático IDEAL (óptimo QUBO exacto), el Modelo Q es peor que C, D y R en alineación, y neutro en Sharpe. |
| 4 | ¿Procede el cambio a Quantum Reservoir? | **Sí, condición activada**. Roadmap en `docs/qr_roadmap.md` (~4-5 días de trabajo; requiere re-cablear la integración: QR es extractor de features, no selector). |

---

## Punto 1 — Benchmarks clásicos (`outputs/tables/benchmarks_v6.csv`)

Mismo split de test, misma recompensa `log_wealth`, mismo harness de evaluación
que las campañas (familia `discrete_policy` directamente comparable con los
agentes RL; familia `static_portfolio` comparable solo vía Sharpe por step).

| Estrategia | Sharpe | topm_hit | Familia |
|---|--:|--:|---|
| **oracle_expost** (solución óptima) | **+1.459** | 0.343 | discreta (fuga deliberada) |
| **momentum_20d** | **+0.120** | **0.241** | discreta, sin fuga |
| equal_weight (1/N) | +0.101 | — | cartera estática |
| markowitz max-Sharpe | +0.092 | — | cartera estática |
| buyhold_best (train) | +0.063 | 0.180 | discreta, sin fuga |
| random policy | +0.062 | 0.201 | discreta |
| — A / B / R / C / D / Q (RL) | +0.033…+0.050 | 0.161…0.201 | discreta |

**Lectura**: en este universo/régimen, NINGÚN agente RL supera a momentum simple
ni a la cartera 1/N en Sharpe; los agentes RL son comparables a una política
aleatoria. El agente captura ~3% del Sharpe del oráculo. La contribución de la
tesis debe enmarcarse como **metodológica** (cómo aislar y evaluar honestamente
módulos cuánticos en RL financiero), no como utilidad financiera.

## Punto 2 — Descomposición: Modelo R (máscara aleatoria)

Modelo R = pipeline idéntico a C/D (mismo grafo, mismo subgrafo, mismo PPO)
pero el "selector" devuelve m nodos **aleatorios** del subgrafo. 10 semillas
pareadas con `campaign_1_v3`, 50k steps.

| Comparación | cand_hit (mean diff) | p_Bonf | Convergencia (diff) | p_Bonf |
|---|--:|--:|--:|--:|
| R − C | **+0.029** | **0.000** | **−8.3 rollouts** | **0.000** |
| R − D | +0.010 | 0.964 (ns) | **−6.4 rollouts** | **0.000** |

**Lectura**: la máscara aleatoria **supera a la caminata clásica** en la métrica
primaria con mayor margen (+0.029) que el que la DTQW exhibía sobre C (+0.019),
y es **estadísticamente indistinguible de la DTQW**. Además converge tan rápido
como el baseline A (8.6 vs 9.0 rollouts), mientras C/D convergen en ~17-19.

**Reinterpretación de H1 (obligada)**: el efecto confirmado D>C en
`candidate_hit_rate` (p_Bonferroni<0.001, replicado n=10) es real, pero su
mecanismo NO es la interferencia cuántica: es la **diversidad/rotación de la
máscara top-m**. Un selector aleatorio — máxima rotación, cero información —
maximiza el efecto. Las caminatas informadas (C, y en menor medida D) lo
*reducen* al concentrar la máscara en los mismos nodos. La cadena de evidencia
completa: v5 ya mostró que Grover uniforme > Householder ponderada (los pesos
no aportaban); v6 cierra el argumento (la propia caminata no aporta sobre azar).

## Punto 3 — Quantum Annealing (Modelo Q)

Implementación: QUBO de selección top-m sobre H_t
(cohesión α·ΣW_ij·x_i·x_j + afinidad al seed β·Σb_i·x_i + penalización de
cardinalidad) en `src/quantum/annealing_walker.py`, con dos modos:

- **`exact`** (usado en campaña): estado fundamental exacto por enumeración
  (M≤16) = límite adiabático ideal T→∞ = **cota superior** del QA real.
- **`evolve`**: evolución adiabática Trotterizada de tiempo finito (estilo
  qibo adiabatic3sat).

**Hallazgo de fidelidad** (`outputs/tables/annealing_fidelity_v6.csv`): la
penalización de cardinalidad comprime los gaps del sector factible ~(M−m)²
veces (156× a M=8; 924× a M=12) → el annealing de tiempo finito practicable
NO alcanza el fundamental (mismatch 18-20/20). Limitación estructural
relevante para QA real (D-Wave) sobre QUBOs con restricción de cardinalidad.

**Resultados de campaña** (modo exact, 10 semillas pareadas):

| Comparación | cand_hit | p_Bonf | topm_hit | p_Bonf | Sharpe | p_Bonf |
|---|--:|--:|--:|--:|--:|--:|
| Q − C | −0.041 | 0.000 | −0.017 | 0.009 | −0.008 | ns |
| Q − D | −0.060 | 0.000 | −0.022 | 0.000 | −0.011 | ns |
| Q − R | −0.069 | 0.000 | −0.026 | 0.000 | −0.014 | ns |

**Lectura**: la selección QUBO-óptima es la PEOR de todos los selectores en
alineación. Causa: el óptimo es estable en el tiempo (determinista) → la máscara
apenas rota → menor cobertura de activos prometedores. Coherente con el
mecanismo identificado en el punto 2. Como el modo exact es la cota superior
del QA, **el QA real de fidelidad finita tampoco aportaría** → condición del
punto 4 activada.

## Punto 4 — Quantum Reservoir: roadmap

`docs/qr_roadmap.md` contiene el diseño condicional completo:
- QR = extractor de features temporales (no selector) → se integra por la
  interfaz `relational_features_fn` ya existente (introducida para el Modelo B);
  cero cambios en el entorno.
- Reservoir de Ising transverso desordenado de 6-8 qubits, simulación matricial
  (mismo patrón que `dtqw.py`); lectura por observables ⟨Z⟩, ⟨ZZ⟩, ⟨X⟩.
- **Controles obligatorios**: ESN clásico de igual dimensión, features aleatorias
  congeladas, Modelo B (ya medido: B−A nulo — advertencia de techo).
- Estimación: 4-5 días de trabajo + ~2 h de cómputo.

## Artefactos

| Artefacto | Ruta |
|---|---|
| Benchmarks | `scripts/run_classical_benchmarks.py` → `outputs/tables/benchmarks_v6.csv` |
| Modelo R | `src/quantum/random_walker.py` |
| Modelo Q | `src/quantum/annealing_walker.py` (+13 tests) |
| Campañas | `scripts/run_variant_campaign_v6.py` → `variant_{R,Q}_v6.csv` |
| Bootstrap pareado | `scripts/paired_variants_v6.py` → `paired_variants_v6.csv` |
| Fidelidad QA | `scripts/annealing_fidelity_study.py` → `annealing_fidelity_v6.csv` |
| Roadmap QR | `docs/qr_roadmap.md` |

Calidad: 277/277 tests verdes; los 3 módulos nuevos pasaron revisión adversarial
multi-agente (7 agentes; 3 hallazgos major detectados y corregidos antes de
ejecutar las campañas definitivas).
