# El proyecto como seis iteraciones: propósito, experimentos y decisión de cada una

Guía de contexto para el agente editor del documento final (sin acceso al
repositorio). Reorganiza el trabajo experimental como **iteraciones guiadas
por una pregunta**, no como versiones de software. Cada iteración sigue el
mismo esqueleto:

1. **Pregunta que la motiva** (qué no sabíamos al empezar).
2. **Qué se decidió medir** (experimentos y controles añadidos).
3. **Qué se encontró.**
4. **Decisión tomada** (qué cambia en la tesis).
5. **Pregunta que abre la siguiente iteración.**

Las etiquetas v1…v8 que aparecen en nombres de campañas, tablas y figuras
son solo el índice cronológico de las rondas de trabajo; la tabla final las
mapea a estas iteraciones. El código es único (etiqueta `v8-final`); lo que
cambia entre iteraciones es el diseño experimental.

---

## Iteración 1 — Construir un banco de pruebas que mida lo que dice medir

**Pregunta.** ¿Puede compararse de forma limpia un filtrado clásico y uno
cuántico dentro de un agente de refuerzo, sobre los mismos datos, la misma
recompensa y la misma interfaz?

**Qué se decidió medir.** La cadena de ablación A→B→C→D (PPO básico; + rasgos
relacionales del grafo; + filtrado clásico por caminata sobre subgrafos; +
filtrado cuántico por caminata de tiempo discreto) sobre 30 activos, con
métricas financieras (Sharpe, retorno), de aprendizaje (convergencia) y de
alineación del conjunto candidato con activos prometedores ex-post
(`candidate_hit`, `topm_hit`). Primera campaña con 5 semillas, barrido de
configuraciones y una configuración alternativa con subgrafo grande
(*sweet spot*, M=16).

**Qué se encontró.** El banco de pruebas tenía tres defectos: la penalización
de riesgo hacía negativo el Sharpe de todos los modelos, el Modelo B era
idéntico a A (los rasgos relacionales no llegaban al estado) y la métrica de
convergencia no variaba entre semillas. Direccionalmente D superaba a C en
alineación, pero sin control de multiplicidad.

**Decisión.** Corregir antes de interpretar: recompensa de riqueza
logarítmica sin penalización (identificada por ablación de λ), Modelo B
funcional, métrica de convergencia rediseñada, bootstrap pareado por semilla
con corrección de Bonferroni sobre once métricas, y bloques de *tuning*
(2018–2022) y *hold-out* (2023–2024) separados. Con el banco corregido, D–C
solo sobrevive a Bonferroni en `candidate_hit` (+0,02); Sharpe y `topm_hit`
no.

**Abre la pregunta.** ¿Ese efecto es real y robusto, o un artefacto de cinco
semillas y de un periodo?

*(Rondas v1–v2; campañas `campaign_1`, `abl_lambda_mini`, `campaign_1_v2`,
`sweet_spot_tuning_v2`, `sweet_spot_holdout_v2`.)*

## Iteración 2 — Establecer el efecto y fijar el veredicto de las hipótesis

**Pregunta.** ¿Se sostiene D > C al duplicar las semillas, bajo ruido de
hardware y en distintos regímenes de mercado? ¿Y se traduce en Sharpe?

**Qué se decidió medir.** Campaña principal ampliada a 10 semillas (la que
usa la memoria: `campaign_1_v3`); ablación de ruido NISQ con cuatro
perfiles; validación temporal *walk-forward* por pliegues anuales
2021–2024; ablación de la moneda cuántica (Householder ponderada, Grover,
Fourier); replicación del hold-out a 10 semillas; contraste B–A completo;
pre-registro formal del criterio confirmatorio.

**Qué se encontró.** El efecto en `candidate_hit` (+0,019) se mantiene con
intervalo más estrecho y es prácticamente invariante al ruido. La ventaja en
Sharpe aparece solo en 2023–2024 y no en la campaña principal: la afirmación
inicial de "H2 confirmada, dependiente del régimen" se retira como HARKing.
Los rasgos relacionales sin filtrado (B–A) no aportan nada. Y una pista
inesperada: la moneda Grover *uniforme* iguala o supera a la ponderada por
afinidad, así que la información financiera codificada en los pesos no
parece ser lo que produce el efecto.

**Decisión.** Veredicto que la memoria mantiene desde entonces: H1
(alineación D > C) confirmada; H2 (Sharpe) no confirmada bajo el criterio
pre-registrado, con el efecto 2023–2024 como evidencia post-hoc; H3
(convergencia) significativa pero marginal y, más tarde, solo descriptiva
por censura.

**Abre la pregunta.** Si los pesos no importan, ¿de dónde sale el efecto?
¿Es siquiera cuántico?

*(Rondas v3–v5; `campaign_1_v3`, `abl_nisq_v3`, `walk_forward_v3/v4`,
`abl_coin_v3`, `sweet_spot_holdout_v3`, `abl_nisq_v5`.)*

## Iteración 3 — Atribuir el mecanismo con controles no cuánticos

**Pregunta.** ¿La mejora de alineación viene de la caminata cuántica o de la
propia formulación (filtrar con una máscara pequeña que cambia con el
tiempo)? ¿Y compite el sistema con estrategias clásicas simples?

**Qué se decidió medir.** Tres controles sobre la misma interfaz: el
**Modelo R** (máscara aleatoria estructurada del mismo tamaño m), el
**Modelo Q** (selección determinista por *Quantum Annealing* formulado como
QUBO, evaluada incluso en su límite adiabático ideal) y un banco de
*benchmarks* clásicos (1/N, momentum, Markowitz, política aleatoria, oráculo
ex-post).

**Qué se encontró.** El resultado que cambia la tesis: la máscara
**aleatoria** iguala o supera a la DTQW (R–C = +0,029 frente a D–C =
+0,019; R–D nulo) y converge antes; el orden de selectores es R > D > C > Q,
es decir, cuanto más informado y estable es el selector, peor alineación; Q
es el peor incluso en el límite ideal. Ningún agente RL bate en Sharpe a 1/N
ni a momentum; el oráculo ex-post marca un techo enorme.

**Decisión.** H1 se mantiene como efecto real pero **se reinterpreta**: el
mecanismo dominante es la **rotación de la máscara**, no una propiedad
cuántica. La contribución de la tesis pasa a ser metodológica (cómo aislar y
refutar un módulo cuántico en RL financiero) y el título cambia de
"exploración" a "filtrado".

**Abre la pregunta.** La atribución es una inferencia observacional
(gradiente R > D > C > Q). ¿Puede demostrarse causalmente? ¿Y por qué el
canal selector no mueve el Sharpe?

*(Ronda v6; `variant_R_v6`, `variant_Q_v6`, `benchmarks_v6`,
`annealing_fidelity_v6`.)*

## Iteración 4 — Demostrar causalmente el mecanismo y acotar el techo del canal

**Pregunta.** Si se manipula *solo* la rotación, con la información fija en
cero, ¿responde la alineación? ¿Explica la rotación las diferencias entre
todos los selectores? ¿Es formal la equivalencia cuántico ≈ azar y se
sostiene fuera de muestra? ¿Qué podría lograr el canal con una máscara
perfecta?

**Qué se decidió medir.** Cinco experimentos: dosis-respuesta con el
selector S(p) (mantiene la máscara y re-sortea cada posición con
probabilidad p; 6 valores × 10 semillas); mediación por replay de las
máscaras de todos los selectores sobre la misma secuencia de evaluación;
prueba de equivalencia (TOST) R–D; validez externa por regímenes anuales
2021–2024 con A, D, R y *benchmarks*; y la máscara-oráculo (PPO entrenado
con la máscara top-m del retorno futuro, fuga ex-post deliberada).

**Qué se encontró.** La alineación crece de forma monótona y saturante con
la rotación (pendiente +0,031, p<0,001); la rotación explica el patrón de
todos los selectores (ρ = 0,93) y la DTQW cae exactamente sobre la curva del
azar; R ≈ D es formal en `topm_hit` y Sharpe (en `candidate_hit` el test
quedó subpotenciado con n=10) y se replica en los cuatro regímenes; con la
máscara perfecta el Sharpe pasa de +0,05 a +1,32, luego la política PPO no
es el cuello de botella: ningún selector real anticipa retornos mejor que
el azar.

**Decisión.** El mecanismo queda demostrado causalmente y cuantificado; se
integra la ablación A→B→C→D como cadena canónica del plan; la ausencia de
mejora en Sharpe se explica por la calidad de selección, no por la
arquitectura.

**Abre la pregunta.** Todo lo anterior son resultados negativos obtenidos
sobre subgrafos irregulares y con máscara dura. ¿Y si la ventaja cuántica
existe pero solo en el régimen donde la teoría la predice (grafos
regulares, dispersión balística) o con una integración más suave? ¿Y tiene
el TOST potencia suficiente en la métrica primaria?

*(Ronda v7; `rotation_sweep_v7`, `mediation_rotation_v7`,
`tost_equivalence_v7`, `regime_validation_v7`, `oracle_v7`.)*

## Iteración 5 — Buscar activamente la ventaja donde debería aparecer (campaña pre-registrada)

**Pregunta.** Si se diseña a propósito el escenario más favorable a la
caminata cuántica y se le da potencia estadística suficiente, ¿aparece
alguna ventaja?

**Qué se decidió medir.** Una campaña complementaria con pre-registro
congelado antes de toda ejecución (seis hipótesis H-v8.1…H-v8.5, familia
Bonferroni k=6, análisis de potencia, 30 semillas nuevas) y tres compuertas
de decisión con confirmación humana antes de tocar la memoria:

- **G1**: métricas desacopladas de la rotación (precision@m, NDCG@m),
  información mutua máscara→futuro, TOST R–D con potencia adecuada, y una
  rotación *calibrada* a la de la DTQW (S(p\*)) con control de manipulación.
- **G2**: cuatro topologías de regularidad controlada (3-regular con pesos
  de afinidad, 3-regular uniforme, ciclo C₈, bipartito) con verificación
  empírica de que la DTQW alcanza el régimen balístico.
- **G3**: integración suave (sesgo de logits con β_soft en lugar de máscara
  dura) y selectores informados con rotación.
- Rigor financiero: Deflated Sharpe Ratio y bootstrap estacionario por
  bloques.

**Qué se encontró.** Los desafíos fallan todos en la misma dirección: R y D
son prácticamente equivalentes en la métrica primaria (réplica independiente
en 30 semillas nuevas; la diferencia residual detectable favorece a R); el
ranking de la DTQW es indistinguible de una permutación aleatoria; ninguna
máscara porta información sobre el futuro; la rotación calibrada reproduce
el efecto de la DTQW casi exactamente; en las topologías regulares el
régimen balístico se alcanza y la ventaja no aparece (D queda por debajo de
R); la integración suave no supera a la máscara dura; ningún selector
informado supera al azar; ningún Sharpe sobrevive a la deflación.

**Decisión.** El veredicto pasa de "no hay evidencia de ventaja cuántica" a
"la ventaja se buscó activamente en el régimen teóricamente favorable, con
potencia pre-registrada y controles calibrados, y no apareció". El régimen
regular deja de ser una explicación disponible para el resultado negativo.

**Abre la pregunta.** ¿Puede un tercero reproducir cada número de la
memoria sin consultar el código?

*(Ronda v8; EXP-1…EXP-7 y EXP-11; `tost_extension_v8`, `mask_metrics_v8`,
`mask_information_v8`, `sticky_calibrated_v8`, `regular_topologies_v8`,
`soft_integration_v8`, `informed_walkers_v8`, `financial_stats_v8`.)*

## Iteración 6 — Auditoría de reproducibilidad y cierre de confundidos

**Pregunta.** ¿Hay confundidos no controlados en los contrastes principales,
y está cada resultado especificado hasta el punto de poder recalcularse
desde la memoria?

**Qué se decidió medir.** Los controles que las auditorías técnicas
sucesivas pidieron: la variante **Crel** (estado relacional + filtrado
clásico) que descompone el contraste compuesto B–C en dos de un solo factor;
el brazo D con **condición inicial centrada en la semilla** (la misma que la
caminata clásica) para aislar la dinámica; **sensibilidad a costes de
transacción** realistas (5 y 10 puntos básicos) con re-entrenamiento
completo; **instrumentación del respaldo** de máscara completa; réplicas
independientes de H-v8.1 y H-v8.2 separadas de los pares piloto; DSR
recalculado con **varianza transversal** entre ensayos y sensibilidad en K;
diagnósticos de normalidad, versión no paramétrica de los TOST y
sensibilidad a los márgenes; figuras deterministas con manifiesto de hashes.

**Qué se encontró.** Ningún confundido explica el efecto: con la condición
inicial igualada la brecha D–C no desaparece (aumenta), las máscaras no
dependen del estado (Crel–C = 0 exacto), el respaldo nunca se activa en la
evaluación, y las conclusiones son robustas a fricción realista (con el
matiz de que la paridad de R en Sharpe se erosiona a 10 pb). El DSR con el
procedimiento correcto sigue sin superar 0,95 para 1/N bajo ningún K y para
momentum si K_eff ≥ 6. Las series diarias de los agentes no son
reconstruibles (no se conservaron registros por paso ni pesos), y se
declara así en lugar de estimarlas.

**Decisión.** Cada valor de la memoria queda anclado a una tabla y un script,
con sus supuestos, semillas y límites declarados; lo que no puede
reproducirse se dice explícitamente.

*(Revisiones técnicas 3ª–6ª; `crel_variant_v8`, `review3_checks_v8`,
`fallback_events`, `dsr_recalculated`, `dsr_k_sensitivity`,
`tost_diagnostics`.)*

---

## El hilo en una frase por iteración

| Iteración | Pregunta | Respuesta | Etiquetas de ronda |
|---|---|---|---|
| 1 Banco de pruebas | ¿Mide lo que dice medir? | Tras corregir recompensa, Modelo B y multiplicidad: solo `candidate_hit` D > C sobrevive | v1–v2 |
| 2 Efecto y veredicto | ¿Es real y robusto? ¿Mueve el Sharpe? | Real y robusto en alineación; no en Sharpe (H2 no confirmada); los pesos no importan | v3–v5 |
| 3 Mecanismo | ¿Es cuántico? | No: una máscara aleatoria lo iguala; el mecanismo es la rotación | v6 |
| 4 Causalidad y techo | ¿Se puede demostrar? ¿Por qué no hay Sharpe? | Dosis-respuesta causal; R ≈ D formal y por régimen; el techo existe pero ningún selector real lo aprovecha | v7 |
| 5 Búsqueda activa | ¿Aparece en el régimen favorable? | No, con potencia pre-registrada y régimen balístico verificado | v8 |
| 6 Reproducibilidad | ¿Hay confundidos? ¿Se puede recalcular todo? | Ningún confundido explica el efecto; todo trazable, lo irreproducible declarado | rev. 3ª–6ª |

## Glosario de identificadores que verá el editor

| Patrón | Significado |
|---|---|
| Modelos `A, B, C, D` | PPO básico; + rasgos relacionales; + filtrado clásico; + filtrado cuántico (DTQW) |
| `R`, `Q`, `S(p)`, `S(p*)`, `Crel`, `ORACLE`, `soft-D/C/R` | Controles: máscara aleatoria; QUBO; rotación parametrizada; rotación calibrada a D; estado B + filtrado C; máscara perfecta ex-post; integración suave |
| `campaign_1_v3` | Campaña principal (10 semillas) que usa la memoria |
| `sweet_spot*` | Configuración alternativa M=16 (anexos de sensibilidad) |
| `EXP-1…EXP-11`, `H-v8.1…H-v8.5`, `G1/G2/G3` | Experimentos, hipótesis y compuertas de la campaña pre-registrada (Iteración 5) |
| H1, H2, H3 | Hipótesis del plan original (veredicto fijado en la Iteración 2); no confundir con H-v8.n |

## Nota sugerida para el manuscrito (lista para insertar)

> **Nota sobre la organización del trabajo experimental.** El trabajo se
> desarrolló en seis iteraciones, cada una motivada por una pregunta que la
> anterior dejó abierta: (1) construir un banco de pruebas válido; (2)
> establecer el efecto D > C y fijar el veredicto pre-registrado de las
> hipótesis; (3) atribuir el mecanismo mediante controles no cuánticos
> (máscara aleatoria, QUBO, *benchmarks*); (4) demostrarlo causalmente y
> acotar el techo del canal selector; (5) buscar activamente la ventaja en el
> régimen teóricamente favorable con una campaña pre-registrada; y (6)
> auditar confundidos y reproducibilidad. Las etiquetas v1–v8 que aparecen en
> identificadores de campañas, tablas y figuras indexan las rondas de trabajo
> dentro de esas iteraciones; el código es único (etiqueta `v8-final`) y lo
> que cambia entre rondas es el diseño experimental, no la implementación.
