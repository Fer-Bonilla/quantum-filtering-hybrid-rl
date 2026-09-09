# Pre-registro — Campaña complementaria v8

**Proyecto**: Priorización por filtrado cuántico estructurado (TFE, UNIR)
**Fecha de congelación**: 2026-07-05
**Commit de congelación**: (se anota en el informe G1 tras el commit de este archivo)

## 1. Objetivo y contexto declarado

La campaña v8 se diseña para **desafiar** las tres conclusiones centrales de la
memoria: (i) el mecanismo dominante es la rotación de máscara, (ii) la métrica
primaria (`candidate_hit_rate`) capturó ese mecanismo, (iii) el filtrado no
mueve el Sharpe. Se declara explícitamente el conocimiento previo: todos los
agregados de v3/v6/v7 ya publicados en la memoria (R−D cand_hit +0,0095
p=0,088; curva dosis-respuesta; mediación ρ=0,93; oráculo ×27). Ningún dato de
v8 ha sido generado antes de congelar este documento.

## 2. Hipótesis confirmatorias (familia Bonferroni v8, k=6)

α global = 0,05; cada contraste confirmatorio se evalúa a α_c = 0,05/6 ≈ 0,00833.
Todo análisis no listado aquí es **exploratorio**.

| ID | Hipótesis | Test | Margen/dirección | Datos |
|----|-----------|------|------------------|-------|
| H-v8.1 | R y D son equivalentes en `candidate_hit` | TOST pareado (Schuirmann) | ±0,019 | n=40 pares (EXP-3) |
| H-v8.2 | D − S(p*) = 0 en `candidate_hit` (sin valor informacional residual) | TOST pareado | ±0,019 | n=10 pares (EXP-4) |
| H-v8.3a | D − R > 0 en `precision@m` | bootstrap pareado unilateral | >0 | n=10 semillas replay (EXP-1) |
| H-v8.3b | D − R > 0 en `NDCG@m` | bootstrap pareado unilateral | >0 | n=10 semillas replay (EXP-1) |
| H-v8.4 | D − R > 0 en `candidate_hit` sobre subgrafos regulares (agregado de topologías) | bootstrap pareado unilateral | >0 | n≥10 (EXP-5, Tier 3) |
| H-v8.5 | Sharpe(soft-D) − Sharpe(soft-R) > 0 | bootstrap pareado unilateral | >0 | n=10 (EXP-6, Tier 3) |

**Justificación del margen ±0,019**: es el MDE del TOST v7 con n=10 (Anexo E de
la memoria) y coincide con el tamaño del efecto D−C confirmado (+0,019): una
diferencia R−D menor que el propio efecto de interés se considera irrelevante
en la práctica. Fijado antes de ver datos v8.

## 3. Análisis de potencia (H-v8.1)

Con los pares por semilla existentes (`variant_R_v6` − `campaign_1_v3` D,
semillas {7, 42, 99, 123, 314, 456, 789, 1024, 1729, 65535}):

- diferencia media observada R−D = +0,00952
- **sd pareada = 0,01952**

n para potencia ≥0,90 del TOST (α=0,05, margen ±0,019):
- bajo δ verdadero = 0: n = (z_0,95 + z_0,95)²·sd²/0,019² = **11,4 → 12**
- bajo δ verdadero = +0,0095 (estimación puntual observada, caso conservador):
  n = (z_0,95 + z_0,90)²·sd²/(0,019−0,0095)² = **36,3 → 37**

**Decisión pre-registrada: n_total = 40 pares** (10 existentes + 30 nuevos),
que da potencia ≈0,92 en el caso conservador y >0,99 bajo δ=0. Las 10 corridas
existentes de R y D se **reutilizan** (misma configuración Tabla 5.4); se
declaran aquí conforme exige la especificación.

**Semillas nuevas (manifiesto para Tabla C.1)**, 30 primos disjuntos de las
existentes:
`11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139`

## 4. Definiciones operativas (EXP-1, EXP-2)

Sea `promising[t, i]` la matriz booleana del Anexo B.4 (horizonte h=5,
percentil 80 sobre el score g_i^{(h)}(t) = media/(sd+ε) de los retornos
t+1…t+h) y `g[t, i]` el score continuo subyacente. Ambos se computan con el
código existente (`compute_promising_matrix`) sin modificación.

- **`precision@m`**: por paso, |máscara ∩ prometedores_t| / m_eff, con m_eff el
  tamaño real de la máscara (m=3 salvo fallback). Se agrega por semilla como
  media sobre los pasos evaluables (t con horizonte disponible), y los
  contrastes son pareados por semilla. Pasos con fallback all-True se excluyen.
- **`NDCG@m`**: relevancia rel_i(t) = max(g[t,i], 0) sobre los nodos de H_t.
  DCG@m = Σ_{j=1..m} rel(item_j)/log2(j+1) sobre el ranking inducido del
  selector; IDCG@m ídem sobre el orden ideal de los nodos de H_t por rel.
  Pasos con IDCG=0 se excluyen. **Fuentes de ranking**: D → P_k de la DTQW
  (`apply_dtqw`, mismo backend/config); C → p_k de la caminata clásica
  (`random_walk_distribution`); R → permutación uniforme de H_t generada con un
  RNG propio e independiente (null de ranking). Q y S(p) carecen de ranking
  intrínseco y se **excluyen** del NDCG (solo exploratorio si se reporta).
- **`candidate_hit` estratificado** (exploratorio): cada paso con máscara
  propia consecutiva se asigna al tercil (bajo/medio/alto) de la distancia de
  Jaccard con la máscara anterior, calculando cand_hit por estrato y selector.
- **Información mutua (EXP-2, exploratorio)**: sobre los pares (paso, activo)
  agrupados: X = 1[i ∈ máscara_t], Y = 1[i ∈ prometedores_t]. Estimador
  plug-in con corrección de Miller–Madow. Umbral de significancia por
  permutación: ≥1 000 réplicas donde la matriz `promising` se desplaza
  circularmente en el tiempo con offset uniforme en [h+1, T−h−1] (preserva la
  estructura marginal de Y y rompe la asociación temporal). IC por bootstrap
  sobre semillas.

**Modo de obtención de máscaras**: las campañas v3/v6 NO registraron máscaras
por paso en MLflow. Se usa el **modo replay** validado en v7 (mediación): la
secuencia de subgrafos H_t no depende de las acciones (candidate_mask
all-True) y el replay con política ficticia reprodujo el cand_hit de v6 con
d=0,000 en los cuatro selectores. Cada corrida v8 se registra en MLflow o CSV
con etiqueta `campaign=v8_<exp_id>`.

## 5. Calibración de p* (EXP-4)

1. La rotación Jaccard realizada de D se mide por semilla con el replay
   (misma maquinaria del punto 4).
2. La curva p → rotación realizada del StickyWalker medida en v7 sobre la
   MISMA secuencia de evaluación (`mediation_rotation_v7.csv`: p∈{0, 0,1,
   0,25, 0,5, 0,75, 1} → rot∈{0,226; 0,325; 0,445; 0,606; 0,719; 0,803}) se
   invierte por interpolación lineal para obtener p* tal que
   rot(S(p*)) = rot(D).
3. **Regla global/por-semilla**: si la sd entre semillas de rot(D) ≤ 0,05 se
   usa un p* global; si supera 0,05, p* por semilla.
4. S(p*) se entrena con las 10 semillas pareadas de `campaign_1_v3`
   (configuración Tabla 5.4; las corridas D existentes se reutilizan como brazo
   D).
5. **Control de manipulación**: |rot(D) − rot(S(p*))| < 0,03 medido sobre el
   replay de evaluación; si falla, se recalibra una vez y se documenta.

## 6. Presupuesto de cómputo estimado (CPU local)

| Bloque | Corridas | Coste unitario | Total |
|---|---|---|---|
| EXP-3 (R y D × 30 semillas nuevas) | 60 | ≈80 s | ≈1,4 h |
| EXP-4 (S(p*) × 10 semillas) | 10 | ≈65 s | ≈11 min |
| Tier 1 (replay, sin entrenamiento) | — | — | ≈10-15 min |

Dentro del presupuesto disponible; no se requiere escalado.

## 7. Criterios de decisión por compuerta (resumen)

- **G1** (Tier 1+2): H-v8.1 equivalente **y** H-v8.2 equivalente **y** H-v8.3
  no significativa ⇒ propuesta Escenario A. Cualquier rechazo direccional a
  favor de D ⇒ propuesta Escenario B. TOST falla a favor de R ⇒ Escenario A
  reforzado.
- **G2** (EXP-5): H-v8.4 rechaza H0 ⇒ Escenario C; si no ⇒ A ampliado.
- **G3** (EXP-6/7): H-v8.5 rechaza ⇒ Escenario C (canal); selector informado+
  rotación > R (FDR) ⇒ paquete D.

La Fase 2 (edición de la memoria) solo se ejecuta tras confirmación humana del
escenario en cada compuerta.
