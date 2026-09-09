# Informe de compuerta G1 — Campaña v8 (Tier 1 + Tier 2)

**Pre-registro**: `docs/preregistro_v8.md`, congelado en commit **`0e3ee5e`**
antes de toda ejecución. Familia Bonferroni k=6 (α_c = 0,05/6 = 0,00833).
**Estado**: Tier 1 (EXP-1, EXP-2) y Tier 2 (EXP-3, EXP-4) completos.
**Cómputo consumido**: 60 corridas EXP-3 (~80 s c/u) + 10 corridas EXP-4 +
replay Tier 1 ≈ 1,8 h CPU, dentro del presupuesto pre-registrado.

---

## Veredictos confirmatorios

| Hipótesis | Resultado | p | Umbral α_c | Veredicto |
|---|---|--:|--:|---|
| **H-v8.1** TOST R−D en `candidate_hit` (±0,019, n=40) | diff +0,0098, IC90 [+0,0059, +0,0138] | p_TOST = **0,00017** | 0,00833 | **EQUIVALENTES** ✅ |
| **H-v8.2** TOST D−S(p*) en `candidate_hit` (±0,019, n=10) | diff +0,0024, IC90 [−0,0095, +0,0143] | p_TOST = 0,0154 | 0,00833 | Equivalente a α=0,05; **no concluye al umbral corregido** (potencia n=10) |
| **H-v8.3a** D−R en `precision@m` (unilateral D>R) | d = −0,0025, IC95 [−0,0070, +0,0009] | p = 0,905 | 0,00833 | **No significativa** (dirección favorece a R) |
| **H-v8.3b** D−R en `NDCG@m` (unilateral D>R) | d = −0,0006, IC95 [−0,0065, +0,0042] | p = 0,571 | 0,00833 | **No significativa** |

Réplicas secundarias de H-v8.1 (n=40): `topm_hit` p_TOST=0,00034 (±0,010) y
Sharpe p_TOST=0,00019 (±0,020) — **equivalentes** ambas al umbral corregido.

## Resultados de soporte

1. **EXP-1** (replay validado — reproduce v6 con d=0,000 en los 4 selectores):
   la DTQW no supera al azar ni en precisión del conjunto (`precision@m`:
   R 0,186 > D 0,184 > C 0,175 > Q 0,158) ni en calidad de ranking
   (`NDCG@m`: D 0,4091 vs permutación aleatoria 0,4096 — **el ranking inducido
   por P_k es indistinguible de un ranking al azar**). D−C sí replica en
   precisión (+0,0083, p<0,001), coherente con el efecto conocido.
2. **EXP-2** (información mutua): **ninguna máscara porta señal** sobre la
   pertenencia futura al quintil prometedor. MI ≈ 10⁻⁴ nats en los 8
   selectores; ninguna significativa frente al null de permutación (todas
   p_perm ≥ 0,06); la MI de D (0,000096) queda por debajo de su propio null
   (0,00018). I_D ≈ I_R: la equivalencia queda explicada a nivel de señal.
3. **EXP-4 — control de manipulación**: rot(D)=0,4530 (sd 0,0068 → regla
   global), p* = 0,2619; rot(S(p*)) medida en replay = 0,4565;
   |Δ| = **0,0034 < 0,03** ✅. S(p*) alcanza cand_hit 0,446 vs D 0,449.
4. **Estratificación por rotación** (exploratorio): dentro de cada selector,
   el tercil de mayor rotación tiende al mayor cand_hit (D: 0,42/0,41/0,52),
   coherente con el mecanismo.
5. **Dirección de H-v8.1**: el IC90 de R−D excluye el cero (+0,0059 > 0):
   además de equivalentes dentro del margen pre-registrado, la diferencia
   direccional favorece a R (p₂ = 0,008 exploratorio) — el azar puro
   (rotación 0,81) queda ligeramente por ENCIMA de la DTQW (rotación 0,45),
   exactamente lo que la curva dosis-respuesta predice.

## Figuras

- `outputs/figures/v8/fig_g1_tost.png` — TOSTs con IC90 vs márgenes (los 4
  contrastes caen íntegramente dentro del margen).
- `outputs/figures/v8/fig_g1_metricas.png` — precision@m / NDCG@m / MI por
  selector.

## Artefactos y trazabilidad

| Artefacto | Ruta |
|---|---|
| Pre-registro congelado | `docs/preregistro_v8.md` (commit `0e3ee5e`) |
| EXP-1 métricas + estratos | `outputs/tables/mask_metrics_v8{,_strata}.csv` |
| EXP-1 máscaras por paso | `outputs/tables/mask_steps_v8.npz` |
| EXP-2 información mutua | `outputs/tables/mask_information_v8.csv` |
| EXP-3 (30 semillas nuevas × R/D) | `outputs/tables/tost_extension_v8.csv` (MLflow `campaign=v8_exp3`) |
| EXP-4 S(p*) | `outputs/tables/sticky_calibrated_v8.csv` (MLflow `campaign=v8_exp4`) |
| Scripts | `scripts/run_{tost_extension,mask_metrics,mask_information,sticky_calibrated}_v8.py`, `analyze_v8_g1.py`, `check_manipulation_v8.py`, `v8_g1_figures.py` |

---

## Propuesta de escenario: **A — veredicto reforzado**

Los cuatro desafíos de la compuerta apuntan en la misma dirección:

- La equivalencia R≈D en la métrica primaria pasa de "subpotenciada" (v7,
  n=10, MDE≈margen) a **formalmente demostrada con potencia pre-registrada**
  (n=40, p_TOST=0,00017), y replica en topm y Sharpe.
- Las métricas desacopladas de la rotación (precisión, ranking) **no revelan
  señal cuántica no capturada** (H-v8.3a/b n.s.; si algo, R≥D).
- El contenido informacional de todas las máscaras es **nulo** (EXP-2).
- La rotación calibrada S(p*) reproduce el efecto de D (diff +0,0024, IC90
  dentro del margen, manipulación verificada) — el argumento de mediación más
  fuerte disponible, con la única reserva de potencia (n=10 no alcanza el
  umbral corregido; sí el nominal).

**Matiz honesto para la memoria**: H-v8.2 debe reportarse como "equivalencia
nominal (α=0,05) con IC dentro del margen; no concluyente al umbral corregido
por n=10". Si se desea cerrar también formalmente, bastaría ampliar S(p*) a
~25-30 semillas (~30 min de cómputo) antes de la Fase 2.

## Qué sigue (requiere tu confirmación)

1. **Confirmar Escenario A** para la Fase 2 (paquete A de ediciones a la
   memoria: A1-A5 + Anexo H), o
2. Pedir la ampliación de EXP-4 a n=30 antes de confirmar, o
3. Continuar con Tier 3 (EXP-5 subgrafos regulares → compuerta G2; EXP-6
   integración suave + EXP-7 informados-con-rotación → G3) antes de editar.

Conforme al pre-registro, **no se edita la memoria hasta recibir confirmación
humana del escenario aplicable**.
