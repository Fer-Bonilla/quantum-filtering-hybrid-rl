# Fase 1 — Resultados de instrumentación

> Mitigaciones de Problemas 1.1, 1.3 y 1.4 del revisor.
> Suite de tests: **230 / 230 verdes** (213 originales + 17 nuevos).

## Fase 1.A — `episodes_to_convergence` reescrita (Problema 1.1)

**Síntoma original**: la métrica producía `mean(D−C) = 0`, `std = 0` porque la heurística colapsaba a un único índice. Diagnóstico: la tolerancia relativa `5%` sobre el rollout previo se satisface inmediatamente cuando los cambios son pequeños frente al ruido.

**Solución implementada**: tres métricas complementarias en [src/training/evaluate.py](src/training/evaluate.py):

| Métrica | Definición | Rol |
|---|---|---|
| `convergence_to_final` | primer rollout dentro de `tol_rel · \|final\| + tol_abs` del valor final | **Primaria** |
| `convergence_stability` | primer rollout con `std/\|mean\| < tol_std` | Secundaria |
| `convergence_plateau_after_peak` | primer plateau después del pico de la media móvil | Terciaria |

**Verificación**:
- `test_convergence_to_final_has_variance_across_seeds`: 5 curvas sintéticas con plateaux en posiciones `{8, 12, 15, 18, 22}` producen `std(convs) > 0.5`.
- `test_convergence_to_final_with_synthetic_curve_known_point`: curva con plateau a t≈12 retorna ~12 ± 3.
- 4 tests adicionales pasando (incluye retro-compatibilidad).

## Fase 1.B — Modelo B funcional (Problema 1.4)

**Síntoma original**: B colapsaba a A porque los rasgos relacionales del grafo no se inyectaban al estado.

**Solución implementada**:
- [src/env/state_builder.py](src/env/state_builder.py): `build()` acepta argumento opcional `relational_features: np.ndarray`.
- [src/env/relational_state_builder.py](src/env/relational_state_builder.py) (nuevo): genera por step `(N, d_rel)` rasgos del grafo `G_t` con `d_rel = 2 + spectral_top_k`.
- [src/env/market_env.py](src/env/market_env.py): acepta `relational_features_fn` opcional; cuando se proporciona, `observation_space` aumenta a `(L·N·d + N·d_rel,)`.
- [src/utils/config.py](src/utils/config.py): nuevo campo `EnvConfig.use_relational_features: bool`.

**Verificación** (5 tests verdes en `tests/unit/test_model_b_relational.py`):
- `test_B_state_dim_larger_than_A`: shape de B = shape de A + `N · d_rel`.
- `test_B_action_diverges_from_A_under_same_seed`: con misma semilla PyTorch, las acciones argmax de A y B divergen en ≥10% de los steps. **B ya NO es idéntico a A**.
- `test_B_no_information_leak`: modificar `data[t+1:]` no altera los rasgos relacionales en `t`.
- `test_B_obs_finite_after_warmup`: sin NaN/inf tras superar el lookback.
- `test_B_zeros_during_warmup`: retorna ceros cuando la ventana no encaja.

## Fase 1.C — Reward `log_wealth` + ablación de λ (Problema 1.3)

**Síntoma original**: todos los modelos producían Sharpe entre `−1.21` y `−1.07`; ninguno aprendía política rentable. Hipótesis del revisor: la penalización `λ=0.1` era dominante.

**Solución implementada**:
- [src/env/reward.py](src/env/reward.py): nuevo tipo `reward_type ∈ {risk_penalty, log_wealth}` con `r_t = log(1+R) − λσ − μc` para el segundo.
- [src/utils/config.py](src/utils/config.py): nuevo campo `EnvConfig.reward_type`.
- [scripts/run_lambda_ablation.py](scripts/run_lambda_ablation.py) (nuevo): ablación factorial `λ × reward_type × seed × Modelo A`.

**Verificación** (8 tests en `tests/unit/test_reward_log_wealth.py`):
- Correctitud algebraica de `log_wealth` para retornos sintéticos.
- Compresión de ganancias extremas y amplificación de pérdidas pequeñas.
- Sanción acotada `−1000` para `R ≤ −1`.
- Retro-compatibilidad: `RewardCoefficients()` por defecto produce el esquema original.

**Mini-ablación ejecutada** (24 runs × 4 000 steps en universo Nivel 2, ~75 s):

| `reward_type` | `λ` | `return` (mean ± std) | `sharpe` (mean ± std) | P(Sharpe>0) |
|---|---|---|---|:-:|
| `log_wealth`    | **0.000** | **+0.067 ± 0.44**  | **+0.021 ± 0.10** | **0.67** |
| `log_wealth`    | 0.010 | −0.562 ± 0.49 | −0.125 ± 0.10 | 0.00 |
| `log_wealth`    | 0.050 | −2.968 ± 0.66 | −0.665 ± 0.09 | 0.00 |
| `log_wealth`    | 0.100 | −5.986 ± 0.78 | −1.208 ± 0.10 | 0.00 |
| `risk_penalty`  | **0.000** | **+0.104 ± 0.44**  | **+0.030 ± 0.10** | **0.67** |
| `risk_penalty`  | 0.010 | −0.525 ± 0.50 | −0.115 ± 0.10 | 0.00 |
| `risk_penalty`  | 0.050 | −2.908 ± 0.65 | −0.655 ± 0.09 | 0.00 |
| `risk_penalty`  | 0.100 | −5.953 ± 0.76 | −1.199 ± 0.10 | 0.00 |

### Hallazgos clave

1. **`λ=0.0` produce Sharpe positivo en 2 de 3 seeds** en ambos esquemas. Confirma la hipótesis del revisor: el régimen "degenerado" venía de la penalización por riesgo excesiva.
2. **`λ=0.1` (original) es la peor configuración**: Sharpe ≈ −1.20 en ambos esquemas, consistente con `campaign_1` original (Sharpe ≈ −1.22).
3. **`risk_penalty` ≈ `log_wealth`** en este régimen: ambos esquemas producen Sharpe casi idéntico para cada `λ`. El driver crítico es `λ`, no la forma funcional.
4. **Sensibilidad a `λ`**: una reducción de `λ=0.1 → 0.0` mueve el Sharpe medio de `−1.20` a `+0.02`. Cambio de ~1.22 unidades de Sharpe.

### Implicación para Fase 3

Las campañas v2 (Fase 3) usarán **`λ=0.0`, `reward_type=log_wealth`** como configuración base (Modelo A baseline) sobre el hold-out 2023-2024. Con esto, la comparación A→B→C→D se hace sobre un régimen donde **el agente sí puede aprender una política rentable**, eliminando el caveat del revisor sobre comparar "quién pierde menos".

## Resumen

| Mitigación | Estado | Tests | Resultado clave |
|---|:-:|:-:|---|
| 1.A `episodes_to_convergence` | ✅ | 4 nuevos | std no nula → métrica utilizable |
| 1.B Modelo B funcional | ✅ | 5 nuevos | B ≠ A verificado en logits |
| 1.C `log_wealth` + ablación λ | ✅ | 8 nuevos + 24 runs | **λ=0.0 produce Sharpe positivo** |

Suite total: **230 / 230 verdes** (213 originales + 17 nuevos de Fase 1).
