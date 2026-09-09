# Roadmap: Quantum Reservoir Computing como módulo cuántico (v6 — Punto 4 del director)

**Estado**: diseño condicional. Se activa SOLO si el Modelo Q (Quantum Annealing,
punto 3) tampoco aporta ventaja medible sobre el Modelo C en la métrica
primaria. Documento de planificación, sin implementación.

---

## 1. Condición de activación

El punto 4 del director establece una cadena de decisión:

```
DTQW (Modelo D)  --sin ventaja-->  QA (Modelo Q)  --sin ventaja-->  QR (rediseño)
```

Criterio operativo (consistente con el pre-registro): el Modelo Q se considera
"sin ventaja" si, tras bootstrap pareado con n=10 semillas y Bonferroni k=11,
ninguna de las métricas de hipótesis (`candidate_hit_rate`, `topm_hit_rate`,
`sharpe_ratio`, `episodes_to_convergence`) muestra `p_Bonferroni < 0.05` a favor
de Q frente a C **ni** frente a R (máscara aleatoria).

## 2. Por qué QR cambia la metodología (no solo el módulo)

Los módulos actuales (DTQW, QA) son **selectores**: reciben el subgrafo H_t y
devuelven un conjunto candidato top-m que actúa como máscara del PPO. Un
Quantum Reservoir es un **extractor de características temporales**: recibe una
serie temporal y devuelve un embedding no lineal de alta dimensión. Esto obliga
a re-cablear el pipeline:

| Componente | Hoy (D/Q) | Con QR |
|---|---|---|
| Entrada al módulo cuántico | Subgrafo estático H_t | Ventana temporal de retornos (secuencia) |
| Salida | Máscara top-m (discreta) | Vector de features (continuo, dim ~2^n_qubits o n_observables) |
| Integración con PPO | Restricción del espacio de acción | **Aumento del estado** (como Modelo B, pero con features cuánticas) |
| Papel del grafo G_t | Define H_t | Opcional: define acoplamientos del reservoir |

Consecuencia: el comparador natural de un Modelo QR no es C (caminata clásica)
sino **B** (estado aumentado clásico) y un **Echo State Network clásico** (ESN,
el reservoir clásico estándar) con el mismo número de features de salida.

## 3. Diseño propuesto (mínimo viable)

### 3.1 Reservoir

- **Sistema**: n_q = 6-8 qubits con Hamiltoniano de Ising transverso
  desordenado: `H = Σ J_ij Z_i Z_j + h Σ X_i`, con `J_ij` fijos aleatorios
  (o derivados de la matriz de afinidad media del universo — conexión natural
  con el grafo existente).
- **Inyección de datos**: en cada paso temporal, codificar el vector de
  retornos del día (proyectado a n_q dims) como rotaciones RY sobre los qubits
  de entrada; evolucionar `e^{-iHτ}`; los qubits restantes retienen memoria.
- **Lectura**: valores esperados `<Z_i>, <Z_i Z_j>, <X_i>` → vector de
  features de dim O(n_q²) por paso. Solo la capa de lectura se entrena
  (regresión lineal o directamente como input extra del PPO).
- **Simulación**: matricial densa (2^8 = 256 amplitudes), mismo patrón de
  implementación que `dtqw.py` y `annealing_walker.py`. Coste por paso ~ms.

### 3.2 Integración (Modelo E propuesto)

- `MarketEnv` ya soporta `relational_features_fn(t) -> (N, d)` (introducido
  para el Modelo B v2). El QR se integra por la MISMA interfaz:
  `qr_features_fn(t) -> (1, d_qr)` con el embedding del reservoir hasta t.
  Cero cambios en el entorno.
- Sin fuga temporal por construcción: el reservoir solo consume retornos
  `data[:t]` (estado recurrente causal).

### 3.3 Controles experimentales obligatorios

1. **ESN clásico** con idéntica dimensión de lectura (aísla "reservoir" de
   "cuántico").
2. **Modelo B** (features relacionales clásicas) — ya medido: B−A nulo.
3. **Features aleatorias congeladas** de la misma dimensión (control de
   "proyección aleatoria no lineal").
4. Pre-registro de la métrica primaria ANTES de correr (consistente con la
   disciplina del trabajo): la hipótesis natural es sobre `sharpe_ratio` y
   `episodes_to_convergence`, porque un QR no produce máscara y no afecta a
   `candidate_hit_rate`.

## 4. Estimación de esfuerzo

| Tarea | Estimación |
|---|---|
| `src/quantum/reservoir.py` + tests (sim. matricial Ising) | 1-2 días |
| Integración vía `relational_features_fn` + config | 0.5 días |
| Controles (ESN clásico, features random) | 1 día |
| Campaña n=10 × 50k + ESN + random-features + bootstrap | ~2 h cómputo |
| Análisis + LaTeX | 1 día |
| **Total** | **~4-5 días de trabajo** |

## 5. Riesgos

- **Riesgo principal**: el hallazgo B−A ≈ 0 (v5) sugiere que *aumentar el
  estado* no mueve la aguja en este universo/régimen — un QR podría heredar
  ese techo. El control ESN es crítico para no atribuir al "quantum" lo que
  sea genérico del reservoir.
- La literatura de QRC en finanzas reporta ventajas principalmente en
  predicción de series, no en RL end-to-end; la traducción a mejora de
  política no está garantizada.

## 6. Referencias de partida

- Fujii & Nakajima (2017), *Harnessing disordered-ensemble quantum dynamics
  for machine learning*. Phys. Rev. Applied.
- Mujal et al. (2021), *Opportunities in quantum reservoir computing and
  extreme learning machines*. Adv. Quantum Technol.
- Yasuda et al. (2023), *Quantum reservoir computing with repeated
  measurements on superconducting devices*.
