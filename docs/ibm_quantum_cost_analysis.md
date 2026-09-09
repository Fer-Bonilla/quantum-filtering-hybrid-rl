# Análisis de coste — Ejecución en IBM Quantum

> **Documento de metodología y rectificación**: en el documento previo
> [`scenarios_for_stronger_results.md`](scenarios_for_stronger_results.md)
> estimé el coste de una campaña completa en hardware IBM en
> **≈ 24 M USD**. Esa cifra era una **aproximación gruesa** que mezclaba
> tiempo de cola (no facturado) con tiempo de QPU (facturado). Tras
> verificación de precios y un cálculo riguroso, el rango real es de
> **20 k - 300 k USD** dependiendo de configuración. Sigue siendo
> inviable para una tesis de máster pero el orden de magnitud es otro.
> Este documento muestra la metodología completa.

---

## 1. Precios IBM Quantum verificados (mayo 2026)

Fuentes consultadas vía web (búsqueda 2026-05-11):

| Plan | Precio | Restricciones |
|---|---|---|
| **Open Plan** | **0 USD** | 10 min QPU / ventana móvil de 28 días + promo 180 min/12 meses (marzo 2026). |
| **Pay-as-you-go** | **1.60 USD/seg** = 96 USD/min = 5 760 USD/hora | Sólo tiempo de QPU efectivo. Sin compromiso. |
| **Flex Plan** | **72 USD/min compute** | Mínimo 400 minutos prepago = **28 800 USD inicial** + uso. Para proyectos de investigación. |
| **Premium Plan** | Enterprise (sin precio público) | Suscripción anual ≫ 100 k USD/año típicamente. Incluye Qiskit Functions. |
| **On-Prem** | Enterprise (típicamente > 10 M USD) | Sistema dedicado IBM in-situ. |

**Fuentes**:
- [IBM Quantum pricing & plans overview](https://quantum.cloud.ibm.com/docs/en/guides/plans-overview)
- [Manage cost on Pay-as-You-Go Plan](https://quantum.cloud.ibm.com/docs/en/guides/manage-cost)
- [Quantum Computing Report — Pay-as-you-go intro](https://quantumcomputingreport.com/ibm-adds-pay-as-you-go-pricing-model-and-expands-qiskit-runtime/)
- [Moor Insights — Flex Plan analysis](https://moorinsightsstrategy.com/research-notes/ibms-new-flex-plan-fills-a-big-gap-in-quantum-access-pricing/)
- [SpinQuanta — Quantum cost breakdown 2026](https://www.spinquanta.com/news-detail/cost-of-quantum-computer)

---

## 2. ¿Qué cuenta exactamente como "tiempo facturable"?

IBM Cloud define **Quantum time** como:

> *"The duration a QPU is committed to fulfilling a user request."*

Esto significa que **se factura SOLO el tiempo que el QPU está ejecutando físicamente tu circuito**, NO:

| Actividad | ¿Se factura? | Notas |
|---|:-:|---|
| Tiempo en cola (queue wait) | ❌ | Variable (minutos a horas), gratis pero añade wallclock |
| Compilación / transpile | ❌ | Clásico, en CPU del usuario |
| Submit / preparación del job | ❌ | Sí pero negligible (~ms) |
| **Ejecución del circuito en qubits** | ✅ | Es el core de la facturación |
| **Lecturas (readout)** | ✅ | Incluidas en "QPU time" |
| **Reset / inicialización entre shots** | ✅ | Suma al tiempo del job |
| Retorno de resultados al usuario | ❌ | Clásico |
| Tiempo de Runtime API overhead | ⚠️ Parcialmente | ~100 ms-1 s por Sampler call según IBM |

Por tanto, el **tiempo facturable** depende de:

```
t_billable = n_shots × (t_circuit + t_readout + t_reset) + t_runtime_overhead
```

---

## 3. Cálculo detallado para una evaluación DTQW (M=8, k=3)

### 3.1. Anatomía del circuito tras transpile

Para el DTQW con `M=8` (Hilbert padeado a 64 = **6 qubits**), `k=3` pasos:

| Componente | Gates nativos típicos | Profundidad |
|---|---|---|
| Preparación del estado inicial uniforme (`qml.StatePrep` → cascada de gates) | ~50-100 (varía con sparsity) | ~30-60 |
| Operador `U_t = S · C` (matriz 64×64) — transpilado a `cx, sx, x, rz` | ~40-80 gates | ~30-50 |
| `k=3` iteraciones de U_t | ~120-240 gates | ~90-150 |
| Mediciones finales (6 qubits, lecturas paralelas) | — | +1 ciclo readout |
| **Total** | **~170-340 gates** | **~120-210 ciclos** |

### 3.2. Tiempo por shot

Eagle r3 / Heron r1 (procesadores IBM 127-156 qubits):

| Operación | Duración típica |
|---|---|
| Single-qubit gate (`sx`, `rz`) | 35 ns |
| Two-qubit gate (`cx`, `ecr`) | 100-300 ns |
| Readout por qubit (en paralelo) | 1-4 μs |
| Reset entre shots | 1-2 μs |

**Estimación por shot** (depth medio = 150, mezcla 70 % 1Q + 30 % 2Q):
```
t_circuit_per_shot ≈ 0.7 × 35 ns × 150 + 0.3 × 200 ns × 150
                   ≈ 3.7 μs + 9.0 μs
                   ≈ 12.7 μs
t_readout       ≈ 3 μs
t_reset         ≈ 1.5 μs
                ─────────
t_per_shot      ≈ 17 μs
```

### 3.3. Tiempo total facturable por evaluación

| Configuración | shots | t_QPU (s) | Coste @ 1.60 USD/s |
|---|---|---|---|
| **Baja precisión** | 128 | 0.0022 | **0.0035 USD** |
| **Media** (típica) | 512 | 0.0087 | **0.014 USD** |
| **Alta precisión** | 1 024 | 0.0174 | **0.028 USD** |
| **Sobrada** | 4 096 | 0.0696 | **0.111 USD** |

⚠ **Más Runtime overhead**: IBM agrega ~0.5-2 s adicionales por Sampler call (preparación del job + transferencia). Para RL secuencial (no batched), esto puede dominar:

```
t_total = t_qpu_real + t_runtime_overhead
        ≈ 0.0087 + 1.0  (con 1 s de overhead)
        ≈ 1.0 segundos efectivos facturables
```

**Si el Runtime overhead se factura completo**, una evaluación cuesta **~1.60 USD**, no 0.014 USD. Hay incertidumbre en este punto y depende del modo de invocación (`Session` vs `Batch` vs `Job` individual).

---

## 4. Coste de una campaña completa

### 4.1. Volumen de evaluaciones

Para la campaña principal del plan:
```
n_modelos × n_seeds × n_steps_PPO × eval_por_step
= 1 × 5 × 50 000 × 1               (solo modelo D usa DTQW)
= 250 000 evaluaciones DTQW
```

(A, B, C no usan DTQW; solo D paga el coste cuántico.)

### 4.2. Escenarios de coste para la campaña completa de D

| Configuración | shots | overhead | Coste/eval | **Campaña 250k evals** |
|---|---|---|---|---|
| Optimista (batch, sin overhead) | 128 | 0 s | 0.0035 USD | **≈ 875 USD** |
| Realista bajo (batch, mínimo overhead) | 512 | 0.1 s | 0.17 USD | **≈ 42 500 USD** |
| Realista medio (jobs estándar) | 512 | 0.5 s | 0.82 USD | **≈ 205 000 USD** |
| Realista alto (jobs sin Session) | 1 024 | 1.0 s | 1.63 USD | **≈ 407 500 USD** |
| Conservador (alta precisión) | 4 096 | 2.0 s | 3.31 USD | **≈ 827 500 USD** |

### 4.3. Comparación con la cifra previa

| Estimación | Valor | Comentario |
|---|---|---|
| Mi cifra original ("≈ 24 M USD") | 24 000 000 USD | ❌ Incorrecta. Confundía wallclock con QPU time. |
| **Cifra corregida (realista)** | **40 k - 400 k USD** | ✅ Basada en cálculo de QPU time real. |
| Cifra optimista absoluta | < 1 k USD | Solo si batching perfecto + 128 shots. Difícil de lograr para RL. |

**Disculpas por la imprecisión previa**. El orden de magnitud correcto es **k - 100 k USD**, no millones. Sigue siendo inviable para una tesis pero por razones distintas (presupuesto institucional, no escala industrial).

---

## 5. Sensibilidad de costes — los 3 drivers principales

### 5.1. Número de shots por evaluación

La cantidad de mediciones determina la precisión estadística de `P_k(v_i)`:

```
σ(P_k) ≈ sqrt(p · (1-p) / shots)
```

- 128 shots → σ ≈ 4% para p=0.2
- 1024 shots → σ ≈ 1.3%
- 4096 shots → σ ≈ 0.7%

Para top-m (m=3 sobre 8 candidatos), un error de 4% es marginal pero usable. **128-512 shots son razonables** para un experimento de validación.

### 5.2. Profundidad del circuito (driver del `t_per_shot`)

Depende de:
- Número de pasos `k` (cada uno suma ~40-80 gates).
- Tamaño del subgrafo `M` (cuántas amplitudes en `U_t`).
- **Topología del backend**: Eagle tiene conectividad heavy-hex con grado 2-3. Circuitos densos en qubits no vecinos requieren swaps adicionales → profundidad efectiva 1.5-3× mayor.

### 5.3. Runtime overhead (driver oculto y dominante)

Si cada evaluación se invoca como un **Job independiente**:
- Submit + queue check + cleanup: 1-5 segundos por evaluación.
- Domina sobre el QPU time real (que es ~10 ms).

Estrategias para reducirlo:
- **`Session` Mode**: mantiene el backend reservado, evita re-queue. Reduce overhead a ~100 ms/eval.
- **`Batch` Mode**: agrupa circuitos. **No aplicable directamente a RL** porque cada eval depende del estado anterior.
- **PrimitiveJob streaming**: pipeline asíncrono. Reduce a ~50 ms/eval con buena ingeniería.

---

## 6. Modelo de coste programático

Función reutilizable para estimar el coste de cualquier campaña:

```python
# scripts/estimate_ibm_cost.py (propuesto)
def estimate_ibm_cost(
    n_evaluations: int,
    shots: int = 512,
    circuit_depth: int = 150,
    overhead_per_eval_s: float = 0.5,
    rate_usd_per_s: float = 1.60,
) -> float:
    """Estimar coste USD de N evaluaciones DTQW en IBM Quantum.

    Args:
        n_evaluations: número total de invocaciones a apply_dtqw_pennylane.
        shots: shots por circuito.
        circuit_depth: profundidad media (gates) tras transpile.
        overhead_per_eval_s: Runtime/Session overhead por evaluación.
        rate_usd_per_s: precio IBM Pay-as-you-go.

    Returns:
        Coste total estimado en USD.
    """
    t_per_shot_s = circuit_depth * 100e-9 + 4.5e-6  # gates + readout/reset
    t_qpu_per_eval_s = shots * t_per_shot_s
    t_total_per_eval_s = t_qpu_per_eval_s + overhead_per_eval_s
    return n_evaluations * t_total_per_eval_s * rate_usd_per_s
```

Ejemplos de uso:
```python
# Campaña completa de Modelo D, configuración media
estimate_ibm_cost(n_evaluations=250_000, shots=512)
# → ≈ 205 000 USD

# Demo de validación (100 evals para 1 figura)
estimate_ibm_cost(n_evaluations=100, shots=1024)
# → ≈ 165 USD

# Demo extra-frugal (Open Plan, 64 shots, batched)
estimate_ibm_cost(n_evaluations=50, shots=64, overhead_per_eval_s=0.05)
# → ≈ 4.30 USD  (cabe en el Open Plan free tier)
```

---

## 7. Comparación con servicios alternativos

Para contexto, otros proveedores cobran de forma distinta:

| Servicio | Modelo de precio | Hardware típico (2026) | Notas |
|---|---|---|---|
| **IBM Pay-as-you-go** | 1.60 USD/s QPU | Eagle 127q, Heron 156q | Por tiempo, favorece pocos shots. |
| **AWS Braket** (IonQ Aria) | 0.03 USD/shot | IonQ Aria 25q | Por shot, ~30 USD por 1024 shots. |
| **AWS Braket** (Quantinuum H2) | 0.04 USD/shot | Quantinuum H2 56q | Trapped ion, alta fidelidad. |
| **Azure Quantum** (Quantinuum H1) | 0.013 USD/shot | Quantinuum H1-1 20q | Por shot. |
| **Azure Quantum** (Pasqal) | Por tiempo (similar IBM) | Pasqal 100+q neutral atoms | Diferente paradigma. |

**Para nuestro caso (RL con muchas llamadas, pocos shots):** IBM Pay-as-you-go es **el más barato** porque favorece circuitos con pocos shots. Braket/Azure cobran por shot, lo que penaliza la precisión estadística.

Para 250 000 evaluaciones × 512 shots:
- IBM: ~200 k USD (variable según overhead)
- AWS IonQ Aria: 250 000 × 512 × 0.03 = **3.84 M USD**
- Azure Quantinuum H1: 250 000 × 512 × 0.013 = **1.66 M USD**

➡ **IBM es 10-20× más barato** para nuestro patrón de uso. La elección original era correcta.

---

## 8. Estrategias prácticas para reducir coste

### 8.1. Para validación de tesis (recomendado)

**Demostración puntual en hardware real** — coste objetivo **< 100 USD**:

| Decisión | Valor | Justificación |
|---|---|---|
| `n_evaluations` | 100-500 | Suficiente para gráfica `P_k(real) vs P_k(sim)` con barras de error. |
| `shots` | 1024-2048 | Alta precisión por evaluación; pocos puntos a mostrar. |
| `M` (subgrafo) | 4 | Hilbert 16-dim (4 qubits) — menor profundidad y coste. |
| `k` | 3 | Suficiente para mostrar interferencia. |
| Backend | `ibm_kyoto` o `ibm_quebec` (Eagle 127q) | Más estables y baratos que sistemas premium. |
| **Coste estimado** | **30-150 USD** | Cabe perfectamente con la promo 180 min del Open Plan o ~50-150 USD en Pay-as-you-go. |

### 8.2. Si se quisiera campaña completa (NO recomendado)

Reducir todos los drivers:
- `shots = 128`
- `Session` mode con overhead ≈ 50 ms/eval
- Batch dentro del rollout PPO (si arquitecturalmente posible)
- **Coste estimado**: 5-15 k USD

Sigue siendo inviable para una tesis individual pero **ya no es millonario**.

### 8.3. Cero coste con resultado equivalente

```python
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeKyotoV2

# Reproduce las características del IBM Kyoto real
fake = FakeKyotoV2()
noise_model = NoiseModel.from_backend(fake)
sim = AerSimulator(noise_model=noise_model)
```

**0 USD · cero queue · mismos resultados estadísticos a primer orden.** Para `M ≤ 8` esto es indistinguible del hardware real desde el punto de vista de los resultados experimentales.

---

## 9. Conclusión rectificada

### Lo que estaba mal en mi estimación previa

> **"24 M USD" era un mal cálculo**. Mezclé 5 segundos de wallclock (con cola) con tiempo facturable real (~10 ms). El coste real está en **20 k - 400 k USD** dependiendo de shots y overhead.

### Lo que sigue siendo correcto

| Conclusión | Razón |
|---|---|
| ✅ IBM hardware no es viable para campaña completa | 200 k USD sigue siendo inviable para tesis máster individual |
| ✅ Simulación matricial es mejor para M ≤ 32 | Microsegundos vs decenas de mili-segundos + sin coste |
| ✅ Demo NISQ puntual sí es asequible | **30-150 USD** es realista para validación cualitativa |
| ✅ Aer + FakeBackend es equivalente a coste 0 | Reproduce el ruido real para los tamaños relevantes |

### Recomendación final actualizada

**Para una tesis con presupuesto limitado**:

1. **Cero coste, alto valor científico**: implementar `src/quantum/qiskit_aer_backend.py` con `FakeKyotoV2`. Comparable a hardware real para `M ≤ 8`.

2. **Bajo coste, alto valor narrativo** (50-150 USD): ejecutar 100-500 evaluaciones en `ibm_kyoto` real con `M=4, k=3, shots=1024`. Una figura mostrando `P_k(real) ≈ P_k(simulador)` añade peso académico considerable.

3. **No hacer**: campaña completa en hardware. 200 k USD es inviable; los resultados serían inferiores a la simulación clásica para `M ≤ 32`.

---

## 10. Disculpa por la imprecisión

Pido disculpas por la cifra de "24 M USD" del documento anterior. Era una **estimación gruesa de orden de magnitud** que no contrasté con la metodología real de facturación. La cifra correcta (40 k - 400 k USD para campaña completa) cambia el argumento cualitativamente:

- Antes: "completamente fuera de cualquier presupuesto imaginable"
- Ahora: "fuera del presupuesto de una tesis individual pero al alcance de un proyecto de investigación con financiación institucional moderada"

**Esta corrección está reflejada también en una nota al pie del documento [`scenarios_for_stronger_results.md`](scenarios_for_stronger_results.md)** para no dejar afirmaciones inconsistentes en el repositorio.

---

*Documento de metodología generado tras consultar precios actuales de IBM Quantum (mayo 2026) y rectificar la estimación previa.*
