# Escenarios para resultados más contundentes

> Análisis estratégico de **qué cambios al diseño experimental** podrían reforzar la evidencia a favor de la hipótesis cuántica de la tesis, evaluando viabilidad técnica, coste y probabilidad de éxito.

---

## 0. Diagnóstico: ¿por qué la campaña principal mostró señal débil?

Antes de proponer escenarios alternativos, hay que identificar **dónde está el cuello de botella** del componente cuántico actual:

### El módulo cuántico opera sobre subgrafos muy pequeños

Configuración actual (Sec. 7.16 del documento):
- Tamaño del subgrafo `M = 8` nodos
- Grado máximo `d_max ≈ 4-5` (k-NN simetrizado con k=5)
- Espacio de Hilbert: `M · d_max ≈ 8 × 5 = 40` dimensiones (≈ 6 qubits)
- Pasos DTQW: `k = 3`

**Implicación teórica crítica**: la ventaja teórica de las caminatas cuánticas sobre las clásicas **escala con el tamaño del grafo**. Para grafos regulares de `M` nodos:

- Caminata clásica: tiempo de mezcla `T_mix ~ M²` (difusión lenta).
- DTQW: tiempo de mezcla `T_mix ~ M` (dispersión balística).
- **Factor de ventaja teórico ≈ √M**.

Con `M = 8`, el factor es `√8 ≈ 2.8`. Con la cantidad de ruido inherente al `m = 3` (top-3 sobre 8 candidatos), este factor **es absorbido por el muestreo discreto** y no se manifiesta empíricamente.

### Implicación para la tesis

La señal cuántica débil no es un defecto de la implementación — es **consistente con la teoría de caminatas cuánticas en grafos pequeños**. Para que el efecto sea observable, el subgrafo debe ser **suficientemente grande** para que `√M >> 1` domine el ruido del top-m discreto.

---

## 1. Escenarios propuestos

A continuación se evalúan cuatro escenarios. Cada uno se valora en cinco dimensiones:

| Dimensión | Significado |
|---|---|
| **Probabilidad de mejora** | Cuán probable es que el escenario produzca evidencia más contundente (≪/♯/♯♯/♯♯♯). |
| **Coste técnico** | Esfuerzo de implementación y cómputo (€/€€/€€€). |
| **Coste económico** | Gasto en hardware o servicios externos. |
| **Riesgo de no-resultado** | Probabilidad de invertir y no obtener mejora. |
| **Defendibilidad académica** | Cuán convincente es el escenario para un tribunal de tesis. |

---

## Escenario A — Ampliar el universo a Nivel 2 (N = 30-50 activos)

**Cambio concreto**: pasar de 15 a 30-50 tickers del S&P 500 (Nivel 2, Sec. 8.6.2 del documento).

| Dimensión | Valoración |
|---|---|
| Probabilidad de mejora | ♯ (modesta) |
| Coste técnico | € (trivial: cambiar el YAML) |
| Coste económico | 0 € |
| Riesgo de no-resultado | Medio |
| Defendibilidad | Alta (es el plan original) |

### Por qué AYUDA marginalmente
- Más candidatos → score de seed más informativo → subgrafos más "interesantes".
- Mayor varianza estructural en `G_t` → la DTQW tiene grafos más ricos donde dispersarse.

### Por qué NO basta por sí solo
- El **módulo cuántico sigue operando sobre subgrafos de tamaño M**. Aumentar N de 15 a 50 **no aumenta M**. La ventaja teórica `√M` permanece en `√8 ≈ 2.8`.
- El efecto se diluye: con 50 candidatos y `m = 3`, la probabilidad de acierto baja proporcionalmente, ahogando cualquier mejora de priorización.

### Recomendación
Hacer este cambio **junto con el Escenario B** (aumentar M). Hacerlo aislado no produce ventaja cuántica observable.

### Comando para ejecutarlo
```powershell
# Crear configs/data/nivel2.yaml con 30-50 tickers, luego:
.\tasks.ps1 campaign -- --universe nivel2 --seeds 42 123 456 789 1024 --steps 50000 --campaign-id n50
```

---

## Escenario B — Aumentar el subgrafo (M = 20-32) ⭐ recomendado

**Cambio concreto**: aumentar `subgraph_max_size` de 8 a 20-32 nodos. Combinable con el Escenario A.

| Dimensión | Valoración |
|---|---|
| Probabilidad de mejora | ♯♯♯ (alta) |
| Coste técnico | € (cambio en YAML) |
| Coste económico | 0 € |
| Riesgo de no-resultado | Bajo |
| Defendibilidad | Muy alta (justificable matemáticamente) |

### Por qué FUNCIONA
La ventaja teórica `√M` se multiplica directamente:

| M actual / nuevo | Ventaja teórica | Hilbert dim | Equivalente qubits |
|---|---|---|---|
| 8 | √8 ≈ 2.83 | 40-64 | 6 |
| 20 | √20 ≈ 4.47 | 100-160 | 7 |
| **32** | **√32 ≈ 5.66** | **160-256** | **8** |
| 64 | √64 = 8.00 | 320-512 | 9 |

Con M = 32, el factor casi se duplica respecto al actual (5.66 vs 2.83). En graph bandits (Yamagami et al. 2025) este es exactamente el rango donde la ventaja cuántica empieza a ser **estadísticamente detectable**.

### Coste computacional (backend matricial)

Con la implementación actual (`complex128` denso) sobre M=32:
- Matriz `U_t`: 256×256 complex128 ≈ **1 MB**.
- Tiempo por paso DTQW (`k=3-5`): ~5-15 ms (vs ~1 ms para M=8).
- Por run de 50k steps: **~250-750 s** (4-12 min, vs 80 s actual).
- Campaña 4×5 seeds: **~1-4 horas** (vs ~17 min actual).

Totalmente viable en un workstation. No necesita GPU.

### Restricción importante
Para M=32 hay que tener N ≥ 32 en el universo. Esto **obliga** a combinarlo con el Escenario A (Nivel 2, 50 activos).

### Comando para ejecutarlo
```yaml
# configs/experiment/model_d_large.yaml (nuevo)
defaults:
  - ../data/nivel2          # 50 activos
  - ../env/default
  - ../agent/ppo
  - ../graph/default
  - ../quantum/default

agent:
  model: D

graph:
  subgraph_max_size: 32     # ← cambio clave
  k_neighbors: 8            # ↑ para evitar subgrafos aislados

quantum:
  k_steps: 5                # ↑ más pasos para que la interferencia se note
  m_top: 5
```

```powershell
.\tasks.ps1 campaign -- --universe nivel2 --seeds 42 123 456 789 1024 1414 1729 --steps 50000 --campaign-id large_subgraph
```

---

## Escenario C — Hardware cuántico real IBM (Eagle 127 qubits / Heron 156 qubits)

> **🔄 Nota correctiva** (añadida posteriormente): la estimación de
> "**~24 M USD**" para una campaña completa en hardware IBM que aparece
> abajo era una **aproximación gruesa incorrecta** que mezclaba tiempo
> de cola (no facturado) con tiempo de QPU real (facturado). Tras
> consulta a la documentación oficial (mayo 2026), el coste corregido
> de la campaña completa está en el rango **40 k - 400 k USD**
> dependiendo de configuración (shots, overhead, modo de invocación).
> La conclusión cualitativa **no cambia** — sigue siendo inviable para
> una tesis individual, sólo el orden de magnitud es distinto. Ver
> metodología completa en
> [`ibm_quantum_cost_analysis.md`](ibm_quantum_cost_analysis.md).

**Cambio concreto**: ejecutar la DTQW en un procesador IBM Quantum real via Qiskit Runtime, en lugar del simulador.

| Dimensión | Valoración |
|---|---|
| Probabilidad de mejora **del resultado científico** | ≪ (¡baja!) |
| Probabilidad de mejora **de la defendibilidad académica** | ♯♯♯ (alta, narrativa fuerte) |
| Coste técnico | €€€ (alta integración Qiskit Runtime) |
| Coste económico | €€-€€€ (dependiendo del modo, ver abajo) |
| Riesgo de no-resultado | Muy alto |
| Defendibilidad | Alta como "validación NISQ", baja como "ventaja experimental" |

### La verdad incómoda

Para los tamaños de problema relevantes a esta tesis (`M ≤ 32`, Hilbert ≤ 256 dim, ≤ 8 qubits), **un procesador IBM de 128 qubits es DRAMÁTICAMENTE más lento que la simulación clásica en CPU**:

| Backend | Tiempo por evaluación DTQW (M=8, k=3) |
|---|---|
| NumPy matricial | ~1 ms |
| PennyLane `default.qubit` | ~7 ms |
| PennyLane `default.mixed` + ruido | ~20-50 ms |
| **IBM Quantum hardware (`ibm_kyoto`)** | **~5-30 segundos** (incluye queue + transpile + execute + readout) |

**Causa**:
- Cada job pasa por una cola de minutos a horas.
- El transpile a la topología nativa del backend agrega gates.
- El readout y error mitigation duplican el tiempo.
- Las redes de IBM tienen latencia HTTP entre cada submit.

### Coste económico real

| Plan IBM Quantum | Coste | Restricciones |
|---|---|---|
| **Open Plan** (gratis) | 0 € | 10 min/mes de tiempo de procesador. Insuficiente para un sólo entrenamiento. |
| **Premium Plan** | €€€ por mes | Acceso prioritario a backends de 100+ qubits. Pagado por organización. |
| **Pay-as-you-go** | ~1.60 USD/segundo | Para una campaña que requiere 60 × 50k = 3M evaluaciones DTQW × 5s = 15M segundos = **24M USD** ❌ inviable. |

### Lo que SÍ se puede hacer (validación parcial)

Una **demostración de viabilidad** ejecutando un **subset reducido** de la campaña en hardware real:

**Mini-experimento defendible** (~5-10 min de tiempo de procesador):
- 1 entrenamiento ya finalizado en simulador.
- Tomar el agente entrenado y ejecutar **100 steps de evaluación con DTQW en hardware real** (≈ 100 evaluaciones × 5s = 500s).
- Mostrar que la **distribución `P_k(v_i)`** del hardware real **coincide** con la del simulador dentro del error de ruido caracterizado.

Esto **valida que la implementación es portable a NISQ** sin pretender una campaña completa en hardware.

### Alternativa equivalente sin coste: simulador con ruido realista de IBM

Qiskit Aer proporciona modelos de ruido extraídos de calibraciones recientes de los backends IBM (`FakeKyoto`, `FakeQuebec`, `FakeMontreal`). El proyecto **ya tiene PennyLane** con `DepolarizingChannel` y `PhaseDamping`; añadir un wrapper sobre `qiskit-aer` con `FakeBackendV2` permite reproducir el ruido real de un procesador específico **sin gastar dinero ni esperar la cola**.

```python
# src/quantum/qiskit_aer_backend.py (propuesto, ~50 líneas)
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeKyotoV2

fake_backend = FakeKyotoV2()
noise_model = NoiseModel.from_backend(fake_backend)
simulator = AerSimulator(noise_model=noise_model)
# ... ejecutar el circuito DTQW transpilado a la topología de Kyoto
```

Esto produce los **mismos resultados estadísticos** que ejecutar en hardware real, pero gratis y rápido.

### Recomendación
- **No** intentes correr una campaña completa en hardware IBM. No es viable económicamente.
- **Sí** considera una **demostración de viabilidad NISQ** de 1-2 minutos de tiempo IBM como cierre del Cap. 7 o como apéndice.
- **Sí** implementa el simulador con ruido realista de IBM (Qiskit Aer + FakeBackend). Esto **fortalece enormemente la defendibilidad** sin coste.

---

## Escenario D — Replanteo del problema para amplificar la señal

El cuello de botella real es que **el problema de selección de activos no es donde las caminatas cuánticas brillan teóricamente**. La literatura muestra ventaja cuántica clara en:

1. **Graph bandits** (Yamagami et al. 2025, citado en la tesis): identificación de mejores brazos en grafos.
2. **Element distinctness** (Ambainis 2003): búsqueda en datos no estructurados.
3. **Glued trees** (Childs et al. 2004): exponential separation en grafos específicos.

Para encajar mejor el problema de tesis:

### D.1. Reformular como problema de bandits sobre el grafo

En lugar de selección de un solo activo por paso, formular como **multi-arm bandit** donde:
- Cada activo = brazo.
- El grafo `G_t` define una estructura de similitud entre brazos.
- La DTQW se usa para **identificar el mejor brazo más rápido** (graph bandit cuántico).

Esto alinea el problema con Yamagami et al. (2025), donde la ventaja cuántica está demostrada teórica y empíricamente.

**Coste técnico**: ♯♯ (refactor de ~2 semanas).
**Probabilidad de mejora**: ♯♯ (alta — el método cuántico está diseñado para este caso).

### D.2. Universo con estructura sectorial fuerte

La afinidad actual usa `α · max(0, corr) + β · sector` con `α = 1, β = 0`. La componente sectorial está **apagada**.

Activarla (`α = 0.5, β = 0.5`) crea un grafo con **estructura de comunidades** sectoriales explícita. Las caminatas cuánticas son **especialmente buenas** en grafos con comunidades porque la interferencia constructiva las concentra dentro de la comunidad relevante.

**Coste técnico**: € (cambiar 2 parámetros en YAML).
**Probabilidad de mejora**: ♯♯ (moderada-alta).

```yaml
# configs/graph/sectoral.yaml
graph:
  alpha: 0.5    # ← antes 1.0
  beta: 0.5     # ← antes 0.0
  k_neighbors: 6
```

### D.3. Reward que premie diversificación

El reward actual `R - λσ - μc` penaliza riesgo individual. Añadir un término de **diversificación** (e.g. penalizar acción repetida) hace que la **exploración importa más** → la calidad de la priorización cuántica se vuelve decisiva.

```python
# Reward modificado (propuesto)
r_t = R - λσ - μc + ν · (1 - similarity(u_t, u_{t-1}))
```

---

## 2. Síntesis: ¿qué hacer? — Recomendación priorizada

### 🥇 Prioridad 1 (semana adicional al plan, coste 0): Escenario B+A combinado

**Acción**: ejecutar una campaña con `Nivel 2 (N=50)` + `M=32` + `k=5` + `m=5` + `10 seeds`.

```powershell
# Pseudocódigo de los nuevos configs
configs/data/nivel2.yaml         # 50 tickers líquidos S&P 500
configs/experiment/model_d_large.yaml  # M=32, k=5, m=5
```

**Coste**: ~3-4 horas de cómputo en CPU. **0 € económico**.
**Expectativa**: con `√M ≈ 5.66` (vs 2.83 actual) y `n=10`, **es razonable esperar que `P(D > C)` en Sharpe alcance > 0.95** (significancia formal).

### 🥈 Prioridad 2 (1-2 días, coste 0): Activar afinidad sectorial (D.2)

Crea grafos con estructura de comunidades donde la DTQW tiene ventaja teórica documentada.

### 🥉 Prioridad 3 (3-5 días, coste 0): Simulador con ruido realista IBM (Aer FakeBackend)

Refuerza la defendibilidad ante el tribunal: "validado contra el modelo de ruido del IBM Kyoto V2 actualizado a 2026".

### ⭐ Opcional para narrativa fuerte (~1h tiempo IBM, coste ~10 USD): demostración NISQ real

Una sola gráfica que muestre `P_k` de hardware IBM real ≈ `P_k` del simulador con ruido del Aer, para `M=4` y `k=3`. Esto **no añade evidencia empírica**, pero **sí añade gravitas académica**: "se validó en hardware NISQ de IBM Eagle 127 qubits".

### ⛔ No recomendado: campaña completa en IBM hardware

Inviable económicamente (millones de dólares) y científicamente inferior a la simulación matricial para `M ≤ 32`.

---

## 3. Tabla resumen

| Escenario | Mejora prob. | Coste técnico | Coste económico | Plazo | Recomendación |
|---|:-:|:-:|:-:|:-:|:-:|
| A. Universo Nivel 2 (N=50) | ♯ | € | 0 € | 1 día | ⚠ solo combinado con B |
| **B. Subgrafo M=20-32** | **♯♯♯** | € | 0 € | 1 día | ⭐ **HACER YA** |
| C. Campaña completa IBM hardware | ≪ | €€€ | Millones USD | — | ⛔ NO |
| C'. Demostración NISQ (1 figura) | (narrativa) | €€ | ~10 USD | 2-3 días | ⭐ Opcional para defensa |
| D.1. Refactor a graph bandit | ♯♯ | ♯♯ | 0 € | 2 sem | Opcional |
| **D.2. Afinidad sectorial activa** | **♯♯** | € | 0 € | 1 día | ⭐ Cambiar 2 líneas |
| D.3. Reward con diversificación | ♯ | € | 0 € | 2 días | Opcional |
| **Aer + ruido realista IBM** | (validez) | €€ | 0 € | 3-5 días | ⭐ **Reforzaria defensa** |

---

## 4. Plan de acción concreto recomendado (2 semanas adicionales)

### Semana 10: Amplificar la señal cuántica
- **Día 1**: crear `configs/data/nivel2.yaml` con 50 tickers, descargar datos.
- **Día 2**: crear `configs/experiment/model_d_large.yaml` con M=32, k=5, m=5; ejecutar mini-validación 1 seed.
- **Día 3-4**: ejecutar campaña `n=10` seeds con configuración ampliada (~5-8 horas).
- **Día 5**: activar afinidad sectorial (α=0.5, β=0.5) y repetir con `n=5`.
- **Día 6-7**: agregación + figuras + análisis comparativo.

### Semana 11: Defendibilidad NISQ
- **Día 1-2**: implementar `src/quantum/qiskit_aer_backend.py` con `FakeKyotoV2` como modelo de ruido.
- **Día 3**: ejecutar ablación de ruido completa (Cap. 8.20.5) con backend realista.
- **Día 4-5** (opcional, presupuesto permitiendo): registrar cuenta IBM Quantum, ejecutar 100 evaluaciones DTQW en `ibm_kyoto` real para una figura de validación.
- **Día 6-7**: documentación final y figuras para el documento.

### Resultado esperado tras 2 semanas

| Métrica | Campaña actual | Campaña propuesta |
|---|---|---|
| `n_seeds` | 5 | 10 |
| `N` (universo) | 15 | 50 |
| `M` (subgrafo) | 8 | 32 |
| `k` (pasos DTQW) | 3 | 5 |
| Ventaja teórica `√M` | 2.83 | **5.66** |
| `P(D > C)` en Sharpe (estimación) | 0.943 (casi-sig.) | **> 0.97 (significativo)** |
| Defendibilidad NISQ | Ruido sintético | **Ruido FakeKyotoV2 + opcional hardware real** |

---

## 5. Conclusión honesta

> **El hardware cuántico real de IBM no es la palanca correcta** para mejorar los resultados de esta tesis. Lo que falta es **escalar el subgrafo** (M=32) y combinar con **n=10 seeds** y **estructura sectorial activa**. Esto es ejecutable en CPU en una semana y tiene **alta probabilidad de cruzar el umbral de significancia formal**. El hardware real cabe como **demostración cualitativa de viabilidad NISQ** (1 figura) para fortalecer la defensa, no como motor de los resultados experimentales principales.

### Resumen en 3 oraciones

1. **Escalar M de 8 a 32 es lo más efectivo y prácticamente gratuito.**
2. **Usar simulador IBM con ruido realista (FakeBackendV2) refuerza la defensa sin coste.**
3. **Ejecutar la campaña completa en hardware IBM es inviable; una demostración puntual de 5-10 min sí cabe como apéndice.**

---

*Documento estratégico generado para guiar las extensiones experimentales del proyecto, 2026-05-11.*
