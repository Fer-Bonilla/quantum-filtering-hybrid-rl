# Playbook — Extender la simulación para mostrar diferenciación clara

> Guía operativa para encontrar el **punto operativo (sweet spot)** donde la
> DTQW se diferencia significativamente de la caminata clásica, sin gastar en
> hardware. Todo ejecutable en CPU con la implementación matricial existente.

---

## Tabla de contenidos

1. [Diagnóstico: 6 palancas disponibles](#1-diagnóstico-6-palancas-disponibles)
2. [Estrategia de exploración por fases](#2-estrategia-de-exploración-por-fases)
3. [Fase A — Construir infraestructura (1 día)](#3-fase-a--construir-infraestructura-1-día)
4. [Fase B — Escaneo rápido de parámetros (medio día)](#4-fase-b--escaneo-rápido-de-parámetros-medio-día)
5. [Fase C — Validación del sweet spot (3-6 horas)](#5-fase-c--validación-del-sweet-spot-3-6-horas)
6. [Fase D — Campaña final extendida (8-15 horas)](#6-fase-d--campaña-final-extendida-8-15-horas)
7. [Estimación de tiempo de cómputo](#7-estimación-de-tiempo-de-cómputo)
8. [Criterios de éxito y decisión](#8-criterios-de-éxito-y-decisión)

---

## 1. Diagnóstico: 6 palancas disponibles

Recordatorio del análisis previo: la **ventaja cuántica teórica escala como `√M`** (tamaño del subgrafo). Con `M=8` la ventaja es `2.83`, absorbida por el ruido discreto del top-m. Para diferenciación clara hay seis palancas:

| Palanca | Símbolo | Valor actual | Rango propuesto | Coste cómputo | Driver |
|---|---|:-:|:-:|:-:|---|
| Tamaño del subgrafo | `M` | 8 | **12 → 32** | Alto (∝ M²) | **Ventaja `√M`** |
| Universo elegible | `N` | 15 | 30 → 50 | Bajo | Permite M grande |
| Pasos DTQW | `k` | 3 | 5 → 10 | Lineal | Interferencia profunda |
| Tamaño top-m | `m` | 3 | 3 → 7 | Ninguno | Sensibilidad al filtrado |
| Afinidad sectorial | `β` | **0.0** | **0.3 → 0.6** | Ninguno | **Estructura de comunidades** |
| Vecinos del grafo | `k_neighbors` | 5 | 8 → 12 | Ninguno | Evita subgrafos aislados |

**Hipótesis ranking** (más impactantes primero):
1. **`β > 0`** (estructura sectorial) — coste 0, alto impacto teórico.
2. **`M = 16-24`** — base de la ventaja teórica.
3. **`k = 5-8`** — más interferencia = más coherencia cuántica.
4. **`N = 30-50`** — habilita M grande.
5. **`m = 5-7`** — filtros más laxos suavizan el muestreo discreto.
6. **`n_seeds = 10`** — solo estadística.

---

## 2. Estrategia de exploración por fases

```mermaid
flowchart LR
    A[Fase A: Setup<br/>nivel2 + configs] --> B[Fase B: Scan rápido<br/>4-9 configs × 3 seeds × 5k steps]
    B --> C{¿P(D>C) sube<br/>en alguna config?}
    C -->|Sí| D[Fase C: Validar sweet spot<br/>5 seeds × 50k steps]
    C -->|No| E[Pivot: probar D.1<br/>graph-bandit reformulation]
    D --> F[Fase D: Campaña final<br/>10 seeds × 100k steps]
    F --> G[Defensa de tesis]
    E -.->|Plan B| D
```

**Filosofía**: ir de lo barato a lo caro. Cada fase descarta combinaciones para que la siguiente trabaje sólo con las candidatas viables.

---

## 3. Fase A — Construir infraestructura (1 día)

### 3.1. Crear el universo Nivel 2

```yaml
# configs/data/nivel2.yaml  ⭐ NUEVO
data:
  tickers:
    # Technology (6)
    - AAPL
    - MSFT
    - NVDA
    - ADBE
    - CRM
    - ORCL
    # Communications (3)
    - GOOGL
    - META
    - NFLX
    # Consumer Discretionary (4)
    - AMZN
    - TSLA
    - HD
    - NKE
    # Consumer Staples (3)
    - KO
    - PG
    - WMT
    # Financials (4)
    - JPM
    - V
    - MA
    - GS
    # Healthcare (4)
    - JNJ
    - UNH
    - PFE
    - LLY
    # Energy (2)
    - XOM
    - CVX
    # Industrials (2)
    - BA
    - CAT
    # Utilities/Materials/Real Estate (2)
    - NEE
    - LIN
  start_date: 2018-01-02
  end_date: 2024-12-31
  source: yahoo
  interval: 1d
  cache_subdir: nivel2
  auto_adjust: true
```

30 tickers con cobertura sectorial balanceada (6 IT, 3 COMM, 4 COND, 3 COST, 4 FIN, 4 HC, 2 ENRG, 2 IND, 2 misc). Permite `M ≤ 28` cómodamente.

### 3.2. Descargar

```powershell
uv run python -c "
import yaml; from datetime import date
from src.data.download import fetch_ohlcv
from src.utils.logging import setup_logging
setup_logging('INFO')
with open('configs/data/nivel2.yaml') as f: cfg = yaml.safe_load(f)['data']
fetch_ohlcv(cfg['tickers'], date.fromisoformat(str(cfg['start_date'])),
            date.fromisoformat(str(cfg['end_date'])), universe='nivel2',
            interval=cfg['interval'], auto_adjust=cfg.get('auto_adjust', True))
"
```

Tiempo: ~2-3 minutos (depende de Yahoo Finance).

### 3.3. Plantillas de configs escalados

Crear configs **paramétricos** que permitan barridos limpios:

```yaml
# configs/graph/large.yaml  ⭐ NUEVO
graph:
  alpha: 0.5
  beta: 0.5
  eps: 1.0e-8
  lookback_window: 60
  k_neighbors: 10           # ↑ para evitar subgrafos aislados con M grande
  sym_mode: avg
  subgraph_max_size: 16     # ← punto medio razonable
  seed_score_window: 20
  update_frequency: 1
```

```yaml
# configs/quantum/large.yaml  ⭐ NUEVO
quantum:
  k_steps: 5                # ↑ más interferencia
  m_top: 5                  # ↑ filtro más permisivo
  backend: matrix
  init_mode: uniform
  renormalize_threshold: 1.0e-9
  noise:
    enabled: false
```

```yaml
# configs/experiment/model_d_large.yaml
defaults:
  - ../data/nivel2
  - ../env/default
  - ../agent/ppo
  - ../graph/large
  - ../quantum/large

agent:
  model: D

training:
  total_steps: 50000
  mlflow_experiment_name: model_d_large

seeds: [42, 123, 456, 789, 1024]
```

Idénticos para `model_c_large.yaml` y opcionalmente `model_a_large.yaml` / `model_b_large.yaml`.

### Criterio de salida de Fase A
- ✅ Datos en `data/raw/nivel2/` (30 parquets + manifest).
- ✅ Configs `large.yaml` y `model_*_large.yaml` creados.
- ✅ Smoke test: 1 run de D al nuevo config con 1k steps completa sin error.

---

## 4. Fase B — Escaneo rápido de parámetros (medio día)

Objetivo: identificar **qué combinación de `M`, `β`, `k`** maximiza el gap `D − C` en Sharpe **sin gastar mucho cómputo**.

### 4.1. Plan factorial reducido

Diseño 2³ (8 configs) en la dimensión más prometedora:

| Variante | M | k | β | m |
|---|:-:|:-:|:-:|:-:|
| baseline (existente) | 8 | 3 | 0.0 | 3 |
| **s1** (mas comunidades) | 8 | 3 | **0.5** | 3 |
| **s2** (mas M) | **16** | 3 | 0.0 | 5 |
| **s3** (mas M + comunidades) | **16** | 3 | **0.5** | 5 |
| **s4** (mas k) | 8 | **5** | 0.0 | 3 |
| **s5** (mas M + k) | **16** | **5** | 0.0 | 5 |
| **s6** (mas M + k + comunidades) ⭐ | **16** | **5** | **0.5** | 5 |
| **s7** (extra grande) | **24** | **5** | **0.5** | 5 |
| **s8** (extra grande + interferencia) | **24** | **8** | **0.5** | 5 |

Cada variante se ejecuta para **C y D** con **3 seeds × ~5 000 steps**:
- 8 variantes × 2 modelos × 3 seeds = **48 runs cortos**
- Tiempo estimado: ~20-40 min (depende de la variante)

### 4.2. Implementación: usar el wrapper existente

El script `run_campaign.py` ya soporta `--max-steps` y `--window-length`. Para variantes con M y k distintos, hay que crear YAMLs o (más rápido) usar overrides en código.

Versión rápida vía sub-scripts:

```powershell
# Variante s1: solo activar beta
.\tasks.ps1 campaign -- --universe nivel2 --seeds 42 123 456 `
    --steps 5000 --campaign-id scan_s1 `
    --models C D
# (requiere editar configs/graph/default.yaml temporalmente o crear scan_s1.yaml)
```

Mejor: crear un script utility `scripts/run_param_scan.py` (propuesto):

```python
# scripts/run_param_scan.py (propuesto, ~80 líneas)
import argparse
from pathlib import Path
from itertools import product
from scripts.run_campaign import _train_and_evaluate_one
from src.utils.config import load_config

VARIANTS = {
    "s1": dict(M=8, k=3, beta=0.5, m=3),
    "s2": dict(M=16, k=3, beta=0.0, m=5),
    "s3": dict(M=16, k=3, beta=0.5, m=5),
    "s4": dict(M=8, k=5, beta=0.0, m=3),
    "s5": dict(M=16, k=5, beta=0.0, m=5),
    "s6": dict(M=16, k=5, beta=0.5, m=5),
    "s7": dict(M=24, k=5, beta=0.5, m=5),
    "s8": dict(M=24, k=8, beta=0.5, m=5),
}

# ... loop over variants × models × seeds, escribir CSV con columna `variant`
```

### 4.3. Análisis del scan

Producir tabla con `mean(Sharpe_D − Sharpe_C)` por variante y barra de error:

| Variante | M | k | β | mean(D−C) Sharpe | n |
|---|---|---|---|---|---|
| s1 | 8 | 3 | 0.5 | ¿+0.04? | 3 |
| s3 | 16 | 3 | 0.5 | ¿+0.08? | 3 |
| **s6** | **16** | **5** | **0.5** | ¿+0.15? | 3 |
| ... | | | | | |

La variante con mayor `mean(D−C)` consistente entre seeds pasa a Fase C.

### Criterio de salida de Fase B
- ✅ CSV `outputs/tables/param_scan.csv` con 48 runs.
- ✅ Al menos **1 variante con `mean(D−C) > 0` en Sharpe** consistente entre 3 seeds.
- ⚠ Si **ninguna variante muestra mejora**: pivote a Plan B (Fase E).

---

## 5. Fase C — Validación del sweet spot (3-6 horas)

Una vez identificada la variante ganadora del scan, ejecutar la campaña completa al **estándar de tesis**:

```powershell
# Usando la variante ganadora (ejemplo: s6 con M=16, k=5, β=0.5)
.\tasks.ps1 campaign -- --universe nivel2 `
    --seeds 42 123 456 789 1024 `
    --steps 50000 `
    --campaign-id sweet_spot `
    --max-steps 252 `
    --window-length 20
.\tasks.ps1 aggregate -- --campaign outputs\tables\sweet_spot.csv
.\tasks.ps1 figures -- --campaign outputs\tables\sweet_spot.csv
```

### Tiempo estimado

Para M=16, k=5, n=5 seeds × 4 modelos × 50k steps:
- Modelo D tardará ~3-5× más que en campaign_1 (M=8) por la matriz más grande.
- Total estimado: **3-6 horas** en CPU (vs 17 min de la campaña original).

### Criterio de salida de Fase C
- ✅ `P(D > C)` en Sharpe **> 0.95** (significancia formal).
- ✅ Bootstrap pareado con IC95 % que no cruza 0.
- ✅ Resultado replicado en al menos 2 métricas (Sharpe + algo más).

---

## 6. Fase D — Campaña final extendida (8-15 horas)

Sólo si Fase C confirma significancia. Para la memoria definitiva:

```powershell
.\tasks.ps1 campaign -- --universe nivel2 `
    --seeds 42 123 456 789 1024 1414 1729 2026 31415 9999 `
    --steps 100000 `
    --campaign-id final_v2 `
    --max-steps 252 --window-length 20
```

- **10 seeds** (vs 5) → IC95 % se contrae por `1/√(N)`.
- **100 000 steps** (vs 50 000) → PPO converge mejor, ruido entre seeds se reduce.
- **5 modelos × 10 seeds = 50 runs**.

### Criterio de salida de Fase D
- ✅ Documento de tesis listado con figuras reproducibles.
- ✅ `P(D > C)` con `n=10` confirmando la dirección de Fase C.

---

## 7. Estimación de tiempo de cómputo

Los costes computacionales del backend matricial dependen de:
- `dim_Hilbert = M × d_max` (dim de matrices densas)
- Cost per step DTQW ∝ `dim²` (1 matvec)

| `M` | `d_max` típico | `dim` | Cost/step (vs M=8) | Run 50k steps (estimado) |
|:-:|:-:|:-:|:-:|---|
| 8 | 5 | 40 | 1.0× | 80 s |
| 12 | 6 | 72 | 3.2× | ~260 s = 4 min |
| 16 | 7 | 112 | 7.8× | ~625 s = 10 min |
| 20 | 8 | 160 | 16.0× | ~1280 s = 21 min |
| 24 | 9 | 216 | 29.2× | ~2340 s = 39 min |
| 32 | 10 | 320 | 64.0× | ~5120 s = 85 min |

**Para Fase C con M=16, n=5 seeds, 4 modelos × 50k steps**:
- A, B: ~3 min × 5 = 15 min
- C: ~5 min × 5 = 25 min (matrix backend más simple)
- D: ~10 min × 5 = 50 min
- **Total: ~1.5 horas**

**Para Fase D con M=16, n=10 seeds × 100k steps**:
- Aproximadamente 4-6 horas.

Si la máquina tiene 8 cores, paralelizar **diferentes seeds en procesos** reduce a 1-2 horas (cada run ya es CPU-bound mono-thread por torch).

---

## 8. Criterios de éxito y decisión

### Indicadores de éxito durante el scan (Fase B)

| Indicador | Umbral mínimo | Acción si se alcanza |
|---|---|---|
| `mean(D−C)` en Sharpe sobre 3 seeds | > 0 con 2/3 seeds positivos | ✅ Promover variante a Fase C |
| `mean(D−C)` en `topm_hit_rate` | > 0.01 | ✅ Buena señal de exploración |
| Reducción de `std(Sharpe)` en D vs C | std_D / std_C < 0.7 | ✅ Estabilidad cuántica confirmada |
| Latencia D / latencia C | < 5× | ✅ Operativamente viable |

### Indicadores de fracaso (decidir pivote)

| Indicador | Implicación |
|---|---|
| Tras escanear las 8 variantes, **ninguna** muestra `mean(D−C) > 0` consistente | Considerar Plan B (graph bandit reformulation) |
| Subgrafos frecuentemente aislados con M > 16 | Aumentar `k_neighbors` o cambiar k-NN por threshold |
| Recompensa colapsa a 0 (agente degenerado) | Reducir `entropy_coef` o aumentar `total_steps` |

### Plan B (si Fase B falla)

**Reformular como graph bandit cuántico** (Yamagami et al. 2025):
- En vez de selección de un solo activo, formular como identificación del mejor brazo.
- La DTQW se usa para identificar el brazo óptimo, no para filtrar candidatos PPO.
- Coste de refactor: ~2 semanas.
- Probabilidad de éxito: alta (alineado con literatura que demuestra ventaja cuántica).

---

## 9. Checklist ejecutivo

Para ejecutar este playbook completo:

- [ ] **A.1** Crear `configs/data/nivel2.yaml` (30 tickers)
- [ ] **A.2** Descargar datos (~3 min)
- [ ] **A.3** Crear `configs/graph/large.yaml` y `configs/quantum/large.yaml`
- [ ] **A.4** Crear `configs/experiment/model_{a,b,c,d}_large.yaml`
- [ ] **A.5** Smoke test (1 run, 1k steps) de D al nuevo config
- [ ] **B.1** Escribir `scripts/run_param_scan.py`
- [ ] **B.2** Ejecutar scan 8 variantes × 2 modelos × 3 seeds × 5k steps (~30-60 min)
- [ ] **B.3** Identificar variante ganadora (mejor `mean(D−C)` en Sharpe)
- [ ] **C.1** Campaña en variante ganadora: 4 modelos × 5 seeds × 50k steps (~3-6h)
- [ ] **C.2** Aggregate + figures
- [ ] **C.3** Verificar `P(D > C) > 0.95`
- [ ] **D.1** Campaña final: 4 modelos × 10 seeds × 100k steps (~10-15h)
- [ ] **D.2** Generar figuras + reporte para el documento de tesis

---

## 10. Resumen ejecutivo en 3 frases

1. **El cambio más impactante es `β > 0` + `M = 16`**: activar afinidad sectorial crea estructura de comunidades donde la DTQW tiene ventaja teórica documentada, y doblar M duplica `√M` casi al doble (2.83 → 4.0).

2. **El scan de la Fase B (~30-60 min) decide si vale la pena**: con 8 configs × 3 seeds × 5k steps se identifica si alguna combinación produce diferenciación detectable; si la respuesta es "sí", se invierte en Fase C (3-6h); si "no", se considera pivote a graph-bandit reformulation.

3. **Coste económico: 0 €. Tiempo de cómputo total: 8-25 horas en CPU**, ejecutable en background mientras se redacta el documento de tesis.

---

*Playbook generado tras el análisis de la campaña principal y los escenarios estratégicos. Operacionalmente concreto y económicamente viable.*
