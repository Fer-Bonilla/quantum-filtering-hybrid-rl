# Priorización por filtrado cuántico estructurado sobre subgrafos dinámicos en RL híbrido para selección de activos

Repositorio público de reproducibilidad del Trabajo Fin de Máster
(Máster en Computación Cuántica, Universidad Internacional de La Rioja, 2026).
Autor: Oscar Fernando Bonilla Suárez.

Contiene **todo lo necesario para reproducir cada número de la memoria**:
la implementación completa, los datos utilizados (con manifiesto de hashes),
los resultados de todas las campañas experimentales, los generadores de
figuras (deterministas y verificables) y la documentación científica de cada
iteración del trabajo.

> **Veredicto de la tesis en una frase.** Un módulo de caminata cuántica de
> tiempo discreto (DTQW) que filtra candidatos sobre subgrafos de un grafo
> dinámico de mercado mejora una métrica operativa de alineación frente a su
> par clásico (D–C = +0,019 en `candidate_hit_rate`, p < 0,001, n = 10), pero
> los controles muestran que el mecanismo dominante es la **rotación de la
> máscara**, no una propiedad cuántica: una máscara aleatoria del mismo tamaño
> iguala o supera a la DTQW, una rotación calibrada reproduce su efecto, el
> ranking cuántico es indistinguible de una permutación aleatoria, ninguna
> máscara porta información sobre el futuro y la ventaja no aparece ni
> siquiera en topologías regulares con régimen balístico verificado. No hay
> mejora confirmatoria en Sharpe y ningún agente supera a 1/N o momentum.

## Contenido

| Carpeta | Qué hay |
|---|---|
| `src/` | Implementación: datos, entorno Gymnasium, grafo dinámico, agentes PPO/híbrido, módulos cuánticos (DTQW, QUBO, controles), entrenamiento y evaluación |
| `configs/` | Configuraciones YAML (datos, entorno, agente, grafo, cuántico, experimentos) |
| `scripts/` | Un script por campaña, análisis y figura (ver [REPRODUCIBILITY.md](REPRODUCIBILITY.md)) |
| `tests/` | Suite de pruebas (313 tests) |
| `data/raw/` | Datos OHLCV utilizados (15 y 30 activos, 2018–2024) con manifiesto SHA-256; ver [data/README.md](data/README.md) |
| `outputs/tables/` | Todas las tablas de resultados (CSV) de las campañas v1–v8 y de las auditorías; ver [outputs/README.md](outputs/README.md) |
| `outputs/figures/` | Figuras generadas; `memoria/` contiene las figuras de la memoria con manifiesto de hashes |
| `outputs/logs/` | Registros de ejecución de las campañas |
| `docs/` | Pre-registros, informes de cada iteración, análisis y guía de contexto |

Documentos de entrada recomendados:

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — mapa tabla/figura de la memoria → script → CSV, comandos exactos y tiempos.
- [PROVENANCE.md](PROVENANCE.md) — versión del código, entorno y hashes que respaldan la memoria.
- [docs/historial_versiones_experimentales.md](docs/historial_versiones_experimentales.md) — el proyecto como seis iteraciones guiadas por pregunta (qué significan las etiquetas v1–v8).
- [docs/preregistro_v8.md](docs/preregistro_v8.md) — pre-registro de la campaña complementaria (hipótesis, márgenes, potencia, semillas).

## Instalación

Requisitos: Python 3.13 y [uv](https://docs.astral.sh/uv/) (0.11 o superior). Solo CPU; no se necesita GPU ni hardware cuántico (las simulaciones cuánticas usan PennyLane/NumPy).

```bash
git clone https://github.com/Fer-Bonilla/quantum-filtering-hybrid-rl.git
cd quantum-filtering-hybrid-rl
uv sync                      # crea .venv con las versiones exactas de uv.lock
uv run pytest -q             # 313 tests
```

En Windows puede usarse el wrapper `tasks.ps1` (`.\tasks.ps1 test`, `.\tasks.ps1 campaign ...`).

## Inicio rápido (5 minutos, sin reentrenar)

Los resultados ya están en `outputs/tables/`; estos comandos los releen y reproducen los análisis y figuras de la memoria:

```bash
# Contrastes pareados de la campaña principal (Tabla maestra del Cap. 5)
uv run python scripts/aggregate_results.py --campaign outputs/tables/campaign_1_v3.csv

# Compuerta G1 de la campaña v8 (TOSTs, precision@m, NDCG@m, MI)
uv run python scripts/analyze_v8_g1.py

# DSR con varianza transversal y sensibilidad en K
uv run python scripts/recalculate_dsr_review.py
uv run python scripts/dsr_k_sensitivity_review.py

# Figuras v8 de la memoria: regeneración determinista + verificación SHA-256
uv run python scripts/v8_thesis_figures.py
uv run python scripts/v8_thesis_figures.py --verify
```

## Arquitectura en una pantalla

```
OHLCV (Yahoo Finance, manifiesto SHA-256)
   └─ features sin fuga temporal (shift(1)) ─► MarketEnv (Gymnasium)
                                                  │  s_t → u_t ∈ {1..N} → r_t = log-riqueza − coste
                                                  ▼
                 grafo dinámico G_t (afinidad k-NN simetrizada) ─► subgrafo H_t (BFS desde la semilla, M=8)
                                                  │
              LocalModule.candidate_set(H_t, k, m) — único punto de variación:
                 C  caminata clásica D⁻¹W        D  DTQW (moneda ponderada)      R  aleatoria
                 Q  QUBO / recocido              S(p) rotación parametrizada     ORACLE ex-post
                                                  │
                                    máscara top-m sobre la política PPO (renormalización)
```

Modelos de la ablación canónica: **A** (PPO), **B** (+ rasgos relacionales del grafo), **C** (+ filtrado clásico), **D** (+ filtrado cuántico DTQW). Controles: **R**, **Q**, **S(p)**, **S(p\*)**, **Crel**, **ORACLE**, integración suave **soft-D/C/R**.

## Cómo está organizado el trabajo experimental

Seis iteraciones, cada una motivada por una pregunta que la anterior dejó abierta (detalle en `docs/historial_versiones_experimentales.md`):

1. **Banco de pruebas** (v1–v2): construir un harness válido; solo `candidate_hit` D > C sobrevive a Bonferroni.
2. **Efecto y veredicto** (v3–v5): n = 10, ruido NISQ, walk-forward, moneda; H1 confirmada, H2 (Sharpe) no, H3 marginal.
3. **Mecanismo** (v6): la máscara aleatoria R iguala a la DTQW; el mecanismo es la rotación.
4. **Causalidad y techo** (v7): dosis-respuesta, mediación, oráculo, TOST, regímenes.
5. **Búsqueda activa** (v8): campaña pre-registrada en el régimen teóricamente favorable; la ventaja no aparece.
6. **Reproducibilidad** (auditorías): confundidos cerrados, DSR transversal, réplicas independientes, figuras verificables.

## Licencia y cita

Código y documentación bajo licencia MIT ([LICENSE](LICENSE)). Los datos de mercado proceden de Yahoo Finance vía `yfinance` y se incluyen únicamente con fines de reproducibilidad académica (ver `data/README.md`). Para citar, use [CITATION.cff](CITATION.cff).
