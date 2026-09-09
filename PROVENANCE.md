# Procedencia

Este repositorio público es un **snapshot curado** del repositorio de
desarrollo privado en el que se ejecutó todo el trabajo experimental.

| Campo | Valor |
|---|---|
| Commit de origen (repositorio privado) | `981549de53c18a5d1918d0eb8253ba4991e7187e` (2026-09-05) |
| Etiqueta de referencia de la memoria | `v8-final` = `5a4385995e487a00e1a1317df007f4ba27683e21` (2026-08-02); este snapshot incluye además las auditorías posteriores (3ª–6ª revisión) |
| Fecha del snapshot | 2026-09-08 |
| Entorno | Python 3.13.13 · uv 0.11.8 · numpy 2.4.4 · scipy 1.17.1 · pandas 2.3.3 · torch 2.11.0 (CPU) · pennylane 0.44.1 · qiskit 2.4.1 · gymnasium 1.3.0 |
| Suite de tests | 313 tests (`uv run pytest -q`) |
| Figuras v8 | `outputs/figures/memoria/manifest_v8_sha256.txt` |
| Datos | `data/raw/*/manifest.json` (SHA-256 por archivo; descarga 2026-05-12) |

Qué se excluyó del snapshot y por qué:

- Las fuentes LaTeX y los PDF de la memoria y de los informes de revisión
  (documentos del depósito académico, no artefactos de reproducibilidad).
- Las guías internas de edición del documento final.
- `outputs/mlruns/` (80 MB; duplica las métricas agregadas de los CSV).
- Ilustraciones conceptuales (JPG) no generadas por el código.
- Tres verificadores estáticos del proyecto LaTeX.

Lo único modificado respecto al código de origen es la ruta de salida de
los dos generadores de figuras de la memoria (`scripts/thesis_figures_es.py`
y `scripts/v8_thesis_figures.py`), que apuntaban a la carpeta de la memoria y
ahora escriben en `outputs/figures/memoria/`. Los hashes del manifiesto no
dependen de esa ruta.
