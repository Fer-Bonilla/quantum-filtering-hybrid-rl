# Datos

## Contenido

| Carpeta | Universo | Activos | Periodo | Uso en la memoria |
|---|---|---|---|---|
| `raw/nivel1/` | Nivel 1 | 15 (AAPL, AMZN, CAT, GOOGL, JNJ, JPM, KO, LIN, MSFT, NEE, NVDA, TSLA, UNH, V, XOM) | 2018-01-02 – 2024-12-30 | pruebas y smoke tests |
| `raw/nivel2/` | Nivel 2 | 30 | 2018-01-02 – 2024-12-30 | **todas las campañas reportadas** |

Cada carpeta contiene un archivo Parquet por activo (OHLCV diario ajustado,
`auto_adjust=True`) y un `manifest.json` con ticker, rango, número de filas
y SHA-256 de cada archivo. Descarga realizada el 2026-05-12 con `yfinance`
1.3.0 mediante `src/data/download.py`.

## Por qué se incluyen los archivos

Los precios ajustados de Yahoo Finance pueden cambiar retroactivamente
(ajustes por dividendos y splits, correcciones), de modo que una nueva
descarga no garantiza los mismos valores. Para que cada número de la memoria
sea reproducible exactamente, se incluye la copia utilizada, verificable
contra el manifiesto.

## Condiciones de uso

Los datos proceden de Yahoo Finance y se redistribuyen aquí **únicamente con
fines de reproducibilidad académica** del trabajo. No están cubiertos por la
licencia MIT del código; su reutilización para otros fines queda sujeta a
los términos de uso de Yahoo Finance. Quien prefiera no usar la copia puede
regenerarla con:

```bash
uv run python -m src.data.download --config configs/data/nivel2.yaml
```

y comparar el manifiesto resultante con el incluido.

## Verificación de integridad

```bash
uv run python - <<'PY'
import json, hashlib, pathlib
for uni in ("nivel1", "nivel2"):
    m = json.load(open(f"data/raw/{uni}/manifest.json"))
    ok = all(hashlib.sha256(open(pathlib.Path("data") / e["file_path"].replace("\\", "/"), "rb").read()).hexdigest() == e["sha256"] for e in m["entries"])
    print(uni, len(m["entries"]), "archivos,", "OK" if ok else "FALLO")
PY
```
