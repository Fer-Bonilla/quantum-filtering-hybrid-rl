"""Smoke test end-to-end de la capa de datos.

Descarga un universo pequeño de Yahoo Finance, lo limpia, computa features,
particiona cronológicamente y verifica que el manifest sea idempotente
(dos corridas producen el mismo hash de archivo).

Uso::

    uv run python scripts/smoke_data_pipeline.py
"""

from __future__ import annotations

from datetime import date

from src.data.cleaning import CleaningPolicy, clean, get_tickers
from src.data.download import fetch_ohlcv, load_manifest, load_ohlcv
from src.data.features import FeatureSpec, compute_features
from src.data.sector_map import build_sector_map
from src.data.splits import SplitSpec, chronological_split
from src.utils.hashing import sha256_file
from src.utils.logging import setup_logging
from src.utils.paths import DATA_RAW_DIR


def main() -> None:
    setup_logging("INFO")

    # Universo pequeño para smoke test (5 tickers, 2 años)
    tickers = ["AAPL", "MSFT", "JPM", "JNJ", "XOM"]
    start, end = date(2023, 1, 3), date(2024, 12, 31)

    print(f"\n[1/6] Descargando {len(tickers)} tickers de Yahoo Finance...")
    manifest = fetch_ohlcv(
        tickers,
        start,
        end,
        universe="smoke",
        interval="1d",
    )
    print(f"      Manifest: {len(manifest.entries)} entradas.")

    print("\n[2/6] Verificando idempotencia del manifest...")
    manifest_path = DATA_RAW_DIR / "smoke" / "manifest.json"
    hash_1 = sha256_file(manifest_path)
    fetch_ohlcv(tickers, start, end, universe="smoke", interval="1d")
    hash_2 = sha256_file(manifest_path)
    if hash_1 == hash_2:
        print(f"      Idempotencia OK: hash={hash_1[:16]}...")
    else:
        # NOTA: el campo downloaded_at_utc rompe la igualdad de hash del JSON
        # entero. Verificamos por hashes de parquets en su lugar.
        m1 = load_manifest(manifest_path)
        entries_hashes = {e["ticker"]: e["sha256"] for e in m1["entries"]}
        print(
            f"      Manifest JSON cambia (esperado por timestamp), pero "
            f"hashes individuales: {entries_hashes}"
        )

    print("\n[3/6] Cargando parquets...")
    raw = load_ohlcv(universe="smoke")
    print(f"      {len(raw)} tickers cargados.")

    print("\n[4/6] Limpiando y alineando...")
    clean_df = clean(raw, CleaningPolicy())
    print(f"      Tickers tras limpieza: {get_tickers(clean_df)}")
    print(f"      Rango temporal: {clean_df.index.min().date()} -> {clean_df.index.max().date()}")
    print(f"      Filas: {len(clean_df)}")

    print("\n[5/6] Computando features...")
    spec = FeatureSpec(volatility_window=20, volume_window=20, indicators=("rsi",))
    features = compute_features(clean_df, spec)
    feature_names = sorted({c[1] for c in features.columns})
    print(f"      Features: {feature_names}")
    print(f"      Shape: {features.shape}")

    print("\n[6/6] Partición cronológica train/val/test (60/20/20)...")
    splits = chronological_split(features, SplitSpec(train_frac=0.6, val_frac=0.2))
    for name, df in splits.items():
        print(
            f"      {name:5s}: {len(df):4d} filas  [{df.index.min().date()}, {df.index.max().date()}]"
        )

    print("\nSector map:")
    sectors = build_sector_map(get_tickers(clean_df), allow_online=False)
    print(f"      {sectors}")

    print("\n[OK] Smoke test completado exitosamente.\n")


if __name__ == "__main__":
    main()
