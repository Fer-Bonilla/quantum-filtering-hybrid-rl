"""Descarga de datos OHLCV desde Yahoo Finance.

Produce un parquet por ticker en ``data/raw/<universe>/`` y un manifest JSON
con metadatos de descarga, periodo, lista de tickers y hashes SHA-256 de
cada archivo. El manifest garantiza idempotencia: dos descargas con los
mismos parámetros producen el mismo hash.

CLI::

    uv run python -m src.data.download --config configs/data/nivel1.yaml
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from importlib.metadata import version as pkg_version
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.hashing import sha256_file
from src.utils.logging import get_logger
from src.utils.paths import DATA_RAW_DIR, ensure_dir

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DownloadEntry:
    """Entrada del manifest para un ticker descargado."""

    ticker: str
    rows: int
    start: str
    end: str
    file_path: str
    sha256: str


@dataclass(slots=True)
class Manifest:
    """Manifest de un snapshot de datos."""

    universe: str
    tickers: list[str]
    start_date: str
    end_date: str
    interval: str
    auto_adjust: bool
    yfinance_version: str
    downloaded_at_utc: str
    entries: list[DownloadEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "universe": self.universe,
            "tickers": sorted(self.tickers),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "interval": self.interval,
            "auto_adjust": self.auto_adjust,
            "yfinance_version": self.yfinance_version,
            "downloaded_at_utc": self.downloaded_at_utc,
            "entries": sorted(
                (asdict(e) for e in self.entries),
                key=lambda d: d["ticker"],
            ),
        }


def fetch_ohlcv(
    tickers: list[str],
    start: date,
    end: date,
    *,
    universe: str = "nivel1",
    interval: str = "1d",
    auto_adjust: bool = True,
    out_dir: Path | None = None,
) -> Manifest:
    """Descargar OHLCV para una lista de tickers.

    Args:
        tickers: Lista de tickers (e.g. ["AAPL", "MSFT", ...]).
        start: Fecha inicial (inclusive).
        end: Fecha final (inclusive en Yahoo Finance).
        universe: Nombre del subdirectorio destino en ``data/raw/``.
        interval: Intervalo OHLCV ("1d", "1wk", "1mo").
        auto_adjust: Si True, ajusta precios por dividendos/splits.
        out_dir: Directorio raíz de salida (por defecto: ``data/raw``).

    Returns:
        Manifest con metadatos del snapshot. También escrito a JSON en disco.

    Raises:
        ValueError: Si la descarga de algún ticker no produce filas válidas.
    """
    if not tickers:
        raise ValueError("Lista de tickers vacía.")
    if end <= start:
        raise ValueError(f"end ({end}) debe ser posterior a start ({start}).")

    base = out_dir if out_dir is not None else DATA_RAW_DIR
    out = ensure_dir(base / universe)

    manifest = Manifest(
        universe=universe,
        tickers=list(tickers),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        interval=interval,
        auto_adjust=auto_adjust,
        yfinance_version=pkg_version("yfinance"),
        downloaded_at_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    for ticker in sorted(tickers):
        _log.info("Descargando %s ...", ticker)
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            raise ValueError(f"Descarga vacía para {ticker} entre {start} y {end}.")

        df = _normalize_ohlcv_columns(df)
        df.index.name = "date"

        parquet_path = out / f"{ticker}.parquet"
        df.to_parquet(parquet_path, engine="pyarrow", index=True)

        entry = DownloadEntry(
            ticker=ticker,
            rows=len(df),
            start=str(df.index.min().date()),
            end=str(df.index.max().date()),
            file_path=str(parquet_path.relative_to(base.parent)),
            sha256=sha256_file(parquet_path),
        )
        manifest.entries.append(entry)
        _log.info(
            "  %s: %d filas [%s, %s] sha256=%s",
            ticker,
            entry.rows,
            entry.start,
            entry.end,
            entry.sha256[:12],
        )

    manifest_path = out / "manifest.json"
    _write_manifest(manifest_path, manifest)
    _log.info("Manifest escrito en %s", manifest_path)
    return manifest


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizar nombres de columnas yfinance a un conjunto estándar.

    yfinance ocasionalmente devuelve MultiIndex (cuando se piden varios
    tickers); esta función aplana y normaliza.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Cuando se descarga un ticker individual con yfinance >=0.2,
        # las columnas pueden ser MultiIndex (ticker, OHLCV). Aplanamos.
        df = df.copy()
        df.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in df.columns]

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    keep = [c for c in ("open", "high", "low", "close", "adj_close", "volume") if c in df.columns]
    if "close" not in keep:
        raise ValueError(f"Descarga sin columna 'close' tras normalización: {df.columns.tolist()}")
    return df[keep].copy()


def _write_manifest(path: Path, manifest: Manifest) -> None:
    """Escribir el manifest como JSON con claves ordenadas (reproducibilidad)."""
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_manifest(path: Path) -> dict:
    """Cargar un manifest desde disco."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_ohlcv(
    universe: str = "nivel1", *, base_dir: Path | None = None
) -> dict[str, pd.DataFrame]:
    """Cargar los parquets OHLCV de un universo en un dict ticker → DataFrame."""
    base = base_dir if base_dir is not None else DATA_RAW_DIR
    universe_dir = base / universe
    if not universe_dir.is_dir():
        raise FileNotFoundError(f"Universo no encontrado: {universe_dir}")

    result: dict[str, pd.DataFrame] = {}
    for parquet in sorted(universe_dir.glob("*.parquet")):
        ticker = parquet.stem
        result[ticker] = pd.read_parquet(parquet)
    if not result:
        raise FileNotFoundError(f"No se encontraron parquets en {universe_dir}")
    return result
