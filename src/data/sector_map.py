"""Mapeo de tickers a sectores GICS.

Para Nivel 1 (10-20 activos líquidos del S&P 500), se usa un mapeo estático
declarado en este módulo para evitar dependencia online en tests y
reproducibilidad determinista. Si un ticker no está en el mapa estático,
se intenta resolver vía yfinance.info (cacheado).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

import yfinance as yf

from src.utils.logging import get_logger

_log = get_logger(__name__)

# Mapeo estático de tickers líquidos a sector GICS (códigos cortos).
# Fuente: clasificación pública GICS 2024. Reproducible y trazable.
_STATIC_SECTOR_MAP: Final[dict[str, str]] = {
    # Technology
    "AAPL": "IT",
    "MSFT": "IT",
    "NVDA": "IT",
    "GOOGL": "COMM",
    "META": "COMM",
    "ADBE": "IT",
    "CRM": "IT",
    # Consumer
    "AMZN": "COMM",  # GICS lo coloca en Consumer Discretionary; aquí simplificamos
    "TSLA": "COND",
    "HD": "COND",
    "NKE": "COND",
    "MCD": "COND",
    "KO": "COST",
    "PEP": "COST",
    "PG": "COST",
    "WMT": "COST",
    # Financials
    "JPM": "FIN",
    "BAC": "FIN",
    "WFC": "FIN",
    "GS": "FIN",
    "MS": "FIN",
    "V": "FIN",
    "MA": "FIN",
    # Healthcare
    "JNJ": "HC",
    "PFE": "HC",
    "UNH": "HC",
    "ABBV": "HC",
    "LLY": "HC",
    # Energy
    "XOM": "ENRG",
    "CVX": "ENRG",
    # Industrials
    "BA": "IND",
    "CAT": "IND",
    "GE": "IND",
    # Utilities
    "NEE": "UTIL",
    "DUK": "UTIL",
    # Materials
    "LIN": "MAT",
    "SHW": "MAT",
    # Real Estate
    "AMT": "REAL",
    "PLD": "REAL",
}


def get_static_sector_map() -> dict[str, str]:
    """Retornar una copia del mapeo estático de tickers → sector."""
    return dict(_STATIC_SECTOR_MAP)


@lru_cache(maxsize=256)
def lookup_sector_online(ticker: str) -> str | None:
    """Consultar el sector de un ticker vía yfinance (cacheado).

    Returns:
        Código corto del sector (e.g. "IT", "FIN") o None si no se puede resolver.
    """
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector")
        if sector:
            # Mapear nombres largos de yfinance a códigos cortos consistentes
            return _normalize_sector_name(sector)
    except Exception as exc:
        _log.warning("No se pudo resolver sector online para %s: %s", ticker, exc)
    return None


def _normalize_sector_name(name: str) -> str:
    """Normalizar nombres de sectores yfinance a códigos cortos GICS."""
    mapping = {
        "Technology": "IT",
        "Information Technology": "IT",
        "Communication Services": "COMM",
        "Consumer Cyclical": "COND",
        "Consumer Discretionary": "COND",
        "Consumer Defensive": "COST",
        "Consumer Staples": "COST",
        "Financial Services": "FIN",
        "Financials": "FIN",
        "Healthcare": "HC",
        "Health Care": "HC",
        "Energy": "ENRG",
        "Industrials": "IND",
        "Utilities": "UTIL",
        "Basic Materials": "MAT",
        "Materials": "MAT",
        "Real Estate": "REAL",
    }
    return mapping.get(name, name[:4].upper())


def build_sector_map(tickers: list[str], *, allow_online: bool = False) -> dict[str, str]:
    """Construir el mapeo ticker → sector para una lista de tickers.

    Args:
        tickers: Lista de tickers (case-sensitive, e.g. "AAPL").
        allow_online: Si True, los tickers no presentes en el mapa estático
            se intentan resolver vía yfinance. Si False, se asigna "UNK".

    Returns:
        Diccionario ``{ticker: sector_code}``. Siempre incluye todos los tickers.
    """
    result: dict[str, str] = {}
    missing: list[str] = []
    for ticker in tickers:
        if ticker in _STATIC_SECTOR_MAP:
            result[ticker] = _STATIC_SECTOR_MAP[ticker]
        elif allow_online:
            online = lookup_sector_online(ticker)
            if online is not None:
                result[ticker] = online
            else:
                result[ticker] = "UNK"
                missing.append(ticker)
        else:
            result[ticker] = "UNK"
            missing.append(ticker)
    if missing:
        _log.warning(
            "Sectores desconocidos para %d ticker(s); usando 'UNK': %s",
            len(missing),
            ", ".join(missing),
        )
    return result
