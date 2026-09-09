"""Tests de src/data/sector_map.py."""

from __future__ import annotations

from src.data.sector_map import build_sector_map, get_static_sector_map


def test_static_map_covers_common_tickers() -> None:
    static = get_static_sector_map()
    # Cobertura básica del Nivel 1
    for ticker in ("AAPL", "MSFT", "NVDA", "JPM", "JNJ", "XOM"):
        assert ticker in static


def test_build_sector_map_all_known() -> None:
    tickers = ["AAPL", "MSFT", "JPM"]
    out = build_sector_map(tickers, allow_online=False)
    assert out == {"AAPL": "IT", "MSFT": "IT", "JPM": "FIN"}


def test_build_sector_map_unknown_returns_unk() -> None:
    out = build_sector_map(["AAPL", "ZZZ_UNKNOWN_TICKER"], allow_online=False)
    assert out["AAPL"] == "IT"
    assert out["ZZZ_UNKNOWN_TICKER"] == "UNK"
