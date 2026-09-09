"""Tests de src/utils/hashing.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from src.utils.hashing import sha256_dataframe, sha256_file, sha256_json


def test_sha256_file_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello world")
    h1 = sha256_file(path)
    h2 = sha256_file(path)
    assert h1 == h2
    assert len(h1) == 64


def test_sha256_file_different_content(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"world")
    assert sha256_file(a) != sha256_file(b)


def test_sha256_json_invariant_to_key_order() -> None:
    obj1 = {"a": 1, "b": 2, "c": [1, 2, 3]}
    obj2 = {"c": [1, 2, 3], "b": 2, "a": 1}
    assert sha256_json(obj1) == sha256_json(obj2)


def test_sha256_dataframe_deterministic() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    h1 = sha256_dataframe(df)
    h2 = sha256_dataframe(df.copy())
    assert h1 == h2


def test_sha256_dataframe_distinguishes_values() -> None:
    df1 = pd.DataFrame({"x": [1, 2, 3]})
    df2 = pd.DataFrame({"x": [1, 2, 4]})
    assert sha256_dataframe(df1) != sha256_dataframe(df2)
