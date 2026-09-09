"""Hashing determinista para reproducibilidad y caches.

Funciones para hashear archivos (manifest de datos), configs serializadas
(cache de grafos) y DataFrames (idempotencia del pipeline).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

_CHUNK_SIZE: int = 65536


def sha256_file(path: Path) -> str:
    """Calcular el hash SHA-256 hexadecimal de un archivo.

    Lee en bloques para soportar archivos grandes sin cargarlos en memoria.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Calcular el hash SHA-256 hexadecimal de bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash determinista de un objeto JSON-serializable.

    Usa `sort_keys=True` y separadores compactos para que dos representaciones
    semánticamente iguales produzcan el mismo hash.
    """
    serialized = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(serialized.encode("utf-8"))


def sha256_dataframe(df: pd.DataFrame) -> str:
    """Hash determinista de un DataFrame.

    Considera índice, columnas y valores. Útil para `test_dataset_idempotent`.
    """
    h = hashlib.sha256()
    # Hashear nombre del índice y de las columnas
    h.update(str(df.index.name).encode("utf-8"))
    for col in df.columns:
        h.update(str(col).encode("utf-8"))
    # Hashear el contenido en bytes (pandas lo serializa de forma estable)
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()
