"""Wrapper minimalista de MLflow para trazabilidad de experimentos.

Por defecto usa el file-store local en ``outputs/mlruns``. Cada run loguea:
- params: config aplanado + seed + hash del manifest de datos.
- metrics: por step (train) y al final (test).
- artifacts: config.yaml serializado, checkpoints, curvas.

Diseñado para ser inocuo en tests: si el directorio no existe, lo crea; si
MLflow no está disponible, el logger se vuelve no-op (útil para CI).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

from src.utils.paths import MLRUNS_DIR, ensure_dir


def _flatten(d: Mapping[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """Aplanar un mapping anidado para logueo de params."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, Mapping):
            out.update(_flatten(v, key, sep=sep))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(list(v), default=str)
        else:
            out[key] = v
    return out


class MLflowLogger:
    """Wrapper de MLflow con un contexto de run gestionado."""

    def __init__(
        self,
        experiment_name: str,
        *,
        tracking_uri: str | None = None,
    ) -> None:
        ensure_dir(MLRUNS_DIR)
        uri = tracking_uri if tracking_uri is not None else MLRUNS_DIR.absolute().as_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        self._experiment_name = experiment_name

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        *,
        tags: dict[str, str] | None = None,
    ) -> Iterator[mlflow.ActiveRun]:
        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            yield run

    @staticmethod
    def log_params(params: Mapping[str, Any]) -> None:
        mlflow.log_params({k: str(v) for k, v in _flatten(dict(params)).items()})

    @staticmethod
    def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items()}
        mlflow.log_metrics(clean, step=step)

    @staticmethod
    def log_artifact(path: Path) -> None:
        mlflow.log_artifact(str(path))

    @staticmethod
    def log_dict(obj: Mapping[str, Any], artifact_path: str) -> None:
        mlflow.log_dict(dict(obj), artifact_path)
