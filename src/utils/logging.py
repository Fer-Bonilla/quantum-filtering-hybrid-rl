"""Configuración de logging estructurado.

Usa el `logging` estándar de Python con formato consistente y nivel configurable
vía variable de entorno `QUANTUM_AI_LOG_LEVEL` (default INFO).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

_DEFAULT_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"
_LOG_LEVEL_ENV_VAR: Final[str] = "QUANTUM_AI_LOG_LEVEL"


def setup_logging(level: str | int | None = None, *, force: bool = False) -> None:
    """Configurar el logger raíz del paquete.

    Args:
        level: Nivel de log ("DEBUG", "INFO", ...) o int. Si None, lee de la
            variable de entorno QUANTUM_AI_LOG_LEVEL o usa INFO por defecto.
        force: Si True, reconfigura aunque ya esté configurado.
    """
    resolved_level = level if level is not None else os.environ.get(_LOG_LEVEL_ENV_VAR, "INFO")
    logging.basicConfig(
        level=resolved_level,
        format=_DEFAULT_FORMAT,
        datefmt=_DEFAULT_DATEFMT,
        stream=sys.stdout,
        force=force,
    )


def get_logger(name: str) -> logging.Logger:
    """Obtener un logger con el namespace del proyecto.

    Convención: usar `__name__` del módulo invocante.
    """
    return logging.getLogger(name)
