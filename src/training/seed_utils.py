"""Utilidades de semilla global para reproducibilidad."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> np.random.Generator:
    """Fijar semillas en todos los RNG relevantes y devolver un Generator NumPy.

    Cubre:
    - ``random.seed`` (Python stdlib).
    - ``np.random.seed`` + crea un ``Generator`` independiente.
    - ``torch.manual_seed`` (CPU y CUDA si disponible).
    - ``PYTHONHASHSEED`` (afecta solo a procesos nuevos, no al actual; se
      fija por seguridad ante imports tardíos).
    """
    if seed < 0:
        raise ValueError(f"seed debe ser >= 0; recibido {seed}.")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)
