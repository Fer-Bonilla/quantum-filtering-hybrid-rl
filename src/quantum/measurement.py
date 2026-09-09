"""Construcción de la señal q_t (Sec. 7.17).

Convierte la distribución P_k del DTQW en el conjunto candidato top-m.
"""

from __future__ import annotations

import numpy as np


def top_m(probs: np.ndarray, m: int, *, exclude_indices: np.ndarray | None = None) -> np.ndarray:
    """Top-m índices con mayor probabilidad.

    Args:
        probs: vector de probabilidades ``(M,)`` no negativo.
        m: tamaño del conjunto candidato. Se recorta a ``min(m, M_disp)``.
        exclude_indices: índices que NO deben aparecer en el top-m (e.g. el
            seed cuando se quiere medir exploración fuera del seed).

    Returns:
        Array ``(m_eff,)`` con índices LOCALES ordenados por probabilidad
        descendente.
    """
    if probs.ndim != 1:
        raise ValueError(f"probs debe ser 1D; recibido {probs.shape}")
    if m < 1:
        raise ValueError(f"m debe ser >= 1; recibido {m}")

    scores = probs.copy()
    if exclude_indices is not None:
        scores[exclude_indices] = -np.inf
    valid = np.isfinite(scores)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("No quedan índices válidos tras exclude_indices.")
    m_eff = min(m, n_valid)
    # argpartition de los m_eff más grandes
    top = np.argpartition(-scores, kth=m_eff - 1)[:m_eff]
    return top[np.argsort(-scores[top])]
