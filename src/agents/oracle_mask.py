"""Máscara-oráculo: cota superior del canal selector (v7 — experimento 4).

A diferencia de los selectores C/D/R/Q/S (que no ven el futuro), el oráculo
selecciona ex-post los ``m`` activos con MAYOR retorno futuro — el mismo retorno
``R_{u,t+1}`` que determina la recompensa del entorno. Es una fuga deliberada y
etiquetada, análoga al benchmark ``oracle_expost`` de v6, pero aplicada como
MÁSCARA al PPO: responde "¿puede el canal selector mover el Sharpe, o el cuello
de botella es la política?".

    oracle-máscara ≈ Modelo A  →  ningún selector podía ayudar; límite = política.
    oracle-máscara ≫ Modelo A  →  el canal tiene techo; el problema es la calidad
                                   de selección alcanzable.

Este módulo expone solo la parte PURA (sin entorno) para poder testearla; el
``step_hook`` que lee ``env.current_t`` vive en ``scripts/run_oracle_campaign_v7.py``.
"""

from __future__ import annotations

import numpy as np


def top_m_future_mask(future_returns: np.ndarray, m: int) -> np.ndarray:
    """Máscara booleana ``[N]`` con True en los ``m`` mayores retornos futuros.

    Args:
        future_returns: vector ``(N,)`` con el retorno del siguiente step por
            activo. Los valores no finitos (NaN/inf) se tratan como ``-inf``
            (nunca se seleccionan si hay alternativas finitas).
        m: número de activos a marcar (recortado a ``min(m, N)``).

    Returns:
        Array booleano ``(N,)`` con exactamente ``min(m, N)`` posiciones True
        (menos si ``m <= 0``).
    """
    fr = np.asarray(future_returns, dtype=np.float64)
    if fr.ndim != 1:
        raise ValueError(f"future_returns debe ser 1D; recibido {fr.shape}")
    n = fr.shape[0]
    mask = np.zeros(n, dtype=bool)
    m_eff = min(m, n)
    if m_eff <= 0:
        return mask
    safe = np.where(np.isfinite(fr), fr, -np.inf)
    top = np.argpartition(-safe, kth=m_eff - 1)[:m_eff]
    mask[top] = True
    return mask
