"""Cálculo de la matriz de afinidad A_t (Sec. 6.4 / 7.5).

Forma adoptada:

    A_t(i, j) = alpha * max{0, rho_t(i, j)} + beta * S(i, j)

con alpha + beta = 1, rho la correlación móvil de retornos y S afinidad
sectorial binaria. Tras normalización a [0, 1].

Toda función es PURA (sin estado) y trabaja sobre arrays NumPy para
máxima velocidad.
"""

from __future__ import annotations

import numpy as np


def positive_correlation(returns: np.ndarray) -> np.ndarray:
    """Correlación de Pearson clipeada a la mitad positiva.

    Args:
        returns: Matriz ``(T, N)`` con retornos de N activos en T pasos.
            Se asume que los NaN ya fueron eliminados.

    Returns:
        Matriz ``(N, N)`` simétrica con ``max(0, corr(i, j))``. La diagonal
        es 1.0.
    """
    if returns.ndim != 2:
        raise ValueError(f"returns debe ser 2D; recibido shape {returns.shape}")
    if returns.shape[0] < 2:
        raise ValueError(f"Se necesitan al menos 2 observaciones; recibido {returns.shape[0]}")
    # np.corrcoef trabaja por filas; necesitamos por columnas.
    corr = np.corrcoef(returns, rowvar=False)
    # Forzar simetría y eliminar pequeñas asimetrías numéricas.
    corr = 0.5 * (corr + corr.T)
    return np.maximum(corr, 0.0)


def sector_affinity(sectors: list[str]) -> np.ndarray:
    """Matriz binaria de afinidad sectorial: 1 si mismo sector, 0 en otro caso.

    Args:
        sectors: lista de códigos de sector para los N activos, en orden.

    Returns:
        Matriz simétrica ``(N, N)``. Diagonal = 1.
    """
    len(sectors)
    arr = np.asarray(sectors)
    return (arr[:, None] == arr[None, :]).astype(np.float64)


def affinity_matrix(
    returns: np.ndarray,
    sectors: list[str] | None,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Combinación convexa: ``alpha * max(0, corr) + beta * sector``.

    Args:
        returns: Matriz ``(T, N)`` de retornos.
        sectors: lista de N sectores. Requerido si ``beta > 0``.
        alpha, beta: pesos no negativos con suma 1 (validado fuera).

    Returns:
        Matriz simétrica no negativa ``(N, N)``.
    """
    if not np.isclose(alpha + beta, 1.0, atol=1e-6):
        raise ValueError(f"alpha + beta debe ser 1.0; recibido {alpha + beta}")
    a = alpha * positive_correlation(returns)
    if beta > 0.0:
        if sectors is None:
            raise ValueError("sectors requerido cuando beta > 0")
        if len(sectors) != returns.shape[1]:
            raise ValueError(f"len(sectors)={len(sectors)} != N={returns.shape[1]}")
        a = a + beta * sector_affinity(sectors)
    return a


def normalize_affinity(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalizar al intervalo [0, 1] dividiendo por el máximo.

    Args:
        matrix: matriz no negativa.
        eps: término pequeño para evitar división por cero.

    Returns:
        Matriz normalizada ``A_t = (matrix + eps) / max(matrix)``.
    """
    max_val = float(matrix.max())
    if max_val <= 0.0:
        # Caso degenerado: matriz toda cero. Devolver epsilon constante.
        return np.full_like(matrix, eps)
    return (matrix + eps) / max_val
