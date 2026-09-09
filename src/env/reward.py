"""Función de recompensa del entorno (Sec. 6.7 / 8.14).

Soporta dos esquemas de recompensa (rev. v2 — Problema 1.3 del revisor):

1. ``risk_penalty`` (esquema original, retro-compatible):

       r_t = R_{u_t, t+1} - lambda * sigma_hat_{u_t, t} - mu * c_t

2. ``log_wealth`` (esquema nuevo, escala log-utilidad):

       r_t = log(1 + R_{u_t, t+1})
             - lambda * sigma_hat_{u_t, t}
             - mu * c_t

   La transformación log estabiliza retornos extremos y se alinea con
   maximización de log-wealth (criterio de Kelly). Bajo este esquema
   se recomienda lambda < 0,1 para que la penalización no domine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

RewardType = Literal["risk_penalty", "log_wealth"]


@dataclass(frozen=True, slots=True)
class RewardCoefficients:
    """Coeficientes de la recompensa."""

    lambda_risk: float = 0.1
    """Penalización por volatilidad."""

    mu_cost: float = 0.001
    """Coeficiente de costos."""

    transaction_cost: float = 0.0005
    """Costo aplicado cuando hay cambio de posición (u_t != u_{t-1})."""

    reward_type: RewardType = "risk_penalty"
    """Esquema de recompensa: ``risk_penalty`` (original) o
    ``log_wealth`` (rev. v2). Por defecto se mantiene el original para
    retro-compatibilidad con campañas anteriores."""


def compute_reward(
    realized_return: float,
    realized_volatility: float,
    cost: float,
    coefs: RewardCoefficients,
) -> float:
    """Computar la recompensa instantánea.

    Args:
        realized_return: R_{u_t, t+1} ya observado.
        realized_volatility: sigma_hat_{u_t, t} estimada con info <= t.
        cost: costo de transacción acumulado en este step (>= 0).
        coefs: coeficientes; el campo ``reward_type`` selecciona el
            esquema (``risk_penalty`` o ``log_wealth``).

    Returns:
        float r_t.
    """
    if coefs.reward_type == "log_wealth":
        return compute_reward_log_wealth(realized_return, realized_volatility, cost, coefs)
    return realized_return - coefs.lambda_risk * realized_volatility - coefs.mu_cost * cost


def compute_reward_log_wealth(
    realized_return: float,
    realized_volatility: float,
    cost: float,
    coefs: RewardCoefficients,
) -> float:
    """Recompensa basada en log-wealth (criterio de Kelly).

    Definición:
        r_t = log(1 + R) - lambda * sigma - mu * c

    Caracterizada por amplificar las pérdidas pequeñas y comprimir las
    ganancias grandes en relación al esquema lineal. Más estable
    numéricamente para retornos extremos y compatible con
    interpretación financiera de log-wealth.

    Args:
        realized_return: R \\in (-1, +\\infty).
        realized_volatility, cost, coefs: como en ``compute_reward``.

    Returns:
        float r_t. Si ``1 + R \\le 0`` (quiebra teórica), retorna
        ``-1e3`` como sanción acotada para evitar -inf en el rollout.
    """
    one_plus_r = 1.0 + realized_return
    if one_plus_r <= 0.0:
        log_term = -1000.0  # sanción acotada por quiebra teórica
    else:
        log_term = math.log(one_plus_r)
    return log_term - coefs.lambda_risk * realized_volatility - coefs.mu_cost * cost


def transaction_cost_for(
    action: int,
    previous_action: int | None,
    base_cost: float,
) -> float:
    """Costo asociado al cambio de posición.

    Si la posición no cambia (o no hay posición previa), se devuelve 0.
    """
    if previous_action is None or action == previous_action:
        return 0.0
    return base_cost
