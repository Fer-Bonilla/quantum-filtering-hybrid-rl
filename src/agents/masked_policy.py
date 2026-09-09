"""Política categórica con máscara de acciones (Sec. 5.7 / 6.9).

Implementa:

    pi_eff(u_t | s_t, C_t) = pi_c(u_t | s_t) * 1[u_t in C_t]
                             / sum_{j in C_t} pi_c(j | s_t)

Detalles críticos:
- ``NEG_INF = -1e9`` (no ``-inf``): evita NaN en ``log_softmax`` cuando una
  fila tiene posiciones enmascaradas. ``-inf`` produciría ``-inf * 0 = NaN``
  en el cálculo de entropía.
- Pre-assert: la máscara debe tener al menos una posición True por fila.
- Gradientes: ``masked_fill`` no propaga gradiente por las posiciones -1e9.
"""

from __future__ import annotations

from typing import Final

import torch
from torch.distributions import Categorical

NEG_INF: Final[float] = -1e9
"""Valor usado para enmascarar logits. -1e9 (no -inf) por estabilidad."""


def apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Aplicar máscara a logits.

    Args:
        logits: Tensor float de shape ``(..., N)``.
        mask: Tensor bool de shape ``(..., N)``; True donde la acción está
            permitida.

    Returns:
        Tensor con la misma shape; valores ``NEG_INF`` donde mask es False.

    Raises:
        ValueError: Si alguna fila de mask es completamente False.
    """
    if logits.shape != mask.shape:
        raise ValueError(
            f"logits y mask deben tener mismo shape; recibido {tuple(logits.shape)} "
            f"vs {tuple(mask.shape)}."
        )
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("Máscara con al menos una fila completamente False.")
    return logits.masked_fill(~mask, NEG_INF)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Construir una ``Categorical`` con logits enmascarados."""
    return Categorical(logits=apply_mask(logits, mask))


def masked_log_prob(
    logits: torch.Tensor,
    mask: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Log-probabilidad de ``actions`` bajo la distribución enmascarada.

    Args:
        logits: ``(..., N)``.
        mask: ``(..., N)`` bool.
        actions: ``(...,)`` long.

    Returns:
        Tensor ``(...,)`` de log-probabilidades.

    Notes:
        El caller DEBE garantizar que ``mask.gather(-1, actions.unsqueeze(-1))``
        sea True para todos los samples (la acción tomada debe ser válida bajo
        la máscara con la que se evalúa). Esto se verifica en el rollout
        buffer con ``test_rollout_mask_consistency``.
    """
    masked_logits = apply_mask(logits, mask)
    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
    return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Entropía de la distribución enmascarada.

    Usa ``probs * log_probs`` con manejo seguro de los términos enmascarados:
    cuando la prob es 0 (porque su logit era ``NEG_INF``), el término
    ``0 * log(0)`` se sustituye por 0 (límite correcto).
    """
    masked_logits = apply_mask(logits, mask)
    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
    probs = torch.exp(log_probs)
    # En posiciones enmascaradas probs ~ 0; evitamos 0 * (-inf) usando where.
    safe_term = torch.where(mask, probs * log_probs, torch.zeros_like(probs))
    return -safe_term.sum(dim=-1)
