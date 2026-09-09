"""Red actor-critic discreta para PPO.

Arquitectura mínima (Sec. 5.5.5):
- Codificador MLP del estado.
- Cabeza de política (logits sobre N acciones).
- Cabeza de valor V(s_t) (scalar).

El MLP se comparte entre las dos cabezas para reducir parámetros y mejorar
estabilidad. Inicialización ortogonal recomendada por el paper original de
PPO y replicada por implementaciones de referencia (stable-baselines3,
CleanRL).
"""

from __future__ import annotations

import torch
from torch import nn


def _ortho_init(layer: nn.Linear, gain: float = 1.0) -> None:
    """Inicialización ortogonal + bias 0 (recomendada para PPO)."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class ActorCritic(nn.Module):
    """Red actor-critic con tronco compartido."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: list[int] | tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        if obs_dim < 1:
            raise ValueError(f"obs_dim debe ser >= 1; recibido {obs_dim}.")
        if n_actions < 2:
            raise ValueError(f"n_actions debe ser >= 2; recibido {n_actions}.")

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layer = nn.Linear(in_dim, h)
            _ortho_init(layer, gain=2.0**0.5)  # sqrt(2) para ReLU
            layers.extend([layer, nn.Tanh()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_dim, n_actions)
        self.value_head = nn.Linear(in_dim, 1)
        # Inicialización pequeña en la política para empezar cerca de uniforme.
        _ortho_init(self.policy_head, gain=0.01)
        _ortho_init(self.value_head, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computar (logits, valor) en un solo forward.

        Args:
            obs: Tensor float ``(B, obs_dim)`` o ``(obs_dim,)``.

        Returns:
            ``logits`` shape ``(B, n_actions)`` y ``value`` shape ``(B,)``.
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        features = self.trunk(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        if squeeze:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value

    def value_only(self, obs: torch.Tensor) -> torch.Tensor:
        """Camino especializado para bootstrap del último estado del rollout."""
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        features = self.trunk(obs)
        v = self.value_head(features).squeeze(-1)
        return v.squeeze(0) if squeeze else v
