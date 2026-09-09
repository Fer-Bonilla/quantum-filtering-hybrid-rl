"""Agente PPO clásico (usado por modelos A y B).

API minimalista para el trainer:
    act(obs, mask, rng) -> (action, log_prob, value)
    evaluate(obs_batch, mask_batch, actions_batch) -> (log_probs, values, entropy)

El agente no conoce el grafo ni el módulo cuántico; la máscara se le
proporciona externamente (en A es todo True; en B viene del entorno con
rasgos relacionales pre-computados).
"""

from __future__ import annotations

import numpy as np
import torch

from src.agents.masked_policy import masked_categorical, masked_entropy, masked_log_prob
from src.agents.policy_network import ActorCritic


class ClassicalAgent:
    """Agente PPO discreto con política enmascarada."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: list[int] | tuple[int, ...] = (128, 128),
        device: str = "cpu",
    ) -> None:
        self._network = ActorCritic(obs_dim, n_actions, hidden_sizes).to(device)
        self._device = torch.device(device)
        self._n_actions = n_actions

    @property
    def network(self) -> ActorCritic:
        return self._network

    @property
    def device(self) -> torch.device:
        return self._device

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, float, float]:
        """Muestrear acción + log_prob + valor para un único estado.

        El muestreo usa el RNG de PyTorch (que comparte semilla con el del
        trainer si éste hace ``torch.manual_seed`` al inicio).
        ``rng`` se acepta por simetría con la futura API del agente híbrido.
        """
        del rng  # parámetro reservado para extensiones híbridas
        obs_t = torch.from_numpy(obs).to(self._device)
        mask_t = torch.from_numpy(mask).to(self._device)
        logits, value = self._network(obs_t)
        dist = masked_categorical(logits.unsqueeze(0), mask_t.unsqueeze(0))
        action = dist.sample().squeeze(0)
        log_prob = dist.log_prob(action).squeeze(0)
        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        """V(s) sin máscara (la cabeza de valor no se enmascara)."""
        obs_t = torch.from_numpy(obs).to(self._device)
        return float(self._network.value_only(obs_t).item())

    def evaluate(
        self,
        obs_batch: torch.Tensor,
        mask_batch: torch.Tensor,
        actions_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluar log_probs, valores y entropía sobre un minibatch.

        Usado en el update PPO. La máscara que se pasa es la **almacenada**
        en el rollout buffer; NO se recalcula.
        """
        obs_batch = obs_batch.to(self._device)
        mask_batch = mask_batch.to(self._device)
        actions_batch = actions_batch.to(self._device)
        logits, values = self._network(obs_batch)
        log_probs = masked_log_prob(logits, mask_batch, actions_batch)
        entropy = masked_entropy(logits, mask_batch)
        return log_probs, values, entropy

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._network.state_dict()

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        self._network.load_state_dict(sd)
