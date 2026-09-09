"""Buffer de rollout para PPO con máscara dinámica.

Crítico (plan riesgo "Máscara dinámica rompe PPO"):
- Guarda ``mask_t`` junto a ``(s_t, a_t, log_pi_t, V_t, r_t, done_t)``.
- En el update, la máscara ALMACENADA se reaplica (no recalcular). Esto
  garantiza que el ratio ``pi_new / pi_old`` se evalúa bajo la misma
  distribución soportada.
- Computa ventajas via GAE (Schulman et al., 2016).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class RolloutBuffer:
    """Buffer plano de rollout con campos pre-allocados."""

    n_steps: int
    obs_dim: int
    n_actions: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = "cpu"

    obs: np.ndarray = None  # type: ignore[assignment]
    actions: np.ndarray = None  # type: ignore[assignment]
    log_probs: np.ndarray = None  # type: ignore[assignment]
    values: np.ndarray = None  # type: ignore[assignment]
    rewards: np.ndarray = None  # type: ignore[assignment]
    dones: np.ndarray = None  # type: ignore[assignment]
    masks: np.ndarray = None  # type: ignore[assignment]
    advantages: np.ndarray = None  # type: ignore[assignment]
    returns: np.ndarray = None  # type: ignore[assignment]
    _ptr: int = 0

    def __post_init__(self) -> None:
        self.obs = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.n_steps, dtype=np.int64)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self.values = np.zeros(self.n_steps, dtype=np.float32)
        self.rewards = np.zeros(self.n_steps, dtype=np.float32)
        self.dones = np.zeros(self.n_steps, dtype=np.float32)
        self.masks = np.ones((self.n_steps, self.n_actions), dtype=bool)
        self.advantages = np.zeros(self.n_steps, dtype=np.float32)
        self.returns = np.zeros(self.n_steps, dtype=np.float32)
        self._ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        mask: np.ndarray,
    ) -> None:
        if self._ptr >= self.n_steps:
            raise RuntimeError("Buffer lleno; llamar reset() antes de add().")
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.log_probs[self._ptr] = log_prob
        self.values[self._ptr] = value
        self.rewards[self._ptr] = reward
        self.dones[self._ptr] = float(done)
        self.masks[self._ptr] = mask
        self._ptr += 1

    @property
    def is_full(self) -> bool:
        return self._ptr >= self.n_steps

    @property
    def size(self) -> int:
        return self._ptr

    def reset(self) -> None:
        self._ptr = 0
        self.advantages.fill(0.0)
        self.returns.fill(0.0)

    def compute_gae(self, last_value: float, last_done: bool) -> None:
        """Computar ventajas via GAE y retornos via TD(λ).

        Args:
            last_value: V(s_T) para bootstrap (V del estado tras el último
                paso del rollout).
            last_done: ``done`` flag del último step.
        """
        if not self.is_full:
            raise RuntimeError("compute_gae() requiere buffer lleno.")
        gae = 0.0
        next_value = last_value
        next_non_terminal = 1.0 - float(last_done)
        for t in reversed(range(self.n_steps)):
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            next_value = self.values[t]
            next_non_terminal = 1.0 - self.dones[t]
        self.returns[:] = self.advantages + self.values

    def iter_minibatches(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> RolloutMinibatchIterator:
        """Iterar minibatches barajados (todos los pasos del rollout)."""
        return RolloutMinibatchIterator(self, batch_size, rng)


class RolloutMinibatchIterator:
    """Iterador de minibatches sobre el buffer (índices barajados)."""

    def __init__(
        self,
        buffer: RolloutBuffer,
        batch_size: int,
        rng: np.random.Generator,
    ) -> None:
        self._buffer = buffer
        self._batch_size = batch_size
        self._indices = rng.permutation(buffer.n_steps)
        self._cursor = 0

    def __iter__(self) -> RolloutMinibatchIterator:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._cursor >= len(self._indices):
            raise StopIteration
        idx = self._indices[self._cursor : self._cursor + self._batch_size]
        self._cursor += self._batch_size
        return {
            "obs": torch.from_numpy(self._buffer.obs[idx]),
            "actions": torch.from_numpy(self._buffer.actions[idx]),
            "old_log_probs": torch.from_numpy(self._buffer.log_probs[idx]),
            "advantages": torch.from_numpy(self._buffer.advantages[idx]),
            "returns": torch.from_numpy(self._buffer.returns[idx]),
            "masks": torch.from_numpy(self._buffer.masks[idx]),
        }
