"""Loop de entrenamiento PPO.

El trainer es ESTRUCTURAL: agnóstico al modelo (A, B, C, D). La diferencia
entre modelos vive en (1) el entorno (si construye o no rasgos relacionales)
y (2) el agente híbrido que sobreescribe la máscara antes de cada step.

Para Modelo A: env devuelve mask=all-True, agente es ``ClassicalAgent``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.classical_agent import ClassicalAgent
from src.agents.rollout_buffer import RolloutBuffer
from src.env.market_env import MarketEnv
from src.training.mlflow_logger import MLflowLogger
from src.utils.logging import get_logger

_log = get_logger(__name__)


class StepHook(Protocol):
    """Hook opcional que se invoca antes de cada step para modificar la máscara.

    Usado por modelos C/D para inyectar el conjunto candidato top-m.
    Para Modelo A/B, el hook es None y la máscara queda como el env la entrega.
    """

    def __call__(self, env: MarketEnv, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """Retornar una máscara bool de tamaño N (sobreescribe info['candidate_mask'])."""
        ...


@dataclass(slots=True)
class PPOHyperparams:
    """Hiperparámetros del optimizador PPO."""

    learning_rate: float = 3e-4
    n_epochs: int = 4
    batch_size: int = 64
    rollout_steps: int = 2048
    clip_coef: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True


@dataclass(slots=True)
class TrainResult:
    """Resultado agregado de una sesión de entrenamiento."""

    total_steps: int
    train_curve: list[float] = field(default_factory=list)
    """Recompensa media por rollout."""

    losses: dict[str, list[float]] = field(default_factory=dict)
    """Series de losses (policy, value, entropy, kl)."""


class PPOTrainer:
    """Trainer PPO discreto con máscara dinámica."""

    def __init__(
        self,
        env: MarketEnv,
        agent: ClassicalAgent,
        hp: PPOHyperparams,
        *,
        rng: np.random.Generator,
        logger: MLflowLogger | None = None,
        step_hook: StepHook | None = None,
    ) -> None:
        self._env = env
        self._agent = agent
        self._hp = hp
        self._rng = rng
        self._logger = logger
        self._step_hook = step_hook
        obs_shape = env.observation_space.shape
        if obs_shape is None or len(obs_shape) != 1:
            raise ValueError("Solo se soporta observation_space.shape de 1 dimensión.")
        self._buffer = RolloutBuffer(
            n_steps=hp.rollout_steps,
            obs_dim=obs_shape[0],
            n_actions=int(env.action_space.n),
            gamma=hp.gamma,
            gae_lambda=hp.gae_lambda,
        )
        self._optimizer = torch.optim.Adam(
            self._agent.network.parameters(),
            lr=hp.learning_rate,
        )

    def train(self, total_steps: int) -> TrainResult:
        """Entrenar durante ``total_steps`` pasos de interacción con el entorno."""
        result = TrainResult(total_steps=total_steps)
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
            result.losses[key] = []

        steps_done = 0
        obs, info = self._env.reset(seed=int(self._rng.integers(0, 1_000_000)))

        while steps_done < total_steps:
            obs, info, ep_rewards = self._collect_rollout(obs, info)
            steps_done += self._buffer.n_steps
            mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            result.train_curve.append(mean_reward)

            losses = self._update()
            for k, v in losses.items():
                result.losses[k].append(v)

            if self._logger is not None:
                self._logger.log_metrics(
                    {
                        "train/reward_mean": mean_reward,
                        **{f"train/{k}": v for k, v in losses.items()},
                    },
                    step=steps_done,
                )

            _log.info(
                "step=%d reward=%.4f policy=%.4f value=%.4f entropy=%.4f kl=%.4f",
                steps_done,
                mean_reward,
                losses["policy_loss"],
                losses["value_loss"],
                losses["entropy"],
                losses["approx_kl"],
            )

        return result

    def _collect_rollout(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any], list[float]]:
        """Recolectar un rollout completo, devolviendo (last_obs, last_info, episode_rewards)."""
        self._buffer.reset()
        episode_rewards: list[float] = []
        current_episode_reward = 0.0

        for _ in range(self._buffer.n_steps):
            mask = self._compute_mask(obs, info)
            action, log_prob, value = self._agent.act(obs, mask, self._rng)
            next_obs, reward, terminated, truncated, next_info = self._env.step(action)
            done = bool(terminated or truncated)
            self._buffer.add(obs, action, log_prob, value, reward, done, mask)
            current_episode_reward += reward
            obs = next_obs
            info = next_info
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                obs, info = self._env.reset(seed=int(self._rng.integers(0, 1_000_000)))

        last_value = self._agent.value(obs)
        last_done = bool(info.get("episode_step", 0) == 0)  # tras reset
        self._buffer.compute_gae(last_value=last_value, last_done=last_done)
        return obs, info, episode_rewards

    def _compute_mask(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """Determinar la máscara final del step.

        - Sin hook: usa ``info["candidate_mask"]`` (que para Modelo A es all-True).
        - Con hook: el hook calcula la máscara (usado en C/D).
        """
        if self._step_hook is None:
            return info["candidate_mask"].astype(bool, copy=False)
        new_mask = self._step_hook(self._env, obs, info)
        if new_mask.dtype != bool or new_mask.shape != info["candidate_mask"].shape:
            raise ValueError(
                f"step_hook debe devolver bool[{info['candidate_mask'].shape}]; "
                f"recibido dtype={new_mask.dtype}, shape={new_mask.shape}."
            )
        if not new_mask.any():
            raise RuntimeError("step_hook devolvió máscara completamente vacía.")
        return new_mask

    def _update(self) -> dict[str, float]:
        """Actualizar la red mediante n_epochs sobre minibatches del buffer."""
        advantages = torch.from_numpy(self._buffer.advantages)
        if self._hp.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            self._buffer.advantages[:] = advantages.numpy()

        agg_policy = 0.0
        agg_value = 0.0
        agg_entropy = 0.0
        agg_kl = 0.0
        n_batches = 0

        for _ in range(self._hp.n_epochs):
            for batch in self._buffer.iter_minibatches(self._hp.batch_size, self._rng):
                obs_b = batch["obs"]
                actions_b = batch["actions"]
                old_log_b = batch["old_log_probs"]
                adv_b = batch["advantages"]
                ret_b = batch["returns"]
                mask_b = batch["masks"]

                log_probs, values, entropy = self._agent.evaluate(obs_b, mask_b, actions_b)
                ratio = torch.exp(log_probs - old_log_b)
                unclipped = ratio * adv_b
                clipped = (
                    torch.clamp(ratio, 1.0 - self._hp.clip_coef, 1.0 + self._hp.clip_coef) * adv_b
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, ret_b)
                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + self._hp.value_coef * value_loss
                    + self._hp.entropy_coef * entropy_loss
                )
                with torch.no_grad():
                    approx_kl = (old_log_b - log_probs).mean()

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._agent.network.parameters(), self._hp.max_grad_norm
                )
                self._optimizer.step()

                agg_policy += float(policy_loss.item())
                agg_value += float(value_loss.item())
                agg_entropy += float(entropy.mean().item())
                agg_kl += float(approx_kl.item())
                n_batches += 1

        return {
            "policy_loss": agg_policy / max(n_batches, 1),
            "value_loss": agg_value / max(n_batches, 1),
            "entropy": agg_entropy / max(n_batches, 1),
            "approx_kl": agg_kl / max(n_batches, 1),
        }
