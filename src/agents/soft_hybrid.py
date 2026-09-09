"""Integración suave de la señal de priorización (v8 — EXP-6).

En lugar de la máscara dura top-m, la señal del selector entra como sesgo
aditivo sobre los logits de la política:

    z' = z + B,   B_i = beta * log(P_i + eps)  si i in H_t;  NEG_INF si no,

con P = P_k de la DTQW (soft-D), p_k clásica (soft-C) o uniforme sobre H_t
(soft-R, el control correcto: subgrafo sin ranking). El sesgo B se almacena
en el búfer y se reaplica en el update, de modo que el cociente PPO queda
definido sobre la misma distribución sesgada (misma regla de consistencia que
la máscara dura).

Todo el camino booleano existente queda intacto: estas clases viven en
paralelo (regla 6 del pre-registro v8).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from src.agents.classical_agent import ClassicalAgent
from src.agents.hybrid_agent import HybridAgent
from src.agents.masked_policy import NEG_INF
from src.agents.rollout_buffer import RolloutBuffer
from src.env.market_env import MarketEnv
from src.graph.classical_walk import random_walk_distribution
from src.graph.graph_builder import build_graph
from src.graph.subgraph_selector import seed_score, select_subgraph
from src.quantum.dtqw import apply_dtqw
from src.training.evaluate import EvalResult, _max_drawdown, _sharpe
from src.training.trainer import PPOTrainer

_SUPPORT_THR = NEG_INF / 2.0
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Operaciones de política con sesgo aditivo
# ---------------------------------------------------------------------------


def apply_bias(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """logits + bias, validando que cada fila conserve soporte."""
    if logits.shape != bias.shape:
        raise ValueError(
            f"logits y bias deben tener mismo shape; {tuple(logits.shape)} "
            f"vs {tuple(bias.shape)}.")
    support = bias > _SUPPORT_THR
    if not torch.all(support.any(dim=-1)):
        raise ValueError("Sesgo con al menos una fila sin soporte.")
    return logits + bias


def biased_categorical(logits: torch.Tensor, bias: torch.Tensor) -> Categorical:
    return Categorical(logits=apply_bias(logits, bias))


def biased_log_prob(logits: torch.Tensor, bias: torch.Tensor,
                    actions: torch.Tensor) -> torch.Tensor:
    lp = torch.nn.functional.log_softmax(apply_bias(logits, bias), dim=-1)
    return lp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def biased_entropy(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    support = bias > _SUPPORT_THR
    lp = torch.nn.functional.log_softmax(apply_bias(logits, bias), dim=-1)
    probs = torch.exp(lp)
    safe = torch.where(support, probs * lp, torch.zeros_like(probs))
    return -safe.sum(dim=-1)


def bool_to_bias(mask: np.ndarray) -> np.ndarray:
    """Máscara booleana → sesgo equivalente (0 dentro, NEG_INF fuera)."""
    return np.where(mask, 0.0, NEG_INF).astype(np.float32)


# ---------------------------------------------------------------------------
# Agente, búfer y trainer sesgados (paralelos a los booleanos)
# ---------------------------------------------------------------------------


class BiasedClassicalAgent(ClassicalAgent):
    """ClassicalAgent cuyo argumento de máscara es un sesgo float32."""

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray,
            rng: np.random.Generator | None = None) -> tuple[int, float, float]:
        del rng
        obs_t = torch.from_numpy(obs).to(self.device)
        bias_t = torch.from_numpy(mask.astype(np.float32)).to(self.device)
        logits, value = self.network(obs_t)
        dist = biased_categorical(logits.unsqueeze(0), bias_t.unsqueeze(0))
        action = dist.sample().squeeze(0)
        log_prob = dist.log_prob(action).squeeze(0)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def evaluate(self, obs_batch: torch.Tensor, mask_batch: torch.Tensor,
                 actions_batch: torch.Tensor,
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_batch = obs_batch.to(self.device)
        bias_batch = mask_batch.to(self.device).float()
        actions_batch = actions_batch.to(self.device)
        logits, values = self.network(obs_batch)
        log_probs = biased_log_prob(logits, bias_batch, actions_batch)
        entropy = biased_entropy(logits, bias_batch)
        return log_probs, values, entropy


@dataclass
class SoftRolloutBuffer(RolloutBuffer):
    """RolloutBuffer con máscaras float32 (sesgos aditivos)."""

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        self.masks = np.zeros((self.n_steps, self.n_actions), dtype=np.float32)


class SoftPPOTrainer(PPOTrainer):
    """PPOTrainer cuyo step_hook devuelve sesgos float32."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        base = self._buffer
        self._buffer = SoftRolloutBuffer(
            n_steps=base.n_steps, obs_dim=base.obs_dim,
            n_actions=base.n_actions, gamma=base.gamma,
            gae_lambda=base.gae_lambda)

    def _compute_mask(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        if self._step_hook is None:
            return bool_to_bias(info["candidate_mask"].astype(bool, copy=False))
        bias = self._step_hook(self._env, obs, info)
        if bias.dtype not in (np.float32, np.float64):
            raise RuntimeError(f"soft step_hook debe devolver float; {bias.dtype}")
        if not (bias > _SUPPORT_THR).any():
            raise RuntimeError("soft step_hook devolvió sesgo sin soporte.")
        return bias.astype(np.float32, copy=False)


class SoftHybridAgent(HybridAgent):
    """HybridAgent que produce sesgos β·log(P+ε) sobre el soporte de H_t.

    ``mode``: 'softD' (P_k DTQW) | 'softC' (p_k clásica) | 'softR' (uniforme).
    """

    def __init__(self, *args, mode: str = "softD", beta_bias: float = 1.0,
                 init_mode: str = "uniform", renorm: float = 1e-9,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if mode not in ("softD", "softC", "softR"):
            raise ValueError(mode)
        self._mode = mode
        self._beta_bias = float(beta_bias)
        self._init_mode = init_mode
        self._renorm = renorm

    def make_mask(self, env: MarketEnv, obs: np.ndarray,
                  info: dict[str, Any]) -> np.ndarray:
        if (self._cached_mask is not None
                and self._step_counter % self._spec.update_frequency != 0):
            self._step_counter += 1
            return self._cached_mask.copy()

        t = int(env.current_t)
        graph_start = max(0, t - self._spec.graph_lookback + 1)
        seed_start = max(0, t - self._spec.seed_score_window + 1)
        n = self._n_tickers
        if graph_start == t or seed_start == t:
            self._step_counter += 1
            bias = np.zeros(n, dtype=np.float32)  # sin restricción (≡ A)
            self._cached_mask = bias
            return bias

        graph = build_graph(self._returns_panel[graph_start: t + 1],
                            self._tickers, self._sectors, self._spec.graph_spec)
        scores = seed_score(self._returns_panel[seed_start: t + 1])
        try:
            sub = select_subgraph(
                graph, scores, max_size=self._spec.subgraph_max_size,
                eligible_mask=info["candidate_mask"].astype(bool, copy=False))
        except ValueError:
            self._step_counter += 1
            bias = np.zeros(n, dtype=np.float32)
            self._cached_mask = bias
            return bias

        if self._mode == "softD":
            p = apply_dtqw(sub.W_local, sub.seed_idx_local, self._spec.k_steps,
                           init_mode=self._init_mode,
                           renormalize_threshold=self._renorm)
        elif self._mode == "softC":
            p = random_walk_distribution(sub.W_local, sub.seed_idx_local,
                                         self._spec.k_steps)
        else:  # softR
            p = np.full(sub.size, 1.0 / sub.size)

        bias = np.full(n, NEG_INF, dtype=np.float32)
        bias[sub.global_node_ids] = self._beta_bias * np.log(p + _EPS)
        self._cached_mask = bias
        self._step_counter += 1
        return bias


# ---------------------------------------------------------------------------
# Evaluación con sesgos (espejo fiel de evaluate_agent)
# ---------------------------------------------------------------------------


def evaluate_soft_agent(env: MarketEnv, agent: BiasedClassicalAgent, *,
                        n_episodes: int = 5, seed: int = 0, step_hook=None,
                        promising_matrix: np.ndarray | None = None) -> EvalResult:
    rng = np.random.default_rng(seed)
    rewards_per_episode: list[float] = []
    all_returns: list[float] = []
    visited: set[int] = set()
    n_steps_total = 0
    latencies: list[float] = []
    topm_hits = cand_hits = cand_evals = 0
    time_to_first = float("inf")

    for _ in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        ep_reward = 0.0
        while True:
            t_idx = env.current_t
            t0 = time.perf_counter()
            if step_hook is not None:
                bias = step_hook(env, obs, info)
            else:
                bias = bool_to_bias(info["candidate_mask"].astype(bool))
            action, _, _ = agent.act(obs, bias, rng)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            visited.add(int(action))
            if promising_matrix is not None and t_idx < promising_matrix.shape[0]:
                fila = promising_matrix[t_idx]
                if fila[int(action)]:
                    topm_hits += 1
                    if np.isinf(time_to_first):
                        time_to_first = float(n_steps_total + 1)
                cand_evals += 1
                support = bias > _SUPPORT_THR
                if (support & fila).any():
                    cand_hits += 1
            obs, reward, term, trunc, info = env.step(action)
            all_returns.append(float(reward))
            ep_reward += float(reward)
            n_steps_total += 1
            if term or trunc:
                break
        rewards_per_episode.append(ep_reward)

    arr = np.array(all_returns, dtype=np.float64)
    return EvalResult(
        cumulative_return=float(np.sum(arr)),
        sharpe_ratio=_sharpe(arr),
        max_drawdown=_max_drawdown(arr),
        mean_reward=float(np.mean(rewards_per_episode)) if rewards_per_episode else 0.0,
        n_episodes=n_episodes,
        n_steps=n_steps_total,
        asset_coverage=len(visited) / max(env.n_tickers, 1),
        mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        topm_hit_rate=(topm_hits / max(n_steps_total, 1)
                       if promising_matrix is not None else float("nan")),
        candidate_hit_rate=(cand_hits / max(cand_evals, 1)
                            if promising_matrix is not None else float("nan")),
        time_to_first_promising=time_to_first,
    )
