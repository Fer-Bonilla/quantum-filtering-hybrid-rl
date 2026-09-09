"""Evaluación en partición de prueba.

Métricas (Sec. 8.16-8.17):
- Financieras: retorno acumulado, Sharpe, máximo drawdown, mean return.
- Aprendizaje: episodios hasta convergencia, recompensa media.
- Exploración: top-m hit rate sobre activos prometedores, cobertura, tiempo
  hasta primer acierto.
- Módulo: latencia por step (ms).

Definición operativa de "activo prometedor" (Sec. 8.17):

    g_i(h) = mean(R_{i, t+1 : t+h}) / (std(R_{i, t+1 : t+h}) + eps)

Un activo i se considera prometedor en t si g_i(h) está en el top 20% del
universo elegible. **Se usa exclusivamente ex-post; nunca como feature.**
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.agents.classical_agent import ClassicalAgent
from src.env.market_env import MarketEnv


class StepHook(Protocol):
    def __call__(self, env: MarketEnv, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Resultado de evaluación en una partición."""

    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    mean_reward: float
    n_episodes: int
    n_steps: int
    asset_coverage: float
    """Fracción de tickers seleccionados al menos una vez."""

    mean_latency_ms: float
    """Latencia media por step (incluye step_hook + agent.act)."""

    topm_hit_rate: float
    """Frecuencia con que la acción elegida está entre activos prometedores."""

    candidate_hit_rate: float
    """Frecuencia con que el conjunto candidato C_t contiene algún activo prometedor."""

    time_to_first_promising: float
    """Pasos hasta seleccionar por primera vez un activo prometedor (inf si nunca)."""


# ---------------------------------------------------------------------------
# Activos prometedores (Sec. 8.17, post-hoc)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromisingSpec:
    """Configuración del criterio post-hoc de activo prometedor."""

    horizon: int = 5
    """h pasos hacia adelante."""

    percentile: float = 80.0
    """Top percentil que define 'prometedor' (e.g. 80 => top 20%)."""

    eps: float = 1e-8


def compute_promising_matrix(
    returns_panel: np.ndarray,
    spec: PromisingSpec,
) -> np.ndarray:
    """Matriz booleana ``(T, N)``: True si activo i es prometedor en t.

    Para cada t, computa el score de horizonte h y selecciona el percentil
    superior. Las últimas `h` filas se rellenan con False (no hay horizonte
    suficiente para evaluar).

    Args:
        returns_panel: matriz ``(T, N)`` con retornos por activo (los mismos
            que el env usa para R_{u_t, t+1}).
        spec: configuración del horizonte y percentil.

    Returns:
        Matriz ``(T, N)`` dtype bool.
    """
    T, N = returns_panel.shape
    out = np.zeros((T, N), dtype=bool)
    h = spec.horizon
    if h >= T:
        return out  # horizonte demasiado largo
    threshold_pct = spec.percentile

    for t in range(T - h):
        window = returns_panel[t + 1 : t + 1 + h]  # h filas
        mean = window.mean(axis=0)
        std = window.std(axis=0, ddof=1) if h > 1 else np.zeros(N)
        score = mean / (std + spec.eps)
        cutoff = float(np.percentile(score, threshold_pct))
        out[t] = score >= cutoff
    return out


# ---------------------------------------------------------------------------
# Convergencia
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Métricas de convergencia (rev. v2 — Problema 1.1 del revisor)
#
# La heurística original (ventana móvil con tolerancia relativa al rollout
# anterior) colapsaba a un valor único entre seeds porque PPO produce
# estimaciones iniciales ruidosas que rápidamente entran en banda del 5 %
# del rollout previo, dando un mismo índice en todos los runs. Para
# resolverlo se implementan tres definiciones complementarias:
#
#   * ``convergence_to_final``  — primer rollout cuya media móvil dista
#       menos que un umbral mixto (tol_rel · |final| + tol_abs) del valor
#       final (mean de los últimos ``window_last`` rollouts).  Es la métrica
#       PRIMARIA reportada.
#   * ``convergence_stability``  — primer rollout en que la desviación
#       estándar dentro de la ventana cae por debajo de tol_std relativo
#       (estabilidad estadística independiente del valor final).
#   * ``convergence_plateau_after_peak`` — primer rollout posterior al
#       pico de la media móvil con pendiente menor que ``slope_tol``
#       (capta plateaux tras alcanzar un máximo, útil cuando hay
#       sobreajuste).
# ---------------------------------------------------------------------------


def convergence_to_final(
    reward_curve: list[float] | np.ndarray,
    *,
    window: int = 3,
    window_last: int = 3,
    tol_rel: float = 0.05,
    tol_abs: float = 1e-3,
) -> float:
    """Primer rollout cuya ventana móvil dista menos que ``tol`` del valor final.

    Definición:
        ``reward_final = mean(curve[-window_last:])``
        ``threshold   = |reward_final| · tol_rel + tol_abs``
        Devuelve el primer ``i`` tal que
        ``|mean(curve[i:i+window]) - reward_final| < threshold``.

    Es la métrica PRIMARIA reportada para Hipótesis 8.4.1 (eficiencia de
    aprendizaje). Produce varianza no nula entre semillas porque depende
    del valor final, no de diferencias rollout-a-rollout.

    Args:
        reward_curve: serie de recompensas medias por rollout.
        window: tamaño de la ventana móvil dentro de la curva.
        window_last: nº de rollouts finales sobre los que se calcula la
            referencia ``reward_final``.
        tol_rel: tolerancia relativa al valor final.
        tol_abs: tolerancia absoluta mínima (estabiliza valores cercanos a 0).

    Returns:
        Índice 1-indexado del primer rollout que cumple el criterio; ``inf``
        si no se alcanza dentro de la curva.
    """
    arr = np.asarray(reward_curve, dtype=np.float64)
    if arr.size < window + window_last:
        return float("inf")
    reward_final = float(np.mean(arr[-window_last:]))
    threshold = abs(reward_final) * tol_rel + tol_abs
    for i in range(arr.size - window):
        mean_w = float(np.mean(arr[i : i + window]))
        if abs(mean_w - reward_final) < threshold:
            return float(i + 1)
    return float("inf")


def convergence_stability(
    reward_curve: list[float] | np.ndarray,
    *,
    window: int = 3,
    tol_std: float = 0.05,
    eps: float = 1e-8,
) -> float:
    """Primer rollout cuya ventana móvil tiene std/|mean| < ``tol_std``.

    Métrica SECUNDARIA: estabilidad estadística independiente del valor
    final. Útil cuando la curva alcanza un plateau ruidoso vs uno limpio.

    Returns:
        Índice 1-indexado; ``inf`` si nunca se alcanza.
    """
    arr = np.asarray(reward_curve, dtype=np.float64)
    if arr.size < window + 1:
        return float("inf")
    for i in range(arr.size - window):
        window_slice = arr[i : i + window]
        mean_w = float(np.mean(window_slice))
        std_w = float(np.std(window_slice, ddof=0))
        denom = max(abs(mean_w), eps)
        if std_w / denom < tol_std:
            return float(i + 1)
    return float("inf")


def convergence_plateau_after_peak(
    reward_curve: list[float] | np.ndarray,
    *,
    window: int = 3,
    slope_tol: float = 0.01,
) -> float:
    """Primer rollout posterior al pico con pendiente local menor que ``slope_tol``.

    Métrica TERCIARIA: detecta plateaux tras un máximo (útil con sobreajuste).
    Returns:
        Índice 1-indexado del primer plateau; ``inf`` si no se detecta.
    """
    arr = np.asarray(reward_curve, dtype=np.float64)
    if arr.size < 2 * window:
        return float("inf")
    # Media móvil de la curva
    cs = np.convolve(arr, np.ones(window) / window, mode="valid")
    peak_idx = int(np.argmax(cs))
    for i in range(peak_idx, len(cs) - window):
        # Pendiente local sobre la siguiente ventana
        y = cs[i : i + window]
        x = np.arange(window, dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0])
        # Normalizar pendiente por la escala de la curva
        scale = max(abs(float(np.mean(cs))), 1e-8)
        if abs(slope) / scale < slope_tol:
            return float(i + 1)
    return float("inf")


def episodes_to_convergence(
    reward_curve: list[float] | np.ndarray,
    *,
    window: int = 3,
    tol: float = 0.05,
) -> float:
    """Compatibilidad con código existente: delega en ``convergence_to_final``.

    La firma antigua se conserva para que tests previos sigan pasando, pero
    el comportamiento actual es el de la métrica primaria reescrita.
    Equivale a ``convergence_to_final(curve, window=window, tol_rel=tol)``.
    """
    return convergence_to_final(reward_curve, window=window, tol_rel=tol)


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------


def evaluate_agent(
    env: MarketEnv,
    agent: ClassicalAgent,
    *,
    n_episodes: int = 10,
    seed: int = 0,
    step_hook: StepHook | None = None,
    promising_matrix: np.ndarray | None = None,
) -> EvalResult:
    """Evaluar un agente en la partición que tiene el env.

    Args:
        env: MarketEnv ya cargado con la partición a evaluar.
        agent: ClassicalAgent entrenado.
        n_episodes: número de episodios.
        seed: semilla del muestreo.
        step_hook: hook opcional (modelos C/D) que sobreescribe la máscara.
        promising_matrix: matriz ``(T, N)`` precalculada con
            ``compute_promising_matrix(env.features['return'], spec)``. Si
            None, las métricas de exploración se devuelven con NaN.

    Returns:
        EvalResult con métricas financieras + exploración + latencia.
    """
    rng = np.random.default_rng(seed)
    rewards_per_episode: list[float] = []
    all_returns: list[float] = []
    visited_actions: set[int] = set()
    n_steps_total = 0
    latencies_ms: list[float] = []

    # Métricas de exploración
    topm_hits = 0
    candidate_hits = 0
    candidate_evals = 0
    time_to_first = float("inf")

    for _ in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        ep_reward = 0.0
        while True:
            t_idx = env.current_t
            t0 = time.perf_counter()
            if step_hook is not None:
                mask = step_hook(env, obs, info)
            else:
                mask = info["candidate_mask"].astype(bool, copy=False)
            action, _, _ = agent.act(obs, mask, rng)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            visited_actions.add(int(action))

            # Exploración: ¿la acción está en activos prometedores?
            if promising_matrix is not None and t_idx < promising_matrix.shape[0]:
                promising_row = promising_matrix[t_idx]
                if promising_row[int(action)]:
                    topm_hits += 1
                    if np.isinf(time_to_first):
                        time_to_first = float(n_steps_total + 1)
                # ¿el conjunto candidato contiene algún activo prometedor?
                candidate_evals += 1
                if (mask & promising_row).any():
                    candidate_hits += 1

            obs, reward, terminated, truncated, info = env.step(action)
            all_returns.append(float(reward))
            ep_reward += float(reward)
            n_steps_total += 1
            if terminated or truncated:
                break
        rewards_per_episode.append(ep_reward)

    returns_arr = np.array(all_returns, dtype=np.float64)
    cum_return = float(np.sum(returns_arr))
    sharpe = _sharpe(returns_arr)
    drawdown = _max_drawdown(returns_arr)
    coverage = len(visited_actions) / max(env.n_tickers, 1)
    mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    topm_rate = topm_hits / max(n_steps_total, 1) if promising_matrix is not None else float("nan")
    cand_rate = (
        candidate_hits / max(candidate_evals, 1) if promising_matrix is not None else float("nan")
    )

    return EvalResult(
        cumulative_return=cum_return,
        sharpe_ratio=sharpe,
        max_drawdown=drawdown,
        mean_reward=float(np.mean(rewards_per_episode)) if rewards_per_episode else 0.0,
        n_episodes=n_episodes,
        n_steps=n_steps_total,
        asset_coverage=coverage,
        mean_latency_ms=mean_latency,
        topm_hit_rate=topm_rate,
        candidate_hit_rate=cand_rate,
        time_to_first_promising=time_to_first,
    )


def _sharpe(returns: np.ndarray, eps: float = 1e-8) -> float:
    """Sharpe ratio (sin anualizar; aplicable a recompensas por step)."""
    if returns.size < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    return mean / (std + eps)


def _max_drawdown(returns: np.ndarray) -> float:
    """Máximo drawdown sobre la curva de retorno acumulado."""
    if returns.size == 0:
        return 0.0
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = running_max - cum
    return float(np.max(drawdown))
