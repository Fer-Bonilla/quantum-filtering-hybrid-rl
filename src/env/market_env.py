"""MarketEnv: entorno Gymnasium para selección discreta de activos.

El entorno NO construye el grafo G_t ni el subgrafo H_t. Eso es
responsabilidad del agente híbrido (modelos C y D). El entorno expone
``info["candidate_mask"]`` que por defecto es todo True; un wrapper o el
agente puede sobreescribir la máscara antes de muestrear la acción.

Contrato (Sec. 5.6 / 6.6):
    reset(seed) -> (obs, info)
    step(action) -> (obs, reward, terminated, truncated, info)
    info["candidate_mask"]: np.ndarray[N] bool   (True donde u in C_t)
    info["asset_ids"]:      np.ndarray[N] str   (tickers ordenados)
    info["t"]:              pd.Timestamp del paso actual

Espacios:
    action_space      = Discrete(N)
    observation_space = Box(low=-inf, high=+inf, shape=(L*N*d,), float32)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.env.episode_sampler import EpisodeSampler
from src.env.reward import RewardCoefficients, compute_reward, transaction_cost_for
from src.env.state_builder import StateBuilder, infer_n_tickers_and_features
from src.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass(slots=True)
class MarketEnvSpec:
    """Especificación del entorno."""

    window_length: int = 20
    max_steps: int = 252
    reward_coefs: RewardCoefficients = field(default_factory=RewardCoefficients)
    episode_sampling: str = "random"
    return_feature_name: str = "return"
    """Nombre de la feature usada como retorno realizado del activo."""

    volatility_feature_name: str = "volatility"
    """Nombre de la feature usada como sigma_hat."""


class MarketEnv(gym.Env):
    """Entorno Gymnasium para selección discreta de un activo por step.

    Cada paso `t`:
        1. El agente recibe ``obs = phi(X_t)``.
        2. Elige una acción ``u_t in {0, ..., N-1}`` (índice del ticker).
        3. El entorno computa la recompensa:
               r_t = R_{u_t, t+1} - lambda * sigma_hat_{u_t, t} - mu * cost
           donde R_{u_t, t+1} es la feature ``return`` en `t+1` del ticker
           seleccionado (anticipándose al primer step temporal).
        4. Se incrementa `t` y se devuelve la nueva observación.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        spec: MarketEnvSpec | None = None,
        *,
        relational_features_fn: Callable[[int], np.ndarray] | None = None,
        relational_features_dim: int = 0,
    ) -> None:
        """Constructor del entorno.

        Args:
            features: panel multi-ticker con MultiIndex (ticker, feature).
            spec: especificación del entorno (ventana, recompensa, etc.).
            relational_features_fn: función opcional que, dado un índice
                temporal ``t``, devuelve una matriz ``(N, d_rel)`` con
                rasgos agregados del grafo dinámico ``G_t``. Cuando se
                proporciona, ``MarketEnv`` la invoca en cada ``step()`` y
                ``reset()`` y concatena el resultado al estado clásico.
                Garantía: la función SOLO debe usar ``data[:t]``
                (responsabilidad del caller; ver
                ``test_B_no_information_leak``).
            relational_features_dim: ``d_rel`` esperado. Necesario para
                pre-calcular ``observation_space``; si ``relational_features_fn``
                es ``None``, se ignora.
        """
        if not isinstance(features.columns, pd.MultiIndex):
            raise TypeError("features debe tener columnas MultiIndex (ticker, feature).")
        if not features.index.is_monotonic_increasing:
            raise ValueError("features debe tener índice monotónicamente creciente.")

        self._features = features
        self._spec = spec if spec is not None else MarketEnvSpec()
        self._tickers: list[str] = sorted({c[0] for c in features.columns})
        self._n_tickers, self._n_features = infer_n_tickers_and_features(features)

        if self._spec.return_feature_name not in {c[1] for c in features.columns}:
            raise ValueError(
                f"Feature '{self._spec.return_feature_name}' no presente "
                f"(disponibles: {sorted({c[1] for c in features.columns})})."
            )

        self._state_builder = StateBuilder(window_length=self._spec.window_length)
        self._episode_sampler = EpisodeSampler(
            n_rows=len(features),
            window_length=self._spec.window_length,
            max_steps=self._spec.max_steps,
            mode=self._spec.episode_sampling,  # type: ignore[arg-type]
        )
        self._relational_features_fn = relational_features_fn
        self._relational_features_dim = relational_features_dim

        self.action_space = spaces.Discrete(self._n_tickers)
        base_shape = self._state_builder.state_shape(self._n_tickers, self._n_features)
        if relational_features_fn is not None and relational_features_dim > 0:
            obs_shape = (base_shape[0] + self._n_tickers * relational_features_dim,)
        else:
            obs_shape = base_shape
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        # Estado interno (poblado en reset)
        self._rng: np.random.Generator | None = None
        self._t: int = 0
        self._episode_start: int = 0
        self._episode_steps: int = 0
        self._previous_action: int | None = None

        # Cache de matrices NumPy para acceso O(1) durante step()
        self._return_matrix: np.ndarray = self._extract_field_matrix(
            self._spec.return_feature_name
        )
        self._volatility_matrix: np.ndarray | None = None
        if self._spec.volatility_feature_name in {c[1] for c in features.columns}:
            self._volatility_matrix = self._extract_field_matrix(
                self._spec.volatility_feature_name
            )
        # Lista posicional de tickers para info["asset_ids"]
        self._asset_ids = np.array(self._tickers, dtype=object)

        # Asegurar que el entorno tiene como mínimo un step accesible
        if self._spec.max_steps + self._spec.window_length >= len(features):
            raise ValueError(
                f"Features insuficientes ({len(features)}) para window_length="
                f"{self._spec.window_length} + max_steps={self._spec.max_steps}."
            )

    # ------------------------------------------------------------------
    # API Gymnasium
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)
        # Sample un único start: cada reset = un episodio independiente
        self._episode_start = next(self._episode_sampler.iter_starts(self._rng, 1))
        self._t = self._episode_start
        self._episode_steps = 0
        self._previous_action = None
        return self._build_obs(), self._build_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Acción {action} fuera de [0, {self._n_tickers}).")
        if self._rng is None:
            raise RuntimeError("step() invocado antes de reset().")

        # 1. Computar recompensa con R_{u_t, t+1}
        t_next = self._t + 1
        if t_next >= len(self._features):
            # Sin retorno futuro disponible -> truncar inmediatamente
            obs = self._build_obs()
            return obs, 0.0, False, True, self._build_info()

        realized_return = float(self._return_matrix[t_next, int(action)])
        if self._volatility_matrix is not None:
            realized_vol = float(self._volatility_matrix[self._t, int(action)])
        else:
            realized_vol = 0.0

        cost = transaction_cost_for(
            int(action),
            self._previous_action,
            self._spec.reward_coefs.transaction_cost,
        )
        reward = compute_reward(
            realized_return,
            realized_vol,
            cost,
            self._spec.reward_coefs,
        )

        # 2. Avanzar el reloj
        self._t = t_next
        self._previous_action = int(action)
        self._episode_steps += 1

        terminated = False
        truncated = (
            self._episode_steps >= self._spec.max_steps
            or self._t + 1 >= len(self._features)
        )
        obs = self._build_obs()
        info = self._build_info()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Accesores públicos
    # ------------------------------------------------------------------

    @property
    def tickers(self) -> list[str]:
        return list(self._tickers)

    @property
    def n_tickers(self) -> int:
        return self._n_tickers

    @property
    def current_t(self) -> int:
        return self._t

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _extract_field_matrix(self, field_name: str) -> np.ndarray:
        """Construir matriz [n_rows, n_tickers] para un nombre de feature."""
        cols = [(t, field_name) for t in self._tickers]
        for col in cols:
            if col not in self._features.columns:
                raise ValueError(f"Columna ausente: {col}")
        return self._features.loc[:, cols].to_numpy(dtype=np.float32, copy=True)

    def _build_obs(self) -> np.ndarray:
        rel = None
        if self._relational_features_fn is not None:
            rel = self._relational_features_fn(self._t)
            if rel.shape != (self._n_tickers, self._relational_features_dim):
                raise ValueError(
                    "relational_features_fn devolvió shape "
                    f"{rel.shape}; esperado ({self._n_tickers}, {self._relational_features_dim})"
                )
        return self._state_builder.build(
            self._features, self._t, relational_features=rel
        )

    def _build_info(self) -> dict[str, Any]:
        mask = np.ones(self._n_tickers, dtype=bool)
        return {
            "candidate_mask": mask,
            "asset_ids": self._asset_ids,
            "t": self._features.index[self._t],
            "episode_step": self._episode_steps,
        }


class RecordingMarketEnv(MarketEnv):
    """Subclase instrumentada para detectar fuga de información futura.

    Cualquier acceso a ``_return_matrix`` o ``_volatility_matrix`` en
    posiciones ``t'`` se registra en ``self.accessed_future_indices`` si
    ``t' > self._t``. Usado por ``test_env_no_future``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.accessed_future_indices: list[int] = []
        self._cutoff: int = -1

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._cutoff = self._t  # snapshot ANTES del step
        # Wrappear las matrices para registrar accesos
        original_return = self._return_matrix
        original_vol = self._volatility_matrix
        self._return_matrix = _AccessRecorder(
            original_return, cutoff=self._cutoff + 1, log=self.accessed_future_indices
        )  # type: ignore[assignment]
        if original_vol is not None:
            self._volatility_matrix = _AccessRecorder(
                original_vol, cutoff=self._cutoff, log=self.accessed_future_indices
            )  # type: ignore[assignment]
        try:
            return super().step(action)
        finally:
            self._return_matrix = original_return
            self._volatility_matrix = original_vol


class _AccessRecorder:
    """Wrapper de np.ndarray que registra accesos a posiciones > cutoff."""

    def __init__(self, arr: np.ndarray, *, cutoff: int, log: list[int]) -> None:
        self._arr = arr
        self._cutoff = cutoff
        self._log = log

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple) and isinstance(key[0], int):
            if key[0] > self._cutoff:
                self._log.append(key[0])
        return self._arr[key]
