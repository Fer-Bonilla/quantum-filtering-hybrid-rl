"""Configuración del experimento validada con pydantic.

Cinco grupos jerárquicos (data, env, agent, graph, quantum) compuestos en
`ExperimentConfig`. Soporta composición vía clave ``defaults:`` similar a
Hydra (loader manual, sin dependencia de Hydra).

Validación cruzada (Apéndice del plan):
    model == "A" ⇒ graph = None ∧ quantum = None
    model == "B" ⇒ graph requerido, quantum = None
    model == "C" ⇒ ambos requeridos, quantum.backend ignorado en runtime
    model == "D" ⇒ ambos requeridos, quantum activo
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.utils.paths import CONFIGS_DIR

# ---------------------------------------------------------------------------
# Grupos individuales
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    """Configuración del pipeline de datos."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(..., min_length=2, description="Tickers del universo")
    start_date: date = Field(..., description="Fecha inicial (inclusive)")
    end_date: date = Field(..., description="Fecha final (inclusive)")
    source: Literal["yahoo"] = "yahoo"
    interval: Literal["1d", "1wk", "1mo"] = "1d"
    cache_subdir: str = Field("nivel1", description="Subdirectorio en data/raw")
    auto_adjust: bool = True

    @model_validator(mode="after")
    def _validate_dates(self) -> DataConfig:
        if self.end_date <= self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) debe ser posterior a start_date ({self.start_date})"
            )
        return self


class FeatureConfig(BaseModel):
    """Configuración de la generación de features (X_t)."""

    model_config = ConfigDict(extra="forbid")

    window_length: int = Field(20, ge=2, description="Longitud L de la ventana del estado")
    volatility_window: int = Field(20, ge=2)
    volume_window: int = Field(20, ge=2)
    return_type: Literal["simple", "log"] = "log"
    indicators: list[Literal["rsi", "macd", "bb_width"]] = Field(default_factory=list)


class SplitConfig(BaseModel):
    """Partición cronológica train/val/test."""

    model_config = ConfigDict(extra="forbid")

    train_frac: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.6
    val_frac: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.2

    @model_validator(mode="after")
    def _validate_fractions(self) -> SplitConfig:
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError(
                f"train_frac+val_frac ({self.train_frac + self.val_frac}) "
                "debe ser < 1.0 para dejar fracción de prueba > 0."
            )
        return self


class EnvConfig(BaseModel):
    """Configuración del entorno Gymnasium."""

    model_config = ConfigDict(extra="forbid")

    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    lambda_risk: float = Field(0.1, ge=0.0, description="Penalización por volatilidad")
    mu_cost: float = Field(0.001, ge=0.0, description="Coeficiente costos de transacción")
    transaction_cost: float = Field(0.0005, ge=0.0)
    max_steps: int = Field(252, ge=10, description="Pasos máximos por episodio (1 año bursátil)")
    episode_sampling: Literal["random", "sequential"] = "random"
    reward_type: Literal["risk_penalty", "log_wealth"] = Field(
        "risk_penalty",
        description=(
            "Esquema de recompensa. 'risk_penalty' (default, retro-compatible) "
            "usa r = R - lambda*sigma - mu*c; 'log_wealth' (rev. v2) usa "
            "r = log(1+R) - lambda*sigma - mu*c."
        ),
    )
    use_relational_features: bool = Field(
        False,
        description=(
            "Si True, activa el state_builder relacional del Modelo B v2. "
            "El estado se aumenta con (N x d_rel) rasgos del grafo G_t."
        ),
    )


class PPOConfig(BaseModel):
    """Hiperparámetros del optimizador PPO."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(3e-4, gt=0.0)
    n_epochs: int = Field(4, ge=1)
    batch_size: int = Field(64, ge=1)
    rollout_steps: int = Field(2048, ge=64)
    clip_coef: float = Field(0.2, gt=0.0, lt=1.0)
    gamma: float = Field(0.99, gt=0.0, le=1.0)
    gae_lambda: float = Field(0.95, gt=0.0, le=1.0)
    entropy_coef: float = Field(0.01, ge=0.0)
    value_coef: float = Field(0.5, ge=0.0)
    max_grad_norm: float = Field(0.5, gt=0.0)
    hidden_sizes: list[int] = Field(default_factory=lambda: [128, 128])


class AgentConfig(BaseModel):
    """Configuración del agente RL."""

    model_config = ConfigDict(extra="forbid")

    model: Literal["A", "B", "C", "D"]
    ppo: PPOConfig = Field(default_factory=PPOConfig)


class GraphConfig(BaseModel):
    """Configuración del grafo dinámico G_t (Sec. 6.4 / 7.5-7.9)."""

    model_config = ConfigDict(extra="forbid")

    alpha: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    beta: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    eps: float = Field(1e-8, gt=0.0)
    lookback_window: int = Field(60, ge=10, description="Lt: ventana para correlación móvil")
    k_neighbors: int = Field(5, ge=1)
    sym_mode: Literal["avg", "max", "mutual"] = "avg"
    subgraph_max_size: int = Field(8, ge=2, le=16, description="M: tamaño máximo del subgrafo")
    seed_score_window: int = Field(20, ge=2, description="Ls: ventana del score semilla")
    update_frequency: int = Field(1, ge=1, description="Reconstruir G_t cada N steps")

    @model_validator(mode="after")
    def _validate_mix_weights(self) -> GraphConfig:
        # Tolerancia para errores de redondeo del YAML
        if abs(self.alpha + self.beta - 1.0) > 1e-6:
            raise ValueError(f"alpha+beta debe sumar 1.0 (recibido {self.alpha + self.beta})")
        return self


class NoiseConfig(BaseModel):
    """Configuración opcional de modelo de ruido para PennyLane/Qiskit."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    depolarizing_prob: float = Field(0.0, ge=0.0, le=1.0)
    dephasing_prob: float = Field(0.0, ge=0.0, le=1.0)


class QuantumConfig(BaseModel):
    """Configuración del módulo cuántico DTQW (Sec. 6.10 / 7.10-7.17)."""

    model_config = ConfigDict(extra="forbid")

    k_steps: int = Field(3, ge=1, le=20)
    m_top: int = Field(3, ge=1, description="Tamaño del conjunto candidato top-m")
    backend: Literal["matrix", "pennylane"] = "matrix"
    shots: int | None = Field(None, ge=1, description="Solo aplica si backend=pennylane")
    init_mode: Literal["uniform", "seed_centered"] = "uniform"
    renormalize_threshold: float = Field(1e-9, gt=0.0, description="Tolerancia |‖ψ‖²-1|")
    noise: NoiseConfig = Field(default_factory=NoiseConfig)


class TrainingConfig(BaseModel):
    """Configuración del loop de entrenamiento y trazabilidad."""

    model_config = ConfigDict(extra="forbid")

    total_steps: int = Field(50_000, ge=100)
    eval_every: int = Field(2_000, ge=10)
    snapshot_every: int = Field(10_000, ge=100)
    mlflow_experiment_name: str = "model_default"
    mlflow_tracking_uri: str | None = Field(
        None, description="None ⇒ outputs/mlruns (file-store local)"
    )


# ---------------------------------------------------------------------------
# Configuración compuesta
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Configuración completa de un experimento."""

    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    env: EnvConfig
    agent: AgentConfig
    graph: GraphConfig | None = None
    quantum: QuantumConfig | None = None
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    seeds: list[int] = Field(default_factory=lambda: [42])

    @model_validator(mode="after")
    def _validate_model_requirements(self) -> ExperimentConfig:
        model = self.agent.model
        if model == "A":
            if self.graph is not None or self.quantum is not None:
                raise ValueError("Modelo A no debe definir graph ni quantum")
        elif model == "B":
            if self.graph is None:
                raise ValueError("Modelo B requiere graph")
            if self.quantum is not None:
                raise ValueError("Modelo B no debe definir quantum")
        elif model == "C":
            if self.graph is None or self.quantum is None:
                raise ValueError("Modelo C requiere graph y quantum (k_steps, m_top)")
        elif model == "D" and (self.graph is None or self.quantum is None):
            raise ValueError("Modelo D requiere graph y quantum")
        return self


# ---------------------------------------------------------------------------
# Loader con composición estilo Hydra (ligero)
# ---------------------------------------------------------------------------


def _resolve_defaults(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolver la clave ``defaults`` mergeando los YAML referenciados.

    Sintaxis admitida en el YAML raíz::

        defaults:
          - ../data/nivel1
          - ../env/default

    El merge es por claves de primer nivel: el archivo raíz tiene prioridad
    sobre los defaults (las claves duplicadas se conservan del raíz).
    """
    defaults = raw.pop("defaults", [])
    if not defaults:
        return raw

    merged: dict[str, Any] = {}
    for ref in defaults:
        ref_path = (base_dir / f"{ref}.yaml").resolve()
        if not ref_path.is_file():
            raise FileNotFoundError(f"defaults referenciado no encontrado: {ref_path}")
        with ref_path.open("r", encoding="utf-8") as fh:
            sub = yaml.safe_load(fh) or {}
        # Cada YAML de defaults aporta una sola sección de primer nivel
        # (e.g. "data:" o "env:"); merge se hace por sección.
        if not isinstance(sub, dict):
            raise ValueError(f"Default {ref_path} debe contener un mapping en su raíz")
        for key, value in sub.items():
            if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

    # Las claves del archivo raíz sobreescriben las heredadas
    for key, value in raw.items():
        if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_config(path: Path | str) -> ExperimentConfig:
    """Cargar y validar un experimento desde un archivo YAML.

    Args:
        path: Ruta al YAML del experimento. Puede ser absoluta o relativa al
            directorio actual; si no existe, se intenta como relativa a
            ``configs/``.

    Returns:
        ExperimentConfig validado.

    Raises:
        FileNotFoundError: Si no se encuentra el archivo.
        pydantic.ValidationError: Si la config no cumple los esquemas.
    """
    path = Path(path)
    if not path.is_absolute():
        candidates = [Path.cwd() / path, CONFIGS_DIR / path]
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"Config no encontrada: {path} (probados: {candidates})")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"El YAML debe contener un mapping en la raíz: {path}")

    merged = _resolve_defaults(raw, base_dir=path.parent)
    return ExperimentConfig.model_validate(merged)
