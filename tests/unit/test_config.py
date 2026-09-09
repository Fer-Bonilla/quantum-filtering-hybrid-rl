"""Tests de src/utils/config.py: validación pydantic y composición YAML."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from src.utils.config import (
    DataConfig,
    ExperimentConfig,
    GraphConfig,
    PPOConfig,
    QuantumConfig,
    load_config,
)
from src.utils.paths import CONFIGS_DIR

# ---------------------------------------------------------------------------
# Validación cruzada modelo ↔ secciones requeridas
# ---------------------------------------------------------------------------


def _base_config(model: str) -> dict:
    return {
        "data": {
            "tickers": ["AAPL", "MSFT"],
            "start_date": "2020-01-01",
            "end_date": "2022-12-31",
        },
        "env": {},
        "agent": {"model": model},
        "seeds": [42],
    }


def test_model_a_rejects_graph_and_quantum() -> None:
    cfg = _base_config("A")
    cfg["graph"] = {}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)


def test_model_b_requires_graph_rejects_quantum() -> None:
    cfg = _base_config("B")
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)  # falta graph

    cfg["graph"] = {}
    cfg["quantum"] = {}
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)


def test_model_c_requires_both() -> None:
    cfg = _base_config("C")
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(cfg)

    cfg["graph"] = {}
    cfg["quantum"] = {}
    parsed = ExperimentConfig.model_validate(cfg)
    assert parsed.agent.model == "C"


def test_model_d_requires_both() -> None:
    cfg = _base_config("D")
    cfg["graph"] = {}
    cfg["quantum"] = {}
    parsed = ExperimentConfig.model_validate(cfg)
    assert parsed.agent.model == "D"


# ---------------------------------------------------------------------------
# Validaciones por sección
# ---------------------------------------------------------------------------


def test_data_end_after_start() -> None:
    with pytest.raises(ValidationError):
        DataConfig.model_validate(
            {
                "tickers": ["AAPL", "MSFT"],
                "start_date": "2022-01-01",
                "end_date": "2021-01-01",
            }
        )


def test_graph_alpha_plus_beta_equals_one() -> None:
    with pytest.raises(ValidationError):
        GraphConfig.model_validate({"alpha": 0.5, "beta": 0.3})


def test_graph_default_alpha_only() -> None:
    cfg = GraphConfig()
    assert cfg.alpha == 1.0
    assert cfg.beta == 0.0


def test_ppo_lr_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        PPOConfig.model_validate({"learning_rate": -0.001})


def test_quantum_default_backend_matrix() -> None:
    cfg = QuantumConfig()
    assert cfg.backend == "matrix"
    assert cfg.k_steps >= 1


# ---------------------------------------------------------------------------
# Loader con composición de defaults
# ---------------------------------------------------------------------------


def test_load_model_a_from_repo() -> None:
    cfg = load_config(CONFIGS_DIR / "experiment" / "model_a.yaml")
    assert cfg.agent.model == "A"
    assert cfg.graph is None
    assert cfg.quantum is None
    assert len(cfg.data.tickers) >= 10


def test_load_model_b_from_repo() -> None:
    cfg = load_config(CONFIGS_DIR / "experiment" / "model_b.yaml")
    assert cfg.agent.model == "B"
    assert cfg.graph is not None
    assert cfg.quantum is None


def test_load_model_c_from_repo() -> None:
    cfg = load_config(CONFIGS_DIR / "experiment" / "model_c.yaml")
    assert cfg.agent.model == "C"
    assert cfg.graph is not None
    assert cfg.quantum is not None
    assert cfg.quantum.k_steps >= 1
    assert cfg.quantum.m_top >= 1


def test_load_model_d_from_repo() -> None:
    cfg = load_config(CONFIGS_DIR / "experiment" / "model_d.yaml")
    assert cfg.agent.model == "D"
    assert cfg.graph is not None
    assert cfg.quantum is not None


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("does/not/exist.yaml"))
