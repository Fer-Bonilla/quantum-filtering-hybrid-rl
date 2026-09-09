"""Tests de src/agents/classical_agent.py."""

from __future__ import annotations

import numpy as np
import torch
from src.agents.classical_agent import ClassicalAgent


def test_classical_agent_act_returns_int_and_floats() -> None:
    agent = ClassicalAgent(obs_dim=12, n_actions=4)
    obs = np.random.default_rng(0).standard_normal(12).astype(np.float32)
    mask = np.array([True, True, False, True])
    a, lp, v = agent.act(obs, mask)
    assert isinstance(a, int)
    assert 0 <= a < 4
    assert a != 2  # acción 2 está enmascarada
    assert isinstance(lp, float) and np.isfinite(lp)
    assert isinstance(v, float) and np.isfinite(v)


def test_classical_agent_evaluate_shapes() -> None:
    agent = ClassicalAgent(obs_dim=6, n_actions=3)
    obs_b = torch.randn(5, 6)
    mask_b = torch.tensor([[True, True, True]] * 5)
    actions_b = torch.tensor([0, 1, 2, 0, 1])
    lp, v, e = agent.evaluate(obs_b, mask_b, actions_b)
    assert lp.shape == (5,)
    assert v.shape == (5,)
    assert e.shape == (5,)
    assert torch.all(torch.isfinite(lp))
    assert torch.all(torch.isfinite(v))
    assert torch.all(torch.isfinite(e))


def test_classical_agent_deterministic_seeded() -> None:
    torch.manual_seed(123)
    agent_a = ClassicalAgent(obs_dim=6, n_actions=3)
    torch.manual_seed(123)
    agent_b = ClassicalAgent(obs_dim=6, n_actions=3)
    obs = np.zeros(6, dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    # Mismo seed PyTorch -> mismos pesos -> misma decisión determinista en muestreo.
    torch.manual_seed(0)
    a1, _, _ = agent_a.act(obs, mask)
    torch.manual_seed(0)
    a2, _, _ = agent_b.act(obs, mask)
    assert a1 == a2
