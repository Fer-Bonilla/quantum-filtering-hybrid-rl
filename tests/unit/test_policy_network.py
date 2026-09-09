"""Tests de src/agents/policy_network.py."""

from __future__ import annotations

import pytest
import torch
from src.agents.policy_network import ActorCritic


def test_actor_critic_forward_batch() -> None:
    net = ActorCritic(obs_dim=10, n_actions=4)
    obs = torch.randn(8, 10)
    logits, values = net(obs)
    assert logits.shape == (8, 4)
    assert values.shape == (8,)
    assert torch.all(torch.isfinite(logits))
    assert torch.all(torch.isfinite(values))


def test_actor_critic_forward_single_obs() -> None:
    net = ActorCritic(obs_dim=10, n_actions=4)
    obs = torch.randn(10)
    logits, values = net(obs)
    assert logits.shape == (4,)
    assert values.shape == ()


def test_actor_critic_value_only_consistent() -> None:
    net = ActorCritic(obs_dim=10, n_actions=4)
    obs = torch.randn(8, 10)
    _, v_full = net(obs)
    v_only = net.value_only(obs)
    torch.testing.assert_close(v_full, v_only, atol=1e-6, rtol=0)


def test_actor_critic_grad_flows() -> None:
    net = ActorCritic(obs_dim=4, n_actions=3)
    obs = torch.randn(2, 4, requires_grad=False)
    logits, values = net(obs)
    loss = logits.mean() + values.mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert all(torch.all(torch.isfinite(g)) for g in grads)


def test_invalid_dims_raise() -> None:
    with pytest.raises(ValueError):
        ActorCritic(obs_dim=0, n_actions=4)
    with pytest.raises(ValueError):
        ActorCritic(obs_dim=10, n_actions=1)
