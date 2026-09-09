"""Tests de la vía suave (v8 — EXP-6) y de los walkers informados (EXP-7)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.masked_policy import (
    NEG_INF,
    masked_categorical,
    masked_entropy,
    masked_log_prob,
)
from src.agents.soft_hybrid import (
    BiasedClassicalAgent,
    SoftRolloutBuffer,
    biased_categorical,
    biased_entropy,
    biased_log_prob,
    bool_to_bias,
)
from src.graph.subgraph_selector import Subgraph
from src.quantum.informed_walkers import (
    MomentumRefreshWalker,
    SoftmaxWalker,
    ThompsonWalker,
)


def _sub(M: int = 8) -> Subgraph:
    rng = np.random.default_rng(0)
    A = rng.uniform(0.05, 1.0, size=(M, M))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0.0)
    return Subgraph(W_local=A, global_node_ids=np.arange(10, 10 + M),
                    seed_idx_local=0)


# ------------------------- vía suave -------------------------


def test_bias_equivale_a_mascara_booleana() -> None:
    """Con sesgo 0/NEG_INF, la política sesgada coincide con la enmascarada."""
    torch.manual_seed(0)
    logits = torch.randn(4, 10)
    mask = torch.zeros(4, 10, dtype=torch.bool)
    mask[:, :5] = True
    bias = torch.from_numpy(bool_to_bias(mask.numpy()))
    acts = torch.randint(0, 5, (4,))
    lp_m = masked_log_prob(logits, mask, acts)
    lp_b = biased_log_prob(logits, bias, acts)
    assert torch.allclose(lp_m, lp_b, atol=1e-5)
    assert torch.allclose(masked_entropy(logits, mask),
                          biased_entropy(logits, bias), atol=1e-5)
    assert torch.allclose(masked_categorical(logits, mask).probs,
                          biased_categorical(logits, bias).probs, atol=1e-6)


def test_bias_gradua_probabilidades() -> None:
    logits = torch.zeros(1, 4)
    bias = torch.tensor([[0.0, np.log(2.0), NEG_INF, NEG_INF]])
    probs = biased_categorical(logits, bias).probs.squeeze(0)
    assert probs[2] < 1e-6 and probs[3] < 1e-6
    assert pytest.approx(float(probs[1] / probs[0]), rel=1e-4) == 2.0


def test_bias_sin_soporte_falla() -> None:
    logits = torch.zeros(1, 3)
    bias = torch.full((1, 3), NEG_INF)
    with pytest.raises(ValueError):
        biased_categorical(logits, bias)


def test_soft_buffer_almacena_float() -> None:
    buf = SoftRolloutBuffer(n_steps=4, obs_dim=3, n_actions=5,
                            gamma=0.99, gae_lambda=0.95)
    assert buf.masks.dtype == np.float32
    bias = np.array([0.0, -0.5, NEG_INF, 0.2, NEG_INF], dtype=np.float32)
    buf.add(np.zeros(3, dtype=np.float32), 0, 0.0, 0.0, 0.0, False, bias)
    assert np.allclose(buf.masks[0], bias)


def test_biased_agent_actua_en_soporte() -> None:
    agent = BiasedClassicalAgent(obs_dim=3, n_actions=6)
    bias = np.full(6, NEG_INF, dtype=np.float32)
    bias[2] = 0.0
    bias[4] = 1.0
    for _ in range(20):
        a, lp, v = agent.act(np.zeros(3, dtype=np.float32), bias)
        assert a in (2, 4)
        assert np.isfinite(lp) and np.isfinite(v)


# ------------------------- walkers informados -------------------------


def test_softmax_walker_muestrea_sin_reemplazo() -> None:
    w = SoftmaxWalker(seed=1, tau=1.0)
    out = w.candidate_set(_sub(), k=3, m=3)
    assert len(out) == len(set(out.tolist())) == 3
    assert set(out.tolist()) <= set(range(10, 18))


def test_softmax_walker_rota_entre_llamadas() -> None:
    w = SoftmaxWalker(seed=2, tau=2.0)
    sets = {tuple(sorted(w.candidate_set(_sub(), 3, 3).tolist()))
            for _ in range(25)}
    assert len(sets) > 1  # estocástico => rotación


def test_momentum_walker_elige_top_momentum() -> None:
    T, N = 60, 20
    panel = np.zeros((T, N))
    panel[:, 12] = 0.05  # global id 12 con momentum máximo
    w = MomentumRefreshWalker(returns_panel=panel)
    w.t = 50
    out = w.candidate_set(_sub(), k=3, m=1)
    assert out.tolist() == [12]


def test_thompson_walker_actualiza_solo_pasado() -> None:
    rng = np.random.default_rng(3)
    panel = rng.normal(0, 0.01, size=(80, 20))
    w = ThompsonWalker(returns_panel=panel, seed=4)
    w.t = 30
    _ = w.candidate_set(_sub(), 3, 3)
    assert w._resolved == 25  # t - horizonte
    total = w._alpha.sum() + w._beta.sum()
    assert total == pytest.approx(2 * 20 + 26 * 20)  # prior + 26 filas resueltas
    out = w.candidate_set(_sub(), 3, 3)
    assert len(out) == 3
