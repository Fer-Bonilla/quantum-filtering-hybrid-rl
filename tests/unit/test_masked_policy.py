"""Tests de src/agents/masked_policy.py."""

from __future__ import annotations

import pytest
import torch
from src.agents.masked_policy import (
    NEG_INF,
    apply_mask,
    masked_categorical,
    masked_entropy,
    masked_log_prob,
)


def test_apply_mask_sets_neg_inf_where_false() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, False, True, False]])
    out = apply_mask(logits, mask)
    assert out[0, 1].item() == NEG_INF
    assert out[0, 3].item() == NEG_INF
    assert out[0, 0].item() == 1.0
    assert out[0, 2].item() == 3.0


def test_apply_mask_rejects_empty_row() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[False, False, False]])
    with pytest.raises(ValueError):
        apply_mask(logits, mask)


def test_apply_mask_shape_mismatch() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False]])
    with pytest.raises(ValueError):
        apply_mask(logits, mask)


def test_masked_categorical_concentrates_probability() -> None:
    """Probabilidades suman 1 sobre el conjunto permitido; 0 fuera de él."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, True, False, False]])
    dist = masked_categorical(logits, mask)
    probs = dist.probs  # type: ignore[attr-defined]
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert probs[0, 2].item() < 1e-9
    assert probs[0, 3].item() < 1e-9
    assert probs[0, 0].item() + probs[0, 1].item() == pytest.approx(1.0, abs=1e-6)


def test_masked_log_prob_correct_value() -> None:
    logits = torch.tensor([[1.0, 2.0]])  # iguales antes de mask
    mask = torch.tensor([[True, True]])
    actions = torch.tensor([0])
    log_p = masked_log_prob(logits, mask, actions)
    # softmax([1,2]) = [exp(1), exp(2)] / (exp(1)+exp(2)) -> action 0 prob ≈ 0.2689
    expected = torch.log(torch.tensor(0.26894142))
    assert torch.isclose(log_p, expected, atol=1e-5)


def test_masked_entropy_no_nan_with_masked_positions() -> None:
    """Test crítico: la entropía debe ser finita aún con posiciones enmascaradas."""
    logits = torch.randn(8, 16)
    mask = torch.rand(8, 16) > 0.5
    # Garantizar al menos una posición True por fila
    mask[:, 0] = True
    h = masked_entropy(logits, mask)
    assert h.shape == (8,)
    assert torch.all(torch.isfinite(h)), "Entropía con NaN o inf"
    assert torch.all(h >= -1e-6)  # entropía >= 0 (modulo error num.)


def test_masked_log_prob_is_differentiable() -> None:
    """Los gradientes deben propagar por las posiciones válidas (no por NEG_INF)."""
    logits = torch.randn(4, 8, requires_grad=True)
    mask = torch.rand(4, 8) > 0.3
    mask[:, 0] = True
    actions = torch.zeros(4, dtype=torch.long)
    log_p = masked_log_prob(logits, mask, actions)
    log_p.sum().backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_masked_categorical_unique_action_deterministic() -> None:
    """Si la máscara deja solo una acción, esa acción tiene prob = 1."""
    logits = torch.randn(2, 5)
    mask = torch.tensor([[False, True, False, False, False], [False, False, False, False, True]])
    dist = masked_categorical(logits, mask)
    probs = dist.probs  # type: ignore[attr-defined]
    assert torch.isclose(probs[0, 1], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(probs[1, 4], torch.tensor(1.0), atol=1e-6)
