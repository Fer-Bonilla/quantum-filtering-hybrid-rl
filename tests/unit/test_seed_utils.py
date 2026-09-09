"""Tests de src/training/seed_utils.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from src.training.seed_utils import set_global_seed


def test_set_global_seed_returns_generator() -> None:
    g = set_global_seed(7)
    assert isinstance(g, np.random.Generator)


def test_set_global_seed_makes_numpy_deterministic() -> None:
    set_global_seed(42)
    a = np.random.random(5)
    set_global_seed(42)
    b = np.random.random(5)
    np.testing.assert_array_equal(a, b)


def test_set_global_seed_makes_torch_deterministic() -> None:
    set_global_seed(42)
    a = torch.randn(5)
    set_global_seed(42)
    b = torch.randn(5)
    torch.testing.assert_close(a, b)


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError):
        set_global_seed(-1)
