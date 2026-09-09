"""Tests de src/agents/oracle_mask.py (máscara-oráculo v7)."""

from __future__ import annotations

import numpy as np
import pytest
from src.agents.oracle_mask import top_m_future_mask


def test_selects_actual_top_m() -> None:
    fr = np.array([0.1, 0.5, 0.2, 0.9, 0.3])
    mask = top_m_future_mask(fr, m=2)
    assert mask.sum() == 2
    assert set(np.flatnonzero(mask).tolist()) == {3, 1}  # 0.9 y 0.5


def test_m_ge_n_all_true() -> None:
    fr = np.array([0.1, 0.2, 0.3])
    np.testing.assert_array_equal(top_m_future_mask(fr, m=10), np.ones(3, dtype=bool))


def test_m_zero_all_false() -> None:
    fr = np.array([0.1, 0.2, 0.3])
    assert top_m_future_mask(fr, m=0).sum() == 0


def test_exact_count() -> None:
    rng = np.random.default_rng(0)
    fr = rng.normal(size=30)
    for m in (1, 3, 5, 10):
        assert top_m_future_mask(fr, m).sum() == m


def test_nan_not_selected_when_finite_available() -> None:
    fr = np.array([np.nan, 0.1, 0.2, np.nan, 0.05])
    mask = top_m_future_mask(fr, m=2)
    assert set(np.flatnonzero(mask).tolist()) == {1, 2}  # ignora los NaN
    assert not mask[0] and not mask[3]


def test_negative_returns_handled() -> None:
    fr = np.array([-0.5, -0.1, -0.9, -0.2])
    mask = top_m_future_mask(fr, m=2)
    assert set(np.flatnonzero(mask).tolist()) == {1, 3}  # los menos negativos


def test_rejects_2d() -> None:
    with pytest.raises(ValueError):
        top_m_future_mask(np.zeros((3, 3)), m=2)
