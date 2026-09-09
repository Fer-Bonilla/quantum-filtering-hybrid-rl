"""Casos límite y propiedades del bootstrap pareado unificado (rev. 7)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from src.utils.paired_stats import paired_bootstrap


def test_all_zero_is_degenerate_without_p():
    r = paired_bootstrap(np.zeros(10), seed=1)
    assert r.degenerate is True
    assert math.isnan(r.p_two) and math.isnan(r.p_greater)
    assert r.ci_lo == r.ci_hi == 0.0


def test_all_equal_nonzero_is_degenerate():
    r = paired_bootstrap(np.full(6, 0.3), seed=1)
    assert r.degenerate is True
    assert r.mean == pytest.approx(0.3)


def test_single_nonzero_gives_p_in_unit_interval_not_zero():
    d = np.array([0.0] * 9 + [0.01])
    r = paired_bootstrap(d, seed=1)
    assert r.degenerate is False
    assert 0.0 < r.p_two <= 1.0
    # la formula estricta antigua daba 0 aqui; la inclusiva no
    assert r.p_two > 0.5


def test_p_never_exceeds_one():
    rng = np.random.default_rng(3)
    for _ in range(50):
        d = rng.choice([-1.0, 0.0, 1.0], size=8)
        if np.all(d == d[0]):
            continue
        r = paired_bootstrap(d, n_boot=500, rng=rng)
        assert 0.0 <= r.p_two <= 1.0
        assert 0.0 <= r.p_greater <= 1.0


def test_n_equal_two():
    r = paired_bootstrap([0.1, -0.1], n_boot=1000, seed=5)
    assert r.n == 2 and not r.degenerate
    assert 0.0 <= r.p_two <= 1.0


def test_strong_effect_small_p_and_ci_excludes_zero():
    rng = np.random.default_rng(11)
    d = rng.normal(0.5, 0.05, 20)
    r = paired_bootstrap(d, rng=rng)
    assert r.p_two < 0.001
    assert r.ci_lo > 0.0


def test_deterministic_with_seed():
    d = np.random.default_rng(0).normal(0.0, 1.0, 10)
    a = paired_bootstrap(d, seed=2026)
    b = paired_bootstrap(d, seed=2026)
    assert a == b


def test_matches_legacy_convention_on_continuous_data():
    """Con datos continuos (sin empates en cero) la regla inclusiva coincide
    con la antigua 2*min(P(>0), 1-P(>0)) de aggregate_results."""
    d = np.random.default_rng(4).normal(0.01, 0.02, 10)
    rng = np.random.default_rng(2026)
    boot = np.array([rng.choice(d, d.size, replace=True).mean() for _ in range(5000)])
    legacy = 2 * min((boot > 0).mean(), 1 - (boot > 0).mean())
    r = paired_bootstrap(d, rng=np.random.default_rng(2026))
    assert r.p_two == pytest.approx(legacy)


def test_rejects_too_few_values():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], seed=1)
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, float("nan")], seed=1)
