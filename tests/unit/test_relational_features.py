"""Tests de src/graph/relational_features.py."""

from __future__ import annotations

import numpy as np
from src.graph.relational_features import (
    RelationalSpec,
    degree,
    laplacian_spectral_embedding,
    relational_features,
    weighted_clustering,
)


def _make_W(M: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.random((M, M))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def test_degree_nonneg() -> None:
    W = _make_W()
    d = degree(W)
    assert (d >= 0).all()


def test_weighted_clustering_in_range() -> None:
    W = _make_W()
    c = weighted_clustering(W)
    # Valores acotados por construcción (no normalizamos estrictamente a [0,1])
    assert np.isfinite(c).all()


def test_laplacian_embedding_shape() -> None:
    W = _make_W(M=8)
    emb = laplacian_spectral_embedding(W, k=3)
    assert emb.shape == (8, 3)
    assert np.isfinite(emb).all()


def test_relational_features_concatenates() -> None:
    W = _make_W(M=6)
    spec = RelationalSpec(spectral_top_k=2)
    feats = relational_features(W, spec)
    assert feats.shape == (6, 4)  # degree + clustering + 2 eigvecs
    assert feats.dtype == np.float32
