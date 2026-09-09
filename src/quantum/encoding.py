"""Codificación cuántica de un subgrafo (Sec. 7.12).

Espacio de Hilbert:
    H = H_pos (dim M) ⊗ H_coin (dim d_max)

Para cada nodo i se enumeran sus vecinos en orden ascendente de índice
local y se les asigna un puerto local ``c in [0, d_i)`` (con d_i = grado
de i). Los puertos restantes hasta ``d_max`` son padding (sin amplitud
inicial, puntos fijos del shift).

La función pública es ``build_port_map``, que devuelve estructuras
necesarias para construir los operadores ``C_t`` (coin) y ``S_t`` (shift):

- ``port_to_neighbor[i, c] = j`` si el puerto c en i conduce a j; -1 si padding.
- ``neighbor_to_port[i, j] = c`` (puerto de i hacia j); -1 si no son vecinos.
- ``degrees[i] = d_i``, ``d_max = max(d_i)``.
- ``reciprocal_port[(i, c)] = (j, c')`` cuando el puerto c en i va a j y c' es
  el puerto recíproco en j hacia i.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PortMap:
    """Mapeo nodo-puerto-vecino para construir S_t y C_t."""

    M: int
    d_max: int
    degrees: np.ndarray
    """dtype int64, shape (M,)."""

    port_to_neighbor: np.ndarray
    """dtype int64, shape (M, d_max). -1 indica padding."""

    neighbor_to_port: np.ndarray
    """dtype int64, shape (M, M). -1 indica no-adyacente."""

    edge_weights: np.ndarray
    """dtype float64, shape (M, d_max). 0.0 en posiciones de padding."""


def build_port_map(W_local: np.ndarray, *, weight_threshold: float = 0.0) -> PortMap:
    """Construir el mapeo de puertos a partir de la adyacencia local.

    Una arista (i, j) existe si ``W_local[i, j] > weight_threshold``. Los
    puertos se asignan en orden ascendente del índice del vecino para
    reproducibilidad.

    Args:
        W_local: matriz simétrica ``(M, M)`` con diagonal 0.
        weight_threshold: peso mínimo para considerar una arista.

    Returns:
        ``PortMap``.

    Raises:
        ValueError: Si W no es simétrica o si todos los nodos quedan aislados.
    """
    if W_local.ndim != 2 or W_local.shape[0] != W_local.shape[1]:
        raise ValueError(f"W_local debe ser cuadrada; recibido {W_local.shape}")
    if not np.allclose(W_local, W_local.T, atol=1e-9):
        raise ValueError("W_local debe ser simétrica.")
    M = W_local.shape[0]

    neighbors: list[list[int]] = []
    for i in range(M):
        row = W_local[i]
        nbrs = np.flatnonzero((row > weight_threshold) & (np.arange(M) != i))
        nbrs.sort()  # orden por índice local
        neighbors.append(nbrs.tolist())

    degrees = np.array([len(ns) for ns in neighbors], dtype=np.int64)
    d_max = int(degrees.max()) if degrees.max() > 0 else 1
    if degrees.sum() == 0:
        raise ValueError("Todos los nodos quedaron aislados (sin aristas).")

    port_to_neighbor = np.full((M, d_max), -1, dtype=np.int64)
    neighbor_to_port = np.full((M, M), -1, dtype=np.int64)
    edge_weights = np.zeros((M, d_max), dtype=np.float64)

    for i, nbrs in enumerate(neighbors):
        for c, j in enumerate(nbrs):
            port_to_neighbor[i, c] = j
            neighbor_to_port[i, j] = c
            edge_weights[i, c] = float(W_local[i, j])

    return PortMap(
        M=M,
        d_max=d_max,
        degrees=degrees,
        port_to_neighbor=port_to_neighbor,
        neighbor_to_port=neighbor_to_port,
        edge_weights=edge_weights,
    )


def hilbert_dim(pmap: PortMap) -> int:
    """Dimensión total del Hilbert space M * d_max."""
    return pmap.M * pmap.d_max


def index_of(i: int, c: int, d_max: int) -> int:
    """Índice plano de ``|i, c⟩`` en el orden (nodo, puerto)."""
    return i * d_max + c


def initial_state_uniform(pmap: PortMap) -> np.ndarray:
    """Estado |ψ_0⟩ = superposición uniforme sobre puertos VÁLIDOS.

    Padding tiene amplitud cero (preserva interpretación local).
    """
    dim = hilbert_dim(pmap)
    psi = np.zeros(dim, dtype=np.complex128)
    valid_count = int((pmap.port_to_neighbor >= 0).sum())
    if valid_count == 0:
        raise ValueError("No hay puertos válidos para inicializar.")
    amp = 1.0 / np.sqrt(valid_count)
    for i in range(pmap.M):
        for c in range(pmap.degrees[i]):
            psi[index_of(i, c, pmap.d_max)] = amp
    return psi


def initial_state_seed_centered(pmap: PortMap, seed_idx: int) -> np.ndarray:
    """Estado inicial concentrado en los puertos del nodo semilla.

    |ψ_0⟩ = (1/√d_seed) Σ_c |seed, c⟩ para c en puertos válidos del seed.
    """
    if not (0 <= seed_idx < pmap.M):
        raise ValueError(f"seed_idx fuera de rango: {seed_idx}")
    d_seed = int(pmap.degrees[seed_idx])
    if d_seed == 0:
        raise ValueError(f"Seed {seed_idx} aislado; sin puertos válidos.")
    dim = hilbert_dim(pmap)
    psi = np.zeros(dim, dtype=np.complex128)
    amp = 1.0 / np.sqrt(d_seed)
    for c in range(d_seed):
        psi[index_of(seed_idx, c, pmap.d_max)] = amp
    return psi
