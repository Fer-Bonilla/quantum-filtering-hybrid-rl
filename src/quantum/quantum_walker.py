"""Fachada del Modelo D: cumple el protocolo ``LocalModule``.

Wrapping minimal sobre ``apply_dtqw`` (matricial) o ``apply_dtqw_pennylane``
(con ruido) + ``top_m`` que convierte índices locales a índices globales.

Selección de backend:
- ``backend="matrix"`` (default): NumPy denso, complex128, sin ruido.
- ``backend="pennylane"``: PennyLane ``default.qubit``; única opción para
  experimentos con ruido (Cap. 8.20.5).

Cuando ``noise.is_active()`` es True, se fuerza el backend PennyLane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.graph.subgraph_selector import Subgraph
from src.quantum.dtqw import apply_dtqw
from src.quantum.measurement import top_m
from src.quantum.noise import NoiseSpec
from src.quantum.pennylane_backend import apply_dtqw_pennylane

Backend = Literal["matrix", "pennylane"]


@dataclass(frozen=True, slots=True)
class QuantumWalker:
    """Implementación del protocolo ``LocalModule`` para Modelo D."""

    init_mode: str = "uniform"
    """Estado inicial: ``uniform`` o ``seed_centered``."""

    renormalize_threshold: float = 1e-9
    """Tolerancia ``|‖ψ‖²-1|`` antes de renormalizar (solo backend matrix)."""

    backend: Backend = "matrix"
    """``matrix`` (rápido, sin ruido) o ``pennylane`` (NISQ-compatible)."""

    noise: NoiseSpec = field(default_factory=NoiseSpec)
    """Configuración de ruido. Solo aplica si ``backend == "pennylane"``."""

    coin_type: str = "weighted_householder"
    """Tipo de moneda DTQW (ver ``src/quantum/dtqw.py:build_coin``):
    ``weighted_householder`` (default, sweet_spot), ``grover``, ``fourier``.
    Solo afecta al backend matrix; pennylane usa la moneda Householder
    ponderada por defecto."""

    @property
    def name(self) -> str:
        if self.noise.is_active():
            return "quantum_noisy"
        return "quantum"

    def _effective_backend(self) -> Backend:
        if self.noise.is_active():
            return "pennylane"
        return self.backend

    def candidate_set(
        self,
        subgraph: Subgraph,
        k: int,
        m: int,
    ) -> np.ndarray:
        """Top-m índices globales según la DTQW sobre H_t.

        Args:
            subgraph: H_t.
            k: pasos de la caminata cuántica.
            m: tamaño del conjunto candidato.

        Returns:
            Array ``(m_eff,)`` con índices globales en U.
        """
        if self._effective_backend() == "pennylane":
            probs = apply_dtqw_pennylane(
                subgraph.W_local,
                subgraph.seed_idx_local,
                k,
                init_mode=self.init_mode,
                noise=self.noise,
            )
        else:
            probs = apply_dtqw(
                subgraph.W_local,
                subgraph.seed_idx_local,
                k,
                init_mode=self.init_mode,
                renormalize_threshold=self.renormalize_threshold,
                coin_type=self.coin_type,
            )
        top_local = top_m(probs, m)
        return subgraph.global_node_ids[top_local].astype(np.int64, copy=False)
