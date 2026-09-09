"""Muestreo de inicios de episodio sobre una partición temporal.

Determinista por semilla. Soporta dos modos:
- ``sequential``: episodios consecutivos sin solapamiento (cobertura total).
- ``random``: inicios muestreados uniformemente del rango válido.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class EpisodeSampler:
    """Genera índices de inicio de episodio reproducibles."""

    n_rows: int
    """Filas totales disponibles en la partición."""

    window_length: int
    """Longitud L de la ventana del estado (se reserva al inicio)."""

    max_steps: int
    """Pasos máximos por episodio."""

    mode: Literal["sequential", "random"] = "random"

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps debe ser >= 1.")
        if self.window_length < 1:
            raise ValueError("window_length debe ser >= 1.")
        if self.n_rows < self.window_length + self.max_steps:
            raise ValueError(
                f"n_rows={self.n_rows} insuficiente para window_length="
                f"{self.window_length} + max_steps={self.max_steps}."
            )

    @property
    def min_start(self) -> int:
        """Primer índice posicional válido (ventana completa hacia atrás)."""
        return self.window_length - 1

    @property
    def max_start(self) -> int:
        """Último índice inicial válido para alojar max_steps adelante."""
        return self.n_rows - self.max_steps - 1

    def __iter__(self) -> Iterator[int]:
        raise TypeError("Usar .iter_starts(rng) en su lugar.")

    def iter_starts(self, rng: np.random.Generator, n_episodes: int) -> Iterator[int]:
        """Generar `n_episodes` índices de inicio."""
        if self.mode == "sequential":
            start = self.min_start
            for _ in range(n_episodes):
                if start > self.max_start:
                    start = self.min_start  # wrap-around determinista
                yield start
                start += self.max_steps
        else:  # random
            for _ in range(n_episodes):
                yield int(rng.integers(self.min_start, self.max_start + 1))
