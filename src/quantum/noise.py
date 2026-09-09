"""Modelos de ruido para experimentos NISQ (Sec. 8.20.5).

Se aplican exclusivamente con el backend PennyLane (``default.mixed``) tras
cada paso DTQW. La filosofía es "ruido por gate": el operador unitario U_t
se aplica primero y luego cada qubit sufre un canal independiente.

Tipos soportados:
- ``depolarizing``: pérdida de coherencia (Sec. 8.20 del documento).
- ``dephasing``: pérdida de fase relativa (PhaseDamping en PennyLane).
"""

from __future__ import annotations

from dataclasses import dataclass

import pennylane as qml


@dataclass(frozen=True, slots=True)
class NoiseSpec:
    """Configuración runtime del modelo de ruido."""

    depolarizing_prob: float = 0.0
    """Probabilidad de aplicar el canal depolarizing por qubit, por paso."""

    dephasing_prob: float = 0.0
    """Probabilidad de aplicar PhaseDamping por qubit, por paso."""

    def is_active(self) -> bool:
        return self.depolarizing_prob > 0.0 or self.dephasing_prob > 0.0


def apply_noise_layer(noise: NoiseSpec, n_qubits: int) -> None:
    """Aplicar un canal de ruido independiente a cada qubit del circuito.

    Esta función está pensada para invocarse dentro de un ``@qml.qnode``
    después de cada aplicación de U_t.
    """
    if not noise.is_active():
        return
    for w in range(n_qubits):
        if noise.depolarizing_prob > 0.0:
            qml.DepolarizingChannel(noise.depolarizing_prob, wires=w)
        if noise.dephasing_prob > 0.0:
            qml.PhaseDamping(noise.dephasing_prob, wires=w)
