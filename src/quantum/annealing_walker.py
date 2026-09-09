"""Selección de candidatos vía Quantum Annealing simulado (v6 — Punto 3).

Sustituye la resolución del subgrafo mediante DTQW por una formulación
QUBO + evolución adiabática simulada, siguiendo el esquema del tutorial
adiabático de Qibo (adiabatic3sat) pero con implementación matricial
densa propia, consistente con ``src/quantum/dtqw.py``.

Formulación QUBO (minimizar sobre x ∈ {0,1}^M):

    E(x) = -alpha · Σ_{i<j} W_ij x_i x_j        (cohesión del conjunto)
           -beta  · Σ_i  b_i x_i                 (afinidad al seed)
           +P     · (Σ_i x_i - m)²               (cardinalidad |S| = m)

donde ``b_i = W[seed, i]`` (y ``b_seed = max_j W[seed, j]`` para que el
seed compita en igualdad). El estado fundamental de E es el conjunto de
``m`` nodos con máxima afinidad conjunta al seed y entre sí: la
*resolución del subgrafo* en el sentido del director.

Evolución adiabática (Trotterizada):

    H(s) = (1-s) · H_B + s · H_P
    H_B  = -Σ_i sigma_x^(i)        (campo transverso; fundamental |+...+>)
    H_P  = diag(E(x))              (Hamiltoniano del problema)

    |psi(0)> = |+>^M ;  para j=1..n_steps:  s = j/n_steps
        |psi> <- exp(-i·s·E·dt) |psi>                    (fase diagonal)
        |psi> <- Prod_i Rx_i(2·(1-s)·dt) |psi>           (mezcla)

Medición: probabilidad marginal por nodo p_i = Σ_{x: x_i=1} |psi_x|²;
el conjunto candidato es el top-m de las marginales.

Coste: O(n_steps · M · 2^M). Viable hasta M=16 con ``update_frequency``
del agente híbrido > 1 (la máscara se cachea entre llamadas).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.graph.subgraph_selector import Subgraph
from src.quantum.measurement import top_m

_MAX_QUBITS = 16
"""Límite duro: 2^16 amplitudes. Por encima, el coste es prohibitivo."""

# Cache módulo-level de las tablas de bits por M (reutilizable entre walkers)
_BITS_CACHE: dict[int, np.ndarray] = {}


def _bits_table(n_qubits: int) -> np.ndarray:
    """Matriz ``(2^M, M)`` con la expansión binaria de cada estado base."""
    if n_qubits not in _BITS_CACHE:
        idx = np.arange(2**n_qubits, dtype=np.int64)
        _BITS_CACHE[n_qubits] = (
            (idx[:, None] >> np.arange(n_qubits)[None, :]) & 1
        ).astype(np.float64)
    return _BITS_CACHE[n_qubits]


def build_qubo_energies(
    W_local: np.ndarray,
    seed_idx: int,
    m: int,
    *,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> np.ndarray:
    """Vector ``(2^M,)`` con la energía QUBO de cada bitstring.

    La penalización de cardinalidad se fija automáticamente al rango
    completo de la parte de afinidad, garantizando que violar
    ``|S| = m`` en 1 unidad cuesta más que cualquier ganancia de
    afinidad posible.
    """
    M = int(W_local.shape[0])
    if M > _MAX_QUBITS:
        raise ValueError(
            f"M={M} excede el máximo de {_MAX_QUBITS} qubits simulables."
        )
    bits = _bits_table(M)

    b = W_local[seed_idx].astype(np.float64).copy()
    b[seed_idx] = float(W_local[seed_idx].max()) if M > 1 else 1.0

    # -alpha/2 · x^T W x  (la diagonal de W es 0; el 1/2 evita doble conteo)
    pair = -0.5 * alpha * np.einsum("si,ij,sj->s", bits, W_local, bits)
    lin = -beta * (bits @ b)
    e_aff = pair + lin

    penalty = float(e_aff.max() - e_aff.min()) + 1e-9
    card = (bits.sum(axis=1) - float(m)) ** 2
    return e_aff + penalty * card


def _apply_rx_all(psi: np.ndarray, theta: float, n_qubits: int) -> np.ndarray:
    """Aplicar Rx(theta) a todos los qubits (campo transverso Trotterizado)."""
    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    for q in range(n_qubits):
        psi = psi.reshape(-1, 2, 2**q)
        a = psi[:, 0, :].copy()
        bq = psi[:, 1, :]
        psi[:, 0, :] = c * a + s * bq
        psi[:, 1, :] = s * a + c * bq
    return psi.reshape(-1)


def anneal_marginals(
    energies: np.ndarray,
    n_qubits: int,
    *,
    n_steps: int = 50,
    total_time: float = 20.0,
    feasible_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Evolución adiabática simulada → probabilidades marginales por qubit.

    Args:
        energies: vector ``(2^M,)`` de energías QUBO.
        n_qubits: M.
        n_steps: pasos de Trotter del schedule lineal s = j/n_steps.
        total_time: tiempo total de annealing T (dt = T/n_steps).
        feasible_mask: máscara booleana ``(2^M,)`` del sector factible
            (e.g. cardinalidad m). Si se proporciona, la energía se escala
            para que el RANGO DEL SECTOR FACTIBLE sea ~M (rango espectral
            del mixer). Sin ella, se usa el rango global — pero el rango
            global está dominado por la penalización de cardinalidad
            (~(M-m)² veces el rango factible), lo que comprime los gaps
            relevantes y rompe la adiabaticidad para M ≳ 8.

    Returns:
        Vector ``(M,)`` con ``p_i = P(x_i = 1)`` en el estado final.
    """
    e = energies - energies.min()
    if feasible_mask is not None and bool(feasible_mask.any()):
        e_sector = e[feasible_mask]
        rng_e = float(e_sector.max() - e_sector.min())
    else:
        rng_e = float(e.max())
    if rng_e > 0.0:
        # Escalar para que el rango RELEVANTE quede comparable al rango
        # espectral del mixer H_B = -Σ sigma_x (~[-M, M]). Los estados
        # infactibles quedan a energía >> M tras el escalado: bien.
        e = (e / rng_e) * float(n_qubits)

    dim = 2**n_qubits
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    dt = total_time / n_steps
    for j in range(1, n_steps + 1):
        s = j / n_steps
        psi = np.exp(-1j * s * e * dt) * psi
        # exp(-i·(1-s)·dt·H_B) con H_B = -Σ sigma_x  ⇒  Rx de ángulo
        # NEGATIVO por qubit (|+...+> es el fundamental de H_B; con el
        # signo invertido se seguiría el autoestado MÁXIMO del mixer).
        psi = _apply_rx_all(psi, -2.0 * (1.0 - s) * dt, n_qubits)
        # Renormalización defensiva (mismo criterio que dtqw.py)
        norm2 = float(np.real(np.vdot(psi, psi)))
        if abs(norm2 - 1.0) > 1e-9:
            psi = psi / np.sqrt(norm2)

    probs = np.abs(psi) ** 2
    bits = _bits_table(n_qubits)
    return bits.T @ probs  # (M,) marginales


@dataclass(slots=True)
class AnnealingWalker:
    """Implementación del protocolo ``LocalModule`` para el Modelo Q.

    Igual interfaz que ``ClassicalWalker``/``QuantumWalker``: dado el
    subgrafo H_t devuelve el conjunto candidato top-m resolviendo el
    QUBO de selección, en lugar de propagar una caminata.

    Modos:

    * ``mode="exact"`` (default, usado en campañas): estado fundamental
      EXACTO del Hamiltoniano del problema por enumeración (viable
      porque M <= 16 y las 2^M energías ya se construyen para el QUBO).
      Es el límite adiabático ideal T -> infinito: la cota SUPERIOR del
      desempeño alcanzable por un annealer real sobre esta formulación.
    * ``mode="evolve"``: evolución adiabática Trotterizada de tiempo
      finito (transverse-field). Usado para el estudio de fidelidad:
      la penalización de cardinalidad comprime los gaps del sector
      factible ~(M-m)^2 veces, de modo que a tiempos de annealing
      practicables las marginales son un ranking heurístico, no el
      fundamental (hallazgo documentado en el anexo de fidelidad).
    """

    alpha: float = 1.0
    """Peso del término de cohesión (pares dentro del conjunto)."""

    beta: float = 2.0
    """Peso del término de afinidad al seed."""

    mode: str = "exact"
    """``"exact"`` (fundamental por enumeración, límite adiabático ideal)
    o ``"evolve"`` (schedule Trotterizado de tiempo finito)."""

    trotter_per_k: int = 10
    """``n_steps >= trotter_per_k · k`` — k conserva su semántica de
    'profundidad' del módulo cuántico."""

    total_time: float = 80.0
    """Tiempo total T del schedule adiabático. El número de pasos de
    Trotter se eleva automáticamente para mantener ``dt <= max_dt``."""

    max_dt: float = 0.5
    """Cota superior del paso temporal: limita el error de Trotter y el
    aliasing de fase de los estados del sector factible (~M·dt rad)."""

    name: str = field(default="annealing", init=False)

    def candidate_set(
        self,
        subgraph: Subgraph,
        k: int,
        m: int,
    ) -> np.ndarray:
        """Top-m índices globales según las marginales del annealing."""
        M = subgraph.size
        m_eff = min(m, M)
        if M == 1:
            return subgraph.global_node_ids[:1].astype(np.int64, copy=False)
        energies = build_qubo_energies(
            subgraph.W_local, subgraph.seed_idx_local, m_eff,
            alpha=self.alpha, beta=self.beta,
        )
        if self.mode == "exact":
            ground = int(np.argmin(energies))
            members = np.array(
                [i for i in range(M) if (ground >> i) & 1], dtype=np.int64
            )
            if members.size != m_eff:
                # Defensivo: la penalización garantiza |S|=m; si no, caer
                # a top-m por afinidad directa al seed.
                top_local = top_m(
                    subgraph.W_local[subgraph.seed_idx_local].astype(float), m_eff
                )
            else:
                # Orden determinista: por afinidad al seed descendente
                b = subgraph.W_local[subgraph.seed_idx_local][members]
                top_local = members[np.argsort(-b)]
        elif self.mode == "evolve":
            feasible = _bits_table(M).sum(axis=1) == float(m_eff)
            n_steps = max(
                20,
                self.trotter_per_k * max(k, 1),
                int(np.ceil(self.total_time / self.max_dt)),
            )
            marginals = anneal_marginals(
                energies, M,
                n_steps=n_steps,
                total_time=self.total_time,
                feasible_mask=feasible,
            )
            top_local = top_m(marginals, m_eff)
        else:
            raise ValueError(f"mode desconocido: {self.mode!r}")
        return subgraph.global_node_ids[top_local].astype(np.int64, copy=False)
