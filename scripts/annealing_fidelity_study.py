"""Estudio de fidelidad del annealing de tiempo finito (v6 — anexo).

Cuantifica el hallazgo de la revisión adversarial: la penalización de
cardinalidad del QUBO comprime los gaps del sector factible ~(M-m)²
veces, de modo que la evolución adiabática Trotterizada con tiempos
practicables NO alcanza el estado fundamental para M >= 8. Esto motiva
el uso del modo ``exact`` (límite adiabático ideal) en las campañas y
documenta la limitación práctica de un annealer real sobre esta
formulación.

Para cada M ∈ {6, 8, 10, 12} genera 20 subgrafos aleatorios (pesos
U[0,1] simetrizados) y compara el top-m del modo ``evolve`` (defaults
del walker) contra el conjunto fundamental exacto del QUBO. Reporta:

* ``set_mismatch_rate`` — fracción de instancias donde el conjunto
  difiere del óptimo.
* ``gap_compression``   — mediana del cociente rango_global /
  rango_sector_factible (factor de compresión del gap).

Salida: ``outputs/tables/annealing_fidelity_v6.csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.graph.subgraph_selector import Subgraph
from src.quantum.annealing_walker import (
    AnnealingWalker,
    _bits_table,
    build_qubo_energies,
)

OUT = Path("outputs/tables/annealing_fidelity_v6.csv")
N_INSTANCES = 20
M_TOP = 3


def main() -> None:
    rows: list[dict] = []
    walker = AnnealingWalker(mode="evolve")
    for M in (6, 8, 10, 12):
        rng = np.random.default_rng(123)
        mismatches = 0
        compressions: list[float] = []
        for _ in range(N_INSTANCES):
            W = rng.uniform(0, 1, (M, M))
            W = (W + W.T) / 2.0
            np.fill_diagonal(W, 0.0)
            sub = Subgraph(
                W_local=W,
                global_node_ids=np.arange(M, dtype=np.int64),
                seed_idx_local=0,
            )
            e = build_qubo_energies(W, 0, M_TOP, alpha=walker.alpha, beta=walker.beta)
            bits = _bits_table(M)
            ground = int(np.argmin(e))
            ground_set = frozenset(i for i in range(M) if (ground >> i) & 1)

            feasible = bits.sum(axis=1) == float(M_TOP)
            e_sector = e[feasible]
            rng_global = float(e.max() - e.min())
            rng_sector = float(e_sector.max() - e_sector.min())
            if rng_sector > 0:
                compressions.append(rng_global / rng_sector)

            out = walker.candidate_set(sub, k=3, m=M_TOP)
            if frozenset(out.tolist()) != ground_set:
                mismatches += 1

        rows.append({
            "M": M,
            "m": M_TOP,
            "n_instances": N_INSTANCES,
            "set_mismatch_rate": mismatches / N_INSTANCES,
            "gap_compression_median": float(np.median(compressions)),
        })
        print(
            f"M={M:>2}: mismatch={mismatches}/{N_INSTANCES} "
            f"compresión_gap(mediana)={np.median(compressions):.0f}x",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
