"""Bootstrap pareado por semilla: especificación única del proyecto.

Todas las comparaciones pareadas (campaña principal, controles v6, campaña
v8 y auditorías) usan esta función, de modo que la fórmula del valor ``p``
coincide con la del Anexo B de la memoria.

Definiciones (``d_i`` = diferencia pareada de la semilla ``i``, ``n`` pares,
``B`` remuestreos con reemplazo de tamaño ``n``, ``m_b`` = media del
remuestreo ``b``):

* Intervalo: percentiles simples 2,5 y 97,5 de ``{m_b}`` (sin BCa). No
  depende de la regla del valor ``p``.
* Valor ``p`` bilateral::

      p2 = min(1, 2 * min(P(m_b <= 0), P(m_b >= 0)))

  Las desigualdades son inclusivas (los empates exactos con cero cuentan en
  ambas colas) y el resultado se acota a 1.
* Valor ``p`` unilateral para H1: media > 0 ::

      p_greater = P(m_b <= 0)

* Caso degenerado: si todas las diferencias son iguales (desviación nula) el
  bootstrap no tiene variabilidad y el contraste NO está definido: se
  devuelve ``degenerate=True`` y ``p2 = p_greater = nan``; el intervalo
  colapsa al valor común. Los informes deben reportar "idéntico por
  construcción" sin valor ``p``.

Calibración conocida (véase ``scripts/bootstrap_calibration_review.py``):
el ``p2`` del bootstrap percentil de la media es anticonservador con
muestras pequeñas (tasa de rechazo ~9 % al 5 % nominal con ``n=10``, ~6 %
con ``n=40``); las pruebas t pareada y de Wilcoxon están calibradas y se
reportan como comprobación cruzada en la memoria.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

DEFAULT_B = 5000


@dataclass(frozen=True, slots=True)
class PairedBootstrap:
    """Resultado del bootstrap pareado."""

    mean: float
    ci_lo: float
    ci_hi: float
    n: int
    p_two: float
    p_greater: float
    p_positive: float
    degenerate: bool
    n_boot: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def paired_bootstrap(
    diffs: np.ndarray | list[float],
    *,
    n_boot: int = DEFAULT_B,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> PairedBootstrap:
    """Bootstrap pareado sobre las diferencias ``diffs`` (una por semilla).

    Args:
        diffs: diferencias pareadas; se descartan los valores no finitos.
        n_boot: número de remuestreos ``B``.
        rng: generador a usar; si es ``None`` se crea con ``seed``.
        seed: semilla para el generador cuando ``rng`` es ``None``.

    Raises:
        ValueError: si quedan menos de 2 diferencias finitas.
    """
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 2:
        raise ValueError(f"Se necesitan al menos 2 diferencias finitas; recibidas {d.size}.")
    if rng is None:
        rng = np.random.default_rng(seed)
    mean = float(d.mean())

    if np.all(d == d[0]):
        return PairedBootstrap(
            mean=mean, ci_lo=mean, ci_hi=mean, n=int(d.size),
            p_two=float("nan"), p_greater=float("nan"),
            p_positive=float("nan"), degenerate=True, n_boot=n_boot,
        )

    boot = np.array(
        [rng.choice(d, size=d.size, replace=True).mean() for _ in range(n_boot)],
        dtype=np.float64,
    )
    p_le = float((boot <= 0.0).mean())
    p_ge = float((boot >= 0.0).mean())
    p_two = min(1.0, 2.0 * min(p_le, p_ge))
    return PairedBootstrap(
        mean=mean,
        ci_lo=float(np.percentile(boot, 2.5)),
        ci_hi=float(np.percentile(boot, 97.5)),
        n=int(d.size),
        p_two=p_two,
        p_greater=p_le,
        p_positive=float((boot > 0.0).mean()),
        degenerate=False,
        n_boot=n_boot,
    )
