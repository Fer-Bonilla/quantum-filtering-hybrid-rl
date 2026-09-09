"""Estimador de coste de ejecución en IBM Quantum Pay-as-you-go.

Metodología detallada en ``docs/ibm_quantum_cost_analysis.md``.

Modelo:
    t_per_shot     = circuit_depth * gate_time + readout_time + reset_time
    t_qpu_per_eval = shots * t_per_shot
    t_total        = t_qpu_per_eval + runtime_overhead_per_eval
    coste_total    = n_evals * t_total * USD_PER_SECOND

Uso (línea de comandos)::

    uv run python scripts/estimate_ibm_cost.py --evals 250000 --shots 512
    uv run python scripts/estimate_ibm_cost.py --evals 100 --shots 1024 --overhead 0.1

Uso (importado en notebook)::

    from scripts.estimate_ibm_cost import estimate_ibm_cost
    estimate_ibm_cost(n_evaluations=250_000, shots=512)
"""

from __future__ import annotations

import argparse

# Tarifa IBM Pay-as-you-go (verificada mayo 2026)
USD_PER_SECOND_DEFAULT = 1.60

# Parámetros de hardware Eagle/Heron (estimaciones razonables)
GATE_TIME_S_DEFAULT = 100e-9  # 100 ns por gate promedio (mezcla 1Q + 2Q)
READOUT_TIME_S_DEFAULT = 3e-6  # 3 μs lecturas paralelas
RESET_TIME_S_DEFAULT = 1.5e-6  # 1.5 μs reset entre shots


def estimate_ibm_cost(
    n_evaluations: int,
    shots: int = 512,
    circuit_depth: int = 150,
    overhead_per_eval_s: float = 0.5,
    rate_usd_per_s: float = USD_PER_SECOND_DEFAULT,
    gate_time_s: float = GATE_TIME_S_DEFAULT,
    readout_time_s: float = READOUT_TIME_S_DEFAULT,
    reset_time_s: float = RESET_TIME_S_DEFAULT,
) -> dict[str, float]:
    """Estimar coste USD de N evaluaciones DTQW en IBM Quantum Pay-as-you-go.

    Args:
        n_evaluations: número total de invocaciones a ``apply_dtqw_pennylane``.
        shots: shots por circuito (más shots = más precisión = más coste lineal).
        circuit_depth: profundidad media (en gates nativos) tras transpile.
            Para M=8, k=3: ~120-210. Para M=32, k=5: ~400-600.
        overhead_per_eval_s: Runtime/Session overhead por evaluación. Valores
            típicos:
            - 0.05 s: Session mode bien afinado (RL streaming).
            - 0.5 s: Sampler estándar con Session.
            - 1.0-2.0 s: Jobs individuales (peor caso).
        rate_usd_per_s: precio Pay-as-you-go (default 1.60 USD/s, mayo 2026).
        gate_time_s, readout_time_s, reset_time_s: parámetros del backend.

    Returns:
        Dict con desglose: ``t_per_shot_us``, ``t_qpu_per_eval_ms``,
        ``t_total_per_eval_s``, ``cost_per_eval_usd``, ``total_cost_usd``.
    """
    t_per_shot_s = circuit_depth * gate_time_s + readout_time_s + reset_time_s
    t_qpu_per_eval_s = shots * t_per_shot_s
    t_total_per_eval_s = t_qpu_per_eval_s + overhead_per_eval_s
    cost_per_eval = t_total_per_eval_s * rate_usd_per_s
    total_cost = n_evaluations * cost_per_eval

    return {
        "t_per_shot_us": t_per_shot_s * 1e6,
        "t_qpu_per_eval_ms": t_qpu_per_eval_s * 1e3,
        "t_total_per_eval_s": t_total_per_eval_s,
        "cost_per_eval_usd": cost_per_eval,
        "total_cost_usd": total_cost,
        "total_qpu_hours": (n_evaluations * t_qpu_per_eval_s) / 3600.0,
        "total_wallclock_hours": (n_evaluations * t_total_per_eval_s) / 3600.0,
    }


def _format_report(n_evals: int, shots: int, result: dict[str, float]) -> str:
    sep = "=" * 63
    return f"""\
{sep}
  Estimacion de coste IBM Quantum Pay-as-you-go (mayo 2026)
{sep}

Configuracion:
    n_evaluations       = {n_evals:>12,}
    shots por circuito  = {shots:>12}
    tarifa              = {USD_PER_SECOND_DEFAULT} USD/s

Tiempo por evaluacion:
    t_per_shot          = {result['t_per_shot_us']:>9.2f} us
    t_QPU efectivo      = {result['t_qpu_per_eval_ms']:>9.2f} ms
    t_total facturable  = {result['t_total_per_eval_s']:>9.3f} s

Coste:
    Por evaluacion      = {result['cost_per_eval_usd']:>12.4f} USD
    Campana completa    = {result['total_cost_usd']:>12,.2f} USD

Tiempos agregados:
    QPU time total      = {result['total_qpu_hours']:>9.2f} horas
    Wallclock estimado  = {result['total_wallclock_hours']:>9.2f} horas

{sep}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals", type=int, default=250_000, help="Número total de evaluaciones DTQW.")
    parser.add_argument("--shots", type=int, default=512, help="Shots por circuito.")
    parser.add_argument("--depth", type=int, default=150, help="Profundidad media tras transpile.")
    parser.add_argument(
        "--overhead",
        type=float,
        default=0.5,
        help="Runtime overhead por evaluación (s). 0.05=Session afinado, 0.5=estándar, 2.0=jobs sin Session.",
    )
    parser.add_argument("--rate", type=float, default=USD_PER_SECOND_DEFAULT, help="USD/segundo Pay-as-you-go.")
    args = parser.parse_args()

    result = estimate_ibm_cost(
        n_evaluations=args.evals,
        shots=args.shots,
        circuit_depth=args.depth,
        overhead_per_eval_s=args.overhead,
        rate_usd_per_s=args.rate,
    )
    print(_format_report(args.evals, args.shots, result))


if __name__ == "__main__":
    main()
