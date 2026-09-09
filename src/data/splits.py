"""Partición cronológica train/val/test.

Función pura, determinista, sin solapamiento. No usa shuffling (la
naturaleza temporal del problema lo prohíbe).

Soporta dos esquemas (rev. v2 — Problema 1.5 del revisor):

1. Partición clásica train/val/test sobre el panel completo
   (``chronological_split``).
2. Partición ``tuning`` + ``holdout`` sobre dos ventanas temporales
   disjuntas (``holdout_split``), para eliminar la circularidad de
   seleccionar hiperparámetros y validar en la misma ventana.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import pandas as pd

from src.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """Especificación de la partición."""

    train_frac: float = 0.6
    val_frac: float = 0.2

    @property
    def test_frac(self) -> float:
        return 1.0 - self.train_frac - self.val_frac

    def __post_init__(self) -> None:
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError(f"train_frac debe estar en (0,1); recibido {self.train_frac}")
        if not (0.0 < self.val_frac < 1.0):
            raise ValueError(f"val_frac debe estar en (0,1); recibido {self.val_frac}")
        if self.test_frac <= 0.0:
            raise ValueError(f"test_frac = 1 - train - val = {self.test_frac:.4f} debe ser > 0")


class SplitDict(TypedDict):
    """Resultado de la partición cronológica."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(df: pd.DataFrame, spec: SplitSpec) -> SplitDict:
    """Particionar un DataFrame temporal en train / val / test.

    El DataFrame debe tener un índice temporalmente ordenado.

    Args:
        df: DataFrame con índice ordenado cronológicamente.
        spec: Especificación de fracciones.

    Returns:
        Dict con tres DataFrames sin solapamiento, en orden cronológico.

    Raises:
        ValueError: Si el DataFrame está vacío o el índice no está ordenado.
    """
    if df.empty:
        raise ValueError("DataFrame vacío.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("El índice del DataFrame no está ordenado cronológicamente.")

    n = len(df)
    n_train = int(n * spec.train_frac)
    n_val = int(n * spec.val_frac)

    if n_train < 1 or n_val < 1 or (n - n_train - n_val) < 1:
        raise ValueError(
            f"DataFrame con {n} filas es demasiado pequeño para fracciones "
            f"({spec.train_frac}, {spec.val_frac}, {spec.test_frac})."
        )

    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]

    _log.info(
        "Split cronológico: train=%d [%s,%s] · val=%d [%s,%s] · test=%d [%s,%s]",
        len(train),
        train.index[0].date(),
        train.index[-1].date(),
        len(val),
        val.index[0].date(),
        val.index[-1].date(),
        len(test),
        test.index[0].date(),
        test.index[-1].date(),
    )

    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Hold-out temporal (rev. v2 — Problema 1.5)
# ---------------------------------------------------------------------------


class HoldoutSplitDict(TypedDict):
    """Dos ventanas temporales disjuntas: ``tuning`` y ``holdout``."""

    tuning: pd.DataFrame
    holdout: pd.DataFrame


def holdout_split(
    df: pd.DataFrame,
    *,
    tuning_end: str | pd.Timestamp,
    holdout_start: str | pd.Timestamp | None = None,
) -> HoldoutSplitDict:
    """Particionar el panel en dos ventanas temporales disjuntas.

    Args:
        df: panel con índice temporalmente ordenado.
        tuning_end: última fecha INCLUSIVE del bloque de tuning. Cualquier
            timestamp posterior pertenece al bloque hold-out.
        holdout_start: primera fecha INCLUSIVE del bloque hold-out. Si
            ``None``, se infiere como el siguiente día disponible tras
            ``tuning_end``.

    Returns:
        ``HoldoutSplitDict`` con dos DataFrames disjuntos. La verificación
        ``tuning ∩ holdout = ∅`` se garantiza mediante el filtrado por
        timestamps (test ``test_holdout_split_no_overlap``).

    Raises:
        ValueError: si el resultado deja un bloque vacío.
    """
    if df.empty:
        raise ValueError("DataFrame vacío.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("El índice del DataFrame no está ordenado cronológicamente.")

    tuning_end_ts = pd.Timestamp(tuning_end)
    tuning = df.loc[df.index <= tuning_end_ts]

    if holdout_start is None:
        # Inferir: primer índice estrictamente posterior a tuning_end.
        candidates = df.index[df.index > tuning_end_ts]
        if len(candidates) == 0:
            raise ValueError(
                f"tuning_end={tuning_end_ts} no deja datos posteriores para hold-out."
            )
        holdout_start_ts = candidates[0]
    else:
        holdout_start_ts = pd.Timestamp(holdout_start)
        if holdout_start_ts <= tuning_end_ts:
            raise ValueError(
                f"holdout_start {holdout_start_ts} debe ser estrictamente posterior a "
                f"tuning_end {tuning_end_ts} para evitar solapamiento."
            )

    holdout = df.loc[df.index >= holdout_start_ts]
    if tuning.empty or holdout.empty:
        raise ValueError(
            f"Bloque vacío: tuning={len(tuning)}, holdout={len(holdout)}."
        )

    _log.info(
        "Hold-out split: tuning=%d [%s,%s] · holdout=%d [%s,%s]",
        len(tuning),
        tuning.index[0].date(),
        tuning.index[-1].date(),
        len(holdout),
        holdout.index[0].date(),
        holdout.index[-1].date(),
    )

    return {"tuning": tuning, "holdout": holdout}
