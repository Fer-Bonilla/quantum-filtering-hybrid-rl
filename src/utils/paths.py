"""Paths del proyecto.

Centraliza el descubrimiento del project root y las rutas a directorios
estándar para evitar paths hardcodeados a lo largo del código.
"""

from __future__ import annotations

from pathlib import Path


def _find_project_root() -> Path:
    """Localizar la raíz del proyecto buscando hacia arriba pyproject.toml.

    Returns:
        Ruta absoluta al directorio que contiene pyproject.toml.

    Raises:
        RuntimeError: Si no se encuentra pyproject.toml en ningún ancestro.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(
        f"No se encontró pyproject.toml en ningún ancestro de {here}. "
        "El proyecto debe contener un pyproject.toml en su raíz."
    )


PROJECT_ROOT: Path = _find_project_root()
"""Ruta absoluta a la raíz del proyecto (donde está pyproject.toml)."""

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_INTERIM_DIR: Path = DATA_DIR / "interim"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
MLRUNS_DIR: Path = OUTPUTS_DIR / "mlruns"
CHECKPOINTS_DIR: Path = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
TABLES_DIR: Path = OUTPUTS_DIR / "tables"


def get_project_root() -> Path:
    """Retornar la raíz del proyecto.

    Función explícita para clarificar dependencia cuando es invocada como API.
    """
    return PROJECT_ROOT


def ensure_dir(path: Path) -> Path:
    """Crear el directorio si no existe (incluye padres) y retornarlo."""
    path.mkdir(parents=True, exist_ok=True)
    return path
