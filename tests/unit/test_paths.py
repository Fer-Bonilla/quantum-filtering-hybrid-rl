"""Tests de src/utils/paths.py."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import (
    CONFIGS_DIR,
    DATA_RAW_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    ensure_dir,
)


def test_project_root_contains_pyproject() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_standard_dirs_under_root() -> None:
    assert CONFIGS_DIR.parent == PROJECT_ROOT
    assert DATA_RAW_DIR.parent.parent == PROJECT_ROOT
    assert OUTPUTS_DIR.parent == PROJECT_ROOT


def test_ensure_dir_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "deep"
    ensure_dir(target)
    assert target.is_dir()
    # Segunda llamada no debe fallar.
    ensure_dir(target)
    assert target.is_dir()
