"""Tests for project documentation completeness."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_readme_has_key_sections():
    """README.md must contain Pipeline, Metrics, API, and Docker sections."""
    readme = PROJECT_ROOT / "README.md"
    assert readme.exists(), "README.md not found"
    content = readme.read_text()
    assert "Pipeline" in content, "README.md missing Pipeline section"
    assert "Métricas" in content, "README.md missing Metrics section"
    assert "API" in content, "README.md missing API section"
    assert "Docker" in content, "README.md missing Docker section"


def test_experimentos_md_exists_and_has_optuna_smote():
    """EXPERIMENTOS.md must exist and contain Optuna and SMOTE."""
    experimentos = PROJECT_ROOT / "EXPERIMENTOS.md"
    assert experimentos.exists(), "EXPERIMENTOS.md not found"
    content = experimentos.read_text()
    assert "Optuna" in content, "EXPERIMENTOS.md missing Optuna"
    assert "SMOTE" in content, "EXPERIMENTOS.md missing SMOTE"
