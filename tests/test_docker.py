"""Tests for Docker configuration and startup behavior."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_uses_python_310_slim():
    """Dockerfile must use python:3.10-slim as base image."""
    dockerfile = PROJECT_ROOT / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile not found"
    content = dockerfile.read_text()
    assert "FROM python:3.10-slim" in content, "Dockerfile does not use python:3.10-slim"


def test_dockerfile_exposes_7860():
    """Dockerfile must expose port 7860."""
    dockerfile = PROJECT_ROOT / "Dockerfile"
    content = dockerfile.read_text()
    assert "EXPOSE 7860" in content, "Dockerfile does not expose port 7860"


def test_dockerfile_cmd_contains_uvicorn_and_port():
    """Dockerfile CMD must contain uvicorn and port 7860."""
    dockerfile = PROJECT_ROOT / "Dockerfile"
    content = dockerfile.read_text()
    assert "uvicorn" in content, "Dockerfile CMD does not contain uvicorn"
    assert "7860" in content, "Dockerfile CMD does not reference port 7860"


def test_dockerignore_excludes_directories():
    """.dockerignore must exclude data/, notebooks/, tests/."""
    dockerignore = PROJECT_ROOT / ".dockerignore"
    assert dockerignore.exists(), ".dockerignore not found"
    content = dockerignore.read_text()
    assert "data/" in content, ".dockerignore does not exclude data/"
    assert "notebooks/" in content, ".dockerignore does not exclude notebooks/"
    assert "tests/" in content, ".dockerignore does not exclude tests/"
