"""Integration tests for the FastAPI prediction service."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src import api as api_module
from src.api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_health_endpoint_model_not_loaded(monkeypatch):
    """When model is not loaded, /health should still return 200 with model_loaded=False."""
    fake_path = MagicMock()
    fake_path.exists.return_value = False
    monkeypatch.setattr(api_module, "ARTIFACT_PATH", fake_path)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is False


def test_predict_intent_valid(client):
    payload = {
        "Administrative": 2,
        "Administrative_Duration": 53.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 23,
        "ProductRelated_Duration": 1668.28,
        "BounceRates": 0.0083,
        "ExitRates": 0.0163,
        "PageValues": 0.0,
        "SpecialDay": 0.0,
        "Month": "Feb",
        "OperatingSystems": 1,
        "Browser": 1,
        "Region": 9,
        "TrafficType": 3,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }
    response = client.post("/api/predict_intent", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "classification" in data
    assert "probability" in data
    assert "human_readable_message" in data
    assert data["classification"] in ("compra", "no_compra")
    assert 0.0 <= data["probability"] <= 1.0
    assert isinstance(data["human_readable_message"], str)
    assert len(data["human_readable_message"]) > 0


def test_predict_intent_response_schema(client):
    payload = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 1,
        "ProductRelated_Duration": 5.0,
        "BounceRates": 0.05,
        "ExitRates": 0.05,
        "PageValues": 0.0,
        "SpecialDay": 0.0,
        "Month": "Jan",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": "New_Visitor",
        "Weekend": True,
    }
    response = client.post("/api/predict_intent", json=payload)
    assert response.status_code == 200
    data = response.json()
    required_fields = {"classification", "probability", "human_readable_message"}
    assert required_fields.issubset(data.keys())


def test_predict_intent_model_not_loaded(monkeypatch):
    """When model is not loaded, POST /api/predict_intent should return 503."""
    fake_path = MagicMock()
    fake_path.exists.return_value = False
    monkeypatch.setattr(api_module, "ARTIFACT_PATH", fake_path)
    with TestClient(app) as client:
        payload = {
            "Administrative": 0,
            "Administrative_Duration": 0.0,
            "Informational": 0,
            "Informational_Duration": 0.0,
            "ProductRelated": 1,
            "ProductRelated_Duration": 5.0,
            "BounceRates": 0.05,
            "ExitRates": 0.05,
            "PageValues": 0.0,
            "SpecialDay": 0.0,
            "Month": "Jan",
            "OperatingSystems": 2,
            "Browser": 2,
            "Region": 1,
            "TrafficType": 1,
            "VisitorType": "New_Visitor",
            "Weekend": True,
        }
        response = client.post("/api/predict_intent", json=payload)
        assert response.status_code == 503
        data = response.json()
        assert data["detail"] == "Model not loaded. Service unavailable."


class DummyModel:
    """Fake model that returns a fixed probability."""

    def __init__(self, proba):
        self._proba = proba

    def predict_proba(self, X):
        return [[1 - self._proba, self._proba]]


def _predict_with_mocked_proba(monkeypatch, proba):
    """Start app with a fake model loaded so lifespan injects DummyModel."""
    fake_path = MagicMock()
    fake_path.exists.return_value = True
    monkeypatch.setattr(api_module, "ARTIFACT_PATH", fake_path)
    monkeypatch.setattr(api_module.joblib, "load", lambda path: DummyModel(proba))
    with TestClient(app) as client:
        payload = {
            "Administrative": 0,
            "Administrative_Duration": 0.0,
            "Informational": 0,
            "Informational_Duration": 0.0,
            "ProductRelated": 1,
            "ProductRelated_Duration": 5.0,
            "BounceRates": 0.05,
            "ExitRates": 0.05,
            "PageValues": 0.0,
            "SpecialDay": 0.0,
            "Month": "Jan",
            "OperatingSystems": 2,
            "Browser": 2,
            "Region": 1,
            "TrafficType": 1,
            "VisitorType": "New_Visitor",
            "Weekend": True,
        }
        return client.post("/api/predict_intent", json=payload)


def test_probability_tier_high(monkeypatch):
    """probability >= 0.70 -> message contains 'bastante probable'."""
    response = _predict_with_mocked_proba(monkeypatch, 0.85)
    assert response.status_code == 200
    data = response.json()
    assert "bastante probable" in data["human_readable_message"]


def test_probability_tier_moderate(monkeypatch):
    """probability 0.50-0.69 -> message contains 'moderadamente probable'."""
    response = _predict_with_mocked_proba(monkeypatch, 0.60)
    assert response.status_code == 200
    data = response.json()
    assert "moderadamente probable" in data["human_readable_message"]


def test_probability_tier_low(monkeypatch):
    """probability < 0.50 -> message contains 'poco probable'."""
    response = _predict_with_mocked_proba(monkeypatch, 0.30)
    assert response.status_code == 200
    data = response.json()
    assert "poco probable" in data["human_readable_message"]
