"""Behavioral tests for training and HPO configuration."""

from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src import train
from src.train import SEARCH_SPACE, build_model_pipeline


class DummySearch:
    """Mock RandomizedSearchCV that avoids heavy computation."""

    last_call_kwargs = {}

    def __init__(self, *args, **kwargs):
        DummySearch.last_call_kwargs = kwargs
        self.best_estimator_ = build_model_pipeline()
        self.best_params_ = {"clf__n_estimators": 100}
        self.best_score_ = 0.95

    def fit(self, X, y=None):
        self.best_estimator_.fit(X, y)
        return self


def _make_mock_df(n_rows: int = 20, seed: int = 42) -> pd.DataFrame:
    """Generate a mock dataset large enough for stratified splits."""
    rng = pd.np.random.default_rng(seed) if hasattr(pd, "np") else __import__("numpy").random.default_rng(seed)
    n_pos = max(4, n_rows // 4)
    n_neg = n_rows - n_pos
    rows = []
    for _ in range(n_neg):
        rows.append(
            {
                "Administrative": rng.integers(0, 5),
                "Administrative_Duration": rng.random() * 100,
                "Informational": rng.integers(0, 3),
                "Informational_Duration": rng.random() * 50,
                "ProductRelated": rng.integers(1, 20),
                "ProductRelated_Duration": rng.random() * 500,
                "BounceRates": rng.random() * 0.1,
                "ExitRates": rng.random() * 0.1,
                "PageValues": rng.random() * 50,
                "SpecialDay": 0.0,
                "Month": rng.choice(["Feb", "Mar", "Jan"]),
                "OperatingSystems": rng.integers(1, 4),
                "Browser": rng.integers(1, 4),
                "Region": rng.integers(1, 5),
                "TrafficType": rng.integers(1, 5),
                "VisitorType": rng.choice(["New_Visitor", "Returning_Visitor"]),
                "Weekend": rng.choice([True, False]),
                "Revenue": False,
            }
        )
    for _ in range(n_pos):
        rows.append(
            {
                "Administrative": rng.integers(0, 5),
                "Administrative_Duration": rng.random() * 100,
                "Informational": rng.integers(0, 3),
                "Informational_Duration": rng.random() * 50,
                "ProductRelated": rng.integers(1, 20),
                "ProductRelated_Duration": rng.random() * 500,
                "BounceRates": rng.random() * 0.1,
                "ExitRates": rng.random() * 0.1,
                "PageValues": rng.random() * 50,
                "SpecialDay": 0.0,
                "Month": rng.choice(["Feb", "Mar", "Jan"]),
                "OperatingSystems": rng.integers(1, 4),
                "Browser": rng.integers(1, 4),
                "Region": rng.integers(1, 5),
                "TrafficType": rng.integers(1, 5),
                "VisitorType": rng.choice(["New_Visitor", "Returning_Visitor"]),
                "Weekend": rng.choice([True, False]),
                "Revenue": True,
            }
        )
    return pd.DataFrame(rows)


def test_randomized_search_cv_config():
    """Verify RandomizedSearchCV is configured with n_iter=30, cv=5, scoring='roc_auc'."""
    with patch("src.train.RandomizedSearchCV", new=DummySearch):
        with patch("src.train.load_data") as mock_load_data:
            with patch("src.train.joblib.dump"):
                mock_load_data.return_value = _make_mock_df(n_rows=24)
                train.train()

    assert DummySearch.last_call_kwargs["n_iter"] == 30
    assert DummySearch.last_call_kwargs["cv"] == 5
    assert DummySearch.last_call_kwargs["scoring"] == "roc_auc"


def test_pipeline_structure():
    """Verify the pipeline contains preprocessor + classifier."""
    pipeline = build_model_pipeline()
    assert isinstance(pipeline, Pipeline)
    step_names = [name for name, _ in pipeline.steps]
    assert "preprocessor" in step_names
    assert "clf" in step_names

    clf = pipeline.named_steps["clf"]
    assert isinstance(clf, RandomForestClassifier)


def test_model_artifact_created_after_training(tmp_path, monkeypatch):
    """Verify model artifact is created after training."""
    artifact_path = tmp_path / "ecommerce_pipeline.pkl"
    monkeypatch.setattr(train, "ARTIFACT_PATH", artifact_path)

    with patch("src.train.RandomizedSearchCV", new=DummySearch):
        with patch("src.train.load_data") as mock_load_data:
            mock_load_data.return_value = _make_mock_df(n_rows=24)
            train.train()

    assert artifact_path.exists()
    loaded = joblib.load(artifact_path)
    assert isinstance(loaded, Pipeline)
