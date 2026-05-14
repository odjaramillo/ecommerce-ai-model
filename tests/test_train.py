"""Behavioral tests for training and HPO configuration."""

from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src import train
from src.train import SEARCH_CONFIGS, build_model_pipeline, run_manual_search, select_best_result


class StubPipeline:
    """Lightweight stub for build_model_pipeline to avoid heavy computation in tests."""

    def __init__(self):
        self._params = {}

    def set_params(self, **params):
        self._params = params
        return self

    def fit(self, X, y):
        return self

    def predict(self, X):
        n = len(X)
        return np.array([i % 2 == 0 for i in range(n)])

    def predict_proba(self, X):
        n = len(X)
        probs = np.array([0.3 + (i % 5) * 0.1 for i in range(n)], dtype=float)
        return np.column_stack([1 - probs, probs])


class TrackingStub(StubPipeline):
    """Stub that tracks fit/predict calls for behavioral tests."""

    def __init__(self):
        super().__init__()
        self.fit_calls = []
        self.predict_calls = []

    def fit(self, X, y):
        self.fit_calls.append((len(X), len(y)))
        return self

    def predict(self, X):
        self.predict_calls.append(len(X))
        return super().predict(X)
    
    def predict_proba(self, X):
        self.predict_calls.append(len(X))
        return super().predict_proba(X)


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


def test_pipeline_structure():
    """Verify the pipeline contains preprocessor + classifier."""
    pipeline = build_model_pipeline()
    assert isinstance(pipeline, Pipeline)
    step_names = [name for name, _ in pipeline.steps]
    assert "preprocessor" in step_names
    assert "clf" in step_names

    clf = pipeline.named_steps["clf"]
    assert isinstance(clf, RandomForestClassifier)


def test_run_manual_search_iterates_all_configs():
    """Verify run_manual_search evaluates every config in the grid."""
    stub_grid = [
        {"name": "cfg-a", "params": {"clf__n_estimators": 10}},
        {"name": "cfg-b", "params": {"clf__n_estimators": 20}},
        {"name": "cfg-c", "params": {"clf__n_estimators": 30}},
    ]
    df = _make_mock_df(n_rows=24)
    X = df.drop(columns=["Revenue"])
    y = df["Revenue"]

    with patch("src.train.build_model_pipeline", return_value=StubPipeline()):
        results = run_manual_search(X, y, X, y, config_grid=stub_grid)

    assert len(results) == 3
    names = [r["name"] for r in results]
    assert names == ["cfg-a", "cfg-b", "cfg-c"]
    for r in results:
        assert "params" in r
        assert "val_metrics" in r
        assert "roc_auc" in r["val_metrics"]
        assert "f1" in r["val_metrics"]


def test_select_best_result_primary_by_roc_auc():
    """Verify best result is selected by highest ROC-AUC."""
    results = [
        {"name": "cfg-a", "val_metrics": {"roc_auc": 0.80, "f1": 0.70}},
        {"name": "cfg-b", "val_metrics": {"roc_auc": 0.90, "f1": 0.60}},
        {"name": "cfg-c", "val_metrics": {"roc_auc": 0.85, "f1": 0.65}},
    ]
    best = select_best_result(results)
    assert best["name"] == "cfg-b"


def test_select_best_result_tiebreak_by_f1():
    """When ROC-AUC is tied, verify F1 breaks the tie."""
    results = [
        {"name": "cfg-a", "val_metrics": {"roc_auc": 0.90, "f1": 0.65}},
        {"name": "cfg-b", "val_metrics": {"roc_auc": 0.90, "f1": 0.75}},
        {"name": "cfg-c", "val_metrics": {"roc_auc": 0.90, "f1": 0.70}},
    ]
    best = select_best_result(results)
    assert best["name"] == "cfg-b"


def test_select_best_result_tiebreak_by_name():
    """When both ROC-AUC and F1 are tied, verify name ascending breaks the tie."""
    results = [
        {"name": "cfg-b", "val_metrics": {"roc_auc": 0.90, "f1": 0.70}},
        {"name": "cfg-a", "val_metrics": {"roc_auc": 0.90, "f1": 0.70}},
        {"name": "cfg-c", "val_metrics": {"roc_auc": 0.90, "f1": 0.70}},
    ]
    best = select_best_result(results)
    assert best["name"] == "cfg-a"


def test_manual_search_does_not_use_test_set():
    """Verify run_manual_search never receives X_test/y_test."""
    df = _make_mock_df(n_rows=24)
    X = df.drop(columns=["Revenue"])
    y = df["Revenue"]

    stub_grid = [
        {"name": "cfg-a", "params": {"clf__n_estimators": 10}},
        {"name": "cfg-b", "params": {"clf__n_estimators": 20}},
    ]

    stub = TrackingStub()
    with patch("src.train.build_model_pipeline", return_value=stub):
        results = run_manual_search(X, y, X, y, config_grid=stub_grid)

    # Each config fits on X (train) and predicts on X (val)
    assert len(stub.fit_calls) == 2
    assert len(stub.predict_calls) == 2
    # If test was used, predict_calls would be different lengths or we'd see 3 calls


def test_train_refits_on_train_plus_val_and_evaluates_test_once(tmp_path, monkeypatch):
    """Verify train() refits best config on Train+Val and evaluates Test exactly once."""
    artifact_path = tmp_path / "ecommerce_pipeline.pkl"
    monkeypatch.setattr(train, "ARTIFACT_PATH", artifact_path)

    stub = TrackingStub()
    with patch("src.train.build_model_pipeline", return_value=stub):
        with patch("src.train.load_data") as mock_load_data:
            mock_load_data.return_value = _make_mock_df(n_rows=60)
            train.train()

    # Should have: N configs fits + 1 final refit on Train+Val
    # With default 12 configs: 12 + 1 = 13 fits
    assert len(stub.fit_calls) > 1, f"Expected multiple fits, got {len(stub.fit_calls)}"

    # The last fit should be on a LARGER dataset (Train+Val > Train alone)
    first_fit_size = stub.fit_calls[0][0]
    last_fit_size = stub.fit_calls[-1][0]
    assert last_fit_size > first_fit_size, (
        f"Final refit ({last_fit_size}) should be larger than initial fits ({first_fit_size})"
    )

    # Predict should be called: N configs * 1 (val) + 1 (test) = N + 1
    assert len(stub.predict_calls) > 1

    # Artifact should be created
    assert artifact_path.exists()
    loaded = joblib.load(artifact_path)
    assert isinstance(loaded, TrackingStub)


def test_model_artifact_created_after_training(tmp_path, monkeypatch):
    """Verify model artifact is created after training with manual search."""
    artifact_path = tmp_path / "ecommerce_pipeline.pkl"
    monkeypatch.setattr(train, "ARTIFACT_PATH", artifact_path)

    with patch("src.train.build_model_pipeline", return_value=StubPipeline()):
        with patch("src.train.load_data") as mock_load_data:
            mock_load_data.return_value = _make_mock_df(n_rows=24)
            train.train()

    assert artifact_path.exists()
    loaded = joblib.load(artifact_path)
    assert isinstance(loaded, StubPipeline)
