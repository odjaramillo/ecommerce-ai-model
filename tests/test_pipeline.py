"""Unit tests for the data pipeline module."""

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
    load_data,
    preprocess_data,
    split_data,
)


def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert TARGET in df.columns


def test_feature_groups_present():
    df = load_data()
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]:
        assert col in df.columns


def test_preprocess_data():
    df = load_data()
    X, y = preprocess_data(df)
    assert TARGET not in X.columns
    assert len(X) == len(y)
    assert y.dtype == bool


def test_split_ratios():
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    total = len(df)
    assert len(X_train) == pytest.approx(total * 0.70, abs=2)
    assert len(X_val) == pytest.approx(total * 0.15, abs=2)
    assert len(X_test) == pytest.approx(total * 0.15, abs=2)
    assert len(X_train) + len(X_val) + len(X_test) == total


def test_stratified_splits():
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    original_mean = y.mean()
    assert y_train.mean() == pytest.approx(original_mean, abs=0.05)
    assert y_val.mean() == pytest.approx(original_mean, abs=0.05)
    assert y_test.mean() == pytest.approx(original_mean, abs=0.05)


def test_build_preprocessor():
    preprocessor = build_preprocessor()
    df = load_data()
    X, _ = preprocess_data(df)
    preprocessor.fit(X)
    transformed = preprocessor.transform(X)
    assert transformed.shape[0] == len(X)


def test_preprocessor_uses_robust_scaler_for_numeric():
    """Numeric pipeline must include RobustScaler."""
    preprocessor = build_preprocessor()
    for name, transformer, columns in preprocessor.transformers:
        if name == "num":
            assert any(
                step_name == "scaler" and step_class.__class__.__name__ == "RobustScaler"
                for step_name, step_class in transformer.steps
            )
            return
    pytest.fail("Numeric transformer 'num' not found")


def test_preprocessor_uses_one_hot_encoder_for_categorical():
    """Categorical pipeline must include OneHotEncoder."""
    preprocessor = build_preprocessor()
    for name, transformer, columns in preprocessor.transformers:
        if name == "cat":
            assert any(
                step_name == "encoder" and step_class.__class__.__name__ == "OneHotEncoder"
                for step_name, step_class in transformer.steps
            )
            return
    pytest.fail("Categorical transformer 'cat' not found")


def test_pipeline_includes_simple_imputer():
    """Both numeric and categorical pipelines must include SimpleImputer."""
    preprocessor = build_preprocessor()

    for name, transformer, columns in preprocessor.transformers:
        if name == "num":
            assert any(
                step_name == "imputer" and step_class.__class__.__name__ == "SimpleImputer"
                for step_name, step_class in transformer.steps
            )
        if name == "cat":
            assert any(
                step_name == "imputer" and step_class.__class__.__name__ == "SimpleImputer"
                for step_name, step_class in transformer.steps
            )
