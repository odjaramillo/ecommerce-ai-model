"""Model training entrypoint for the e-commerce purchase intent predictor."""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.pipeline import build_preprocessor, load_data, preprocess_data, split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "ecommerce_pipeline.pkl"

SEARCH_SPACE = {
    "clf__n_estimators": [100, 200, 300, 400, 500],
    "clf__max_depth": [10, 20, 30, None],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
}


def build_model_pipeline() -> Pipeline:
    """Assemble the full sklearn Pipeline (preprocessor + classifier)."""
    preprocessor = build_preprocessor()
    clf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )


def evaluate(y_true: pd.Series, y_pred, y_proba) -> dict:
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def print_confusion_matrix(y_true, y_pred, label="Test"):
    """Print confusion matrix in a readable format."""
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix ({label}):")
    logger.info(f"                 Pred: No  Pred: Sí")
    logger.info(f"Actual: No      {cm[0][0]:6d}    {cm[0][1]:6d}")
    logger.info(f"Actual: Sí      {cm[1][0]:6d}    {cm[1][1]:6d}")
    logger.info(f"Total samples: {cm.sum()}")


def train() -> Pipeline:
    """Run the full training workflow: load, split, HPO, evaluate, serialize."""
    logger.info("Loading data...")
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    logger.info(f"Splits -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    pipeline = build_model_pipeline()

    logger.info("Starting RandomizedSearchCV (n_iter=30, cv=5, scoring=roc_auc)...")
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=SEARCH_SPACE,
        n_iter=30,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV ROC AUC: {search.best_score_:.4f}")

    best = search.best_estimator_

    # Validation evaluation
    val_pred = best.predict(X_val)
    val_proba = best.predict_proba(X_val)[:, 1]
    val_metrics = evaluate(y_val, val_pred, val_proba)
    logger.info(f"Validation metrics: {json.dumps(val_metrics, indent=2)}")

    # Test evaluation
    test_pred = best.predict(X_test)
    test_proba = best.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_pred, test_proba)
    logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")
    print_confusion_matrix(y_test, test_pred, label="Test")

    # Serialize
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, ARTIFACT_PATH)
    logger.info(f"Model serialized to {ARTIFACT_PATH}")

    return best, val_metrics, test_metrics


if __name__ == "__main__":
    train()
