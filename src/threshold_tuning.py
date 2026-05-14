"""Threshold tuning experiment on best Random Forest model (rf-i2-16)."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.pipeline import build_preprocessor, load_data, preprocess_data, split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "ecommerce_pipeline.pkl"

# Best config from iteration 2
BEST_CONFIG = {
    "name": "rf-i2-16",
    "params": {
        "clf__n_estimators": 550,
        "clf__max_depth": None,
        "clf__min_samples_split": 3,
        "clf__min_samples_leaf": 3,
        "clf__criterion": "entropy",
    },
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


def evaluate(y_true, y_pred, y_proba) -> dict:
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def find_optimal_threshold(y_true, y_proba, metric="accuracy") -> tuple[float, float]:
    """Find the threshold that maximizes the given metric.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        metric: Metric to optimize ("accuracy", "f1", "precision", "recall")
    
    Returns:
        (best_threshold, best_metric_value)
    """
    best_threshold = 0.5
    best_value = 0.0
    
    logger.info(f"Searching optimal threshold for metric: {metric}")
    for threshold in np.arange(0.10, 0.91, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == "accuracy":
            value = accuracy_score(y_true, y_pred)
        elif metric == "f1":
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            value = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            value = recall_score(y_true, y_pred, zero_division=0)
        else:
            value = accuracy_score(y_true, y_pred)
        
        if value > best_value:
            best_value = value
            best_threshold = threshold
    
    logger.info(f"Optimal threshold: {best_threshold:.2f} ({metric}: {best_value:.4f})")
    return best_threshold, best_value


def main():
    logger.info("Loading data...")
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    logger.info(f"Splits -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # Build and train model
    logger.info(f"Training {BEST_CONFIG['name']}...")
    pipeline = build_model_pipeline()
    pipeline.set_params(**BEST_CONFIG["params"])
    pipeline.fit(X_train, y_train)

    # Get validation probabilities
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    
    # Default threshold (0.5) metrics
    val_pred_default = (val_proba >= 0.5).astype(int)
    metrics_default = evaluate(y_val, val_pred_default, val_proba)
    logger.info(f"Validation metrics (threshold=0.50): {json.dumps(metrics_default, indent=2)}")

    # Find optimal thresholds for different metrics
    thresholds = {}
    for metric in ["accuracy", "f1", "precision", "recall"]:
        threshold, value = find_optimal_threshold(y_val, val_proba, metric=metric)
        thresholds[metric] = {"threshold": threshold, "value": value}
    
    # Show all thresholds
    logger.info("\nThreshold comparison:")
    for metric, data in thresholds.items():
        logger.info(f"  {metric:12s}: threshold={data['threshold']:.2f}, {metric}={data['value']:.4f}")

    # Test with different thresholds
    logger.info("\nTest set evaluation with different thresholds:")
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    
    for metric, data in thresholds.items():
        threshold = data["threshold"]
        test_pred = (test_proba >= threshold).astype(int)
        test_metrics = evaluate(y_test, test_pred, test_proba)
        logger.info(f"\nThreshold optimized for {metric} ({threshold:.2f}):")
        logger.info(f"  Test metrics: {json.dumps(test_metrics, indent=2)}")
        
        cm = confusion_matrix(y_test, test_pred)
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    Pred: No    Pred: Sí")
        logger.info(f"    Actual: No  {cm[0][0]:6d}    {cm[0][1]:6d}")
        logger.info(f"    Actual: Sí  {cm[1][0]:6d}    {cm[1][1]:6d}")

    # Save best model
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_PATH)
    logger.info(f"\nModel saved to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
