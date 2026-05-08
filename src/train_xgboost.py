"""XGBoost experiment with scale_pos_weight for imbalanced classification.

This script trains an XGBClassifier using the same preprocessing pipeline
as the main model, but uses gradient boosting instead of Random Forest.
The scale_pos_weight hyperparameter is set dynamically based on the
class distribution of the training set to mitigate class imbalance.
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.pipeline import build_preprocessor, load_data, preprocess_data, split_data

# Patch XGBClassifier for sklearn 1.8+ compatibility: estimator_type is not set
# correctly in xgboost 2.1.1, causing RandomizedSearchCV to treat it as a regressor.
_original_sklearn_tags = XGBClassifier.__sklearn_tags__

def _patched_sklearn_tags(self):
    from dataclasses import replace
    tags = _original_sklearn_tags(self)
    return replace(tags, estimator_type="classifier")

XGBClassifier.__sklearn_tags__ = _patched_sklearn_tags

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "ecommerce_pipeline_xgboost.pkl"
MAIN_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "ecommerce_pipeline.pkl"

SEARCH_SPACE = {
    "clf__n_estimators": [100, 200, 300, 400, 500],
    "clf__max_depth": [3, 5, 7, 10],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "clf__subsample": [0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.8, 0.9, 1.0],
}


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """Calculate scale_pos_weight as ratio of negative to positive class."""
    counts = y_train.value_counts().sort_index()
    negative = counts.get(False, counts.get(0, 1))
    positive = counts.get(True, counts.get(1, 1))
    weight = negative / positive
    logger.info(f"Class distribution -> negative: {negative}, positive: {positive}, scale_pos_weight: {weight:.4f}")
    return weight


def build_model_pipeline(scale_pos_weight: float) -> Pipeline:
    """Assemble the full sklearn Pipeline (preprocessor + XGBoost classifier)."""
    preprocessor = build_preprocessor()
    clf = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
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


def print_confusion_matrix(y_true, y_pred, label="Test"):
    """Print confusion matrix in a readable format."""
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix ({label}):")
    logger.info(f"                 Pred: No  Pred: Sí")
    logger.info(f"Actual: No      {cm[0][0]:6d}    {cm[0][1]:6d}")
    logger.info(f"Actual: Sí      {cm[1][0]:6d}    {cm[1][1]:6d}")
    logger.info(f"Total samples: {cm.sum()}")


def train_xgboost() -> Pipeline:
    """Run the XGBoost training workflow: load, split, HPO, evaluate, serialize."""
    logger.info("Loading data...")
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    logger.info(f"Splits -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    pipeline = build_model_pipeline(scale_pos_weight)

    logger.info("Starting RandomizedSearchCV (n_iter=30, cv=5, scoring=roc_auc)...")
    # NOTE: n_jobs=1 for the search to avoid multiprocessing issues with
    # XGBClassifier sklearn tag compatibility. The XGBClassifier itself
    # still uses n_jobs=-1 for parallel tree construction.
    roc_auc_scorer = make_scorer(roc_auc_score, response_method="predict_proba")
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=SEARCH_SPACE,
        n_iter=30,
        cv=5,
        scoring=roc_auc_scorer,
        n_jobs=1,
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
    logger.info(f"XGBoost model serialized to {ARTIFACT_PATH}")

    # Load main model metrics for comparison if available
    if MAIN_ARTIFACT_PATH.exists():
        main_pipeline = joblib.load(MAIN_ARTIFACT_PATH)
        main_test_proba = main_pipeline.predict_proba(X_test)[:, 1]
        main_test_pred = main_pipeline.predict(X_test)
        main_test_metrics = evaluate(y_test, main_test_pred, main_test_proba)

        logger.info("=" * 50)
        logger.info("COMPARISON: Main Model vs XGBoost")
        logger.info("=" * 50)
        for metric in main_test_metrics:
            main_val = main_test_metrics[metric]
            xgb_val = test_metrics[metric]
            delta = xgb_val - main_val
            logger.info(
                f"{metric:12s} | Main: {main_val:.4f} | XGBoost: {xgb_val:.4f} | Δ {delta:+.4f}"
            )
    else:
        logger.warning("Main model artifact not found; skipping comparison.")

    return best, val_metrics, test_metrics


if __name__ == "__main__":
    train_xgboost()
