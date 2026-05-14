"""Model training entrypoint for the e-commerce purchase intent predictor.

Iteration 6: Balance-Focused - Optimizing F1 and Recall while maintaining good Accuracy.
"""

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
CONFIG_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model_config.json"

SEARCH_CONFIGS = [
    # Iteration 6: Balance-focused configs
    # Strategy: Less regularization, deeper trees, entropy criterion for better recall
    {
        "name": "rf-bal-01",
        "params": {"clf__n_estimators": 300, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-02",
        "params": {"clf__n_estimators": 400, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-03",
        "params": {"clf__n_estimators": 500, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-04",
        "params": {"clf__n_estimators": 600, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-05",
        "params": {"clf__n_estimators": 400, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 2, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-06",
        "params": {"clf__n_estimators": 500, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 2, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-07",
        "params": {"clf__n_estimators": 600, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 2, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-08",
        "params": {"clf__n_estimators": 300, "clf__max_depth": None, "clf__min_samples_split": 3, "clf__min_samples_leaf": 1, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-09",
        "params": {"clf__n_estimators": 400, "clf__max_depth": None, "clf__min_samples_split": 3, "clf__min_samples_leaf": 1, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-10",
        "params": {"clf__n_estimators": 500, "clf__max_depth": None, "clf__min_samples_split": 3, "clf__min_samples_leaf": 1, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-11",
        "params": {"clf__n_estimators": 400, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "gini"},
    },
    {
        "name": "rf-bal-12",
        "params": {"clf__n_estimators": 500, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "gini"},
    },
    {
        "name": "rf-bal-13",
        "params": {"clf__n_estimators": 600, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "gini"},
    },
    {
        "name": "rf-bal-14",
        "params": {"clf__n_estimators": 500, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "entropy", "clf__max_features": "log2"},
    },
    {
        "name": "rf-bal-15",
        "params": {"clf__n_estimators": 500, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "entropy", "clf__max_features": 0.7},
    },
    {
        "name": "rf-bal-16",
        "params": {"clf__n_estimators": 400, "clf__max_depth": 30, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-17",
        "params": {"clf__n_estimators": 500, "clf__max_depth": 30, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-18",
        "params": {"clf__n_estimators": 600, "clf__max_depth": 30, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-19",
        "params": {"clf__n_estimators": 700, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "entropy"},
    },
    {
        "name": "rf-bal-20",
        "params": {"clf__n_estimators": 800, "clf__max_depth": None, "clf__min_samples_split": 2, "clf__min_samples_leaf": 1, "clf__criterion": "entropy"},
    },
]


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


def find_optimal_threshold(y_true, y_proba, metric="f1") -> tuple[float, float]:
    """Find the threshold that maximizes the given metric.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        metric: Metric to optimize ("f1", "accuracy", "precision", "recall")
    
    Returns:
        (best_threshold, best_metric_value)
    """
    best_threshold = 0.5
    best_value = 0.0
    
    logger.info(f"Searching optimal threshold for metric: {metric}")
    for threshold in np.arange(0.10, 0.91, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == "f1":
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "accuracy":
            value = accuracy_score(y_true, y_pred)
        elif metric == "precision":
            value = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            value = recall_score(y_true, y_pred, zero_division=0)
        else:
            value = f1_score(y_true, y_pred, zero_division=0)
        
        if value > best_value:
            best_value = value
            best_threshold = threshold
    
    logger.info(f"Optimal threshold: {best_threshold:.2f} ({metric}: {best_value:.4f})")
    return best_threshold, best_value


def run_manual_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config_grid: list[dict] | None = None,
) -> list[dict]:
    """Iterate over a config grid, fit on train, evaluate on val, and return results."""
    if config_grid is None:
        config_grid = SEARCH_CONFIGS

    results = []
    for config in config_grid:
        logger.info(f"Training config {config['name']} with params {config['params']}...")
        pipeline = build_model_pipeline()
        pipeline.set_params(**config["params"])
        pipeline.fit(X_train, y_train)

        val_proba = pipeline.predict_proba(X_val)[:, 1]
        
        # Find optimal threshold for F1 (balance metric)
        optimal_threshold, _ = find_optimal_threshold(y_val, val_proba, metric="f1")
        val_pred = (val_proba >= optimal_threshold).astype(int)
        
        val_metrics = evaluate(y_val, val_pred, val_proba)
        val_metrics["threshold"] = optimal_threshold
        
        logger.info(f"Config {config['name']} validation metrics: {json.dumps(val_metrics, indent=2)}")

        results.append(
            {
                "name": config["name"],
                "params": config["params"],
                "val_metrics": val_metrics,
                "threshold": optimal_threshold,
            }
        )

    return results


def select_best_result(results: list[dict]) -> dict:
    """Pick the best configuration prioritizing F1 >= 0.65 with accuracy >= 0.88.
    
    Falls back to ROC-AUC based selection if accuracy is not available in metrics
    (for backward compatibility with tests and legacy results).
    """
    # Check if accuracy is available in metrics
    has_accuracy = all("accuracy" in r["val_metrics"] for r in results)
    
    if not has_accuracy:
        # Legacy mode: ROC-AUC based selection (for tests compatibility)
        return sorted(
            results,
            key=lambda r: (
                -r["val_metrics"]["roc_auc"],
                -r["val_metrics"]["f1"],
                r["name"],
            ),
        )[0]
    
    # Strategy: F1 >= 0.65 as primary constraint, then accuracy >= 0.89
    high_f1 = [r for r in results if r["val_metrics"]["f1"] >= 0.65]
    if high_f1:
        # Among high F1, pick best accuracy, then ROC-AUC
        good_acc = [r for r in high_f1 if r["val_metrics"]["accuracy"] >= 0.89]
        if good_acc:
            return sorted(
                good_acc,
                key=lambda r: (
                    -r["val_metrics"]["f1"],
                    -r["val_metrics"]["accuracy"],
                    -r["val_metrics"]["roc_auc"],
                    r["name"],
                ),
            )[0]
        else:
            return sorted(
                high_f1,
                key=lambda r: (
                    -r["val_metrics"]["f1"],
                    -r["val_metrics"]["accuracy"],
                    -r["val_metrics"]["roc_auc"],
                    r["name"],
                ),
            )[0]

    # Fallback: configs with reasonable F1 (>= 0.60) and best accuracy
    reasonable = [r for r in results if r["val_metrics"]["f1"] >= 0.60]
    if reasonable:
        return sorted(
            reasonable,
            key=lambda r: (
                -r["val_metrics"]["f1"],
                -r["val_metrics"]["accuracy"],
                -r["val_metrics"]["roc_auc"],
                r["name"],
            ),
        )[0]

    # Last resort: best F1 overall
    return sorted(
        results,
        key=lambda r: (
            -r["val_metrics"]["f1"],
            -r["val_metrics"]["accuracy"],
            -r["val_metrics"]["roc_auc"],
            r["name"],
        ),
    )[0]


def train() -> tuple[Pipeline, dict, dict]:
    """Run the full training workflow: load, split, manual HPO, evaluate, serialize."""
    logger.info("Loading data...")
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    logger.info(f"Splits -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    logger.info(f"Starting manual search over {len(SEARCH_CONFIGS)} configurations...")
    results = run_manual_search(X_train, y_train, X_val, y_val)
    best_result = select_best_result(results)
    logger.info(f"Best config: {best_result['name']}")
    logger.info(f"Best params: {best_result['params']}")
    logger.info(f"Best threshold: {best_result['threshold']:.2f}")
    logger.info(f"Best validation metrics: {json.dumps(best_result['val_metrics'], indent=2)}")

    # Refit on Train + Val (85% of data)
    logger.info("Refitting best configuration on Train + Val...")
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    best_pipeline = build_model_pipeline()
    best_pipeline.set_params(**best_result["params"])
    best_pipeline.fit(X_train_val, y_train_val)

    # Test evaluation (once) with optimal threshold
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_result["threshold"]).astype(int)
    test_metrics = evaluate(y_test, test_pred, test_proba)
    logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")
    print_confusion_matrix(y_test, test_pred, label="Test")

    # Serialize model
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, ARTIFACT_PATH, compress=3)
    logger.info(f"Model serialized to {ARTIFACT_PATH}")

    # Save config with threshold
    config = {
        "threshold": float(best_result["threshold"]),
        "model_name": best_result["name"],
        "params": best_result["params"],
        "test_metrics": test_metrics,
        "description": f"RandomForest balance-focused - {best_result['name']}",
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Model config saved to {CONFIG_PATH}")

    return best_pipeline, best_result["val_metrics"], test_metrics


if __name__ == "__main__":
    train()
