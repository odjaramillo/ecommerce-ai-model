"""Optuna-based Hyperparameter Optimization experiment.

Uses a TPE sampler with 30 trials to optimize RandomForest hyperparameters
for the e-commerce purchase intent classifier.
"""

import json
import logging
from pathlib import Path

import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.pipeline import build_preprocessor, load_data, preprocess_data, split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "ecommerce_pipeline_optuna.pkl"
MAIN_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "ecommerce_pipeline.pkl"


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


def print_confusion_matrix(y_true, y_pred, label="Test"):
    """Print confusion matrix in a readable format."""
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix ({label}):")
    logger.info(f"                 Pred: No  Pred: Sí")
    logger.info(f"Actual: No      {cm[0][0]:6d}    {cm[0][1]:6d}")
    logger.info(f"Actual: Sí      {cm[1][0]:6d}    {cm[1][1]:6d}")
    logger.info(f"Total samples: {cm.sum()}")


def objective(trial, X_train, y_train):
    """Optuna objective optimizing mean CV ROC-AUC on the training set."""
    params = {
        "clf__n_estimators": trial.suggest_int("clf__n_estimators", 100, 500, step=100),
        "clf__max_depth": trial.suggest_categorical("clf__max_depth", [10, 20, 30, None]),
        "clf__min_samples_split": trial.suggest_int("clf__min_samples_split", 2, 10),
        "clf__min_samples_leaf": trial.suggest_int("clf__min_samples_leaf", 1, 4),
    }

    pipeline = build_model_pipeline()
    pipeline.set_params(**params)

    scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
    )
    return scores.mean()


def train_optuna() -> Pipeline:
    """Run the Optuna HPO experiment and serialize the best model."""
    logger.info("Loading data...")
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    logger.info(f"Splits -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=42),
    )

    logger.info("Starting Optuna optimization (30 trials, TPE sampler)...")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=30,
        show_progress_bar=True,
    )

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best CV ROC AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {json.dumps(study.best_params, indent=2)}")

    # Refit best pipeline on full training data
    best_pipeline = build_model_pipeline()
    best_pipeline.set_params(**study.best_params)
    best_pipeline.fit(X_train, y_train)

    # Validation evaluation
    val_pred = best_pipeline.predict(X_val)
    val_proba = best_pipeline.predict_proba(X_val)[:, 1]
    val_metrics = evaluate(y_val, val_pred, val_proba)
    logger.info(f"Validation metrics: {json.dumps(val_metrics, indent=2)}")

    # Test evaluation
    test_pred = best_pipeline.predict(X_test)
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_pred, test_proba)
    logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")
    print_confusion_matrix(y_test, test_pred, label="Test")

    # Serialize
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, ARTIFACT_PATH)
    logger.info(f"Optuna model serialized to {ARTIFACT_PATH}")

    # Load main model metrics for comparison if available
    if MAIN_ARTIFACT_PATH.exists():
        main_pipeline = joblib.load(MAIN_ARTIFACT_PATH)
        main_test_proba = main_pipeline.predict_proba(X_test)[:, 1]
        main_test_pred = main_pipeline.predict(X_test)
        main_test_metrics = evaluate(y_test, main_test_pred, main_test_proba)

        logger.info("=" * 50)
        logger.info("COMPARISON: Main Model vs Optuna HPO")
        logger.info("=" * 50)
        for metric in main_test_metrics:
            main_val = main_test_metrics[metric]
            optuna_val = test_metrics[metric]
            delta = optuna_val - main_val
            logger.info(
                f"{metric:12s} | Main: {main_val:.4f} | Optuna: {optuna_val:.4f} | Δ {delta:+.4f}"
            )
    else:
        logger.warning("Main model artifact not found; skipping comparison.")

    return best_pipeline, val_metrics, test_metrics


if __name__ == "__main__":
    train_optuna()
