"""SMOTE + HPO experiment: oversampling followed by hyperparameter search.

This script combines SMOTE oversampling with RandomizedSearchCV:
1. Fits the preprocessor on original training data.
2. Transforms training data.
3. Applies SMOTE to the processed features.
4. Runs RandomizedSearchCV (30 iterations) on the SMOTE-augmented data.
5. Evaluates on the ORIGINAL val/test sets (preprocessed but NOT SMOTE'd).

This differs from train_smote.py, which used fixed hyperparameters.
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from imblearn.over_sampling import SMOTE

from src.pipeline import build_preprocessor, load_data, preprocess_data, split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "ecommerce_pipeline_smote_hpo.pkl"
MAIN_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "ecommerce_pipeline.pkl"
SMOTE_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "ecommerce_pipeline_smote.pkl"

SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


class SmotePipeline:
    """Lightweight wrapper that preprocesses then classifies.

    Needed because SMOTE is applied offline on the training set only;
    it cannot be part of the inference pipeline.
    """

    def __init__(self, preprocessor, classifier):
        self.preprocessor = preprocessor
        self.classifier = classifier

    def predict(self, X):
        X_proc = self.preprocessor.transform(X)
        return self.classifier.predict(X_proc)

    def predict_proba(self, X):
        X_proc = self.preprocessor.transform(X)
        return self.classifier.predict_proba(X_proc)


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


def train_smote_hpo():
    """Run the SMOTE + HPO experiment and serialize the best model."""
    logger.info("Loading data...")
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    logger.info(f"Splits -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    logger.info("Fitting preprocessor on original training data...")
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    logger.info(f"Processed train shape: {X_train_processed.shape}")

    logger.info("Applying SMOTE on processed training features...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
    logger.info(f"After SMOTE -> shape: {X_train_smote.shape}, class balance: {pd.Series(y_train_smote).value_counts().to_dict()}")

    logger.info("Starting RandomizedSearchCV on SMOTE-augmented data (n_iter=30, cv=5, scoring=roc_auc)...")
    base_clf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=SEARCH_SPACE,
        n_iter=30,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        refit=True,
        verbose=1,
    )
    search.fit(X_train_smote, y_train_smote)
    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV ROC AUC: {search.best_score_:.4f}")

    best_clf = search.best_estimator_

    # Validation evaluation
    X_val_processed = preprocessor.transform(X_val)
    val_pred = best_clf.predict(X_val_processed)
    val_proba = best_clf.predict_proba(X_val_processed)[:, 1]
    val_metrics = evaluate(y_val, val_pred, val_proba)
    logger.info(f"Validation metrics: {json.dumps(val_metrics, indent=2)}")

    # Test evaluation
    X_test_processed = preprocessor.transform(X_test)
    test_pred = best_clf.predict(X_test_processed)
    test_proba = best_clf.predict_proba(X_test_processed)[:, 1]
    test_metrics = evaluate(y_test, test_pred, test_proba)
    logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")
    print_confusion_matrix(y_test, test_pred, label="Test")

    # Serialize wrapper
    wrapper = SmotePipeline(preprocessor, best_clf)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapper, ARTIFACT_PATH)
    logger.info(f"SMOTE+HPO model serialized to {ARTIFACT_PATH}")

    # Load main model metrics for comparison if available
    if MAIN_ARTIFACT_PATH.exists():
        main_pipeline = joblib.load(MAIN_ARTIFACT_PATH)
        main_test_proba = main_pipeline.predict_proba(X_test)[:, 1]
        main_test_pred = main_pipeline.predict(X_test)
        main_test_metrics = evaluate(y_test, main_test_pred, main_test_proba)

        logger.info("=" * 60)
        logger.info("COMPARISON: Main Model vs SMOTE+HPO Experiment")
        logger.info("=" * 60)
        for metric in main_test_metrics:
            main_val = main_test_metrics[metric]
            smote_hpo_val = test_metrics[metric]
            delta = smote_hpo_val - main_val
            logger.info(
                f"{metric:12s} | Main: {main_val:.4f} | SMOTE+HPO: {smote_hpo_val:.4f} | Δ {delta:+.4f}"
            )
    else:
        logger.warning("Main model artifact not found; skipping main comparison.")

    # Load previous SMOTE experiment metrics for comparison if available
    if SMOTE_ARTIFACT_PATH.exists():
        smote_pipeline = joblib.load(SMOTE_ARTIFACT_PATH)
        smote_test_proba = smote_pipeline.predict_proba(X_test)[:, 1]
        smote_test_pred = smote_pipeline.predict(X_test)
        smote_test_metrics = evaluate(y_test, smote_test_pred, smote_test_proba)

        logger.info("=" * 60)
        logger.info("COMPARISON: Previous SMOTE vs SMOTE+HPO Experiment")
        logger.info("=" * 60)
        for metric in smote_test_metrics:
            smote_val = smote_test_metrics[metric]
            smote_hpo_val = test_metrics[metric]
            delta = smote_hpo_val - smote_val
            logger.info(
                f"{metric:12s} | SMOTE: {smote_val:.4f} | SMOTE+HPO: {smote_hpo_val:.4f} | Δ {delta:+.4f}"
            )
    else:
        logger.warning("Previous SMOTE artifact not found; skipping SMOTE comparison.")

    return wrapper, val_metrics, test_metrics


if __name__ == "__main__":
    train_smote_hpo()
