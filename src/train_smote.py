"""SMOTE oversampling experiment on preprocessed features.

This script fits the preprocessor on the original training data, transforms it,
applies SMOTE to the processed features (including One-Hot encoded columns),
and trains a RandomForest on the augmented data.

SMOTE is applied ONLY to the training set; validation and test sets are
preprocessed with the already-fitted preprocessor but never resampled.
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from imblearn.over_sampling import SMOTE

from src.pipeline import build_preprocessor, load_data, preprocess_data, split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "ecommerce_pipeline_smote.pkl"
MAIN_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "ecommerce_pipeline.pkl"


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


def train_smote():
    """Run the SMOTE experiment and serialize the best model."""
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

    logger.info("Training RandomForest on SMOTE-augmented data...")
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_smote, y_train_smote)

    # Validation evaluation
    X_val_processed = preprocessor.transform(X_val)
    val_pred = clf.predict(X_val_processed)
    val_proba = clf.predict_proba(X_val_processed)[:, 1]
    val_metrics = evaluate(y_val, val_pred, val_proba)
    logger.info(f"Validation metrics: {json.dumps(val_metrics, indent=2)}")

    # Test evaluation
    X_test_processed = preprocessor.transform(X_test)
    test_pred = clf.predict(X_test_processed)
    test_proba = clf.predict_proba(X_test_processed)[:, 1]
    test_metrics = evaluate(y_test, test_pred, test_proba)
    logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")
    print_confusion_matrix(y_test, test_pred, label="Test")

    # Serialize wrapper
    wrapper = SmotePipeline(preprocessor, clf)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapper, ARTIFACT_PATH)
    logger.info(f"SMOTE model serialized to {ARTIFACT_PATH}")

    # Load main model metrics for comparison if available
    if MAIN_ARTIFACT_PATH.exists():
        main_pipeline = joblib.load(MAIN_ARTIFACT_PATH)
        main_test_proba = main_pipeline.predict_proba(X_test)[:, 1]
        main_test_pred = main_pipeline.predict(X_test)
        main_test_metrics = evaluate(y_test, main_test_pred, main_test_proba)

        logger.info("=" * 50)
        logger.info("COMPARISON: Main Model vs SMOTE Experiment")
        logger.info("=" * 50)
        for metric in main_test_metrics:
            main_val = main_test_metrics[metric]
            smote_val = test_metrics[metric]
            delta = smote_val - main_val
            logger.info(
                f"{metric:12s} | Main: {main_val:.4f} | SMOTE: {smote_val:.4f} | Δ {delta:+.4f}"
            )
    else:
        logger.warning("Main model artifact not found; skipping comparison.")

    return wrapper, val_metrics, test_metrics


if __name__ == "__main__":
    train_smote()
