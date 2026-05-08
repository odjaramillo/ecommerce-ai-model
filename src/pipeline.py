"""Data pipeline for e-commerce purchase intent prediction."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

TARGET = "Revenue"

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset_shop.csv"


def load_data(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw dataset from disk."""
    path = Path(csv_path) if csv_path else DATA_PATH
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target. Missing values are handled inside the Pipeline."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def _cast_to_object(X):
    """Cast DataFrame to object dtype so SimpleImputer accepts mixed types."""
    return X.astype(object)


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer with imputation and scaling/encoding.

    Numeric features: SimpleImputer(median) -> RobustScaler
    Categorical features: cast to object -> SimpleImputer(most_frequent) -> OneHotEncoder
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("to_object", FunctionTransformer(_cast_to_object, validate=False)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified 70/15/15 train/val/test split on the target."""
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=train_size,
        stratify=y,
        random_state=random_state,
    )

    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=val_ratio,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df)
    preprocessor = build_preprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
