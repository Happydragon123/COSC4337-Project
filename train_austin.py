"""
Train and evaluate a severity classifier on Austin traffic accident data.

Usage:
    pip install -r requirements.txt
    python train_austin.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parent / "austin_data.csv"

NUMERIC_FEATURES = [
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Distance(mi)",
    "Start_Lat",
    "Start_Lng",
]

CATEGORICAL_FEATURES = [
    "Weather_Condition",
    "Wind_Direction",
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Source",
    "Airport_Code",
]

BOOLEAN_FEATURES = [
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",
    "Turning_Loop",
]


def load_and_engineer(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, low_memory=False)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["month"] = df["Start_Time"].dt.month
    df["hour"] = df["Start_Time"].dt.hour
    df["dayofweek"] = df["Start_Time"].dt.dayofweek

    feature_cols = (
        NUMERIC_FEATURES
        + ["month", "hour", "dayofweek"]
        + CATEGORICAL_FEATURES
        + BOOLEAN_FEATURES
    )
    X = df[feature_cols].copy()
    y = df["Severity"].copy()

    for c in BOOLEAN_FEATURES:
        if X[c].dtype == object:
            X[c] = X[c].map({"True": True, "False": False})
        X[c] = X[c].astype(bool).astype(np.int8)

    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].fillna("missing").astype(str)

    mask = y.notna() & X["month"].notna()
    X, y = X.loc[mask], y.loc[mask]
    # Median imputation in the pipeline handles missing numeric weather/road fields.

    return X, y


def build_pipeline() -> Pipeline:
    numeric_all = NUMERIC_FEATURES + ["month", "hour", "dayofweek"]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    min_frequency=0.001,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_all),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=24,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def majority_baseline(y_train: pd.Series, y_test: pd.Series) -> float:
    mode = y_train.mode().iloc[0]
    pred = np.full(shape=len(y_test), fill_value=mode)
    return accuracy_score(y_test, pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train severity model on Austin data.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to austin_data.csv",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Data file not found: {args.data}")

    print(f"Loading {args.data} ...")
    X, y = load_and_engineer(args.data)
    print(f"Samples after cleaning: {len(X):,}")
    print("Severity distribution:\n", y.value_counts().sort_index().to_string(), sep="")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )

    baseline_acc = majority_baseline(y_train, y_test)
    print(f"\nMajority-class baseline (test accuracy): {baseline_acc:.4f}")

    print("\nTraining Random Forest pipeline ...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")

    print("\n--- Results ---")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Test macro-F1:  {test_macro_f1:.4f}")
    print(
        f"Gap (train - test): {train_acc - test_acc:.4f}  "
        "(large gap may indicate overfitting)"
    )

    print("\nConfusion matrix (rows=true, cols=predicted):")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    print(pd.DataFrame(cm, index=[f"true_{i}" for i in labels], columns=[f"pred_{i}" for i in labels]))

    print("\nClassification report (test set):")
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=labels,
            digits=4,
        )
    )


if __name__ == "__main__":
    main()
