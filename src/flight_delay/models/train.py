"""Model training, evaluation, and persistence for flight delay prediction.

Supports three classifiers:
- ``'lr'``      — Logistic Regression
- ``'rf'``      — Random Forest
- ``'xgboost'`` — XGBoost

Provides functions to train, evaluate (accuracy, precision, recall, F1,
ROC-AUC, PR-AUC), save/load models, and a convenience function to
train all three at once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from flight_delay.utils.config import (
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TRAIN_MONTHS,
    TEST_MONTHS,
    XGBOOST_DEFAULT_PARAMS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    model_type: str = "xgboost",
) -> Any:
    """Train a classification model.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples,)
        Training target vector.
    model_type : str, optional
        Model type — ``'lr'`` for Logistic Regression, ``'rf'`` for
        Random Forest, ``'xgboost'`` for XGBoost (default).

    Returns
    -------
    estimator
        Fitted scikit-learn or XGBoost model.

    Raises
    ------
    ValueError
        If *model_type* is not one of ``'lr'``, ``'rf'``, ``'xgboost'``.
    """
    logger.info("Training model (type=%s) on %d samples …", model_type, len(y_train))

    if model_type == "lr":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
        model.fit(X_train, y_train)

    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)

    elif model_type == "xgboost":
        params = {**XGBOOST_DEFAULT_PARAMS}
        n_estimators = params.pop("n_estimators", 500)

        # Handle class imbalance
        n_pos = int(np.sum(np.asarray(y_train) == 1))
        n_neg = int(np.sum(np.asarray(y_train) == 0))
        if n_pos > 0:
            params.setdefault("scale_pos_weight", n_neg / n_pos)

        model = XGBClassifier(n_estimators=n_estimators, **params)
        model.fit(X_train, y_train, verbose=False)

    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            "Choose from 'lr', 'rf', 'xgboost'."
        )

    logger.info("Model training complete (type=%s).", model_type)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Evaluate a fitted model on a test set.

    Parameters
    ----------
    model : estimator
        Fitted classifier with ``predict`` and ``predict_proba`` methods.
    X_test : array-like of shape (n_samples, n_features)
        Test feature matrix.
    y_test : array-like of shape (n_samples,)
        Test target vector.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: ``'accuracy'``, ``'precision'``, ``'recall'``,
        ``'f1'``, ``'roc_auc'``, ``'pr_auc'``.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    logger.info(
        "Evaluation — Accuracy=%.4f  F1=%.4f  ROC-AUC=%.4f  PR-AUC=%.4f",
        metrics["accuracy"],
        metrics["f1"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: Any, filepath: Path) -> Path:
    """Save a model to disk using joblib.

    Parameters
    ----------
    model : estimator
        Fitted model to persist.
    filepath : Path
        Destination file path.

    Returns
    -------
    Path
        The path the model was saved to.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Model saved → %s", filepath)
    return filepath


def load_model(filepath: Path) -> Any:
    """Load a previously saved model from disk.

    Parameters
    ----------
    filepath : Path
        Path to the saved model file.

    Returns
    -------
    estimator
        The loaded model.
    """
    filepath = Path(filepath)
    model = joblib.load(filepath)
    logger.info("Model loaded ← %s", filepath)
    return model


# ---------------------------------------------------------------------------
# Train all models
# ---------------------------------------------------------------------------


def train_all_models(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
) -> dict[str, dict[str, Any]]:
    """Train all three model types, evaluate, and save to MODELS_DIR.

    Parameters
    ----------
    X_train, y_train : array-like
        Training data.
    X_test, y_test : array-like
        Test data for evaluation.

    Returns
    -------
    dict[str, dict[str, Any]]
        Keyed by model type (``'lr'``, ``'rf'``, ``'xgboost'``), each
        value is a dict with ``'model'``, ``'metrics'``, and ``'path'``.
    """
    results: dict[str, dict[str, Any]] = {}

    for model_type in ("lr", "rf", "xgboost"):
        logger.info("── Training %s ──", model_type)
        model = train_model(X_train, y_train, model_type=model_type)
        metrics = evaluate_model(model, X_test, y_test)
        path = save_model(model, MODELS_DIR / f"{model_type}_model.joblib")

        results[model_type] = {
            "model": model,
            "metrics": metrics,
            "path": path,
        }

        logger.info(
            "%s — F1=%.4f  ROC-AUC=%.4f",
            model_type,
            metrics["f1"],
            metrics["roc_auc"],
        )

    logger.info("All three models trained, evaluated, and saved.")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    from flight_delay.data.preprocess import preprocess_pipeline
    from flight_delay.features.pipeline import build_features, get_feature_columns
    from flight_delay.utils.config import RAW_DIR

    # Load and preprocess
    raw_path = RAW_DIR / "synthetic_flights.parquet"
    print(f"Loading data from {raw_path} …")
    df = preprocess_pipeline(raw_path)

    # Build features
    df = build_features(df)

    # Temporal split
    df["fl_date"] = pd.to_datetime(df["fl_date"])
    month = df["fl_date"].dt.month
    df_train = df[month.isin(TRAIN_MONTHS)].reset_index(drop=True)
    df_test = df[month.isin(TEST_MONTHS)].reset_index(drop=True)

    print(f"Train: {len(df_train):,} rows | Test: {len(df_test):,} rows")

    # Prepare feature matrices
    feature_cols = get_feature_columns()
    # Drop categorical features for numeric models
    numeric_features = [c for c in feature_cols if c not in ("carrier", "origin", "dest")]

    X_train = df_train[numeric_features].fillna(0)
    y_train = df_train[TARGET_COL]
    X_test = df_test[numeric_features].fillna(0)
    y_test = df_test[TARGET_COL]

    # Train all models
    results = train_all_models(X_train, y_train, X_test, y_test)

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    for name, info in results.items():
        m = info["metrics"]
        print(
            f"  {name:>8s} — "
            f"Acc={m['accuracy']:.4f}  "
            f"F1={m['f1']:.4f}  "
            f"ROC-AUC={m['roc_auc']:.4f}  "
            f"PR-AUC={m['pr_auc']:.4f}"
        )
    print("=" * 70)
