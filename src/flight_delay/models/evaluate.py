"""Model evaluation: metrics, comparison tables, and publication-quality plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from flight_delay.utils.config import REPORTS_DIR

logger = logging.getLogger(__name__)

# ── Global plot style ────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15, rc={"figure.dpi": 150})
_PALETTE = sns.color_palette("husl", 5)
_LINE_STYLES = ["-", "--", "-.", ":"]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
) -> dict[str, float]:
    """Compute classification metrics for a fitted model.

    Parameters
    ----------
    model : estimator
        Fitted model with ``predict`` and ``predict_proba`` methods.
    X_test : array-like of shape (n_samples, n_features)
        Test features.
    y_test : array-like of shape (n_samples,)
        True labels.
    model_name : str
        Label used for logging.

    Returns
    -------
    dict[str, float]
        Dictionary of metric name → value.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }
    logger.info(
        "%s — F1: %.4f | ROC-AUC: %.4f | PR-AUC: %.4f",
        model_name,
        metrics["f1"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def compare_models(
    models_dict: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Evaluate all models and return a side-by-side comparison DataFrame.

    Parameters
    ----------
    models_dict : dict[str, estimator]
        Mapping of model names to fitted estimators.
    X_test, y_test : array-like
        Test data.

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = metrics.
    """
    rows: list[dict[str, Any]] = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        rows.append({"model": name, **metrics})

    df = pd.DataFrame(rows).set_index("model")
    df = df.sort_values("pr_auc", ascending=False)

    # Pretty-print the comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(df.round(4).to_string())
    print("=" * 70 + "\n")

    return df


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------


def plot_roc_curves(
    models_dict: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot ROC curves for all models on the same axes.

    Parameters
    ----------
    models_dict : dict[str, estimator]
        Mapping of model names to fitted estimators.
    X_test, y_test : array-like
        Test data.
    save_path : str or Path, optional
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, (name, model) in enumerate(models_dict.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(
            fpr,
            tpr,
            label=f"{name}  (AUC = {auc_val:.3f})",
            color=_PALETTE[idx % len(_PALETTE)],
            linestyle=_LINE_STYLES[idx % len(_LINE_STYLES)],
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    _save_figure(fig, save_path, default_name="roc_curves.png")
    return fig


# ---------------------------------------------------------------------------
# Precision-Recall curves
# ---------------------------------------------------------------------------


def plot_precision_recall_curves(
    models_dict: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot Precision-Recall curves for all models.

    Parameters
    ----------
    models_dict : dict[str, estimator]
        Mapping of model names to fitted estimators.
    X_test, y_test : array-like
        Test data.
    save_path : str or Path, optional
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    baseline = np.mean(y_test)

    for idx, (name, model) in enumerate(models_dict.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(
            recall,
            precision,
            label=f"{name}  (AP = {ap:.3f})",
            color=_PALETTE[idx % len(_PALETTE)],
            linestyle=_LINE_STYLES[idx % len(_LINE_STYLES)],
            linewidth=2,
        )

    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.4, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()

    _save_figure(fig, save_path, default_name="pr_curves.png")
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot a normalised confusion matrix heatmap.

    Parameters
    ----------
    model : estimator
        Fitted model.
    X_test, y_test : array-like
        Test data.
    model_name : str
        Used in the plot title.
    save_path : str or Path, optional
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=["On-Time", "Delayed"],
        yticklabels=["On-Time", "Delayed"],
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()

    _save_figure(fig, save_path, default_name=f"confusion_matrix_{model_name}.png")
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Bar plot of feature importances (Random Forest / XGBoost).

    Parameters
    ----------
    model : estimator
        A model exposing ``feature_importances_``.
    feature_names : list[str]
        Feature column names.
    top_n : int
        Number of top features to display.
    save_path : str or Path, optional
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(
        range(len(idx)),
        importances[idx][::-1],
        color=sns.color_palette("viridis", len(idx)),
    )
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()

    _save_figure(fig, save_path, default_name="feature_importance.png")
    return fig


# ---------------------------------------------------------------------------
# Optimal threshold search
# ---------------------------------------------------------------------------


def find_optimal_threshold(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "f1",
) -> float:
    """Find the classification threshold that maximises a given metric.

    Parameters
    ----------
    model : estimator
        Fitted model with ``predict_proba``.
    X_val, y_val : array-like
        Validation data.
    metric : str
        One of ``'f1'``, ``'precision'``, ``'recall'``.

    Returns
    -------
    float
        Optimal threshold in [0.1, 0.9].
    """
    metric_fn = {
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score,
    }
    if metric not in metric_fn:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(metric_fn)}")

    y_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.91, 0.01)
    scores = [
        metric_fn[metric](y_val, (y_prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_idx = int(np.argmax(scores))
    best_threshold = float(thresholds[best_idx])
    logger.info(
        "Optimal threshold for %s: %.2f (score=%.4f)",
        metric,
        best_threshold,
        scores[best_idx],
    )
    return best_threshold


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_figure(
    fig: plt.Figure,
    save_path: Optional[str | Path],
    default_name: str,
) -> None:
    """Save a figure to *save_path* or to ``REPORTS_DIR / default_name``."""
    if save_path is not None:
        path = Path(save_path)
    else:
        path = REPORTS_DIR / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    logger.info("Figure saved → %s", path)
