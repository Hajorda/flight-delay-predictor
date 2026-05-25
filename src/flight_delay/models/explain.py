"""SHAP explainability for flight delay prediction models.

Provides functions to compute SHAP values, visualise feature contributions
at global and instance levels, extract top contributing features, and
generate natural-language explanation text.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from flight_delay.utils.config import REPORTS_DIR

logger = logging.getLogger(__name__)

# Force non-interactive backend for headless environments
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Explainer factory
# ---------------------------------------------------------------------------


def get_shap_explainer(model: Any) -> shap.Explainer:
    """Create an appropriate SHAP explainer for the given model.

    Uses ``TreeExplainer`` for tree-based models (XGBoost, Random Forest)
    and ``LinearExplainer`` for linear models (Logistic Regression).

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn or XGBoost model.

    Returns
    -------
    shap.Explainer
        A SHAP explainer object.
    """
    cls_name = type(model).__name__.lower()

    if "xgb" in cls_name or "randomforest" in cls_name:
        logger.info("Creating TreeExplainer for %s.", type(model).__name__)
        return shap.TreeExplainer(model)
    elif "logistic" in cls_name:
        logger.info("Creating LinearExplainer for %s.", type(model).__name__)
        return shap.LinearExplainer(model, shap.maskers.Independent(
            np.zeros((1, model.n_features_in_)),
            max_samples=100,
        ))
    else:
        logger.warning(
            "Unknown model type '%s' — falling back to generic Explainer.",
            type(model).__name__,
        )
        return shap.Explainer(model)


# ---------------------------------------------------------------------------
# SHAP value computation
# ---------------------------------------------------------------------------


def get_shap_values(
    model: Any,
    X: pd.DataFrame | np.ndarray,
) -> shap.Explanation:
    """Compute SHAP values for a model on a feature matrix.

    Parameters
    ----------
    model : estimator
        Fitted model.
    X : pd.DataFrame or np.ndarray
        Feature matrix to explain.

    Returns
    -------
    shap.Explanation
        SHAP explanation with ``.values``, ``.base_values``, ``.data``,
        and ``.feature_names`` attributes.
    """
    logger.info("Computing SHAP values for %d samples …", len(X))

    explainer = get_shap_explainer(model)
    shap_values = explainer(X)

    # For binary classifiers, TreeExplainer may return 3-D values;
    # keep only the positive-class dimension.
    if (
        isinstance(shap_values.values, np.ndarray)
        and shap_values.values.ndim == 3
    ):
        shap_values = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=(
                shap_values.base_values[:, 1]
                if shap_values.base_values.ndim > 1
                else shap_values.base_values
            ),
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )

    logger.info("SHAP computation complete.")
    return shap_values


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_shap_waterfall(
    shap_values: shap.Explanation,
    idx: int = 0,
) -> plt.Figure:
    """Create a waterfall plot for a single prediction.

    Parameters
    ----------
    shap_values : shap.Explanation
        Pre-computed SHAP values.
    idx : int, optional
        Index of the sample to explain (default: 0).

    Returns
    -------
    matplotlib.figure.Figure
        The waterfall plot figure.
    """
    shap.plots.waterfall(shap_values[idx], show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    fig.tight_layout()
    logger.info("Generated waterfall plot for sample %d.", idx)
    return fig


def plot_shap_summary(shap_values: shap.Explanation) -> plt.Figure:
    """Create a beeswarm summary plot of SHAP values.

    Parameters
    ----------
    shap_values : shap.Explanation
        Pre-computed SHAP values.

    Returns
    -------
    matplotlib.figure.Figure
        The summary plot figure.
    """
    shap.summary_plot(shap_values.values, shap_values.data, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    fig.tight_layout()
    logger.info("Generated SHAP summary plot.")
    return fig


# ---------------------------------------------------------------------------
# Top features
# ---------------------------------------------------------------------------


def get_top_features(
    shap_values: shap.Explanation,
    idx: int = 0,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Get the top contributing features for a single prediction.

    Parameters
    ----------
    shap_values : shap.Explanation
        Pre-computed SHAP values.
    idx : int, optional
        Index of the sample to inspect (default: 0).
    top_n : int, optional
        Number of top features to return (default: 5).

    Returns
    -------
    list[tuple[str, float]]
        List of ``(feature_name, shap_value)`` tuples, sorted by
        absolute SHAP value descending.
    """
    values = shap_values.values[idx]
    feature_names = shap_values.feature_names

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(values))]

    # Sort by absolute value, descending
    abs_vals = np.abs(values)
    top_indices = np.argsort(abs_vals)[::-1][:top_n]

    top_features = [
        (feature_names[i], float(values[i]))
        for i in top_indices
    ]

    logger.info(
        "Top %d features for sample %d: %s",
        top_n,
        idx,
        [(name, f"{val:+.4f}") for name, val in top_features],
    )
    return top_features


# ---------------------------------------------------------------------------
# Natural language explanation
# ---------------------------------------------------------------------------


def generate_explanation_text(
    top_features: list[tuple[str, float]],
) -> str:
    """Generate a natural-language explanation from top SHAP features.

    Parameters
    ----------
    top_features : list[tuple[str, float]]
        List of ``(feature_name, shap_value)`` tuples, as returned by
        :func:`get_top_features`.

    Returns
    -------
    str
        Human-readable explanation string.
    """
    if not top_features:
        return "No feature contributions available for explanation."

    # Determine overall prediction direction
    total_shap = sum(val for _, val in top_features)
    if total_shap > 0:
        intro = "This flight is likely delayed because:"
    else:
        intro = "This flight is likely on time because:"

    reasons: list[str] = []
    for feature_name, shap_val in top_features:
        # Make feature names more readable
        readable_name = feature_name.replace("_", " ").title()

        if shap_val > 0:
            reasons.append(
                f"  • {readable_name} increases delay risk "
                f"(contribution: +{shap_val:.3f})"
            )
        else:
            reasons.append(
                f"  • {readable_name} decreases delay risk "
                f"(contribution: {shap_val:.3f})"
            )

    explanation = "\n".join([intro] + reasons)
    return explanation
