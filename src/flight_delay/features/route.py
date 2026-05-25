"""Route-level feature engineering for flight delay prediction.

Provides target-encoded route and carrier delay rates with Bayesian
smoothing, a distance categoriser, and a reusable ``RouteEncoder`` that is
fitted on training data and applied to unseen data without leakage.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from flight_delay.utils.config import ROUTE_FEATURES, TARGET_COL

logger = logging.getLogger(__name__)

# Smoothing parameter for target encoding — acts as a regularisation
# strength (pseudo-count added from the global mean).
_DEFAULT_SMOOTHING: int = 100

# Distance category thresholds (statute miles)
_SHORT_MI: int = 500
_LONG_MI: int = 1_500


def _distance_category(distance: pd.Series) -> pd.Series:
    """Classify flight distance into short / medium / long.

    Parameters
    ----------
    distance : pd.Series
        Distance in statute miles.

    Returns
    -------
    pd.Series
        Categorical distance labels.
    """
    conditions = [
        distance < _SHORT_MI,
        (distance >= _SHORT_MI) & (distance <= _LONG_MI),
    ]
    choices = ["short", "medium"]
    return pd.Series(
        np.select(conditions, choices, default="long"),
        index=distance.index,
        dtype="category",
    )


class RouteEncoder:
    """Target-encodes route and carrier features with Bayesian smoothing.

    The encoding formula for a given category *c* is:

        encoded(c) = (n_c * mean_c + m * global_mean) / (n_c + m)

    where *n_c* is the number of training samples in category *c*,
    *mean_c* is the category-level target mean, *m* is the smoothing
    parameter, and *global_mean* is the overall target mean.

    Parameters
    ----------
    smoothing : int, optional
        Smoothing parameter *m* (default ``100``).
    """

    def __init__(self, smoothing: int = _DEFAULT_SMOOTHING) -> None:
        self.smoothing = smoothing
        self._global_mean: float = 0.0
        self._route_stats: Dict[str, float] = {}
        self._carrier_stats: Dict[str, float] = {}
        self._is_fitted: bool = False

    # ── helpers ──────────────────────────────────

    @staticmethod
    def _encode_map(
        groups: pd.core.groupby.SeriesGroupBy,
        global_mean: float,
        smoothing: int,
    ) -> Dict[str, float]:
        """Build a dict mapping category → smoothed target-encoded value."""
        agg = groups.agg(["mean", "count"])
        encoded = (
            (agg["count"] * agg["mean"] + smoothing * global_mean)
            / (agg["count"] + smoothing)
        )
        return encoded.to_dict()

    # ── public API ───────────────────────────────

    def fit(self, df: pd.DataFrame) -> "RouteEncoder":
        """Learn target-encoded statistics from training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with ``Origin``, ``Dest``, ``Reporting_Airline``
            (or ``carrier``), and ``TARGET_COL``.

        Returns
        -------
        RouteEncoder
            Self, for method chaining.
        """
        logger.info("Fitting RouteEncoder on %d training rows …", len(df))

        # Build route key
        route = df["Origin"].str.cat(df["Dest"], sep="-")

        self._global_mean = df[TARGET_COL].mean()

        self._route_stats = self._encode_map(
            df.groupby(route)[TARGET_COL],
            self._global_mean,
            self.smoothing,
        )

        carrier_col = self._resolve_carrier_col(df)
        self._carrier_stats = self._encode_map(
            df.groupby(df[carrier_col])[TARGET_COL],
            self._global_mean,
            self.smoothing,
        )

        self._is_fitted = True
        logger.info(
            "RouteEncoder fitted: %d routes, %d carriers, global_mean=%.4f",
            len(self._route_stats),
            len(self._carrier_stats),
            self._global_mean,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned target encodings to a DataFrame.

        Unseen categories fall back to the global mean.

        Parameters
        ----------
        df : pd.DataFrame
            Flight data with ``Origin``, ``Dest``, and a carrier column.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with ``route``, ``route_avg_delay``,
            ``carrier_delay_rate``, and ``distance_category`` added.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "RouteEncoder must be fitted before transform."
            )

        df = df.copy()

        # Route string
        df["route"] = df["Origin"].str.cat(df["Dest"], sep="-")

        # Target-encoded features (unseen → global mean)
        df["route_avg_delay"] = (
            df["route"]
            .map(self._route_stats)
            .fillna(self._global_mean)
        )

        carrier_col = self._resolve_carrier_col(df)
        df["carrier_delay_rate"] = (
            df[carrier_col]
            .map(self._carrier_stats)
            .fillna(self._global_mean)
        )

        # Distance category
        if "Distance" in df.columns:
            df["distance_category"] = _distance_category(df["Distance"])
        elif "distance" in df.columns:
            df["distance_category"] = _distance_category(df["distance"])
        else:
            logger.warning(
                "No distance column found — skipping distance_category."
            )

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit on *df* then transform it.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        pd.DataFrame
            Transformed training data.
        """
        return self.fit(df).transform(df)

    # ── internals ────────────────────────────────

    @staticmethod
    def _resolve_carrier_col(df: pd.DataFrame) -> str:
        """Return the carrier column name present in *df*.

        Looks for ``Reporting_Airline``, then ``carrier``, then
        ``Marketing_Airline_Network``.
        """
        for candidate in (
            "Reporting_Airline",
            "carrier",
            "Marketing_Airline_Network",
            "IATA_CODE_Reporting_Airline",
        ):
            if candidate in df.columns:
                return candidate
        raise KeyError(
            "Cannot find a carrier column in the DataFrame. "
            "Expected one of: Reporting_Airline, carrier, "
            "Marketing_Airline_Network, IATA_CODE_Reporting_Airline."
        )


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add route-level features using a **fresh** RouteEncoder.

    .. warning::
        This convenience function fits and transforms on the *same*
        DataFrame — suitable only for quick exploration.  For proper
        train/test separation use :class:`RouteEncoder` directly.

    Parameters
    ----------
    df : pd.DataFrame
        Flight data with ``Origin``, ``Dest``, a carrier column,
        ``Distance``, and ``TARGET_COL``.

    Returns
    -------
    pd.DataFrame
        Copy with route features added.
    """
    encoder = RouteEncoder()
    return encoder.fit_transform(df)
