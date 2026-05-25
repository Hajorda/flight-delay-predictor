"""Feature engineering pipeline for flight delay prediction.

Provides standalone functions for each feature group (temporal, weather,
airport, route) and a ``build_features`` function that chains them all.

Also exposes:
- :func:`get_feature_columns` — canonical list of feature column names
- :func:`prepare_for_prediction` — transform a user input dict into a
  model-ready single-row DataFrame
"""

from __future__ import annotations

import logging
from typing import Any

import holidays
import numpy as np
import pandas as pd

from flight_delay.utils.config import (
    AIRPORT_FEATURES,
    CATEGORICAL_FEATURES,
    DELAY_THRESHOLD_MINUTES,
    HUB_AIRPORTS,
    RANDOM_STATE,
    ROUTE_FEATURES,
    TARGET_COL,
    TEMPORAL_FEATURES,
    WEATHER_FEATURES,
)

logger = logging.getLogger(__name__)

# US federal holidays (module-level cache)
_US_HOLIDAYS = holidays.UnitedStates()


# ──────────────────────────────────────────────
# Temporal features
# ──────────────────────────────────────────────


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from ``fl_date`` and ``crs_dep_time``.

    Adds: ``hour_of_day``, ``day_of_week``, ``month``, ``is_weekend``,
    ``is_holiday``, ``hour_sin``, ``hour_cos``, ``month_sin``, ``month_cos``.

    Parameters
    ----------
    df : pd.DataFrame
        Flight data with ``fl_date`` and ``crs_dep_time`` columns.

    Returns
    -------
    pd.DataFrame
        Copy with temporal feature columns added.
    """
    df = df.copy()
    logger.info("Adding temporal features …")

    # Ensure datetime
    df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce")

    # Hour from HHMM integer format
    df["hour_of_day"] = (df["crs_dep_time"].astype(int) // 100).clip(upper=23)

    # Calendar features
    df["day_of_week"] = df["fl_date"].dt.dayofweek  # 0=Mon … 6=Sun
    df["month"] = df["fl_date"].dt.month

    # Boolean flags
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = (
        df["fl_date"].map(lambda d: d in _US_HOLIDAYS).astype(int)
    )

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    logger.info("Temporal features added: %s", ", ".join(TEMPORAL_FEATURES))
    return df


# ──────────────────────────────────────────────
# Weather features
# ──────────────────────────────────────────────


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute IFR condition flags from weather columns.

    Adds ``ifr_conditions_origin`` and ``ifr_conditions_dest`` — set to 1
    when visibility < 3 statute miles or ceiling < 1 000 ft.

    If raw weather columns are missing, defaults are filled and IFR is set
    to 0.

    Parameters
    ----------
    df : pd.DataFrame
        Flight data with optional weather columns.

    Returns
    -------
    pd.DataFrame
        Copy with IFR condition columns added.
    """
    df = df.copy()
    logger.info("Adding weather features …")

    # Ensure weather columns exist (fill defaults if absent)
    _weather_defaults = {
        "origin_visibility": 10.0,
        "origin_ceiling": 25_000,
        "dest_visibility": 10.0,
        "dest_ceiling": 25_000,
    }
    for col, default in _weather_defaults.items():
        if col not in df.columns:
            df[col] = default
            logger.warning("'%s' not found — filled with default %.1f.", col, default)

    # IFR conditions: visibility < 3 or ceiling < 1000
    df["ifr_conditions_origin"] = (
        (df["origin_visibility"] < 3) | (df["origin_ceiling"] < 1000)
    ).astype(int)

    df["ifr_conditions_dest"] = (
        (df["dest_visibility"] < 3) | (df["dest_ceiling"] < 1000)
    ).astype(int)

    logger.info("Weather features added: ifr_conditions_origin, ifr_conditions_dest.")
    return df


# ──────────────────────────────────────────────
# Airport features
# ──────────────────────────────────────────────


def add_airport_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add airport-level features: hub flags, congestion, rolling delay rates.

    Adds: ``is_origin_hub``, ``is_dest_hub``, ``origin_flights_per_hour``,
    ``dest_flights_per_hour``, ``origin_delay_rate_rolling``,
    ``dest_delay_rate_rolling``.

    Parameters
    ----------
    df : pd.DataFrame
        Flight data with ``origin``, ``dest``, ``fl_date``, ``hour_of_day``.
        If ``is_delayed`` exists, rolling delay rates are computed;
        otherwise they default to 0.

    Returns
    -------
    pd.DataFrame
        Copy with airport feature columns added.
    """
    df = df.copy()
    logger.info("Adding airport features …")

    # Ensure datetime
    df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce")

    # Hub flags
    df["is_origin_hub"] = df["origin"].isin(HUB_AIRPORTS).astype(int)
    df["is_dest_hub"] = df["dest"].isin(HUB_AIRPORTS).astype(int)

    # Congestion: flights per hour at each airport on each date
    if "hour_of_day" in df.columns:
        for prefix, col in [("origin", "origin_flights_per_hour"),
                            ("dest", "dest_flights_per_hour")]:
            counts = (
                df.groupby([prefix, "fl_date", "hour_of_day"])
                .size()
                .rename(col)
                .reset_index()
            )
            merged = df[[prefix, "fl_date", "hour_of_day"]].merge(
                counts,
                on=[prefix, "fl_date", "hour_of_day"],
                how="left",
            )
            df[col] = merged[col].fillna(0).astype(int).values
    else:
        df["origin_flights_per_hour"] = 0
        df["dest_flights_per_hour"] = 0

    # Rolling delay rates (leakage-safe: shifted by 1 day)
    if TARGET_COL in df.columns:
        for prefix, col in [("origin", "origin_delay_rate_rolling"),
                            ("dest", "dest_delay_rate_rolling")]:
            daily = (
                df.groupby([prefix, "fl_date"])[TARGET_COL]
                .mean()
                .rename("daily_rate")
                .reset_index()
                .sort_values("fl_date")
            )
            daily[col] = (
                daily.groupby(prefix)["daily_rate"]
                .transform(
                    lambda s: s.shift(1).rolling(window=7, min_periods=1).mean()
                )
            )
            merged = df[[prefix, "fl_date"]].merge(
                daily[[prefix, "fl_date", col]],
                on=[prefix, "fl_date"],
                how="left",
            )
            df[col] = merged[col].fillna(0.0).values
    else:
        df["origin_delay_rate_rolling"] = 0.0
        df["dest_delay_rate_rolling"] = 0.0

    logger.info("Airport features added: %s", ", ".join(AIRPORT_FEATURES))
    return df


# ──────────────────────────────────────────────
# Route features
# ──────────────────────────────────────────────


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add route-level features: average delay and carrier delay rate.

    Computes target-encoded ``route_avg_delay`` and ``carrier_delay_rate``
    using Bayesian smoothing (smoothing parameter = 100).

    If ``is_delayed`` is not present, both columns default to 0.

    Parameters
    ----------
    df : pd.DataFrame
        Flight data with ``origin``, ``dest``, ``carrier``, and optionally
        ``is_delayed``.

    Returns
    -------
    pd.DataFrame
        Copy with route feature columns added.
    """
    df = df.copy()
    logger.info("Adding route features …")

    smoothing = 100

    if TARGET_COL in df.columns:
        global_mean = df[TARGET_COL].mean()

        # Route average delay (target-encoded)
        route_key = df["origin"].astype(str) + "-" + df["dest"].astype(str)
        route_agg = df.groupby(route_key)[TARGET_COL].agg(["mean", "count"])
        route_encoded = (
            (route_agg["count"] * route_agg["mean"] + smoothing * global_mean)
            / (route_agg["count"] + smoothing)
        )
        df["route_avg_delay"] = route_key.map(route_encoded).fillna(global_mean)

        # Carrier delay rate (target-encoded)
        carrier_agg = df.groupby("carrier")[TARGET_COL].agg(["mean", "count"])
        carrier_encoded = (
            (carrier_agg["count"] * carrier_agg["mean"] + smoothing * global_mean)
            / (carrier_agg["count"] + smoothing)
        )
        df["carrier_delay_rate"] = (
            df["carrier"].astype(str).map(carrier_encoded).fillna(global_mean)
        )
    else:
        df["route_avg_delay"] = 0.0
        df["carrier_delay_rate"] = 0.0

    logger.info("Route features added: route_avg_delay, carrier_delay_rate.")
    return df


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Chain all feature engineering steps.

    Applies temporal → weather → airport → route feature functions
    sequentially.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed flight data (output of ``preprocess_pipeline``).

    Returns
    -------
    pd.DataFrame
        Fully featurised DataFrame.
    """
    logger.info("Building features for %d rows …", len(df))

    df = add_temporal_features(df)
    df = add_weather_features(df)
    df = add_airport_features(df)
    df = add_route_features(df)

    logger.info("Feature engineering complete — %d columns.", df.shape[1])
    return df


# ──────────────────────────────────────────────
# Feature column list
# ──────────────────────────────────────────────


def get_feature_columns() -> list[str]:
    """Return the canonical list of feature column names.

    Combines temporal, weather, airport, route, and categorical feature
    lists defined in ``config.py``.

    Returns
    -------
    list[str]
        Ordered list of all non-target feature column names.
    """
    return (
        TEMPORAL_FEATURES
        + WEATHER_FEATURES
        + AIRPORT_FEATURES
        + ROUTE_FEATURES
        + CATEGORICAL_FEATURES
    )


# ──────────────────────────────────────────────
# Prediction preparation
# ──────────────────────────────────────────────


def prepare_for_prediction(input_dict: dict[str, Any]) -> pd.DataFrame:
    """Transform a user input dictionary into a model-ready DataFrame.

    Takes a dictionary of user-provided values (e.g. from a Streamlit form)
    and produces a single-row DataFrame with all expected feature columns,
    filling defaults for any missing ones.

    Parameters
    ----------
    input_dict : dict[str, Any]
        User inputs. Expected keys may include ``fl_date``, ``carrier``,
        ``origin``, ``dest``, ``crs_dep_time``, ``distance``, and
        weather columns.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame with all feature columns suitable for
        ``model.predict()``.
    """
    logger.info("Preparing single-row prediction input …")

    # Build a single-row DataFrame
    df = pd.DataFrame([input_dict])

    # Ensure required columns exist with safe defaults
    defaults: dict[str, Any] = {
        "fl_date": pd.Timestamp.now(),
        "carrier": "AA",
        "origin": "ATL",
        "dest": "LAX",
        "crs_dep_time": 1200,
        "distance": 1000,
        "arr_delay": 0,
        "cancelled": 0,
        # Weather defaults
        "origin_wind_speed": 8.0,
        "origin_visibility": 10.0,
        "origin_ceiling": 25_000,
        "origin_temp": 60.0,
        "origin_precip": 0.0,
        "origin_has_thunderstorm": 0,
        "origin_has_snow": 0,
        "origin_has_fog": 0,
        "dest_wind_speed": 8.0,
        "dest_visibility": 10.0,
        "dest_ceiling": 25_000,
        "dest_temp": 60.0,
        "dest_precip": 0.0,
        "dest_has_thunderstorm": 0,
        "dest_has_snow": 0,
        "dest_has_fog": 0,
    }

    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # Run feature engineering
    df = build_features(df)

    # Select only feature columns, filling any still-missing with 0
    feature_cols = get_feature_columns()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    result = df[feature_cols].copy()
    logger.info("Prediction input ready: %d features.", len(feature_cols))
    return result


# ──────────────────────────────────────────────
# Feature Pipeline Class & Helpers
# ──────────────────────────────────────────────

from flight_delay.features.route import RouteEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

def _map_casing(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names to what the feature modules expect."""
    df = df.copy()
    mapping = {
        "fl_date": "FlightDate",
        "crs_dep_time": "CRSDepTime",
        "origin": "Origin",
        "dest": "Dest",
        "distance": "Distance",
        "cancelled": "Cancelled",
        "arr_delay": "arr_delay",
    }
    for old_col, new_col in mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    return df


class FlightDelayFeaturePipeline:
    """Orchestrates the entire feature engineering process from raw to model-ready features."""

    def __init__(self, smoothing: int = 100) -> None:
        self.route_encoder = RouteEncoder(smoothing=smoothing)
        self._is_fitted = bool = False

    def fit(self, df: pd.DataFrame) -> "FlightDelayFeaturePipeline":
        """Fit any stateful transformers on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training flight data.

        Returns
        -------
        FlightDelayFeaturePipeline
            Self.
        """
        df_mapped = _map_casing(df)
        self.route_encoder.fit(df_mapped)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations.

        Parameters
        ----------
        df : pd.DataFrame
            Flight data to transform.

        Returns
        -------
        pd.DataFrame
            Fully transformed DataFrame.
        """
        df_mapped = _map_casing(df)
        
        # Build features using local pipeline.py functions
        df_feat = build_features(df_mapped)
        
        # Apply fitted RouteEncoder
        if self._is_fitted:
            # We already have route_avg_delay and carrier_delay_rate in df_feat, 
            # but we overwrite them with the fitted RouteEncoder mapping to prevent leak.
            df_feat = self.route_encoder.transform(df_mapped)
            # Combine other features that were built
            other_feats = build_features(df_mapped)
            for col in other_feats.columns:
                if col not in df_feat.columns:
                    df_feat[col] = other_feats[col]
        else:
            logger.warning("Transforming without fitting RouteEncoder first!")

        # Ensure all feature columns exist in df_feat, copying from df_mapped or filling 0
        feature_cols = get_feature_columns()
        for col in feature_cols:
            if col not in df_feat.columns:
                if col in df_mapped.columns:
                    df_feat[col] = df_mapped[col]
                else:
                    df_feat[col] = 0

        # Keep only the target col if it exists and feature columns
        cols_to_keep = [col for col in feature_cols]
        if TARGET_COL in df.columns:
            df_feat[TARGET_COL] = df[TARGET_COL]
            cols_to_keep = [TARGET_COL] + cols_to_keep
            
        return df_feat[cols_to_keep]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Parameters
        ----------
        df : pd.DataFrame
            Flight data.

        Returns
        -------
        pd.DataFrame
            Fully transformed DataFrame.
        """
        return self.fit(df).transform(df)


def build_sklearn_pipeline() -> ColumnTransformer:
    """Build an sklearn ColumnTransformer for encoding categorical features.

    Categorical columns ('carrier', 'origin', 'dest') are encoded using OrdinalEncoder.
    Numerical, boolean, and other columns are passed through.
    """
    feature_cols = get_feature_columns()
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    passthrough_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                categorical_cols,
            ),
            ("num", "passthrough", passthrough_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split flight data temporally into train and test sets.

    Uses config.TRAIN_MONTHS (Jan-Oct) and config.TEST_MONTHS (Nov-Dec).

    Parameters
    ----------
    df : pd.DataFrame
        Flight data.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames.
    """
    df = df.copy()
    if "FlightDate" in df.columns:
        dates = pd.to_datetime(df["FlightDate"])
    elif "fl_date" in df.columns:
        dates = pd.to_datetime(df["fl_date"])
    else:
        raise KeyError("Cannot find date column ('FlightDate' or 'fl_date') for splitting.")

    months = dates.dt.month

    from flight_delay.utils.config import TEST_MONTHS, TRAIN_MONTHS

    train_mask = months.isin(TRAIN_MONTHS)
    test_mask = months.isin(TEST_MONTHS)

    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    logger.info("Split dataset: Train=%d rows, Test=%d rows", len(train_df), len(test_df))
    return train_df, test_df


def get_X_y(
    df: pd.DataFrame, feature_cols: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Featurised DataFrame.
    feature_cols : list[str] or None, optional
        List of feature columns to keep. If None, uses get_feature_columns().

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target series y.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    # Keep only feature columns that exist in df
    existing_features = [c for c in feature_cols if c in df.columns]

    X = df[existing_features].copy()
    y = (
        df[TARGET_COL].copy()
        if TARGET_COL in df.columns
        else pd.Series(0, index=df.index)
    )

    return X, y

