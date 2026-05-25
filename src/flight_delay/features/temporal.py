"""Temporal feature engineering for flight delay prediction.

Derives time-based features from CRSDepTime (scheduled departure in HHMM format)
and FlightDate columns. Includes cyclical encodings, holiday flags, and time-block
categorisation.
"""

from __future__ import annotations

import logging
from typing import List

import holidays
import numpy as np
import pandas as pd

from flight_delay.utils.config import TEMPORAL_FEATURES

logger = logging.getLogger(__name__)

# US federal holidays instance (cached at module level for performance)
_US_HOLIDAYS = holidays.UnitedStates()

# Time-block boundaries (start_hour, end_hour, label)
_TIME_BLOCKS: List[tuple[int, int, str]] = [
    (5, 8, "early_morning"),
    (8, 12, "morning"),
    (12, 17, "afternoon"),
    (17, 21, "evening"),
    # night wraps: 21-24 and 0-5
]


def _parse_hour(crs_dep_time: pd.Series) -> pd.Series:
    """Extract the hour from CRSDepTime (HHMM integer format).

    Parameters
    ----------
    crs_dep_time : pd.Series
        Scheduled departure times in HHMM integer format (e.g. 1430 = 14:30).

    Returns
    -------
    pd.Series
        Hour of day (0–23). Times encoded as 2400 are mapped to 0.
    """
    raw_hour = crs_dep_time.astype(int) // 100
    return raw_hour.clip(upper=23)  # 2400 → 24 → clip to 23; safest mapping


def _assign_time_block(hour: pd.Series) -> pd.Series:
    """Map an hour-of-day series to a categorical time-block label.

    Parameters
    ----------
    hour : pd.Series
        Integer hour values 0–23.

    Returns
    -------
    pd.Series
        Categorical time-block labels.
    """
    conditions = [
        (hour >= 5) & (hour < 8),
        (hour >= 8) & (hour < 12),
        (hour >= 12) & (hour < 17),
        (hour >= 17) & (hour < 21),
    ]
    choices = ["early_morning", "morning", "afternoon", "evening"]
    return pd.Series(
        np.select(conditions, choices, default="night"),
        index=hour.index,
        dtype="category",
    )


def _is_near_holiday(dates: pd.Series) -> pd.Series:
    """Check if a date is a US federal holiday OR the day before/after one.

    Parameters
    ----------
    dates : pd.Series
        Datetime-like dates.

    Returns
    -------
    pd.Series[bool]
        True when the date is a holiday, day-before, or day-after.
    """
    dt = pd.to_datetime(dates)
    day_before = dt - pd.Timedelta(days=1)
    day_after = dt + pd.Timedelta(days=1)

    is_hol = dt.map(lambda d: d in _US_HOLIDAYS).astype(bool)
    is_before = day_before.map(lambda d: d in _US_HOLIDAYS).astype(bool)
    is_after = day_after.map(lambda d: d in _US_HOLIDAYS).astype(bool)

    return is_hol | is_before | is_after


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features to a flight-delay DataFrame.

    Expects the input to contain at least:
    - ``CRSDepTime``: scheduled departure in HHMM integer format
    - ``FlightDate``: date string or datetime

    Parameters
    ----------
    df : pd.DataFrame
        Raw flight data.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the following columns added:
        ``hour_of_day``, ``day_of_week``, ``month``, ``is_weekend``,
        ``is_holiday``, ``hour_sin``, ``hour_cos``, ``month_sin``,
        ``month_cos``, ``time_block``.
    """
    df = df.copy()
    logger.info("Adding temporal features …")

    # --- Ensure FlightDate is datetime ---
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    # --- Basic extractions ---
    df["hour_of_day"] = _parse_hour(df["CRSDepTime"])
    df["day_of_week"] = df["FlightDate"].dt.dayofweek  # 0=Mon … 6=Sun
    df["month"] = df["FlightDate"].dt.month  # 1–12

    # --- Boolean flags ---
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = _is_near_holiday(df["FlightDate"]).astype(int)

    # --- Cyclical encodings ---
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Time block ---
    df["time_block"] = _assign_time_block(df["hour_of_day"])

    logger.info(
        "Temporal features added: %s + time_block",
        ", ".join(TEMPORAL_FEATURES),
    )
    return df
