"""Weather-derived feature engineering for flight delay prediction.

Computes IFR conditions, wind-severity categories, and severe-weather
indicators from raw weather columns.  When raw weather data is absent the
module fills in safe defaults and emits a warning.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from flight_delay.utils.config import WEATHER_FEATURES

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Raw weather column expectations
# ──────────────────────────────────────────────
_RAW_WEATHER_COLS: Dict[str, List[str]] = {
    "origin": [
        "origin_wind_speed",
        "origin_visibility",
        "origin_ceiling",
        "origin_temp",
        "origin_precip",
        "origin_has_thunderstorm",
        "origin_has_snow",
        "origin_has_fog",
    ],
    "dest": [
        "dest_wind_speed",
        "dest_visibility",
        "dest_ceiling",
        "dest_temp",
        "dest_precip",
        "dest_has_thunderstorm",
        "dest_has_snow",
        "dest_has_fog",
    ],
}

# Default values when raw data is missing
_DEFAULTS: Dict[str, float | int | bool] = {
    "wind_speed": 8.0,        # calm default (knots)
    "visibility": 10.0,       # clear visibility (statute miles)
    "ceiling": 25_000,        # clear sky (feet)
    "temp": 60.0,             # ~15°C
    "precip": 0.0,
    "has_thunderstorm": 0,
    "has_snow": 0,
    "has_fog": 0,
}

# IFR thresholds (FAA)
_IFR_CEILING_FT: int = 1_000
_IFR_VISIBILITY_MI: float = 3.0

# Wind-speed thresholds (knots)
_WIND_CALM_KT: int = 10
_WIND_STRONG_KT: int = 25
_WIND_SEVERE_KT: int = 35


def _ensure_raw_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Fill in missing raw weather columns with defaults.

    Parameters
    ----------
    df : pd.DataFrame
        Working copy of the flight DataFrame.
    prefix : str
        ``'origin'`` or ``'dest'``.

    Returns
    -------
    pd.DataFrame
        *df* with any missing weather columns added.
    """
    expected = _RAW_WEATHER_COLS[prefix]
    missing = [c for c in expected if c not in df.columns]

    if missing:
        logger.warning(
            "Missing weather columns for %s: %s — filling with defaults.",
            prefix,
            ", ".join(missing),
        )
        for col in missing:
            # Extract the suffix after the prefix + '_'
            suffix = col.replace(f"{prefix}_", "", 1)
            df[col] = _DEFAULTS.get(suffix, 0)

    return df


def _wind_category(wind_speed: pd.Series) -> pd.Series:
    """Classify wind speed into calm / moderate / strong.

    Parameters
    ----------
    wind_speed : pd.Series
        Wind speed in knots.

    Returns
    -------
    pd.Series
        Categorical wind labels.
    """
    conditions = [
        wind_speed < _WIND_CALM_KT,
        (wind_speed >= _WIND_CALM_KT) & (wind_speed <= _WIND_STRONG_KT),
    ]
    choices = ["calm", "moderate"]
    return pd.Series(
        np.select(conditions, choices, default="strong"),
        index=wind_speed.index,
        dtype="category",
    )


def _ifr_conditions(
    ceiling: pd.Series,
    visibility: pd.Series,
) -> pd.Series:
    """Determine Instrument Flight Rules (IFR) conditions.

    IFR applies when the ceiling is below 1 000 ft **or** visibility is
    below 3 statute miles.

    Parameters
    ----------
    ceiling : pd.Series
        Cloud ceiling height in feet.
    visibility : pd.Series
        Visibility in statute miles.

    Returns
    -------
    pd.Series[int]
        1 when IFR conditions apply, 0 otherwise.
    """
    return ((ceiling < _IFR_CEILING_FT) | (visibility < _IFR_VISIBILITY_MI)).astype(int)


def _severe_weather(
    has_thunderstorm: pd.Series,
    has_snow: pd.Series,
    wind_speed: pd.Series,
) -> pd.Series:
    """Flag severe weather: thunderstorm OR snow OR wind > 35 kt.

    Parameters
    ----------
    has_thunderstorm : pd.Series
        Boolean/int indicator.
    has_snow : pd.Series
        Boolean/int indicator.
    wind_speed : pd.Series
        Wind speed in knots.

    Returns
    -------
    pd.Series[int]
        1 when severe weather is present.
    """
    return (
        has_thunderstorm.astype(bool)
        | has_snow.astype(bool)
        | (wind_speed > _WIND_SEVERE_KT)
    ).astype(int)


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-derived features to a flight-delay DataFrame.

    When the expected raw weather columns are present the function computes
    derived indicators; otherwise it fills defaults and logs a warning.

    Parameters
    ----------
    df : pd.DataFrame
        Flight data (raw weather columns optional).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the following columns added or updated:
        ``ifr_conditions_origin``, ``ifr_conditions_dest``,
        ``origin_wind_category``, ``dest_wind_category``,
        ``severe_weather_origin``, ``severe_weather_dest``.
    """
    df = df.copy()
    logger.info("Adding weather features …")

    # Ensure all raw columns are present (fill defaults if not)
    for prefix in ("origin", "dest"):
        df = _ensure_raw_columns(df, prefix)

    # --- IFR conditions ---
    df["ifr_conditions_origin"] = _ifr_conditions(
        df["origin_ceiling"], df["origin_visibility"]
    )
    df["ifr_conditions_dest"] = _ifr_conditions(
        df["dest_ceiling"], df["dest_visibility"]
    )

    # --- Wind categories ---
    df["origin_wind_category"] = _wind_category(df["origin_wind_speed"])
    df["dest_wind_category"] = _wind_category(df["dest_wind_speed"])

    # --- Severe weather ---
    df["severe_weather_origin"] = _severe_weather(
        df["origin_has_thunderstorm"],
        df["origin_has_snow"],
        df["origin_wind_speed"],
    )
    df["severe_weather_dest"] = _severe_weather(
        df["dest_has_thunderstorm"],
        df["dest_has_snow"],
        df["dest_wind_speed"],
    )

    logger.info(
        "Weather features added: ifr_conditions_*, *_wind_category, severe_weather_*"
    )
    return df
