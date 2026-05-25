"""Airport-level feature engineering for flight delay prediction.

Computes congestion proxies (flights-per-hour), rolling delay rates with
leakage-safe shifting, and hub-airport flags.
"""

from __future__ import annotations

import logging

import pandas as pd

from flight_delay.utils.config import (
    AIRPORT_FEATURES,
    DELAY_THRESHOLD_MINUTES,
    HUB_AIRPORTS,
    TARGET_COL,
)

logger = logging.getLogger(__name__)

# Rolling-window size (days) for delay-rate averages
_ROLLING_WINDOW_DAYS: int = 7


def _flights_per_hour(
    df: pd.DataFrame,
    airport_col: str,
    new_col: str,
) -> pd.Series:
    """Count scheduled flights at an airport in a given hour on a given date.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``FlightDate``, ``hour_of_day``, and *airport_col*.
    airport_col : str
        Column identifying the airport (``Origin`` or ``Dest``).
    new_col : str
        Name for the resulting count column (used only for logging).

    Returns
    -------
    pd.Series
        Per-row counts aligned to *df*'s index.
    """
    counts = (
        df.groupby([airport_col, "FlightDate", "hour_of_day"])
        .size()
        .rename(new_col)
        .reset_index()
    )
    merged = df[[airport_col, "FlightDate", "hour_of_day"]].merge(
        counts,
        on=[airport_col, "FlightDate", "hour_of_day"],
        how="left",
    )
    return merged[new_col].fillna(0).astype(int)


def _rolling_delay_rate(
    df: pd.DataFrame,
    airport_col: str,
    new_col: str,
    window: int = _ROLLING_WINDOW_DAYS,
) -> pd.Series:
    """Compute a leakage-safe rolling average delay rate per airport.

    For each airport-date the delay rate is computed, then a rolling mean
    over the preceding *window* days is applied.  ``shift(1)`` ensures
    the current day's information is never used.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``FlightDate``, *airport_col*, and ``TARGET_COL``.
    airport_col : str
        Airport column (``Origin`` or ``Dest``).
    new_col : str
        Name for the result column.
    window : int, optional
        Number of trailing days to average over (default 7).

    Returns
    -------
    pd.Series
        Rolling delay rate aligned to *df*'s index.
    """
    # Daily delay rate per airport
    daily = (
        df.groupby([airport_col, "FlightDate"])[TARGET_COL]
        .mean()
        .rename("daily_rate")
        .reset_index()
        .sort_values("FlightDate")
    )

    # Rolling mean with shift(1) to prevent leakage
    daily[new_col] = (
        daily.groupby(airport_col)["daily_rate"]
        .transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
    )

    # Merge back to flight-level rows
    merged = df[[airport_col, "FlightDate"]].merge(
        daily[[airport_col, "FlightDate", new_col]],
        on=[airport_col, "FlightDate"],
        how="left",
    )
    return merged[new_col].fillna(0.0)


def add_airport_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add airport-level features to a flight-delay DataFrame.

    Expects the DataFrame to already contain:
    - ``Origin``, ``Dest``: IATA airport codes
    - ``FlightDate``: datetime
    - ``hour_of_day``: integer (from temporal feature engineering)
    - ``is_delayed`` (TARGET_COL): binary delay indicator

    Parameters
    ----------
    df : pd.DataFrame
        Flight data with temporal features already added.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the following columns added:
        ``origin_flights_per_hour``, ``dest_flights_per_hour``,
        ``origin_delay_rate_rolling``, ``dest_delay_rate_rolling``,
        ``is_origin_hub``, ``is_dest_hub``.
    """
    df = df.copy()
    logger.info("Adding airport features …")

    # Ensure FlightDate is datetime for grouping
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    # --- Congestion: flights per hour ---
    df["origin_flights_per_hour"] = _flights_per_hour(
        df, "Origin", "origin_flights_per_hour"
    )
    df["dest_flights_per_hour"] = _flights_per_hour(
        df, "Dest", "dest_flights_per_hour"
    )

    # --- Rolling delay rates (leakage-safe) ---
    df["origin_delay_rate_rolling"] = _rolling_delay_rate(
        df, "Origin", "origin_delay_rate_rolling"
    )
    df["dest_delay_rate_rolling"] = _rolling_delay_rate(
        df, "Dest", "dest_delay_rate_rolling"
    )

    # --- Hub flags ---
    df["is_origin_hub"] = df["Origin"].isin(HUB_AIRPORTS).astype(int)
    df["is_dest_hub"] = df["Dest"].isin(HUB_AIRPORTS).astype(int)

    logger.info("Airport features added: %s", ", ".join(AIRPORT_FEATURES))
    return df
