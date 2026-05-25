"""Data preprocessing pipeline for flight delay prediction.

Provides functions to load raw data, cast column types, filter cancelled
flights, create the binary delay target, handle missing values, and chain
all steps into a single preprocessing pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from flight_delay.utils.config import (
    DELAY_THRESHOLD_MINUTES,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load raw flight data from CSV or Parquet.

    Parameters
    ----------
    filepath : Path
        Path to a ``.csv`` or ``.parquet`` file.

    Returns
    -------
    pd.DataFrame
        Raw flight data.

    Raises
    ------
    ValueError
        If the file extension is not ``.csv`` or ``.parquet``.
    """
    filepath = Path(filepath)
    logger.info("Loading raw data from %s …", filepath)

    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
    elif filepath.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(
            f"Unsupported file format '{filepath.suffix}'. "
            "Expected '.csv' or '.parquet'."
        )

    logger.info("Loaded %d rows × %d columns.", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Type casting
# ---------------------------------------------------------------------------


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to appropriate types.

    - ``carrier``, ``origin``, ``dest`` → string / category
    - Numeric columns → float
    - ``fl_date`` → datetime

    Parameters
    ----------
    df : pd.DataFrame
        Raw flight data.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected dtypes.
    """
    df = df.copy()
    logger.info("Casting column types …")

    # Categorical / string columns
    for col in ("carrier", "origin", "dest"):
        if col in df.columns:
            df[col] = df[col].astype(str).astype("category")

    # Parse dates
    if "fl_date" in df.columns:
        df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce")

    # Numeric columns to float
    numeric_candidates = [
        "arr_delay",
        "distance",
        "crs_dep_time",
        "origin_wind_speed",
        "origin_visibility",
        "origin_ceiling",
        "origin_temp",
        "origin_precip",
        "dest_wind_speed",
        "dest_visibility",
        "dest_ceiling",
        "dest_temp",
        "dest_precip",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Type casting complete.")
    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_cancelled(df: pd.DataFrame) -> pd.DataFrame:
    """Remove cancelled flights from the dataset.

    Drops rows where the ``cancelled`` column equals ``1`` or ``True``.

    Parameters
    ----------
    df : pd.DataFrame
        Flight data with a ``cancelled`` column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame without cancelled flights.
    """
    if "cancelled" not in df.columns:
        logger.warning("Column 'cancelled' not found — skipping filter.")
        return df

    n_before = len(df)
    mask = ~df["cancelled"].astype(bool)
    df = df.loc[mask].reset_index(drop=True)

    n_removed = n_before - len(df)
    logger.info(
        "Filtered cancelled flights: removed %d / %d rows (%.1f%%).",
        n_removed,
        n_before,
        100 * n_removed / max(n_before, 1),
    )
    return df


# ---------------------------------------------------------------------------
# Target creation
# ---------------------------------------------------------------------------


def create_delay_target(
    df: pd.DataFrame,
    threshold: int = DELAY_THRESHOLD_MINUTES,
) -> pd.DataFrame:
    """Create binary delay target from arrival delay.

    A flight is considered delayed if ``arr_delay >= threshold``.

    Parameters
    ----------
    df : pd.DataFrame
        Flight data with an ``arr_delay`` column.
    threshold : int, optional
        Delay threshold in minutes (default: ``DELAY_THRESHOLD_MINUTES``).

    Returns
    -------
    pd.DataFrame
        DataFrame with the ``is_delayed`` column appended.
    """
    df = df.copy()

    if "arr_delay" not in df.columns:
        raise KeyError(
            "Column 'arr_delay' is required to create the delay target."
        )

    df[TARGET_COL] = (df["arr_delay"] >= threshold).astype(int)

    n_delayed = df[TARGET_COL].sum()
    pct = 100 * n_delayed / max(len(df), 1)
    logger.info(
        "Created '%s': %d delayed (%.1f%%) at threshold ≥%d min.",
        TARGET_COL,
        n_delayed,
        pct,
        threshold,
    )
    return df


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset.

    - Numeric columns: fill NaN with the column median.
    - Categorical / object columns: fill NaN with ``'UNKNOWN'``.

    Parameters
    ----------
    df : pd.DataFrame
        Flight data with potential missing values.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values filled.
    """
    df = df.copy()

    # Numeric columns — fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(
                "Filled %d NaN in '%s' with median=%.4f.",
                n_missing,
                col,
                median_val,
            )

    # Categorical / object columns — fill with 'UNKNOWN'
    cat_cols = df.select_dtypes(
        include=["category", "object"]
    ).columns.tolist()
    for col in cat_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            if df[col].dtype.name == "category":
                # Add 'UNKNOWN' to categories if not present
                if "UNKNOWN" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories("UNKNOWN")
            df[col] = df[col].fillna("UNKNOWN")
            logger.debug(
                "Filled %d NaN in '%s' with 'UNKNOWN'.",
                n_missing,
                col,
            )

    total_remaining = df.isna().sum().sum()
    logger.info("Missing value handling complete. Remaining NaN: %d.", total_remaining)
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def preprocess_pipeline(filepath: Path) -> pd.DataFrame:
    """Run the full preprocessing pipeline from raw file to clean DataFrame.

    Chains the following steps:
    1. Load raw data
    2. Cast column types
    3. Filter cancelled flights
    4. Create binary delay target
    5. Handle missing values

    Parameters
    ----------
    filepath : Path
        Path to the raw data file.

    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed flight data.
    """
    logger.info("Running preprocessing pipeline on %s …", filepath)

    df = load_raw_data(filepath)
    df = cast_types(df)
    df = filter_cancelled(df)
    df = create_delay_target(df)
    df = handle_missing(df)

    logger.info(
        "Preprocessing complete: %d rows × %d columns.",
        *df.shape,
    )
    return df
