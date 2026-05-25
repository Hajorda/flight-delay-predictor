"""Synthetic flight data generator for development and testing.

Generates ~50,000 realistic synthetic flight records with columns matching
the expected schema: fl_date, carrier, origin, dest, crs_dep_time,
arr_delay, cancelled, distance, and weather columns for origin and
destination airports.

Usage
-----
Run as a module::

    python -m flight_delay.data.synthetic

Or call :func:`generate_synthetic_data` directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from flight_delay.utils.config import (
    HUB_AIRPORTS,
    RANDOM_STATE,
    RAW_DIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CARRIERS: list[str] = [
    "AA", "UA", "DL", "WN", "B6", "AS", "NK", "F9", "HA", "G4",
]

# Convert set to sorted list for reproducible indexing
AIRPORTS: list[str] = sorted(HUB_AIRPORTS)

N_ROWS: int = 50_000

OUTPUT_FILENAME: str = "synthetic_flights.parquet"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_synthetic_data(
    n_rows: int = N_ROWS,
    seed: int = RANDOM_STATE,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate synthetic flight data and save to Parquet.

    Parameters
    ----------
    n_rows : int, optional
        Number of rows to generate (default: 50 000).
    seed : int, optional
        Random seed for reproducibility (default: ``RANDOM_STATE``).
    output_path : Path or None, optional
        Where to save the Parquet file. Defaults to
        ``RAW_DIR / 'synthetic_flights.parquet'``.

    Returns
    -------
    pd.DataFrame
        The generated synthetic flight DataFrame.
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic flight records (seed=%d) …", n_rows, seed)

    # --- Flight dates across a full year ---
    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2023-12-31")
    date_range = pd.date_range(start_date, end_date, freq="D")
    fl_dates = rng.choice(date_range, size=n_rows)

    # --- Carrier ---
    carriers = rng.choice(CARRIERS, size=n_rows)

    # --- Origin and Destination (ensure origin != dest) ---
    origins = rng.choice(AIRPORTS, size=n_rows)
    dests = rng.choice(AIRPORTS, size=n_rows)
    # Re-roll destinations that match origins
    same_mask = origins == dests
    while same_mask.any():
        dests[same_mask] = rng.choice(AIRPORTS, size=same_mask.sum())
        same_mask = origins == dests

    # --- Scheduled departure time (HHMM integer format) ---
    hours = rng.integers(5, 24, size=n_rows)
    minutes = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], size=n_rows)
    crs_dep_time = hours * 100 + minutes

    # --- Distance (statute miles) ---
    # Realistic range: 100–3000 miles with a right skew
    distance = rng.lognormal(mean=6.5, sigma=0.7, size=n_rows).clip(100, 5000).astype(int)

    # --- Arrival delay ---
    # Mix of on-time (normal around -5) and delayed (exponential)
    is_delayed_gen = rng.random(size=n_rows) < 0.25  # ~25% delayed
    arr_delay = np.where(
        is_delayed_gen,
        rng.exponential(scale=45, size=n_rows) + 15,  # delayed: 15+ minutes
        rng.normal(loc=-5, scale=10, size=n_rows),  # on-time / early
    )
    arr_delay = np.round(arr_delay, 1)

    # --- Cancelled (small percentage) ---
    cancelled = (rng.random(size=n_rows) < 0.02).astype(int)

    # --- Weather columns (origin) ---
    origin_wind_speed = rng.uniform(0, 40, size=n_rows).round(1)
    origin_visibility = rng.uniform(0, 15, size=n_rows).round(1)
    origin_ceiling = rng.uniform(0, 30_000, size=n_rows).astype(int)
    origin_temp = rng.uniform(10, 100, size=n_rows).round(1)
    origin_precip = rng.exponential(scale=0.05, size=n_rows).round(3)
    origin_has_thunderstorm = (rng.random(size=n_rows) < 0.05).astype(int)
    origin_has_snow = (rng.random(size=n_rows) < 0.08).astype(int)
    origin_has_fog = (rng.random(size=n_rows) < 0.10).astype(int)

    # --- Weather columns (destination) ---
    dest_wind_speed = rng.uniform(0, 40, size=n_rows).round(1)
    dest_visibility = rng.uniform(0, 15, size=n_rows).round(1)
    dest_ceiling = rng.uniform(0, 30_000, size=n_rows).astype(int)
    dest_temp = rng.uniform(10, 100, size=n_rows).round(1)
    dest_precip = rng.exponential(scale=0.05, size=n_rows).round(3)
    dest_has_thunderstorm = (rng.random(size=n_rows) < 0.05).astype(int)
    dest_has_snow = (rng.random(size=n_rows) < 0.08).astype(int)
    dest_has_fog = (rng.random(size=n_rows) < 0.10).astype(int)

    # --- Assemble DataFrame ---
    df = pd.DataFrame(
        {
            "fl_date": fl_dates,
            "carrier": carriers,
            "origin": origins,
            "dest": dests,
            "crs_dep_time": crs_dep_time,
            "arr_delay": arr_delay,
            "cancelled": cancelled,
            "distance": distance,
            # Origin weather
            "origin_wind_speed": origin_wind_speed,
            "origin_visibility": origin_visibility,
            "origin_ceiling": origin_ceiling,
            "origin_temp": origin_temp,
            "origin_precip": origin_precip,
            "origin_has_thunderstorm": origin_has_thunderstorm,
            "origin_has_snow": origin_has_snow,
            "origin_has_fog": origin_has_fog,
            # Destination weather
            "dest_wind_speed": dest_wind_speed,
            "dest_visibility": dest_visibility,
            "dest_ceiling": dest_ceiling,
            "dest_temp": dest_temp,
            "dest_precip": dest_precip,
            "dest_has_thunderstorm": dest_has_thunderstorm,
            "dest_has_snow": dest_has_snow,
            "dest_has_fog": dest_has_fog,
        }
    )

    # Sort by date for temporal consistency
    df = df.sort_values("fl_date").reset_index(drop=True)

    # --- Save ---
    if output_path is None:
        output_path = RAW_DIR / OUTPUT_FILENAME
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(
        "Synthetic data saved → %s (%d rows × %d columns).",
        output_path,
        *df.shape,
    )

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    df = generate_synthetic_data()
    print(f"\n✅ Generated {len(df):,} synthetic flight records.")
    print(f"   Saved to: {RAW_DIR / OUTPUT_FILENAME}")
    print(f"   Columns:  {list(df.columns)}")
    print(f"   Date range: {df['fl_date'].min()} → {df['fl_date'].max()}")
