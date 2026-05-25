"""BTS On-Time Performance data download and loading utilities.

The Bureau of Transportation Statistics (BTS) On-Time Performance database
requires manual download via a web form. This module provides:

- Step-by-step download instructions
- CSV loading with correct dtypes and cleanup
- Bulk loading of multiple monthly files

Data source: https://www.transtats.bts.gov/DL_SelectFields.aspx
Database: Airline On-Time Performance Data
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from flight_delay.utils.config import DEFAULT_YEAR, RAW_DIR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Fields to select when downloading from BTS
# ──────────────────────────────────────────────────────────────
BTS_FIELDS: list[str] = [
    "Year",
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "ArrDelay",
    "ArrDelayMinutes",
    "Cancelled",
    "Diverted",
    "CRSElapsedTime",
    "Distance",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

# ──────────────────────────────────────────────────────────────
# Pandas dtypes for raw BTS CSV columns
# ──────────────────────────────────────────────────────────────
BTS_DTYPES: dict[str, str] = {
    "Year": "int16",
    "Quarter": "int8",
    "Month": "int8",
    "DayofMonth": "int8",
    "DayOfWeek": "int8",
    "FlightDate": "str",
    "Reporting_Airline": "str",
    "Origin": "str",
    "Dest": "str",
    "CRSDepTime": "int16",
    "DepTime": "float32",       # can be NaN for cancelled
    "DepDelay": "float32",
    "DepDelayMinutes": "float32",
    "ArrDelay": "float32",
    "ArrDelayMinutes": "float32",
    "Cancelled": "float32",     # 0.0 / 1.0
    "Diverted": "float32",
    "CRSElapsedTime": "float32",
    "Distance": "float32",
    "CarrierDelay": "float32",  # NaN when not delayed
    "WeatherDelay": "float32",
    "NASDelay": "float32",
    "SecurityDelay": "float32",
    "LateAircraftDelay": "float32",
}


# ──────────────────────────────────────────────────────────────
# Download helper (manual — BTS requires a browser session)
# ──────────────────────────────────────────────────────────────
def download_bts_data(
    year: int = DEFAULT_YEAR,
    months: list[int] | None = None,
) -> None:
    """Print step-by-step instructions for downloading BTS On-Time data.

    BTS requires interactive browser access (CAPTCHA / session cookies),
    so this function cannot automate the download.  It creates the target
    directory and prints a checklist the user can follow.

    Parameters
    ----------
    year : int
        Calendar year to download (default from config).
    months : list[int] | None
        Specific months (1-12).  ``None`` means all 12 months.
    """
    target_dir = RAW_DIR / "bts"
    target_dir.mkdir(parents=True, exist_ok=True)

    months = months or list(range(1, 13))
    month_str = ", ".join(str(m) for m in months)

    instructions = f"""
╔══════════════════════════════════════════════════════════════╗
║           BTS On-Time Performance — Download Guide          ║
╚══════════════════════════════════════════════════════════════╝

1. Open your browser and navigate to:
   https://www.transtats.bts.gov/DL_SelectFields.aspx

2. Under "Filter Geography", leave defaults (All).

3. Under "Filter Year & Period":
   • Year  → {year}
   • Month → download each month separately: {month_str}

4. Check the following fields (uncheck everything else first):
{chr(10).join(f'   ☐ {f}' for f in BTS_FIELDS)}

5. Click "Download" — you will get a ZIP file for each month.

6. Extract each CSV into:
   {target_dir}

   Suggested naming convention:
     On_Time_Reporting_{year}_1.csv
     On_Time_Reporting_{year}_2.csv
     ...

7. After downloading, use  load_all_bts_data({year})  to load.
"""
    print(instructions)
    logger.info("Download target directory ready: %s", target_dir)


# ──────────────────────────────────────────────────────────────
# Load a single BTS CSV
# ──────────────────────────────────────────────────────────────
def load_bts_csv(filepath: str | Path) -> pd.DataFrame:
    """Load a single raw BTS On-Time Performance CSV file.

    Handles the trailing comma that BTS appends to every row (which
    creates a phantom unnamed column) and strips whitespace from
    string columns.

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw flight data with proper dtypes.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"BTS CSV not found: {filepath}")

    logger.info("Loading BTS CSV: %s", filepath.name)

    df = pd.read_csv(
        filepath,
        dtype=BTS_DTYPES,
        low_memory=False,
        encoding="utf-8",
    )

    # BTS CSVs have a trailing comma → pandas creates an unnamed column
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        logger.debug("Dropped trailing-comma columns: %s", unnamed_cols)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    logger.info(
        "Loaded %s rows × %s cols from %s",
        f"{len(df):,}",
        len(df.columns),
        filepath.name,
    )
    return df


# ──────────────────────────────────────────────────────────────
# Load all BTS CSVs from the raw directory
# ──────────────────────────────────────────────────────────────
def load_all_bts_data(year: int | None = None) -> pd.DataFrame:
    """Load and concatenate all BTS CSV files from ``data/raw/bts/``.

    Parameters
    ----------
    year : int | None
        If provided, only CSVs whose filename contains this year are
        loaded.  Otherwise every ``*.csv`` in the directory is loaded.

    Returns
    -------
    pd.DataFrame
        Concatenated raw flight data.

    Raises
    ------
    FileNotFoundError
        If no CSV files are found.
    """
    bts_dir = RAW_DIR / "bts"
    if not bts_dir.exists():
        raise FileNotFoundError(
            f"BTS data directory does not exist: {bts_dir}\n"
            "Run download_bts_data() for instructions."
        )

    pattern = f"*{year}*.csv" if year else "*.csv"
    csv_files = sorted(bts_dir.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching '{pattern}' in {bts_dir}\n"
            "Run download_bts_data() for instructions."
        )

    logger.info("Found %d BTS CSV file(s) to load.", len(csv_files))

    frames: list[pd.DataFrame] = []
    for fp in csv_files:
        frames.append(load_bts_csv(fp))

    df = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined BTS dataset: %s rows × %s cols",
        f"{len(df):,}",
        len(df.columns),
    )
    return df
