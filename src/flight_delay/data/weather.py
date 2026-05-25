"""NOAA Integrated Surface Database (ISD) weather data integration.

Fetches hourly surface observations from the NOAA ISD archive and maps
them to US airport IATA codes for merging with flight data.

Data source: https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
Station inventory: https://www1.ncdc.noaa.gov/pub/data/noaa/isd-history.csv
Hourly data: https://www1.ncdc.noaa.gov/pub/data/noaa/{year}/{usaf}-{wban}-{year}.gz
"""

from __future__ import annotations

import gzip
import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from flight_delay.utils.config import EXTERNAL_DIR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# IATA → ICAO mapping for the 50 busiest US airports
# ──────────────────────────────────────────────────────────────
IATA_TO_ICAO: dict[str, str] = {
    "ATL": "KATL", "DFW": "KDFW", "DEN": "KDEN", "ORD": "KORD",
    "LAX": "KLAX", "JFK": "KJFK", "LAS": "KLAS", "MCO": "KMCO",
    "MIA": "KMIA", "CLT": "KCLT", "SEA": "KSEA", "PHX": "KPHX",
    "EWR": "KEWR", "SFO": "KSFO", "IAH": "KIAH", "BOS": "KBOS",
    "FLL": "KFLL", "MSP": "KMSP", "LGA": "KLGA", "DTW": "KDTW",
    "PHL": "KPHL", "SLC": "KSLC", "DCA": "KDCA", "SAN": "KSAN",
    "BWI": "KBWI", "TPA": "KTPA", "AUS": "KAUS", "IAD": "KIAD",
    "BNA": "KBNA", "MDW": "KMDW", "HNL": "PHNL", "DAL": "KDAL",
    "PDX": "KPDX", "STL": "KSTL", "RDU": "KRDU", "HOU": "KHOU",
    "SJC": "KSJC", "SMF": "KSMF", "MCI": "KMCI", "OAK": "KOAK",
    "MSY": "KMSY", "CLE": "KCLE", "SAT": "KSAT", "PIT": "KPIT",
    "IND": "KIND", "CMH": "KCMH", "CVG": "KCVG", "RSW": "KRSW",
    "JAX": "KJAX", "ABQ": "KABQ",
}

# URLs
_ISD_HISTORY_URL = "https://www1.ncdc.noaa.gov/pub/data/noaa/isd-history.csv"
_ISD_DATA_URL = "https://www1.ncdc.noaa.gov/pub/data/noaa/{year}/{usaf}-{wban}-{year}.gz"

# Retry / timeout defaults
_TIMEOUT = 60
_MAX_RETRIES = 3


class NOAAWeatherFetcher:
    """Download and parse NOAA ISD hourly weather observations.

    Observations are mapped to IATA airport codes so they can be merged
    with BTS flight data on ``(airport, date, hour)``.

    Parameters
    ----------
    cache_dir : Path | None
        Local directory for caching downloaded files.  Defaults to
        ``data/external/weather/``.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or EXTERNAL_DIR / "weather"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._station_inventory: pd.DataFrame | None = None
        self._airport_station_map: dict[str, str] | None = None
        self._session = requests.Session()

    # ── station inventory ─────────────────────────────────────

    def fetch_station_inventory(self) -> pd.DataFrame:
        """Download (or load cached) ISD station history.

        Returns
        -------
        pd.DataFrame
            Columns: USAF, WBAN, STATION NAME, CTRY, STATE, ICAO,
            LAT, LON, ELEV(M), BEGIN, END.
        """
        cache_file = self.cache_dir / "isd-history.csv"

        if cache_file.exists():
            logger.debug("Loading cached station inventory: %s", cache_file)
            self._station_inventory = pd.read_csv(
                cache_file, dtype={"USAF": str, "WBAN": str, "ICAO": str},
            )
            return self._station_inventory

        logger.info("Downloading ISD station inventory …")
        resp = self._download_with_retry(_ISD_HISTORY_URL)
        cache_file.write_text(resp.text, encoding="utf-8")

        self._station_inventory = pd.read_csv(
            StringIO(resp.text),
            dtype={"USAF": str, "WBAN": str, "ICAO": str},
        )
        logger.info(
            "Station inventory: %d stations.", len(self._station_inventory),
        )
        return self._station_inventory

    # ── airport → station mapping ─────────────────────────────

    def _build_airport_station_map(self) -> dict[str, str]:
        """Map IATA airport codes to the nearest NOAA station ID.

        Strategy
        --------
        1. For each IATA code, derive the ICAO code (e.g. ATL → KATL).
        2. Look up that ICAO in the station inventory.
        3. If no match, fall back to geographic proximity using lat/lon
           of the airport (approximate coordinates embedded in this module).

        Returns
        -------
        dict[str, str]
            ``{iata_code: "USAF-WBAN"}``
        """
        if self._station_inventory is None:
            self.fetch_station_inventory()

        inv = self._station_inventory
        mapping: dict[str, str] = {}

        # Filter to US stations with valid coordinates
        us_stations = inv[
            (inv["CTRY"] == "US")
            & inv["LAT"].notna()
            & inv["LON"].notna()
        ].copy()

        for iata, icao in IATA_TO_ICAO.items():
            # Try direct ICAO match
            match = us_stations[us_stations["ICAO"] == icao]
            if not match.empty:
                row = match.iloc[0]
                station_id = f"{row['USAF']}-{row['WBAN']}"
                mapping[iata] = station_id
                continue

            # Fallback: nearest station by lat/lon
            coords = _AIRPORT_APPROX_COORDS.get(iata)
            if coords is None:
                logger.warning("No coordinates for %s — skipping.", iata)
                continue

            lat, lon = coords
            dists = (
                (us_stations["LAT"] - lat) ** 2
                + (us_stations["LON"] - lon) ** 2
            )
            idx = dists.idxmin()
            row = us_stations.loc[idx]
            station_id = f"{row['USAF']}-{row['WBAN']}"
            mapping[iata] = station_id
            logger.debug(
                "%s → %s (geo fallback, dist²=%.4f)", iata, station_id, dists[idx],
            )

        self._airport_station_map = mapping
        logger.info("Mapped %d airports to NOAA stations.", len(mapping))
        return mapping

    def map_airport_to_station(self, iata_code: str) -> str | None:
        """Return the NOAA station ID for a given IATA airport code.

        Parameters
        ----------
        iata_code : str
            Three-letter IATA code (e.g. ``"ATL"``).

        Returns
        -------
        str | None
            Station ID as ``"USAF-WBAN"`` or ``None`` if unmapped.
        """
        if self._airport_station_map is None:
            self._build_airport_station_map()
        return self._airport_station_map.get(iata_code.upper())

    # ── hourly weather data ───────────────────────────────────

    def fetch_hourly_weather(
        self,
        station_id: str,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """Download and parse hourly ISD weather for one station/year.

        The full year file is downloaded once and cached.  Records are
        then filtered to the requested month.

        Parameters
        ----------
        station_id : str
            ``"USAF-WBAN"`` format (e.g. ``"722190-13874"``).
        year : int
            Four-digit year.
        month : int
            Month number (1–12).

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, temp_c, dew_point_c, wind_speed_ms,
            wind_gust_ms, visibility_m, ceiling_m, precip_mm,
            present_weather.
        """
        usaf, wban = station_id.split("-")
        cache_path = self.cache_dir / f"{usaf}-{wban}-{year}.gz"

        # Download if not cached
        if not cache_path.exists():
            url = _ISD_DATA_URL.format(year=year, usaf=usaf, wban=wban)
            logger.info("Downloading ISD data: %s", url)
            resp = self._download_with_retry(url)
            cache_path.write_bytes(resp.content)

        # Parse the gzipped file
        records: list[dict] = []
        with gzip.open(cache_path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parsed = self.parse_isd_record(line)
                if parsed is not None:
                    records.append(parsed)

        if not records:
            logger.warning(
                "No valid records for station %s, %d.", station_id, year,
            )
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter to requested month
        df = df[df["timestamp"].dt.month == month].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "Station %s, %d-%02d: %d hourly records.",
            station_id, year, month, len(df),
        )
        return df

    # ── ISD fixed-width parser ────────────────────────────────

    @staticmethod
    def parse_isd_record(line: str) -> dict | None:
        """Parse a single line of the ISD fixed-width format.

        Key field positions (0-indexed) in the mandatory-data section
        (first 105 characters):

        ======  =====  ==========================================
        Start   End    Field
        ======  =====  ==========================================
          4      10    USAF station ID
         10      15    WBAN
         15      23    Date (YYYYMMDD)
         23      27    Time (HHMM)
         60      63    Wind direction (degrees)
         63      64    Wind direction quality
         65      69    Wind speed (m/s × 10)
         69      70    Wind speed quality
         78      83    Ceiling height (m, 99999 = unlimited)
         84      90    Visibility distance (m, 999999 = unlimited)
         87      92    Air temperature (°C × 10, leading +/-)
         93      98    Dew point temperature (°C × 10)
        ======  =====  ==========================================

        Parameters
        ----------
        line : str
            A single ISD record line.

        Returns
        -------
        dict | None
            Parsed observation or ``None`` for malformed records.
        """
        if len(line) < 105:
            return None

        try:
            date_str = line[15:23]
            time_str = line[23:27]
            timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}"

            # Wind speed (m/s × 10)
            raw_ws = line[65:69].strip()
            wind_speed = float(raw_ws) / 10.0 if raw_ws and raw_ws != "9999" else None

            # Ceiling (metres)
            raw_ceil = line[78:83].strip()
            ceiling = float(raw_ceil) if raw_ceil and raw_ceil != "99999" else None

            # Visibility (metres)
            raw_vis = line[84:90].strip()
            visibility = float(raw_vis) if raw_vis and raw_vis != "999999" else None

            # Temperature (°C × 10)
            raw_temp = line[87:92].strip()
            temp = float(raw_temp) / 10.0 if raw_temp and raw_temp not in ("+9999", "-9999", "9999") else None

            # Dew point (°C × 10)
            raw_dew = line[93:98].strip()
            dew_point = float(raw_dew) / 10.0 if raw_dew and raw_dew not in ("+9999", "-9999", "9999") else None

            return {
                "timestamp": timestamp,
                "temp_c": temp,
                "dew_point_c": dew_point,
                "wind_speed_ms": wind_speed,
                "wind_gust_ms": None,  # gust is in additional data section
                "visibility_m": visibility,
                "ceiling_m": ceiling,
                "precip_mm": None,  # precipitation is in additional data section
                "present_weather": None,  # present weather codes in additional data
            }
        except (ValueError, IndexError):
            return None

    # ── bulk weather for a flights DataFrame ──────────────────

    def get_weather_for_flights(
        self,
        flights_df: pd.DataFrame,
        airport_col: str = "origin",
        date_col: str = "FlightDate",
        hour_col: str = "hour_of_day",
    ) -> pd.DataFrame:
        """Fetch weather for every unique airport–month in a flights table.

        Parameters
        ----------
        flights_df : pd.DataFrame
            Must contain columns named by *airport_col*, *date_col*.
        airport_col : str
            Column containing IATA airport codes.
        date_col : str
            Column containing flight dates (datetime or string).
        hour_col : str
            Column containing hour of day (0-23).

        Returns
        -------
        pd.DataFrame
            Weather data with columns: airport, timestamp, temp_c,
            dew_point_c, wind_speed_ms, wind_gust_ms, visibility_m,
            ceiling_m, precip_mm, present_weather — ready for
            ``merge_asof`` on ``(airport, timestamp)``.
        """
        flights_df = flights_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(flights_df[date_col]):
            flights_df[date_col] = pd.to_datetime(flights_df[date_col])

        # Unique (airport, year, month) combos
        combos = (
            flights_df[[airport_col, date_col]]
            .assign(
                year=lambda d: d[date_col].dt.year,
                month=lambda d: d[date_col].dt.month,
            )
            .groupby([airport_col, "year", "month"])
            .size()
            .reset_index()
            .rename(columns={airport_col: "airport"})
        )[["airport", "year", "month"]]

        frames: list[pd.DataFrame] = []
        for _, row in combos.iterrows():
            iata = row["airport"]
            station = self.map_airport_to_station(iata)
            if station is None:
                logger.warning("No NOAA station for %s — skipping.", iata)
                continue
            wx = self.fetch_hourly_weather(station, int(row["year"]), int(row["month"]))
            if not wx.empty:
                wx["airport"] = iata
                frames.append(wx)

        if not frames:
            logger.warning("No weather data retrieved.")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values(["airport", "timestamp"]).reset_index(drop=True)
        logger.info("Weather data: %d rows for %d airports.", len(result), combos["airport"].nunique())
        return result

    # ── HTTP helper with retries ──────────────────────────────

    def _download_with_retry(self, url: str) -> requests.Response:
        """GET *url* with up to ``_MAX_RETRIES`` attempts."""
        import time as _time

        backoff = 1.0
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = self._session.get(url, timeout=_TIMEOUT)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                logger.warning(
                    "Download failed (attempt %d/%d): %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                if attempt == _MAX_RETRIES:
                    raise
                _time.sleep(backoff)
                backoff *= 2

        # Unreachable, satisfies type checker.
        raise RuntimeError("Max retries exceeded.")


# ──────────────────────────────────────────────────────────────
# Approximate airport lat/lon (for geo-fallback matching)
# ──────────────────────────────────────────────────────────────
_AIRPORT_APPROX_COORDS: dict[str, tuple[float, float]] = {
    "ATL": (33.64, -84.43), "DFW": (32.90, -97.04), "DEN": (39.86, -104.67),
    "ORD": (41.97, -87.91), "LAX": (33.94, -118.41), "JFK": (40.64, -73.78),
    "LAS": (36.08, -115.15), "MCO": (28.43, -81.31), "MIA": (25.80, -80.29),
    "CLT": (35.21, -80.94), "SEA": (47.45, -122.31), "PHX": (33.44, -112.01),
    "EWR": (40.69, -74.17), "SFO": (37.62, -122.38), "IAH": (29.98, -95.34),
    "BOS": (42.37, -71.01), "FLL": (26.07, -80.15), "MSP": (44.88, -93.22),
    "LGA": (40.78, -73.87), "DTW": (42.21, -83.35), "PHL": (39.87, -75.24),
    "SLC": (40.79, -111.98), "DCA": (38.85, -77.04), "SAN": (32.73, -117.19),
    "BWI": (39.18, -76.67), "TPA": (27.98, -82.53), "AUS": (30.20, -97.67),
    "IAD": (38.95, -77.46), "BNA": (36.13, -86.68), "MDW": (41.79, -87.75),
    "HNL": (21.32, -157.92), "DAL": (32.85, -96.85), "PDX": (45.59, -122.60),
    "STL": (38.75, -90.37), "RDU": (35.88, -78.79), "HOU": (29.65, -95.28),
    "SJC": (37.36, -121.93), "SMF": (38.70, -121.59), "MCI": (39.30, -94.71),
    "OAK": (37.72, -122.22), "MSY": (29.99, -90.26), "CLE": (41.41, -81.85),
    "SAT": (29.53, -98.47), "PIT": (40.49, -80.23), "IND": (39.72, -86.29),
    "CMH": (40.00, -82.89), "CVG": (39.05, -84.67), "RSW": (26.54, -81.76),
    "JAX": (30.49, -81.69), "ABQ": (35.04, -106.61),
}
