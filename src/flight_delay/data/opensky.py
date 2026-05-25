"""OpenSky Network API client for ADS-B flight enrichment data.

Provides authenticated (OAuth2 client-credentials) and anonymous access
to the OpenSky REST API for:

- Airport departures / arrivals
- All flights in a time interval
- Airport congestion (aircraft count in bounding box)

**Important**: OpenSky does **not** provide delay data.  It supplies
ADS-B-derived flight events (actual departure / arrival times) which can
be used for congestion metrics and route validation.

Rate limits:
    - Anonymous : 400  credits / day
    - Authenticated: 4 000 credits / day

API docs: https://openskynetwork.github.io/opensky-api/rest.html
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import requests

from flight_delay.utils.config import OPENSKY_AUTH_URL, OPENSKY_BASE_URL

logger = logging.getLogger(__name__)

# Max interval the /flights/all endpoint accepts (seconds).
_MAX_ALL_FLIGHTS_INTERVAL = 2 * 3600  # 2 hours

# ──────────────────────────────────────────────────────────────
# Approximate coordinates for the 50 busiest US airports (ICAO)
# Used by get_airport_congestion to build a bounding box.
# ──────────────────────────────────────────────────────────────
AIRPORT_COORDS: dict[str, tuple[float, float]] = {
    "KATL": (33.6407, -84.4277),
    "KDFW": (32.8998, -97.0403),
    "KDEN": (39.8561, -104.6737),
    "KORD": (41.9742, -87.9073),
    "KLAX": (33.9425, -118.4081),
    "KJFK": (40.6413, -73.7781),
    "KLAS": (36.0840, -115.1537),
    "KMCO": (28.4312, -81.3081),
    "KMIA": (25.7959, -80.2870),
    "KCLT": (35.2140, -80.9431),
    "KSEA": (47.4502, -122.3088),
    "KPHX": (33.4373, -112.0078),
    "KEWR": (40.6895, -74.1745),
    "KSFO": (37.6213, -122.3790),
    "KIAH": (29.9844, -95.3414),
    "KBOS": (42.3656, -71.0096),
    "KFLL": (26.0742, -80.1506),
    "KMSP": (44.8848, -93.2223),
    "KLGA": (40.7769, -73.8740),
    "KDTW": (42.2124, -83.3534),
    "KPHL": (39.8744, -75.2424),
    "KSLC": (40.7884, -111.9778),
    "KDCA": (38.8512, -77.0402),
    "KSAN": (32.7338, -117.1933),
    "KBWI": (39.1754, -76.6684),
    "KTPA": (27.9755, -82.5332),
    "KAUS": (30.1975, -97.6664),
    "KIAD": (38.9531, -77.4565),
    "KBNA": (36.1263, -86.6774),
    "KMDW": (41.7868, -87.7522),
    "PHNL": (21.3187, -157.9225),
    "KDAL": (32.8471, -96.8518),
    "KPDX": (45.5898, -122.5951),
    "KSTL": (38.7487, -90.3700),
    "KRDU": (35.8776, -78.7875),
    "KHOU": (29.6454, -95.2789),
    "KSJC": (37.3626, -121.9290),
    "KSMF": (38.6954, -121.5908),
    "KMCI": (39.2976, -94.7139),
    "KOAK": (37.7213, -122.2208),
    "KMSY": (29.9934, -90.2580),
    "KCLE": (41.4117, -81.8498),
    "KSAT": (29.5337, -98.4698),
    "KPIT": (40.4915, -80.2329),
    "KIND": (39.7173, -86.2944),
    "KCMH": (39.9980, -82.8919),
    "KCVG": (39.0488, -84.6678),
    "KRSW": (26.5362, -81.7552),
    "KJAX": (30.4941, -81.6879),
    "KABQ": (35.0402, -106.6090),
}

# Quick IATA → ICAO lookup (subset for bounding-box queries)
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


# ──────────────────────────────────────────────────────────────
# Token manager (OAuth2 client-credentials flow)
# ──────────────────────────────────────────────────────────────
@dataclass
class TokenManager:
    """Manages OAuth2 tokens for the OpenSky Network API.

    Tokens are cached in memory and automatically refreshed 30 seconds
    before expiry.
    """

    client_id: str
    client_secret: str
    _access_token: str = field(default="", init=False, repr=False)
    _expires_at: float = field(default=0.0, init=False, repr=False)

    # Refresh margin — fetch a new token this many seconds before expiry.
    _MARGIN_SECONDS: int = 30

    def _fetch_token(self) -> dict:
        """Request a new access token from the OpenSky auth endpoint."""
        logger.debug("Requesting new OpenSky access token …")
        resp = requests.post(
            OPENSKY_AUTH_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    @property
    def is_expired(self) -> bool:
        """Return ``True`` if the cached token is missing or nearly expired."""
        return (
            not self._access_token
            or time.time() >= self._expires_at - self._MARGIN_SECONDS
        )

    def get_token(self) -> str:
        """Return a valid access token, refreshing if necessary."""
        if self.is_expired:
            data = self._fetch_token()
            self._access_token = data["access_token"]
            self._expires_at = time.time() + data.get("expires_in", 300)
            logger.info(
                "OpenSky token refreshed — expires in %ds",
                data.get("expires_in", 0),
            )
        return self._access_token


# ──────────────────────────────────────────────────────────────
# Rate-limit tracker
# ──────────────────────────────────────────────────────────────
@dataclass
class _RateLimitTracker:
    """Simple daily credit-usage tracker."""

    daily_limit: int
    _used: int = field(default=0, init=False)
    _day: str = field(default="", init=False)

    def _reset_if_new_day(self) -> None:
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self._used = 0

    def consume(self, credits: int = 1) -> None:
        self._reset_if_new_day()
        self._used += credits
        pct = self._used / self.daily_limit * 100
        if pct >= 80:
            logger.warning(
                "OpenSky rate limit: %d / %d credits used (%.0f%%)",
                self._used,
                self.daily_limit,
                pct,
            )

    @property
    def remaining(self) -> int:
        self._reset_if_new_day()
        return max(0, self.daily_limit - self._used)


# ──────────────────────────────────────────────────────────────
# OpenSky API client
# ──────────────────────────────────────────────────────────────
class OpenSkyClient:
    """Client for the OpenSky Network REST API.

    Supports both anonymous and authenticated (OAuth2 client-credentials)
    access.  Credentials can be passed directly or read from environment
    variables ``OPENSKY_CLIENT_ID`` / ``OPENSKY_CLIENT_SECRET``.

    Notes
    -----
    OpenSky does **not** provide delay data.  It supplies ADS-B-derived
    flight events (actual departure / arrival Unix timestamps) which are
    useful for:

    - Airport congestion metrics (how many aircraft on the ground)
    - Actual flight counts per route / hour
    - Route validation and enrichment

    Rate limits
    -----------
    - Anonymous:      400 credits / day
    - Authenticated: 4 000 credits / day
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        cid = client_id or os.environ.get("OPENSKY_CLIENT_ID")
        csec = client_secret or os.environ.get("OPENSKY_CLIENT_SECRET")

        self._token_mgr: TokenManager | None = None
        if cid and csec:
            self._token_mgr = TokenManager(client_id=cid, client_secret=csec)
            self._rate = _RateLimitTracker(daily_limit=4_000)
            logger.info("OpenSkyClient initialised (authenticated).")
        else:
            self._rate = _RateLimitTracker(daily_limit=400)
            logger.info("OpenSkyClient initialised (anonymous — 400 credits/day).")

        self._session = requests.Session()
        self._max_retries = 3

    # ── internal helpers ──────────────────────────────────────

    def _get_headers(self) -> dict[str, str]:
        """Build HTTP headers, including Bearer token when authenticated."""
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._token_mgr is not None:
            headers["Authorization"] = f"Bearer {self._token_mgr.get_token()}"
        return headers

    def _request(
        self,
        endpoint: str,
        params: dict | None = None,
    ) -> dict | list:
        """Perform a rate-limited GET request with retries.

        Parameters
        ----------
        endpoint : str
            Path appended to ``OPENSKY_BASE_URL`` (e.g. ``/flights/departure``).
        params : dict | None
            Query parameters.

        Returns
        -------
        dict | list
            Parsed JSON response.

        Raises
        ------
        requests.HTTPError
            On non-retryable HTTP errors.
        RuntimeError
            When daily rate limit is exhausted.
        """
        if self._rate.remaining <= 0:
            raise RuntimeError(
                "OpenSky daily rate limit exhausted. "
                "Wait until UTC midnight or use authenticated access."
            )

        url = f"{OPENSKY_BASE_URL}{endpoint}"
        backoff = 1.0

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._session.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=60,
                )
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", backoff))
                    logger.warning(
                        "Rate-limited (429) — retrying in %.1fs …", wait,
                    )
                    time.sleep(wait)
                    backoff *= 2
                    continue

                resp.raise_for_status()
                self._rate.consume()
                return resp.json()

            except requests.ConnectionError as exc:
                logger.warning(
                    "Connection error (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt == self._max_retries:
                    raise
                time.sleep(backoff)
                backoff *= 2

        # Should not reach here, but satisfy type checker.
        raise RuntimeError("Max retries exceeded for OpenSky request.")

    # ── public methods ────────────────────────────────────────

    def get_airport_departures(
        self,
        airport_icao: str,
        begin: int,
        end: int,
    ) -> list[dict]:
        """Retrieve departure flights for an airport.

        Parameters
        ----------
        airport_icao : str
            ICAO code (e.g. ``"KATL"``).
        begin : int
            Start of time range as Unix timestamp.
        end : int
            End of time range as Unix timestamp.
            Maximum window: 7 days.

        Returns
        -------
        list[dict]
            List of flight records from the API.
        """
        return self._request(
            "/flights/departure",
            params={"airport": airport_icao, "begin": begin, "end": end},
        )

    def get_airport_arrivals(
        self,
        airport_icao: str,
        begin: int,
        end: int,
    ) -> list[dict]:
        """Retrieve arrival flights for an airport.

        Parameters
        ----------
        airport_icao : str
            ICAO code (e.g. ``"KJFK"``).
        begin : int
            Start of time range as Unix timestamp.
        end : int
            End of time range as Unix timestamp.
            Maximum window: 7 days.

        Returns
        -------
        list[dict]
            List of flight records from the API.
        """
        return self._request(
            "/flights/arrival",
            params={"airport": airport_icao, "begin": begin, "end": end},
        )

    def get_flights_in_interval(
        self,
        begin: int,
        end: int,
    ) -> list[dict]:
        """Retrieve all flights in a time interval.

        The API enforces a 2-hour maximum window.  This method
        automatically chunks longer intervals.

        Parameters
        ----------
        begin : int
            Start Unix timestamp.
        end : int
            End Unix timestamp.

        Returns
        -------
        list[dict]
            Aggregated list of flight records.
        """
        results: list[dict] = []
        chunk_start = begin

        while chunk_start < end:
            chunk_end = min(chunk_start + _MAX_ALL_FLIGHTS_INTERVAL, end)
            logger.debug(
                "Fetching flights %s → %s",
                datetime.fromtimestamp(chunk_start, tz=timezone.utc),
                datetime.fromtimestamp(chunk_end, tz=timezone.utc),
            )
            chunk = self._request(
                "/flights/all",
                params={"begin": chunk_start, "end": chunk_end},
            )
            if isinstance(chunk, list):
                results.extend(chunk)
            chunk_start = chunk_end

        return results

    def get_airport_congestion(
        self,
        airport_icao: str,
        timestamp: int,
        bbox_deg: float = 0.5,
    ) -> int:
        """Estimate airport congestion by counting nearby aircraft.

        Queries ``/states/all`` with a geographic bounding box centred on
        the airport.

        Parameters
        ----------
        airport_icao : str
            ICAO code for which coordinates are known.
        timestamp : int
            Unix timestamp for the state snapshot.
        bbox_deg : float
            Half-width of the bounding box in degrees (default 0.5 ≈ 55 km).

        Returns
        -------
        int
            Number of aircraft in the bounding box at that time.
        """
        if airport_icao not in AIRPORT_COORDS:
            logger.warning("No coordinates for %s — returning 0.", airport_icao)
            return 0

        lat, lon = AIRPORT_COORDS[airport_icao]
        data = self._request(
            "/states/all",
            params={
                "time": timestamp,
                "lamin": lat - bbox_deg,
                "lamax": lat + bbox_deg,
                "lomin": lon - bbox_deg,
                "lomax": lon + bbox_deg,
            },
        )
        states = data.get("states", []) if isinstance(data, dict) else []
        count = len(states)
        logger.debug("%s congestion at t=%d: %d aircraft", airport_icao, timestamp, count)
        return count

    # ── DataFrame converters ──────────────────────────────────

    @staticmethod
    def departures_to_dataframe(departures: list[dict]) -> pd.DataFrame:
        """Convert departure API records to a tidy DataFrame.

        Parameters
        ----------
        departures : list[dict]
            Raw API response list.

        Returns
        -------
        pd.DataFrame
            Columns: icao24, callsign, estDepartureAirport,
            estArrivalAirport, firstSeen, lastSeen,
            estDepartureAirportHorizDistance, etc.
        """
        if not departures:
            return pd.DataFrame()
        df = pd.DataFrame(departures)
        for col in ("firstSeen", "lastSeen"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit="s", utc=True)
        return df

    @staticmethod
    def arrivals_to_dataframe(arrivals: list[dict]) -> pd.DataFrame:
        """Convert arrival API records to a tidy DataFrame.

        Parameters
        ----------
        arrivals : list[dict]
            Raw API response list.

        Returns
        -------
        pd.DataFrame
            Same schema as :meth:`departures_to_dataframe`.
        """
        if not arrivals:
            return pd.DataFrame()
        df = pd.DataFrame(arrivals)
        for col in ("firstSeen", "lastSeen"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit="s", utc=True)
        return df
