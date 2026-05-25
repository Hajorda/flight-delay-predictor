"""Configuration: paths, constants, and thresholds."""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # flight-delay-predictor/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Delay definition
# ──────────────────────────────────────────────
DELAY_THRESHOLD_MINUTES = 15  # FAA standard: ≥15 min = delayed

# ──────────────────────────────────────────────
# Data parameters
# ──────────────────────────────────────────────
DEFAULT_YEAR = 2023
TOP_N_AIRPORTS = 50  # Focus on busiest airports

# Random state for reproducibility
RANDOM_STATE = 42

# ──────────────────────────────────────────────
# Feature columns
# ──────────────────────────────────────────────
TARGET_COL = "is_delayed"

TEMPORAL_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

WEATHER_FEATURES = [
    "origin_wind_speed",
    "origin_visibility",
    "origin_ceiling",
    "origin_temp",
    "origin_precip",
    "origin_has_thunderstorm",
    "origin_has_snow",
    "origin_has_fog",
    "dest_wind_speed",
    "dest_visibility",
    "dest_ceiling",
    "dest_temp",
    "dest_precip",
    "dest_has_thunderstorm",
    "dest_has_snow",
    "dest_has_fog",
    "ifr_conditions_origin",
    "ifr_conditions_dest",
]

AIRPORT_FEATURES = [
    "origin_flights_per_hour",
    "dest_flights_per_hour",
    "origin_delay_rate_rolling",
    "dest_delay_rate_rolling",
    "is_origin_hub",
    "is_dest_hub",
]

ROUTE_FEATURES = [
    "distance",
    "route_avg_delay",
    "carrier_delay_rate",
]

CATEGORICAL_FEATURES = [
    "carrier",
    "origin",
    "dest",
]

# ──────────────────────────────────────────────
# Top 50 US airports by passenger traffic
# ──────────────────────────────────────────────
HUB_AIRPORTS = {
    "ATL", "DFW", "DEN", "ORD", "LAX", "JFK", "LAS", "MCO", "MIA", "CLT",
    "SEA", "PHX", "EWR", "SFO", "IAH", "BOS", "FLL", "MSP", "LGA", "DTW",
    "PHL", "SLC", "DCA", "SAN", "BWI", "TPA", "AUS", "IAD", "BNA", "MDW",
    "HNL", "DAL", "PDX", "STL", "RDU", "HOU", "SJC", "SMF", "MCI", "OAK",
    "MSY", "CLE", "SAT", "PIT", "IND", "CMH", "CVG", "RSW", "JAX", "ABQ",
}

# ──────────────────────────────────────────────
# Model parameters
# ──────────────────────────────────────────────
TRAIN_MONTHS = list(range(1, 11))  # Jan–Oct for training
TEST_MONTHS = [11, 12]  # Nov–Dec for testing

XGBOOST_DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ──────────────────────────────────────────────
# OpenSky API (enrichment — no delay data)
# ──────────────────────────────────────────────
OPENSKY_BASE_URL = "https://opensky-network.org/api"
OPENSKY_AUTH_URL = (
    "https://auth.opensky-network.org/auth/realms/"
    "opensky-network/protocol/openid-connect/token"
)
