"""Feature engineering modules.

Public API
----------
- :func:`add_temporal_features`
- :func:`add_airport_features`
- :func:`add_weather_features`
- :func:`add_route_features`
- :class:`RouteEncoder`
- :class:`FlightDelayFeaturePipeline`
- :func:`build_sklearn_pipeline`
- :func:`prepare_train_test_split`
- :func:`get_X_y`
"""

from flight_delay.features.temporal import add_temporal_features
from flight_delay.features.airport import add_airport_features
from flight_delay.features.weather_features import add_weather_features
from flight_delay.features.route import add_route_features, RouteEncoder
from flight_delay.features.pipeline import (
    FlightDelayFeaturePipeline,
    build_sklearn_pipeline,
    prepare_train_test_split,
    get_X_y,
)

__all__ = [
    "add_temporal_features",
    "add_airport_features",
    "add_weather_features",
    "add_route_features",
    "RouteEncoder",
    "FlightDelayFeaturePipeline",
    "build_sklearn_pipeline",
    "prepare_train_test_split",
    "get_X_y",
]
