import pytest
import pandas as pd
import numpy as np
from flight_delay.features.temporal import add_temporal_features
from flight_delay.features.weather_features import add_weather_features
from flight_delay.features.pipeline import FlightDelayFeaturePipeline, get_feature_columns
from flight_delay.utils.config import TARGET_COL

@pytest.fixture
def sample_preprocessed_data():
    """Create a mock preprocessed flight dataset."""
    return pd.DataFrame({
        "fl_date": pd.to_datetime(["2023-07-04", "2023-11-25", "2023-12-25"]), # holiday dates + standard
        "carrier": ["AA", "UA", "DL"],
        "origin": ["ATL", "ORD", "LAX"],
        "dest": ["DFW", "JFK", "SFO"],
        "crs_dep_time": [800, 1830, 2300],
        "distance": [731, 740, 337],
        "arr_delay": [5.0, 45.0, -10.0],
        "cancelled": [0, 0, 0],
        # weather
        "origin_wind_speed": [12.0, 28.0, 5.0],
        "origin_visibility": [10.0, 2.0, 8.0],
        "origin_ceiling": [25000, 800, 15000],
        "origin_temp": [85.0, 32.0, 55.0],
        "origin_precip": [0.0, 0.25, 0.0],
        "origin_has_thunderstorm": [0, 0, 0],
        "origin_has_snow": [0, 1, 0],
        "origin_has_fog": [0, 1, 0],
    })

def test_add_temporal_features(sample_preprocessed_data):
    """Test temporal feature calculations, cyclicals, and holidays."""
    # Add target first
    df = sample_preprocessed_data.copy()
    df[TARGET_COL] = (df["arr_delay"] >= 15).astype(int)
    
    # Run temporal
    # Standardise column casing for the legacy subagent features modules
    df_upper = df.rename(columns={
        "fl_date": "FlightDate",
        "crs_dep_time": "CRSDepTime",
    })
    df_temp = add_temporal_features(df_upper)
    
    assert "hour_of_day" in df_temp.columns
    assert "day_of_week" in df_temp.columns
    assert "is_holiday" in df_temp.columns
    assert "hour_sin" in df_temp.columns
    
    # 2023-07-04 is July 4th (US Independence Day) - should be detected as holiday
    assert df_temp.loc[0, "is_holiday"] == 1
    # 800 -> 8:00 AM -> hour_of_day=8
    assert df_temp.loc[0, "hour_of_day"] == 8

def test_add_weather_features(sample_preprocessed_data):
    """Test IFR and severe weather conditions indicators."""
    df_weather = add_weather_features(sample_preprocessed_data)
    
    assert "ifr_conditions_origin" in df_weather.columns
    assert "severe_weather_origin" in df_weather.columns
    
    # Second flight has visibility 2 miles (<3) and ceiling 800 (<1000) -> IFR
    assert df_weather.loc[1, "ifr_conditions_origin"] == 1
    # Second flight has snow and high wind -> Severe
    assert df_weather.loc[1, "severe_weather_origin"] == 1
    # Third flight is clear -> Not IFR or severe
    assert df_weather.loc[2, "ifr_conditions_origin"] == 0

def test_feature_pipeline_fit_transform(sample_preprocessed_data):
    """Test end-to-end fit_transform of our feature pipeline."""
    df = sample_preprocessed_data.copy()
    df[TARGET_COL] = (df["arr_delay"] >= 15).astype(int)
    
    pipeline = FlightDelayFeaturePipeline()
    df_transformed = pipeline.fit_transform(df)
    
    assert TARGET_COL in df_transformed.columns
    
    feature_cols = get_feature_columns()
    for col in feature_cols:
        assert col in df_transformed.columns
