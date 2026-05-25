import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from flight_delay.data.preprocess import (
    cast_types,
    filter_cancelled,
    create_delay_target,
    handle_missing,
)
from flight_delay.utils.config import TARGET_COL

@pytest.fixture
def sample_raw_data():
    """Create a small mock raw flight dataset."""
    return pd.DataFrame({
        "fl_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "carrier": ["AA", "DL", "UA"],
        "origin": ["ATL", "ORD", "JFK"],
        "dest": ["LAX", "DFW", "SFO"],
        "crs_dep_time": [1200, 1530, 900],
        "arr_delay": [5.0, 45.0, np.nan],
        "cancelled": [0, 0, 1],
        "distance": [1947, 731, 2586],
        "origin_wind_speed": [12.5, np.nan, 8.0],
    })

def test_cast_types(sample_raw_data):
    """Test data types are cast correctly."""
    df_cast = cast_types(sample_raw_data)
    
    assert pd.api.types.is_datetime64_any_dtype(df_cast["fl_date"])
    assert isinstance(df_cast["carrier"].dtype, pd.CategoricalDtype)
    assert isinstance(df_cast["origin"].dtype, pd.CategoricalDtype)
    assert isinstance(df_cast["dest"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_numeric_dtype(df_cast["arr_delay"])
    assert pd.api.types.is_numeric_dtype(df_cast["crs_dep_time"])

def test_filter_cancelled(sample_raw_data):
    """Test cancelled flights are filtered out."""
    df_filtered = filter_cancelled(sample_raw_data)
    
    assert len(df_filtered) == 2
    assert (df_filtered["cancelled"] == 0).all()

def test_create_delay_target(sample_raw_data):
    """Test creation of the binary delay target."""
    # Filter cancelled first and drop nan to make test simple
    df_clean = sample_raw_data.dropna(subset=["arr_delay"]).copy()
    df_target = create_delay_target(df_clean, threshold=15)
    
    assert TARGET_COL in df_target.columns
    assert df_target.loc[0, TARGET_COL] == 0  # 5 min is on time
    assert df_target.loc[1, TARGET_COL] == 1  # 45 min is delayed

def test_handle_missing(sample_raw_data):
    """Test missing values are filled with median for numeric columns."""
    df_filled = handle_missing(sample_raw_data)
    
    assert not df_filled["arr_delay"].isna().any()
    assert not df_filled["origin_wind_speed"].isna().any()
    
    # Value filled should be the median
    expected_wind_median = np.nanmedian(sample_raw_data["origin_wind_speed"])
    assert df_filled.loc[1, "origin_wind_speed"] == expected_wind_median
