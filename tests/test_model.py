import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import joblib
from flight_delay.models.train import train_model, evaluate_model, save_model, load_model
from flight_delay.utils.config import RANDOM_STATE

@pytest.fixture
def mock_dataset():
    """Create a small mock feature matrix and target vector."""
    np.random.seed(RANDOM_STATE)
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series(np.random.choice([0, 1], size=100, p=[0.7, 0.3]))
    return X, y

def test_train_logistic_regression(mock_dataset):
    """Test training Logistic Regression model."""
    X, y = mock_dataset
    model = train_model(X, y, model_type="lr")
    
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

def test_train_random_forest(mock_dataset):
    """Test training Random Forest model."""
    X, y = mock_dataset
    model = train_model(X, y, model_type="rf")
    
    assert model is not None
    assert hasattr(model, "predict")
    
def test_train_xgboost(mock_dataset):
    """Test training XGBoost model."""
    X, y = mock_dataset
    model = train_model(X, y, model_type="xgboost")
    
    assert model is not None
    assert hasattr(model, "predict")

def test_evaluate_model(mock_dataset):
    """Test evaluation output keys and correctness."""
    X, y = mock_dataset
    model = train_model(X, y, model_type="xgboost")
    metrics = evaluate_model(model, X, y)
    
    expected_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    for key in expected_keys:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

def test_save_load_model(mock_dataset):
    """Test saving and loading model roundtrip."""
    X, y = mock_dataset
    model = train_model(X, y, model_type="lr")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_model.joblib"
        save_model(model, filepath)
        
        assert filepath.exists()
        
        loaded_model = load_model(filepath)
        assert loaded_model is not None
        assert hasattr(loaded_model, "predict")
        
        # Test predictions match
        np.testing.assert_array_equal(model.predict(X), loaded_model.predict(X))
