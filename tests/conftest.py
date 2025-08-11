"""
Test configuration and fixtures for pytest.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_time_series():
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Create data with a change point at day 50
    values = np.concatenate([
        np.random.normal(10, 1, 50),  # First regime
        np.random.normal(15, 1.5, 50)  # Second regime
    ])
    
    return pd.DataFrame({
        'date': dates,
        'value': values
    })

@pytest.fixture
def sample_oil_data():
    """Create sample oil price data for testing."""
    np.random.seed(123)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Simulate oil price with trend and volatility
    trend = np.linspace(50, 80, 200)
    noise = np.random.normal(0, 5, 200)
    prices = trend + noise
    
    return pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
