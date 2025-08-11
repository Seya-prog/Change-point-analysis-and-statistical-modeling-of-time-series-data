"""
Tests for change point detection modules.
"""
import pytest
import pandas as pd
import numpy as np

# Try to import change point modules, skip if not available
try:
    from change_point.bayesian_detector import BayesianChangePointDetector
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

try:
    from change_point.model_factory import ModelFactory
    MODEL_FACTORY_AVAILABLE = True
except ImportError:
    MODEL_FACTORY_AVAILABLE = False


class TestBayesianChangePointDetector:
    """Test Bayesian change point detection."""
    
    def test_detector_initialization(self):
        """Test detector can be initialized."""
        if not BAYESIAN_AVAILABLE:
            pytest.skip("BayesianChangePointDetector not available")
        
        detector = BayesianChangePointDetector()
        assert detector is not None


class TestModelFactory:
    """Test ModelFactory functionality."""
    
    def test_model_factory_initialization(self):
        """Test ModelFactory can be initialized."""
        if not MODEL_FACTORY_AVAILABLE:
            pytest.skip("ModelFactory not available")
        
        factory = ModelFactory()
        assert factory is not None
