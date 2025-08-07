"""
Bayesian Change-Point Detection Module

This module implements state-of-the-art Bayesian change-point detection
algorithms for analyzing Brent oil price structural breaks.
"""

from .bayesian_detector import BayesianChangePointDetector
from .model_factory import ModelFactory
from .diagnostics import ModelDiagnostics

__all__ = [
    "BayesianChangePointDetector",
    "ModelFactory", 
    "ModelDiagnostics"
]
