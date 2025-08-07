"""
Change-Point Analysis and Statistical Modeling of Time Series Data

This package provides tools for detecting change points in time series data
and applying various statistical models for temporal analysis.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import data_processing
from . import change_point
from . import visualization
from . import utils

__all__ = [
    "data_processing",
    "change_point", 
    "visualization",
    "utils"
]
