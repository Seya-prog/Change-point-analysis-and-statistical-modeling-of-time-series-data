"""
Simple plotting utilities for time series analysis.

Matplotlib-only implementation for basic visualization needs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from datetime import datetime

# Set basic style
plt.style.use('default')


class TimeSeriesPlotter:
    """Simple time series plotter using matplotlib."""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
    
    def plot_price_series(self, data: pd.DataFrame, title: str = "Brent Oil Prices"):
        """Plot basic price time series."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(data['date'], data['price'], color='blue', linewidth=1.5)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD/barrel)', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


class ChangePointPlotter:
    """Simple change-point plotter."""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
    
    def plot_changepoints(self, data: pd.DataFrame, changepoints: List[datetime], 
                         title: str = "Detected Change-Points"):
        """Plot time series with change-points."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(data['date'], data['price'], color='blue', linewidth=1.5, label='Price')
        
        for i, cp in enumerate(changepoints):
            ax.axvline(cp, color='red', linestyle='--', linewidth=2, alpha=0.7,
                      label='Change-Point' if i == 0 else '')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD/barrel)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig


class RegimePlotter:
    """Simple regime plotter."""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = ['red', 'green', 'blue', 'purple', 'orange']
    
    def plot_regimes(self, data: pd.DataFrame, regime_labels: np.ndarray,
                    title: str = "Market Regimes"):
        """Plot time series colored by regime."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        unique_regimes = np.unique(regime_labels)
        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_data = data[mask]
            ax.scatter(regime_data['date'], regime_data['price'],
                      c=self.colors[regime % len(self.colors)],
                      s=10, alpha=0.6, label=f'Regime {regime + 1}')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD/barrel)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig
