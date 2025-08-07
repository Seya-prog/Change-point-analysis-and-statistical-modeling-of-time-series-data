"""
Dashboard Builder for Oil Price Analysis.

Simple dashboard components using matplotlib for visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from datetime import datetime


class DashboardBuilder:
    """
    Simple dashboard builder using matplotlib.
    
    Creates multi-panel visualizations for comprehensive analysis.
    """
    
    def __init__(self, figsize=(15, 10)):
        """Initialize dashboard builder."""
        self.figsize = figsize
    
    def create_summary_dashboard(self, data: pd.DataFrame, 
                               changepoints: List[datetime],
                               title: str = "Brent Oil Analysis Dashboard") -> plt.Figure:
        """
        Create a comprehensive summary dashboard.
        
        Args:
            data: Price data DataFrame
            changepoints: List of detected change-points
            title: Dashboard title
            
        Returns:
            Matplotlib figure with multiple subplots
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Price series with change-points
        ax1 = axes[0, 0]
        ax1.plot(data['date'], data['price'], color='blue', linewidth=1.5)
        for cp in changepoints:
            ax1.axvline(cp, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('Price Series with Change-Points')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD/barrel)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Price distribution
        ax2 = axes[0, 1]
        ax2.hist(data['price'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.set_title('Price Distribution')
        ax2.set_xlabel('Price (USD/barrel)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling volatility
        ax3 = axes[1, 0]
        rolling_vol = data['price'].rolling(window=30).std()
        ax3.plot(data['date'], rolling_vol, color='orange', linewidth=1.5)
        ax3.set_title('30-Day Rolling Volatility')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Volatility')
        ax3.grid(True, alpha=0.3)
        
        # 4. Price changes
        ax4 = axes[1, 1]
        price_changes = data['price'].pct_change().dropna()
        ax4.hist(price_changes, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_title('Daily Price Changes Distribution')
        ax4.set_xlabel('Price Change (%)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_regime_dashboard(self, data: pd.DataFrame,
                              regime_labels: np.ndarray,
                              title: str = "Regime Analysis Dashboard") -> plt.Figure:
        """
        Create regime analysis dashboard.
        
        Args:
            data: Price data DataFrame
            regime_labels: Array of regime labels
            title: Dashboard title
            
        Returns:
            Matplotlib figure with regime analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        unique_regimes = np.unique(regime_labels)
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        
        # 1. Regimes over time
        ax1 = axes[0, 0]
        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_data = data[mask]
            ax1.scatter(regime_data['date'], regime_data['price'],
                       c=colors[regime % len(colors)], s=10, alpha=0.6,
                       label=f'Regime {regime + 1}')
        ax1.set_title('Market Regimes Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD/barrel)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Regime statistics
        ax2 = axes[0, 1]
        regime_means = [data[regime_labels == r]['price'].mean() for r in unique_regimes]
        regime_stds = [data[regime_labels == r]['price'].std() for r in unique_regimes]
        
        x_pos = np.arange(len(unique_regimes))
        ax2.bar(x_pos, regime_means, yerr=regime_stds, alpha=0.7, 
               color=[colors[i % len(colors)] for i in range(len(unique_regimes))])
        ax2.set_title('Regime Statistics (Mean ± Std)')
        ax2.set_xlabel('Regime')
        ax2.set_ylabel('Price (USD/barrel)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'Regime {r+1}' for r in unique_regimes])
        ax2.grid(True, alpha=0.3)
        
        # 3. Regime durations
        ax3 = axes[1, 0]
        regime_changes = np.diff(regime_labels, prepend=regime_labels[0])
        change_points = np.where(regime_changes != 0)[0]
        
        if len(change_points) > 1:
            durations = np.diff(change_points)
            ax3.hist(durations, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_title('Regime Duration Distribution')
        ax3.set_xlabel('Duration (days)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 4. Regime transition matrix (if applicable)
        ax4 = axes[1, 1]
        if len(unique_regimes) > 1:
            # Simple transition counting
            transitions = np.zeros((len(unique_regimes), len(unique_regimes)))
            for i in range(1, len(regime_labels)):
                from_regime = regime_labels[i-1]
                to_regime = regime_labels[i]
                transitions[from_regime, to_regime] += 1
            
            # Normalize
            row_sums = transitions.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            transitions = transitions / row_sums
            
            im = ax4.imshow(transitions, cmap='Blues', aspect='auto')
            ax4.set_title('Regime Transition Probabilities')
            ax4.set_xlabel('To Regime')
            ax4.set_ylabel('From Regime')
            ax4.set_xticks(range(len(unique_regimes)))
            ax4.set_yticks(range(len(unique_regimes)))
            ax4.set_xticklabels([f'R{r+1}' for r in unique_regimes])
            ax4.set_yticklabels([f'R{r+1}' for r in unique_regimes])
            
            # Add text annotations
            for i in range(len(unique_regimes)):
                for j in range(len(unique_regimes)):
                    ax4.text(j, i, f'{transitions[i, j]:.2f}', 
                           ha='center', va='center', color='white' if transitions[i, j] > 0.5 else 'black')
        
        plt.tight_layout()
        return fig
