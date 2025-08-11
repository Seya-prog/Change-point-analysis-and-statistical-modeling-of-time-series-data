"""
Report Generator for Oil Price Analysis.

Creates comprehensive text and HTML reports from analysis results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class ReportGenerator:
    """
    Generate comprehensive analysis reports.
    
    Creates formatted reports with statistics, findings, and recommendations.
    """
    
    def __init__(self):
        """Initialize report generator."""
        pass
    
    def generate_summary_report(self, 
                              data: pd.DataFrame,
                              changepoints: List[datetime],
                              model_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            data: Price data DataFrame
            changepoints: List of detected change-points
            model_results: Optional model results dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("# Brent Oil Price Change-Point Analysis Report")
        report.append("=" * 60)
        report.append("")
        
        # Data summary
        report.append("## Data Summary")
        report.append(f"- **Analysis Period**: {data['date'].min()} to {data['date'].max()}")
        report.append(f"- **Total Observations**: {len(data):,}")
        report.append(f"- **Price Range**: ${data['price'].min():.2f} - ${data['price'].max():.2f}")
        report.append(f"- **Average Price**: ${data['price'].mean():.2f}")
        report.append(f"- **Price Volatility (Std)**: ${data['price'].std():.2f}")
        report.append("")
        
        # Change-point analysis
        report.append("## Change-Point Analysis")
        report.append(f"- **Number of Change-Points Detected**: {len(changepoints)}")
        
        if changepoints:
            report.append("- **Change-Point Dates**:")
            for i, cp in enumerate(changepoints, 1):
                report.append(f"  {i}. {cp.strftime('%Y-%m-%d')}")
            
            # Regime statistics
            report.append("")
            report.append("### Regime Statistics")
            
            # Calculate regime periods
            regime_periods = []
            start_date = data['date'].min()
            
            for cp in changepoints:
                regime_periods.append((start_date, cp))
                start_date = cp
            
            # Add final regime
            regime_periods.append((start_date, data['date'].max()))
            
            for i, (start, end) in enumerate(regime_periods, 1):
                regime_data = data[(data['date'] >= start) & (data['date'] <= end)]
                if len(regime_data) > 0:
                    duration = (end - start).days
                    mean_price = regime_data['price'].mean()
                    volatility = regime_data['price'].std()
                    
                    report.append(f"**Regime {i}** ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}):")
                    report.append(f"  - Duration: {duration} days")
                    report.append(f"  - Average Price: ${mean_price:.2f}")
                    report.append(f"  - Volatility: ${volatility:.2f}")
                    report.append("")
        
        # Model results (if available)
        if model_results:
            report.append("## Model Performance")
            
            if 'diagnostics' in model_results:
                diagnostics = model_results['diagnostics']
                report.append(f"- **Model Type**: {model_results.get('model_type', 'Unknown')}")
                
                if 'convergence' in diagnostics:
                    conv = diagnostics['convergence']
                    report.append(f"- **MCMC Convergence**: {'✓ Converged' if conv.get('converged', False) else '✗ Not Converged'}")
                    if 'r_hat_max' in conv:
                        report.append(f"- **Max R-hat**: {conv['r_hat_max']:.3f}")
                
                if 'model_fit' in diagnostics:
                    fit = diagnostics['model_fit']
                    if 'waic' in fit:
                        report.append(f"- **WAIC**: {fit['waic']:.2f}")
                    if 'loo' in fit:
                        report.append(f"- **LOO**: {fit['loo']:.2f}")
        
        report.append("")
        
        # Key findings
        report.append("## Key Findings")
        
        if len(changepoints) == 0:
            report.append("- No significant change-points detected in the analyzed period.")
            report.append("- Oil prices appear to follow a relatively stable pattern.")
        else:
            report.append(f"- {len(changepoints)} significant structural breaks identified in oil prices.")
            
            # Calculate price trends between change-points
            if len(changepoints) >= 2:
                first_regime = data[data['date'] <= changepoints[0]]['price']
                last_regime = data[data['date'] >= changepoints[-1]]['price']
                
                if len(first_regime) > 0 and len(last_regime) > 0:
                    price_change = last_regime.mean() - first_regime.mean()
                    change_pct = (price_change / first_regime.mean()) * 100
                    
                    trend = "increased" if price_change > 0 else "decreased"
                    report.append(f"- Overall price trend: {trend} by ${abs(price_change):.2f} ({abs(change_pct):.1f}%)")
        
        # Calculate volatility periods
        rolling_vol = data['price'].rolling(window=30).std()
        high_vol_threshold = rolling_vol.quantile(0.8)
        high_vol_periods = rolling_vol > high_vol_threshold
        
        if high_vol_periods.any():
            high_vol_count = high_vol_periods.sum()
            high_vol_pct = (high_vol_count / len(data)) * 100
            report.append(f"- High volatility periods: {high_vol_pct:.1f}% of the time")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("- **For Investors**: Consider the identified regime changes when making investment decisions.")
        report.append("- **For Risk Management**: Pay attention to high volatility periods around change-points.")
        report.append("- **For Policy Analysis**: Investigate geopolitical events coinciding with change-points.")
        
        if len(changepoints) > 0:
            report.append("- **Further Analysis**: Correlate change-points with historical geopolitical events.")
        
        report.append("")
        report.append("---")
        report.append(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(report)
    
    def generate_technical_report(self, model_results: Dict[str, Any]) -> str:
        """
        Generate technical model report.
        
        Args:
            model_results: Dictionary containing model results and diagnostics
            
        Returns:
            Technical report string
        """
        report = []
        report.append("# Technical Model Report")
        report.append("=" * 40)
        report.append("")
        
        # Model information
        report.append("## Model Configuration")
        report.append(f"- **Model Type**: {model_results.get('model_type', 'Unknown')}")
        report.append(f"- **Sampling Method**: {model_results.get('sampling_method', 'Unknown')}")
        
        if 'config' in model_results:
            config = model_results['config']
            report.append(f"- **Number of Samples**: {config.get('draws', 'Unknown')}")
            report.append(f"- **Number of Chains**: {config.get('chains', 'Unknown')}")
            report.append(f"- **Tune Steps**: {config.get('tune', 'Unknown')}")
        
        report.append("")
        
        # Diagnostics
        if 'diagnostics' in model_results:
            diagnostics = model_results['diagnostics']
            report.append("## Model Diagnostics")
            
            # Convergence
            if 'convergence' in diagnostics:
                conv = diagnostics['convergence']
                report.append("### Convergence Diagnostics")
                report.append(f"- **Converged**: {'Yes' if conv.get('converged', False) else 'No'}")
                
                if 'r_hat_max' in conv:
                    report.append(f"- **Maximum R-hat**: {conv['r_hat_max']:.4f}")
                if 'r_hat_mean' in conv:
                    report.append(f"- **Mean R-hat**: {conv['r_hat_mean']:.4f}")
                if 'ess_bulk_min' in conv:
                    report.append(f"- **Minimum Bulk ESS**: {conv['ess_bulk_min']:.0f}")
                if 'ess_tail_min' in conv:
                    report.append(f"- **Minimum Tail ESS**: {conv['ess_tail_min']:.0f}")
                
                report.append("")
            
            # Model fit
            if 'model_fit' in diagnostics:
                fit = diagnostics['model_fit']
                report.append("### Model Fit Statistics")
                
                if 'waic' in fit:
                    report.append(f"- **WAIC**: {fit['waic']:.2f}")
                if 'loo' in fit:
                    report.append(f"- **LOO**: {fit['loo']:.2f}")
                if 'bpic' in fit:
                    report.append(f"- **BPIC**: {fit['bpic']:.2f}")
                
                report.append("")
        
        # Parameter estimates
        if 'posterior_summary' in model_results:
            summary = model_results['posterior_summary']
            report.append("## Parameter Estimates")
            
            for param, stats in summary.items():
                if isinstance(stats, dict):
                    report.append(f"### {param}")
                    if 'mean' in stats:
                        report.append(f"- **Mean**: {stats['mean']:.4f}")
                    if 'std' in stats:
                        report.append(f"- **Std**: {stats['std']:.4f}")
                    if 'hdi_3%' in stats and 'hdi_97%' in stats:
                        report.append(f"- **94% HDI**: [{stats['hdi_3%']:.4f}, {stats['hdi_97%']:.4f}]")
                    report.append("")
        
        return "\n".join(report)
    
    def save_report(self, report_content: str, filename: str, format: str = 'txt'):
        """
        Save report to file.
        
        Args:
            report_content: Report content string
            filename: Output filename
            format: Output format ('txt' or 'html')
        """
        if format.lower() == 'html':
            # Convert markdown-style report to HTML
            html_content = self._markdown_to_html(report_content)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert simple markdown to HTML."""
        html = ["<html><head><title>Analysis Report</title></head><body>"]
        
        lines = markdown_text.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('# '):
                html.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith('## '):
                html.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith('### '):
                html.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith('- '):
                html.append(f"<li>{line[2:]}</li>")
            elif line.startswith('='):
                html.append("<hr>")
            elif line.startswith('---'):
                html.append("<hr>")
            elif line:
                html.append(f"<p>{line}</p>")
            else:
                html.append("<br>")
        
        html.append("</body></html>")
        return '\n'.join(html)
