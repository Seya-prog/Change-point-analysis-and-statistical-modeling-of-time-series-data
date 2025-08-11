"""
Model Diagnostics and Validation for Change-Point Detection.

Comprehensive diagnostics suite for validating Bayesian change-point models
including convergence checks, posterior predictive checks, and model comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# PyMC imports with fallback
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


class ModelDiagnostics:
    """
    Comprehensive diagnostics for change-point detection models.
    
    Provides tools for:
    - MCMC convergence assessment
    - Posterior predictive checks
    - Model comparison and selection
    - Change-point validation
    - Regime stability analysis
    """
    
    def __init__(self, model_results: Dict[str, Any], original_data: pd.Series):
        """
        Initialize diagnostics with model results and original data.
        
        Args:
            model_results: Results from fitted change-point model
            original_data: Original time series data
        """
        self.results = model_results
        self.data = original_data
        self.diagnostics = {}
        
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """
        Run complete diagnostic suite.
        
        Returns:
            Dictionary with all diagnostic results
        """
        logger.info("Running comprehensive model diagnostics...")
        
        diagnostics = {}
        
        # MCMC diagnostics (if available)
        if 'summary_stats' in self.results:
            diagnostics['mcmc'] = self._mcmc_diagnostics()
        
        # Model fit diagnostics
        diagnostics['fit'] = self._model_fit_diagnostics()
        
        # Change-point validation
        diagnostics['change_points'] = self._change_point_diagnostics()
        
        # Regime analysis
        diagnostics['regimes'] = self._regime_diagnostics()
        
        # Posterior predictive checks
        if PYMC_AVAILABLE:
            diagnostics['posterior_predictive'] = self._posterior_predictive_checks()
        
        # Overall model assessment
        diagnostics['overall'] = self._overall_assessment(diagnostics)
        
        self.diagnostics = diagnostics
        return diagnostics
    
    def _mcmc_diagnostics(self) -> Dict[str, Any]:
        """MCMC-specific diagnostics."""
        logger.info("Running MCMC diagnostics...")
        
        mcmc_diag = {}
        
        if 'model_diagnostics' in self.results:
            model_diag = self.results['model_diagnostics']
            
            mcmc_diag.update({
                'rhat_assessment': {
                    'max_rhat': model_diag.get('rhat_max', None),
                    'mean_rhat': model_diag.get('rhat_mean', None),
                    'rhat_ok': model_diag.get('rhat_max', 2.0) < 1.1,
                    'interpretation': self._interpret_rhat(model_diag.get('rhat_max', 2.0))
                },
                'effective_sample_size': {
                    'min_ess': model_diag.get('ess_min', None),
                    'mean_ess': model_diag.get('ess_mean', None),
                    'ess_ok': model_diag.get('ess_min', 0) > 400,
                    'interpretation': self._interpret_ess(model_diag.get('ess_min', 0))
                },
                'energy_diagnostics': {
                    'bfmi': model_diag.get('bfmi', None),
                    'energy_ok': model_diag.get('bfmi', 0) > 0.2,
                    'interpretation': self._interpret_energy(model_diag.get('bfmi', 0))
                },
                'overall_convergence': model_diag.get('converged', False)
            })
        
        return mcmc_diag
    
    def _model_fit_diagnostics(self) -> Dict[str, Any]:
        """Model fit quality diagnostics."""
        logger.info("Assessing model fit quality...")
        
        fit_diag = {}
        
        # Basic fit metrics
        if 'method' in self.results and self.results['method'] != 'fallback':
            # For Bayesian models, we'd need posterior predictive samples
            # For now, provide placeholder structure
            fit_diag['residual_analysis'] = self._residual_analysis()
            fit_diag['goodness_of_fit'] = self._goodness_of_fit()
        else:
            # Fallback diagnostics
            fit_diag['basic_metrics'] = self._basic_fit_metrics()
        
        return fit_diag
    
    def _change_point_diagnostics(self) -> Dict[str, Any]:
        """Change-point specific diagnostics."""
        logger.info("Analyzing change-point detection quality...")
        
        cp_diag = {}
        
        if 'change_point_probabilities' in self.results:
            probs = self.results['change_point_probabilities']
            
            # Find significant change points
            threshold = 0.5  # Configurable threshold
            significant_cps = np.where(probs > threshold)[0]
            
            cp_diag.update({
                'n_significant_changepoints': len(significant_cps),
                'changepoint_locations': significant_cps.tolist(),
                'max_probability': float(np.max(probs)),
                'mean_probability': float(np.mean(probs)),
                'changepoint_strength': self._assess_changepoint_strength(probs),
                'temporal_distribution': self._analyze_temporal_distribution(significant_cps)
            })
            
        elif 'change_points' in self.results:
            # Fallback method results
            cps = self.results['change_points']
            cp_diag.update({
                'n_changepoints': len(cps),
                'changepoint_locations': cps,
                'method': 'fallback'
            })
        
        return cp_diag
    
    def _regime_diagnostics(self) -> Dict[str, Any]:
        """Regime analysis diagnostics."""
        logger.info("Analyzing regime characteristics...")
        
        regime_diag = {}
        
        if 'regime_means' in self.results and 'regime_volatilities' in self.results:
            means = self.results['regime_means']
            vols = self.results['regime_volatilities']
            
            regime_diag.update({
                'regime_stability': self._assess_regime_stability(means, vols),
                'regime_differences': self._assess_regime_differences(means, vols),
                'volatility_clustering': self._assess_volatility_clustering(vols)
            })
        
        return regime_diag
    
    def _posterior_predictive_checks(self) -> Dict[str, Any]:
        """Posterior predictive checks."""
        logger.info("Running posterior predictive checks...")
        
        # Placeholder for posterior predictive checks
        # In practice, this would involve sampling from the posterior predictive distribution
        
        return {
            'ppc_summary': "Posterior predictive checks require full model trace",
            'implemented': False
        }
    
    def _overall_assessment(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Overall model assessment."""
        logger.info("Computing overall model assessment...")
        
        assessment = {
            'model_quality': 'unknown',
            'recommendations': [],
            'warnings': [],
            'confidence_level': 'medium'
        }
        
        # MCMC assessment
        if 'mcmc' in diagnostics:
            mcmc = diagnostics['mcmc']
            if not mcmc.get('overall_convergence', False):
                assessment['warnings'].append("MCMC convergence issues detected")
                assessment['recommendations'].append("Increase number of samples or tune MCMC parameters")
        
        # Change-point assessment
        if 'change_points' in diagnostics:
            cp = diagnostics['change_points']
            n_cps = cp.get('n_significant_changepoints', cp.get('n_changepoints', 0))
            
            if n_cps == 0:
                assessment['warnings'].append("No significant change points detected")
            elif n_cps > 20:
                assessment['warnings'].append("Very high number of change points - possible overfitting")
                assessment['recommendations'].append("Consider reducing model complexity")
        
        # Determine overall quality
        n_warnings = len(assessment['warnings'])
        if n_warnings == 0:
            assessment['model_quality'] = 'good'
            assessment['confidence_level'] = 'high'
        elif n_warnings <= 2:
            assessment['model_quality'] = 'acceptable'
            assessment['confidence_level'] = 'medium'
        else:
            assessment['model_quality'] = 'poor'
            assessment['confidence_level'] = 'low'
            assessment['recommendations'].append("Consider alternative model specification")
        
        return assessment
    
    def _interpret_rhat(self, rhat: float) -> str:
        """Interpret R-hat convergence diagnostic."""
        if rhat < 1.01:
            return "Excellent convergence"
        elif rhat < 1.05:
            return "Good convergence"
        elif rhat < 1.1:
            return "Acceptable convergence"
        else:
            return "Poor convergence - increase samples"
    
    def _interpret_ess(self, ess: float) -> str:
        """Interpret effective sample size."""
        if ess > 1000:
            return "Excellent effective sample size"
        elif ess > 400:
            return "Good effective sample size"
        elif ess > 100:
            return "Acceptable effective sample size"
        else:
            return "Low effective sample size - increase samples"
    
    def _interpret_energy(self, bfmi: float) -> str:
        """Interpret energy diagnostics."""
        if bfmi > 0.3:
            return "Good energy diagnostics"
        elif bfmi > 0.2:
            return "Acceptable energy diagnostics"
        else:
            return "Poor energy diagnostics - check model specification"
    
    def _residual_analysis(self) -> Dict[str, Any]:
        """Analyze model residuals."""
        # Placeholder - would need posterior predictive samples
        return {
            'residual_autocorrelation': None,
            'residual_normality': None,
            'heteroscedasticity_test': None
        }
    
    def _goodness_of_fit(self) -> Dict[str, Any]:
        """Goodness of fit metrics."""
        # Placeholder - would need posterior predictive samples
        return {
            'waic': None,
            'loo': None,
            'posterior_predictive_p_value': None
        }
    
    def _basic_fit_metrics(self) -> Dict[str, Any]:
        """Basic fit metrics for fallback methods."""
        if 'change_points' not in self.results:
            return {}
        
        # Simple piecewise constant fit evaluation
        change_points = self.results['change_points']
        data_values = self.data.values
        
        # Create piecewise constant prediction
        prediction = np.zeros_like(data_values)
        
        if len(change_points) == 0:
            prediction[:] = np.mean(data_values)
        else:
            # First segment
            prediction[:change_points[0]] = np.mean(data_values[:change_points[0]])
            
            # Middle segments
            for i in range(len(change_points) - 1):
                start, end = change_points[i], change_points[i + 1]
                prediction[start:end] = np.mean(data_values[start:end])
            
            # Last segment
            prediction[change_points[-1]:] = np.mean(data_values[change_points[-1]:])
        
        # Calculate metrics
        mse = mean_squared_error(data_values, prediction)
        mae = mean_absolute_error(data_values, prediction)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'r_squared': float(1 - np.var(data_values - prediction) / np.var(data_values))
        }
    
    def _assess_changepoint_strength(self, probabilities: np.ndarray) -> Dict[str, Any]:
        """Assess the strength of detected change points."""
        return {
            'max_strength': float(np.max(probabilities)),
            'n_strong_changepoints': int(np.sum(probabilities > 0.7)),
            'n_moderate_changepoints': int(np.sum((probabilities > 0.3) & (probabilities <= 0.7))),
            'strength_distribution': {
                'mean': float(np.mean(probabilities)),
                'std': float(np.std(probabilities)),
                'skewness': float(stats.skew(probabilities))
            }
        }
    
    def _analyze_temporal_distribution(self, change_points: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal distribution of change points."""
        if len(change_points) < 2:
            return {'spacing': 'insufficient_data'}
        
        spacings = np.diff(change_points)
        
        return {
            'mean_spacing': float(np.mean(spacings)),
            'std_spacing': float(np.std(spacings)),
            'min_spacing': int(np.min(spacings)),
            'max_spacing': int(np.max(spacings)),
            'clustering_coefficient': float(np.std(spacings) / np.mean(spacings))
        }
    
    def _assess_regime_stability(self, means: Dict[str, Any], volatilities: Dict[str, Any]) -> Dict[str, Any]:
        """Assess stability of detected regimes."""
        if 'std' not in means or 'std' not in volatilities:
            return {'assessment': 'insufficient_data'}
        
        mean_stability = np.mean(means['std']) / np.mean(np.abs(means['mean']))
        vol_stability = np.mean(volatilities['std']) / np.mean(volatilities['mean'])
        
        return {
            'mean_stability_ratio': float(mean_stability),
            'volatility_stability_ratio': float(vol_stability),
            'overall_stability': 'high' if mean_stability < 0.1 and vol_stability < 0.2 else 'moderate' if mean_stability < 0.3 and vol_stability < 0.5 else 'low'
        }
    
    def _assess_regime_differences(self, means: Dict[str, Any], volatilities: Dict[str, Any]) -> Dict[str, Any]:
        """Assess differences between regimes."""
        if 'mean' not in means or 'mean' not in volatilities:
            return {'assessment': 'insufficient_data'}
        
        mean_range = np.max(means['mean']) - np.min(means['mean'])
        vol_range = np.max(volatilities['mean']) - np.min(volatilities['mean'])
        
        return {
            'mean_range': float(mean_range),
            'volatility_range': float(vol_range),
            'regime_separation': 'high' if mean_range > 10 else 'moderate' if mean_range > 5 else 'low'
        }
    
    def _assess_volatility_clustering(self, volatilities: Dict[str, Any]) -> Dict[str, Any]:
        """Assess volatility clustering patterns."""
        if 'mean' not in volatilities:
            return {'assessment': 'insufficient_data'}
        
        vol_means = volatilities['mean']
        
        return {
            'volatility_ratio': float(np.max(vol_means) / np.min(vol_means)) if np.min(vol_means) > 0 else float('inf'),
            'volatility_clustering': 'high' if np.max(vol_means) / np.min(vol_means) > 3 else 'moderate' if np.max(vol_means) / np.min(vol_means) > 1.5 else 'low'
        }
    
    def generate_diagnostic_report(self) -> str:
        """Generate a comprehensive diagnostic report."""
        if not self.diagnostics:
            self.run_full_diagnostics()
        
        report = []
        report.append("# Change-Point Model Diagnostic Report")
        report.append("=" * 50)
        report.append("")
        
        # Overall assessment
        overall = self.diagnostics.get('overall', {})
        report.append(f"**Overall Model Quality:** {overall.get('model_quality', 'unknown').upper()}")
        report.append(f"**Confidence Level:** {overall.get('confidence_level', 'unknown').upper()}")
        report.append("")
        
        # Warnings
        warnings = overall.get('warnings', [])
        if warnings:
            report.append("## ⚠️ Warnings")
            for warning in warnings:
                report.append(f"- {warning}")
            report.append("")
        
        # Recommendations
        recommendations = overall.get('recommendations', [])
        if recommendations:
            report.append("## 💡 Recommendations")
            for rec in recommendations:
                report.append(f"- {rec}")
            report.append("")
        
        # Change-point summary
        cp_diag = self.diagnostics.get('change_points', {})
        if cp_diag:
            report.append("## 📊 Change-Point Analysis")
            n_cps = cp_diag.get('n_significant_changepoints', cp_diag.get('n_changepoints', 0))
            report.append(f"- **Number of Change Points:** {n_cps}")
            
            if 'max_probability' in cp_diag:
                report.append(f"- **Maximum Probability:** {cp_diag['max_probability']:.3f}")
            
            report.append("")
        
        # MCMC diagnostics
        mcmc_diag = self.diagnostics.get('mcmc', {})
        if mcmc_diag:
            report.append("## 🔄 MCMC Diagnostics")
            
            rhat = mcmc_diag.get('rhat_assessment', {})
            if rhat:
                report.append(f"- **R-hat:** {rhat.get('max_rhat', 'N/A')} ({rhat.get('interpretation', 'N/A')})")
            
            ess = mcmc_diag.get('effective_sample_size', {})
            if ess:
                report.append(f"- **Effective Sample Size:** {ess.get('min_ess', 'N/A')} ({ess.get('interpretation', 'N/A')})")
            
            report.append("")
        
        return "\n".join(report)
    
    def save_diagnostics(self, filepath: str) -> None:
        """Save diagnostics to file."""
        if not self.diagnostics:
            self.run_full_diagnostics()
        
        # Save as JSON
        import json
        with open(filepath, 'w') as f:
            json.dump(self.diagnostics, f, indent=2, default=str)
        
        logger.info(f"Diagnostics saved to {filepath}")
        
        # Also save report
        report_path = filepath.replace('.json', '_report.md')
        with open(report_path, 'w') as f:
            f.write(self.generate_diagnostic_report())
        
        logger.info(f"Diagnostic report saved to {report_path}")
