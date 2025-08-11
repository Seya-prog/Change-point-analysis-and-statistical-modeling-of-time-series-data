"""
Time Series Preprocessing for Oil Price Analysis.

Comprehensive preprocessing pipeline including stationarity tests,
volatility analysis, trend detection, and data transformation utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import warnings

# Statistical tests
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from scipy import stats
from loguru import logger


class TimeSeriesPreprocessor:
    """
    Comprehensive time series preprocessing for oil price analysis.
    
    Provides stationarity testing, volatility analysis, trend detection,
    outlier detection, and data transformation utilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.preprocessing_results = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration."""
        return {
            'stationarity': {
                'adf_alpha': 0.05,
                'kpss_alpha': 0.05,
                'max_diff_order': 2
            },
            'volatility': {
                'window_sizes': [20, 60, 252],
                'percentile_threshold': 95
            },
            'outliers': {
                'method': 'iqr',  # 'iqr', 'zscore', 'modified_zscore'
                'threshold': 3.0,
                'iqr_multiplier': 1.5
            },
            'trend': {
                'window_size': 252,  # 1 year for daily data
                'polynomial_degree': 2
            },
            'transformations': {
                'log_transform': True,
                'difference_order': 1,
                'standardize': False
            }
        }
    
    def run_full_preprocessing(self, data: pd.Series) -> Dict[str, Any]:
        """
        Run complete preprocessing pipeline.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with preprocessing results
        """
        logger.info("Starting comprehensive time series preprocessing...")
        
        results = {
            'original_data': data.copy(),
            'stationarity_tests': {},
            'volatility_analysis': {},
            'trend_analysis': {},
            'outlier_analysis': {},
            'transformations': {},
            'recommendations': []
        }
        
        # 1. Stationarity analysis
        results['stationarity_tests'] = self.test_stationarity(data)
        
        # 2. Volatility analysis
        results['volatility_analysis'] = self.analyze_volatility(data)
        
        # 3. Trend analysis
        results['trend_analysis'] = self.analyze_trend(data)
        
        # 4. Outlier detection
        results['outlier_analysis'] = self.detect_outliers(data)
        
        # 5. Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        self.preprocessing_results = results
        return results
    
    def test_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive stationarity testing.
        
        Args:
            data: Time series data
            
        Returns:
            Stationarity test results
        """
        logger.info("Testing stationarity...")
        
        results = {
            'is_stationary': False,
            'tests': {},
            'differencing_required': 0,
            'transformed_series': None
        }
        
        if not HAS_STATSMODELS:
            logger.warning("Statsmodels not available, using simplified stationarity test")
            # Simple variance-based test
            rolling_mean = data.rolling(window=252).mean()
            rolling_var = data.rolling(window=252).var()
            
            mean_stability = rolling_mean.std() / data.mean()
            var_stability = rolling_var.std() / data.var()
            
            results['tests']['simple_stability'] = {
                'mean_stability': float(mean_stability),
                'variance_stability': float(var_stability),
                'is_stable': mean_stability < 0.1 and var_stability < 0.2
            }
            
            results['is_stationary'] = results['tests']['simple_stability']['is_stable']
            return results
        
        # ADF Test
        try:
            adf_result = adfuller(data.dropna())
            results['tests']['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < self.config['stationarity']['adf_alpha']
            }
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            results['tests']['adf'] = {'error': str(e)}
        
        # KPSS Test
        try:
            kpss_result = kpss(data.dropna())
            results['tests']['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > self.config['stationarity']['kpss_alpha']
            }
        except Exception as e:
            logger.warning(f"KPSS test failed: {e}")
            results['tests']['kpss'] = {'error': str(e)}
        
        # Determine overall stationarity
        adf_stationary = results['tests'].get('adf', {}).get('is_stationary', False)
        kpss_stationary = results['tests'].get('kpss', {}).get('is_stationary', False)
        
        results['is_stationary'] = adf_stationary and kpss_stationary
        
        # Test differencing if not stationary
        if not results['is_stationary']:
            results['differencing_required'] = self._find_differencing_order(data)
            
            if results['differencing_required'] > 0:
                diff_data = data.copy()
                for i in range(results['differencing_required']):
                    diff_data = diff_data.diff().dropna()
                results['transformed_series'] = diff_data
        
        return results
    
    def analyze_volatility(self, data: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive volatility analysis.
        
        Args:
            data: Time series data
            
        Returns:
            Volatility analysis results
        """
        logger.info("Analyzing volatility patterns...")
        
        results = {
            'rolling_volatility': {},
            'volatility_clustering': {},
            'high_volatility_periods': [],
            'volatility_statistics': {}
        }
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Rolling volatility for different windows
        for window in self.config['volatility']['window_sizes']:
            if len(returns) >= window:
                rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
                results['rolling_volatility'][f'{window}d'] = {
                    'mean': float(rolling_vol.mean()),
                    'std': float(rolling_vol.std()),
                    'min': float(rolling_vol.min()),
                    'max': float(rolling_vol.max()),
                    'series': rolling_vol
                }
        
        # Volatility clustering analysis
        abs_returns = np.abs(returns)
        if len(abs_returns) > 1:
            # Autocorrelation of absolute returns (proxy for volatility clustering)
            autocorr_lags = min(20, len(abs_returns) // 4)
            autocorrs = [abs_returns.autocorr(lag=i) for i in range(1, autocorr_lags + 1)]
            
            results['volatility_clustering'] = {
                'autocorrelations': autocorrs,
                'significant_clustering': any(abs(ac) > 0.1 for ac in autocorrs if not pd.isna(ac))
            }
        
        # High volatility periods
        if '20d' in results['rolling_volatility']:
            vol_20d = results['rolling_volatility']['20d']['series']
            threshold = np.percentile(vol_20d.dropna(), self.config['volatility']['percentile_threshold'])
            
            high_vol_mask = vol_20d > threshold
            high_vol_periods = []
            
            # Find continuous high volatility periods
            in_period = False
            start_date = None
            
            for date, is_high_vol in high_vol_mask.items():
                if is_high_vol and not in_period:
                    start_date = date
                    in_period = True
                elif not is_high_vol and in_period:
                    high_vol_periods.append({
                        'start': start_date,
                        'end': date,
                        'duration': (date - start_date).days,
                        'max_volatility': float(vol_20d[start_date:date].max())
                    })
                    in_period = False
            
            results['high_volatility_periods'] = high_vol_periods
        
        # Overall volatility statistics
        results['volatility_statistics'] = {
            'annualized_volatility': float(returns.std() * np.sqrt(252)),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'jarque_bera_test': stats.jarque_bera(returns.dropna())._asdict() if len(returns.dropna()) > 0 else {}
        }
        
        return results
    
    def analyze_trend(self, data: pd.Series) -> Dict[str, Any]:
        """
        Trend analysis and decomposition.
        
        Args:
            data: Time series data
            
        Returns:
            Trend analysis results
        """
        logger.info("Analyzing trends and seasonality...")
        
        results = {
            'linear_trend': {},
            'polynomial_trend': {},
            'seasonal_decomposition': {},
            'trend_changes': []
        }
        
        # Linear trend
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data.values)
        
        results['linear_trend'] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing'
        }
        
        # Polynomial trend
        degree = self.config['trend']['polynomial_degree']
        poly_coeffs = np.polyfit(x, data.values, degree)
        poly_trend = np.polyval(poly_coeffs, x)
        
        results['polynomial_trend'] = {
            'coefficients': poly_coeffs.tolist(),
            'r_squared': float(1 - np.sum((data.values - poly_trend)**2) / np.sum((data.values - data.mean())**2))
        }
        
        # Seasonal decomposition (if enough data)
        if HAS_STATSMODELS and len(data) >= 2 * 252:  # At least 2 years of daily data
            try:
                decomposition = seasonal_decompose(data, model='additive', period=252)
                results['seasonal_decomposition'] = {
                    'trend_strength': float(1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.trend.dropna() + decomposition.resid.dropna())),
                    'seasonal_strength': float(1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.seasonal.dropna() + decomposition.resid.dropna())),
                    'has_seasonality': True
                }
            except Exception as e:
                logger.warning(f"Seasonal decomposition failed: {e}")
                results['seasonal_decomposition'] = {'error': str(e), 'has_seasonality': False}
        else:
            results['seasonal_decomposition'] = {'has_seasonality': False, 'reason': 'insufficient_data'}
        
        return results
    
    def detect_outliers(self, data: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive outlier detection.
        
        Args:
            data: Time series data
            
        Returns:
            Outlier detection results
        """
        logger.info("Detecting outliers...")
        
        results = {
            'method_used': self.config['outliers']['method'],
            'outliers': [],
            'outlier_statistics': {},
            'cleaned_data': None
        }
        
        method = self.config['outliers']['method']
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            multiplier = self.config['outliers']['iqr_multiplier']
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            threshold = self.config['outliers']['threshold']
            outlier_mask = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            threshold = self.config['outliers']['threshold']
            outlier_mask = np.abs(modified_z_scores) > threshold
            
        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            return results
        
        # Extract outlier information
        outliers = data[outlier_mask]
        results['outliers'] = [
            {
                'date': str(date),
                'value': float(value),
                'deviation': float(value - data.median())
            }
            for date, value in outliers.items()
        ]
        
        results['outlier_statistics'] = {
            'total_outliers': len(outliers),
            'outlier_percentage': float(len(outliers) / len(data) * 100),
            'outlier_dates': [str(date) for date in outliers.index]
        }
        
        # Create cleaned data
        results['cleaned_data'] = data[~outlier_mask]
        
        return results
    
    def _find_differencing_order(self, data: pd.Series, max_order: int = None) -> int:
        """Find optimal differencing order for stationarity."""
        max_order = max_order or self.config['stationarity']['max_diff_order']
        
        for order in range(1, max_order + 1):
            diff_data = data.copy()
            for i in range(order):
                diff_data = diff_data.diff().dropna()
            
            if len(diff_data) < 50:  # Not enough data
                break
                
            # Simple stationarity test
            if HAS_STATSMODELS:
                try:
                    adf_result = adfuller(diff_data)
                    if adf_result[1] < self.config['stationarity']['adf_alpha']:
                        return order
                except:
                    pass
            else:
                # Simple variance test
                rolling_var = diff_data.rolling(window=min(20, len(diff_data)//4)).var()
                if rolling_var.std() / rolling_var.mean() < 0.5:  # Stable variance
                    return order
        
        return max_order
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate preprocessing recommendations based on analysis."""
        recommendations = []
        
        # Stationarity recommendations
        if not results['stationarity_tests']['is_stationary']:
            diff_order = results['stationarity_tests']['differencing_required']
            if diff_order > 0:
                recommendations.append(f"Apply {diff_order}-order differencing to achieve stationarity")
            else:
                recommendations.append("Consider log transformation or detrending")
        
        # Volatility recommendations
        vol_analysis = results['volatility_analysis']
        if vol_analysis.get('volatility_clustering', {}).get('significant_clustering', False):
            recommendations.append("Consider GARCH modeling for volatility clustering")
        
        if len(vol_analysis.get('high_volatility_periods', [])) > 0:
            recommendations.append("Investigate high volatility periods for structural breaks")
        
        # Trend recommendations
        trend_analysis = results['trend_analysis']
        if trend_analysis['linear_trend']['is_significant']:
            direction = trend_analysis['linear_trend']['trend_direction']
            recommendations.append(f"Strong {direction} trend detected - consider detrending")
        
        # Outlier recommendations
        outlier_pct = results['outlier_analysis']['outlier_statistics']['outlier_percentage']
        if outlier_pct > 5:
            recommendations.append(f"High outlier percentage ({outlier_pct:.1f}%) - consider outlier treatment")
        
        return recommendations
    
    def apply_transformations(self, data: pd.Series, 
                            transformations: Optional[List[str]] = None) -> pd.Series:
        """
        Apply specified transformations to the data.
        
        Args:
            data: Input time series
            transformations: List of transformations to apply
            
        Returns:
            Transformed time series
        """
        if transformations is None:
            transformations = []
            if self.config['transformations']['log_transform']:
                transformations.append('log')
            if self.config['transformations']['difference_order'] > 0:
                transformations.append('difference')
            if self.config['transformations']['standardize']:
                transformations.append('standardize')
        
        transformed_data = data.copy()
        
        for transform in transformations:
            if transform == 'log':
                # Ensure positive values
                if (transformed_data <= 0).any():
                    min_val = transformed_data.min()
                    transformed_data = transformed_data - min_val + 1
                transformed_data = np.log(transformed_data)
                
            elif transform == 'difference':
                order = self.config['transformations']['difference_order']
                for _ in range(order):
                    transformed_data = transformed_data.diff().dropna()
                    
            elif transform == 'standardize':
                transformed_data = (transformed_data - transformed_data.mean()) / transformed_data.std()
                
            elif transform == 'normalize':
                transformed_data = (transformed_data - transformed_data.min()) / (transformed_data.max() - transformed_data.min())
        
        return transformed_data
