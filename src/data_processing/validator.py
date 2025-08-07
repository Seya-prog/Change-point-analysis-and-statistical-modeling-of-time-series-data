"""
Data Validation for Oil Price Analysis.

Comprehensive validation using Pydantic models for data quality assurance
and schema validation of price and event data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, field_validator, Field
from pydantic import ConfigDict
import re
from datetime import datetime, date
from loguru import logger


class PriceDataSchema(BaseModel):
    """Schema for oil price data validation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    date: Union[datetime, date]
    price: float = Field(gt=0, description="Price must be positive")
    
    @field_validator('price')
    @classmethod
    def validate_price_range(cls, v):
        """Validate price is within reasonable range."""
        if v < 0.1 or v > 1000:
            raise ValueError(f"Price {v} is outside reasonable range (0.1-1000)")
        return v
    
    @field_validator('date')
    @classmethod
    def validate_date_range(cls, v):
        """Validate date is within expected range."""
        min_date = datetime(1980, 1, 1)
        max_date = datetime(2030, 12, 31)
        
        if isinstance(v, date):
            v = datetime.combine(v, datetime.min.time())
        
        if v < min_date or v > max_date:
            raise ValueError(f"Date {v} is outside expected range ({min_date}-{max_date})")
        return v


class EventDataSchema(BaseModel):
    """Schema for geopolitical event data validation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    date: Union[datetime, date]
    event: str = Field(min_length=5, description="Event description must be at least 5 characters")
    category: str = Field(pattern=r'^(war|sanctions|opec|economic|political|other)$')
    impact: str = Field(pattern=r'^(high|medium|low)$')
    description: Optional[str] = None
    
    @field_validator('event')
    @classmethod
    def validate_event_description(cls, v):
        """Validate event description is meaningful."""
        if len(v.strip()) < 5:
            raise ValueError("Event description too short")
        return v.strip()
    
    @field_validator('date')
    @classmethod
    def validate_event_date(cls, v):
        """Validate event date."""
        min_date = datetime(1980, 1, 1)
        max_date = datetime.now()
        
        if isinstance(v, date):
            v = datetime.combine(v, datetime.min.time())
        
        if v < min_date or v > max_date:
            raise ValueError(f"Event date {v} is outside valid range")
        return v


class DataValidator:
    """
    Comprehensive data validator for oil price analysis.
    
    Validates data quality, schema compliance, and business rules
    for both price data and geopolitical events.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.validation_results = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            'price_validation': {
                'min_price': 0.1,
                'max_price': 1000.0,
                'max_missing_percentage': 5.0,
                'max_consecutive_missing': 7,
                'outlier_threshold': 5.0  # Standard deviations
            },
            'event_validation': {
                'required_categories': ['war', 'sanctions', 'opec', 'economic', 'political'],
                'min_events': 5,
                'max_events': 100
            },
            'temporal_validation': {
                'min_date': datetime(1980, 1, 1),
                'max_date': datetime(2030, 12, 31),
                'expected_frequency': 'D'  # Daily
            }
        }
    
    def validate_price_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate oil price data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Validation results dictionary
        """
        logger.info("Validating price data...")
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'schema_validation': []
        }
        
        try:
            # Basic structure validation
            if data.empty:
                results['errors'].append("Data is empty")
                results['valid'] = False
                return results
            
            # Required columns
            required_cols = ['date', 'price']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                results['errors'].append(f"Missing required columns: {missing_cols}")
                results['valid'] = False
                return results
            
            # Schema validation using Pydantic
            schema_errors = []
            valid_rows = 0
            
            for idx, row in data.iterrows():
                try:
                    PriceDataSchema(date=row['date'], price=row['price'])
                    valid_rows += 1
                except Exception as e:
                    schema_errors.append(f"Row {idx}: {str(e)}")
            
            results['schema_validation'] = schema_errors
            if len(schema_errors) > len(data) * 0.1:  # More than 10% errors
                results['errors'].append(f"Too many schema validation errors: {len(schema_errors)}")
                results['valid'] = False
            
            # Data quality checks
            price_series = pd.to_numeric(data['price'], errors='coerce')
            
            # Missing values
            missing_count = price_series.isnull().sum()
            missing_percentage = (missing_count / len(data)) * 100
            
            if missing_percentage > self.config['price_validation']['max_missing_percentage']:
                results['errors'].append(f"Too many missing values: {missing_percentage:.1f}%")
                results['valid'] = False
            elif missing_count > 0:
                results['warnings'].append(f"Found {missing_count} missing values ({missing_percentage:.1f}%)")
            
            # Price range validation
            valid_prices = price_series.dropna()
            if len(valid_prices) > 0:
                min_price = valid_prices.min()
                max_price = valid_prices.max()
                
                config_min = self.config['price_validation']['min_price']
                config_max = self.config['price_validation']['max_price']
                
                if min_price < config_min:
                    results['warnings'].append(f"Minimum price {min_price:.2f} below expected range")
                if max_price > config_max:
                    results['warnings'].append(f"Maximum price {max_price:.2f} above expected range")
            
            # Outlier detection
            if len(valid_prices) > 10:
                z_scores = np.abs((valid_prices - valid_prices.mean()) / valid_prices.std())
                outliers = z_scores > self.config['price_validation']['outlier_threshold']
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    outlier_percentage = (outlier_count / len(valid_prices)) * 100
                    if outlier_percentage > 5:
                        results['warnings'].append(f"High number of outliers: {outlier_count} ({outlier_percentage:.1f}%)")
                    else:
                        results['warnings'].append(f"Found {outlier_count} potential outliers")
            
            # Temporal validation
            if 'date' in data.columns:
                date_series = pd.to_datetime(data['date'], errors='coerce')
                
                # Date range
                min_date = date_series.min()
                max_date = date_series.max()
                
                config_min_date = self.config['temporal_validation']['min_date']
                config_max_date = self.config['temporal_validation']['max_date']
                
                if min_date < config_min_date:
                    results['warnings'].append(f"Data starts before expected range: {min_date}")
                if max_date > config_max_date:
                    results['warnings'].append(f"Data extends beyond expected range: {max_date}")
                
                # Frequency analysis
                if len(date_series.dropna()) > 1:
                    date_diffs = date_series.dropna().diff().dropna()
                    most_common_diff = date_diffs.mode()
                    
                    if len(most_common_diff) > 0:
                        freq_days = most_common_diff.iloc[0].days
                        if freq_days != 1:  # Not daily
                            results['warnings'].append(f"Data frequency appears to be {freq_days} days, expected daily")
            
            # Statistics
            if len(valid_prices) > 0:
                results['statistics'] = {
                    'total_observations': len(data),
                    'valid_prices': len(valid_prices),
                    'missing_values': int(missing_count),
                    'missing_percentage': float(missing_percentage),
                    'price_range': {
                        'min': float(valid_prices.min()),
                        'max': float(valid_prices.max()),
                        'mean': float(valid_prices.mean()),
                        'std': float(valid_prices.std())
                    },
                    'date_range': {
                        'start': str(date_series.min()) if not date_series.empty else None,
                        'end': str(date_series.max()) if not date_series.empty else None
                    }
                }
            
        except Exception as e:
            results['errors'].append(f"Validation error: {str(e)}")
            results['valid'] = False
        
        self.validation_results['price_data'] = results
        return results
    
    def validate_event_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate geopolitical event data.
        
        Args:
            data: DataFrame with event data
            
        Returns:
            Validation results dictionary
        """
        logger.info("Validating event data...")
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'schema_validation': []
        }
        
        try:
            if data.empty:
                results['warnings'].append("No event data provided")
                return results
            
            # Required columns
            required_cols = ['date', 'event', 'category', 'impact']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                results['errors'].append(f"Missing required columns: {missing_cols}")
                results['valid'] = False
                return results
            
            # Schema validation
            schema_errors = []
            valid_rows = 0
            
            for idx, row in data.iterrows():
                try:
                    EventDataSchema(
                        date=row['date'],
                        event=row['event'],
                        category=row['category'],
                        impact=row['impact'],
                        description=row.get('description', None)
                    )
                    valid_rows += 1
                except Exception as e:
                    schema_errors.append(f"Row {idx}: {str(e)}")
            
            results['schema_validation'] = schema_errors
            if len(schema_errors) > len(data) * 0.2:  # More than 20% errors
                results['errors'].append(f"Too many schema validation errors: {len(schema_errors)}")
                results['valid'] = False
            
            # Business rule validation
            min_events = self.config['event_validation']['min_events']
            max_events = self.config['event_validation']['max_events']
            
            if len(data) < min_events:
                results['warnings'].append(f"Few events provided: {len(data)} (minimum recommended: {min_events})")
            elif len(data) > max_events:
                results['warnings'].append(f"Many events provided: {len(data)} (maximum recommended: {max_events})")
            
            # Category distribution
            if 'category' in data.columns:
                category_counts = data['category'].value_counts()
                required_categories = self.config['event_validation']['required_categories']
                
                missing_categories = [cat for cat in required_categories if cat not in category_counts.index]
                if missing_categories:
                    results['warnings'].append(f"Missing event categories: {missing_categories}")
            
            # Impact distribution
            if 'impact' in data.columns:
                impact_counts = data['impact'].value_counts()
                if len(impact_counts) < 2:
                    results['warnings'].append("All events have the same impact level - consider more variation")
            
            # Temporal distribution
            if 'date' in data.columns:
                event_dates = pd.to_datetime(data['date'], errors='coerce').dropna()
                if len(event_dates) > 1:
                    date_range = event_dates.max() - event_dates.min()
                    if date_range.days < 365:
                        results['warnings'].append("Events span less than one year - consider longer time range")
            
            # Statistics
            results['statistics'] = {
                'total_events': len(data),
                'valid_events': valid_rows,
                'categories': data['category'].value_counts().to_dict() if 'category' in data.columns else {},
                'impact_levels': data['impact'].value_counts().to_dict() if 'impact' in data.columns else {},
                'date_range': {
                    'start': str(event_dates.min()) if len(event_dates) > 0 else None,
                    'end': str(event_dates.max()) if len(event_dates) > 0 else None
                }
            }
            
        except Exception as e:
            results['errors'].append(f"Event validation error: {str(e)}")
            results['valid'] = False
        
        self.validation_results['event_data'] = results
        return results
    
    def validate_combined_data(self, price_data: pd.DataFrame, event_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate combined price and event data for consistency.
        
        Args:
            price_data: Price data DataFrame
            event_data: Optional event data DataFrame
            
        Returns:
            Combined validation results
        """
        logger.info("Validating combined data consistency...")
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'price_validation': {},
            'event_validation': {},
            'consistency_checks': {}
        }
        
        # Validate individual datasets
        results['price_validation'] = self.validate_price_data(price_data)
        
        if event_data is not None and not event_data.empty:
            results['event_validation'] = self.validate_event_data(event_data)
            
            # Consistency checks
            try:
                price_dates = pd.to_datetime(price_data['date'], errors='coerce').dropna()
                event_dates = pd.to_datetime(event_data['date'], errors='coerce').dropna()
                
                if len(price_dates) > 0 and len(event_dates) > 0:
                    price_start, price_end = price_dates.min(), price_dates.max()
                    event_start, event_end = event_dates.min(), event_dates.max()
                    
                    # Check if events fall within price data range
                    events_outside_range = (event_dates < price_start) | (event_dates > price_end)
                    if events_outside_range.any():
                        outside_count = events_outside_range.sum()
                        results['warnings'].append(f"{outside_count} events fall outside price data range")
                    
                    # Check for temporal alignment
                    overlap_start = max(price_start, event_start)
                    overlap_end = min(price_end, event_end)
                    
                    if overlap_start <= overlap_end:
                        overlap_days = (overlap_end - overlap_start).days
                        total_days = (price_end - price_start).days
                        overlap_percentage = (overlap_days / total_days) * 100
                        
                        results['consistency_checks']['temporal_overlap'] = {
                            'overlap_days': overlap_days,
                            'overlap_percentage': float(overlap_percentage),
                            'sufficient_overlap': overlap_percentage > 50
                        }
                        
                        if overlap_percentage < 50:
                            results['warnings'].append(f"Low temporal overlap between price and event data: {overlap_percentage:.1f}%")
                    else:
                        results['errors'].append("No temporal overlap between price and event data")
                        results['valid'] = False
                
            except Exception as e:
                results['errors'].append(f"Consistency check error: {str(e)}")
        
        # Overall validation status
        if not results['price_validation']['valid']:
            results['valid'] = False
        
        if event_data is not None and not results['event_validation']['valid']:
            results['valid'] = False
        
        return results
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        if not self.validation_results:
            return "No validation results available. Run validation methods first."
        
        report = []
        report.append("# Data Validation Report")
        report.append("=" * 50)
        
        # Price data validation
        if 'price_data' in self.validation_results:
            price_results = self.validation_results['price_data']
            report.append("\n## Price Data Validation")
            
            status = "✅ PASSED" if price_results['valid'] else "❌ FAILED"
            report.append(f"**Status:** {status}")
            
            if price_results['errors']:
                report.append(f"\n**Errors ({len(price_results['errors'])}):**")
                for error in price_results['errors']:
                    report.append(f"- {error}")
            
            if price_results['warnings']:
                report.append(f"\n**Warnings ({len(price_results['warnings'])}):**")
                for warning in price_results['warnings']:
                    report.append(f"- {warning}")
            
            if 'statistics' in price_results:
                stats = price_results['statistics']
                report.append(f"\n**Statistics:**")
                report.append(f"- Total observations: {stats.get('total_observations', 0):,}")
                report.append(f"- Valid prices: {stats.get('valid_prices', 0):,}")
                report.append(f"- Missing values: {stats.get('missing_values', 0)} ({stats.get('missing_percentage', 0):.1f}%)")
        
        # Event data validation
        if 'event_data' in self.validation_results:
            event_results = self.validation_results['event_data']
            report.append("\n## Event Data Validation")
            
            status = "✅ PASSED" if event_results['valid'] else "❌ FAILED"
            report.append(f"**Status:** {status}")
            
            if event_results['errors']:
                report.append(f"\n**Errors ({len(event_results['errors'])}):**")
                for error in event_results['errors']:
                    report.append(f"- {error}")
            
            if event_results['warnings']:
                report.append(f"\n**Warnings ({len(event_results['warnings'])}):**")
                for warning in event_results['warnings']:
                    report.append(f"- {warning}")
        
        return "\n".join(report)
