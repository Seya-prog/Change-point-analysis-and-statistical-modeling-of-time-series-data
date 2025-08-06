"""
Data loader for Brent oil price analysis.
Implements modern data loading practices with validation and error handling.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from loguru import logger
import yaml
from pydantic import BaseModel, validator
from datetime import datetime, date


class DataConfig(BaseModel):
    """Configuration model for data loading."""
    start_date: str
    end_date: str
    source: str = "yahoo"
    symbol: str = "BZ=F"
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')


class DataLoader:
    """
    Modern data loader for Brent oil prices with comprehensive validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the data loader with configuration."""
        # Find the project root directory
        self.project_root = self._find_project_root()
        self.config_path = config_path or str(self.project_root / "config" / "config.yaml")
        self.config = self._load_config()
        self.data_config = DataConfig(**self.config['data']['brent_oil'])
        
    def _find_project_root(self) -> Path:
        """Find the project root directory by looking for key files."""
        current_path = Path(__file__).resolve()
        
        # Look for project root indicators
        root_indicators = ['config', 'src', 'requirements.txt', '.git']
        
        for parent in current_path.parents:
            if any((parent / indicator).exists() for indicator in root_indicators):
                return parent
        
        # Fallback to current directory
        return Path.cwd()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def fetch_brent_oil_data(self, save_to_file: bool = True) -> pd.DataFrame:
        """
        Fetch Brent oil price data from Yahoo Finance.
        
        Args:
            save_to_file: Whether to save the data to CSV file
            
        Returns:
            DataFrame with Brent oil price data
        """
        logger.info(f"Fetching Brent oil data from {self.data_config.start_date} to {self.data_config.end_date}")
        
        try:
            # Fetch data using yfinance
            ticker = yf.Ticker(self.data_config.symbol)
            data = ticker.history(
                start=self.data_config.start_date,
                end=self.data_config.end_date,
                interval="1d"
            )
            
            if data.empty:
                raise ValueError("No data retrieved from Yahoo Finance")
            
            # Clean and prepare the data
            data = self._clean_price_data(data)
            
            if save_to_file:
                self._save_data(data, "brent_oil_prices.csv")
            
            logger.info(f"Successfully fetched {len(data)} data points")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Brent oil data: {e}")
            raise
    
    def _clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the price data.
        
        Args:
            data: Raw price data from yfinance
            
        Returns:
            Cleaned DataFrame
        """
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Rename columns to match our schema
        data = data.rename(columns={
            'Date': 'date',
            'Close': 'price',
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Volume': 'volume'
        })
        
        # Select relevant columns
        columns_to_keep = ['date', 'price', 'open_price', 'high_price', 'low_price', 'volume']
        data = data[columns_to_keep]
        
        # Handle missing values
        missing_count = data['price'].isna().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing price values. Interpolating...")
            data['price'] = data['price'].interpolate(method='linear')
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        # Ensure date is datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Data cleaned. Shape: {data.shape}")
        return data
    
    def load_existing_data(self, filename: str = "BrentOilPrices.csv") -> pd.DataFrame:
        """
        Load existing data from CSV file.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        file_path = self.project_root / self.config['data']['raw_data_path'] / filename
        
        try:
            # Try to read the CSV file
            data = pd.read_csv(file_path)
            
            # Handle different possible column names and standardize them
            column_mapping = {
                'Date': 'date',
                'Price': 'price', 
                'Close': 'price',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Volume': 'volume'
            }
            
            # Rename columns to standard format
            data = data.rename(columns=column_mapping)
            
            # Convert date column to datetime
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            logger.info(f"Loaded existing data from {file_path}. Shape: {data.shape}")
            logger.info(f"Columns: {list(data.columns)}")
            return data
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def _save_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save data to CSV file.
        
        Args:
            data: DataFrame to save
            filename: Name of the output file
        """
        # Ensure directory exists
        output_dir = self.project_root / self.config['data']['raw_data_path']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / filename
        data.to_csv(file_path, index=False)
        logger.info(f"Data saved to {file_path}")
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the data.
        
        Args:
            data: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_observations': len(data),
            'date_range': {
                'start': data['date'].min().strftime('%Y-%m-%d'),
                'end': data['date'].max().strftime('%Y-%m-%d')
            },
            'price_statistics': {
                'mean': data['price'].mean(),
                'std': data['price'].std(),
                'min': data['price'].min(),
                'max': data['price'].max(),
                'median': data['price'].median()
            },
            'missing_values': data.isna().sum().to_dict(),
            'data_types': data.dtypes.to_dict()
        }
        
        return summary


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Try to load existing data first
    data = loader.load_existing_data()
    
    if data.empty:
        # Fetch new data if no existing data found
        data = loader.fetch_brent_oil_data()
    
    # Print summary
    summary = loader.get_data_summary(data)
    print("Data Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
