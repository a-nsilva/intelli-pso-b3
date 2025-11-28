"""
Data Collector Module for Intelli PSO B3

This module handles robust data collection from Yahoo Finance for Brazilian stocks (B3).
Supports both preset periods and custom date ranges with comprehensive validation.

Author: Alexandre do Nascimento Silva
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import logging

import numpy as np
import pandas as pd
import yfinance as yf


# ENUMS
# ============================================================================
class DataSource(Enum):
    """Available data sources for financial data"""
    YAHOO = "yfinance"

class PeriodType(Enum):
    """Types of period configuration"""
    PRESET = "preset"
    CUSTOM = "custom"

# DATACLASSES
# ============================================================================
@dataclass
class DataCollectionConfig:
    """
    Configuration for data collection process.
    """
    symbols: List[str]
    period_type: PeriodType
    period_value: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    source: DataSource = DataSource.YAHOO
    min_observations: int = 50
    timeout: int = 30

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.period_type == PeriodType.PRESET and not self.period_value:
            raise ValueError("period_value required for PRESET period type")
        
        if self.period_type == PeriodType.CUSTOM:
            if not self.start_date or not self.end_date:
                raise ValueError("start_date and end_date required for CUSTOM period type")


@dataclass
class CollectionMetadata:
    """
    Metadata about the data collection process.
    """
    period_requested: str
    period_actual: str
    observations: int
    assets_requested: int
    assets_collected: int
    collection_time: str
    source: str
    failed_symbols: List[str] = field(default_factory=list)


@dataclass
class CollectionResult:
    """
    Result of data collection process.
    """
    data: Optional[pd.DataFrame] = None
    returns: Optional[pd.DataFrame] = None
    symbols: List[str] = field(default_factory=list)
    metadata: Optional[CollectionMetadata] = None
    success: bool = False
    errors: List[str] = field(default_factory=list)

class DataCollector:
    """
    Data collector for financial assets from Yahoo Finance.
    """
    
    def __init__(self, config: DataCollectionConfig):
        """
        Initialize DataCollector.
        
        Args:
            config: DataCollectionConfig instance with collection parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DataCollector initialized with {len(config.symbols)} symbols")
    
    @classmethod
    def from_preset(
        cls,
        symbols: List[str],
        period: str = '5y',
        min_observations: int = 50,
        timeout: int = 30
    ) -> 'DataCollector':
        """
        Create collector with preset period.
        
        Args:
            symbols: List of stock symbols (e.g., ['PETR4.SA', 'VALE3.SA'])
            period: Preset period ('5y', '3y', '2y', '1y')
            min_observations: Minimum required data points per asset
            timeout: Timeout in seconds for data download
            
        Returns:
            DataCollector instance ready to use
        """
        config = DataCollectionConfig(
            symbols=symbols,
            period_type=PeriodType.PRESET,
            period_value=period,
            min_observations=min_observations,
            timeout=timeout
        )
        return cls(config)
    
    @classmethod
    def from_custom(
        cls,
        symbols: List[str],
        start_date: str,
        end_date: str,
        min_observations: int = 50,
        timeout: int = 30
    ) -> 'DataCollector':
        """
        Create collector with custom date range.
        
        Args:
            symbols: List of stock symbols (e.g., ['PETR4.SA', 'VALE3.SA'])
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            min_observations: Minimum required data points per asset
            timeout: Timeout in seconds for data download
            
        Returns:
            DataCollector instance ready to use
        """
        config = DataCollectionConfig(
            symbols=symbols,
            period_type=PeriodType.CUSTOM,
            start_date=start_date,
            end_date=end_date,
            min_observations=min_observations,
            timeout=timeout
        )
        return cls(config)
    
    def collect(self) -> CollectionResult:
        """
        Execute complete data collection process.
        
        Returns:
            CollectionResult with collected data and metadata
        """
        self.logger.info("Starting data collection process")
        
        result = CollectionResult()
        successful_data = {}
        
        # Download each asset
        for symbol in self.config.symbols:
            self.logger.info(f"Downloading {symbol}")
            
            try:
                data = self._download_asset(symbol)
                
                if data is not None and self._validate_data(data, symbol):
                    successful_data[symbol] = data
                    self.logger.info(f"✓ {symbol}: {len(data)} observations")
                else:
                    result.errors.append(f"{symbol}: Insufficient data")
                    self.logger.warning(f"✗ {symbol}: Validation failed")
                    
            except Exception as e:
                error_msg = f"{symbol}: {str(e)[:50]}"
                result.errors.append(error_msg)
                self.logger.error(f"✗ {symbol}: {str(e)}")
        
        # Check minimum assets requirement
        if len(successful_data) < 5:
            result.success = False
            result.errors.append(f"Insufficient assets: {len(successful_data)}/10")
            self.logger.error(f"Collection failed: only {len(successful_data)} assets")
            return result
        
        # Align and process data
        try:
            aligned_data = self._align_data(successful_data)
            returns = self._calculate_returns(aligned_data)
            
            # Create metadata
            metadata = self._create_metadata(
                aligned_data, 
                successful_data,
                self.config.symbols
            )
            
            # Populate result
            result.data = aligned_data
            result.returns = returns
            result.symbols = list(successful_data.keys())
            result.metadata = metadata
            result.success = True
            
            self.logger.info(f"Collection successful: {len(result.symbols)} assets, "
                           f"{len(aligned_data)} observations")
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Data processing error: {str(e)}")
            self.logger.error(f"Data processing failed: {str(e)}")
        
        return result
    
    def _download_asset(self, symbol: str) -> Optional[pd.Series]:
        """
        Download data for a single asset.
        
        Args:
            symbol: Stock symbol to download
            
        Returns:
            Series with price data or None if download fails
        """
        try:
            if self.config.period_type == PeriodType.CUSTOM:
                stock = yf.download(
                    symbol,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    progress=False,
                    timeout=self.config.timeout
                )
            else:
                stock = yf.download(
                    symbol,
                    period=self.config.period_value,
                    progress=False,
                    timeout=self.config.timeout
                )
            
            # Handle empty download
            if stock is None or len(stock) == 0:
                return None
            
            # Extract price column
            if 'Adj Close' in stock.columns:
                prices = stock['Adj Close']
            elif 'Close' in stock.columns:
                prices = stock['Close']
            else:
                return None
            
            # Clean data
            clean_prices = prices.dropna()
            
            return clean_prices if len(clean_prices) > 0 else None
            
        except Exception as e:
            self.logger.error(f"Download failed for {symbol}: {str(e)}")
            return None
    
    def _validate_data(self, data: pd.Series, symbol: str) -> bool:
        """
        Validate collected data quality.
        
        Args:
            data: Price data series
            symbol: Stock symbol being validated
            
        Returns:
            True if data passes validation, False otherwise
        """
        # Check minimum observations
        if len(data) < self.config.min_observations:
            self.logger.warning(f"{symbol}: Only {len(data)} observations "
                              f"(minimum: {self.config.min_observations})")
            return False
        
        # Check for excessive NaN values
        nan_ratio = data.isna().sum() / len(data)
        if nan_ratio > 0.1:
            self.logger.warning(f"{symbol}: {nan_ratio*100:.1f}% NaN values")
            return False
        
        # Check for zero/negative prices
        if (data <= 0).any():
            self.logger.warning(f"{symbol}: Contains zero or negative prices")
            return False
        
        # Check for extreme volatility (potential data errors)
        returns = data.pct_change().dropna()
        if (abs(returns) > 0.5).any():
            self.logger.warning(f"{symbol}: Extreme price changes detected")
            return False
        
        return True
    
    def _align_data(self, data_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Align multiple asset data to common dates.
        
        Args:
            data_dict: Dictionary mapping symbols to price series
            
        Returns:
            DataFrame with aligned data for all assets
        """
        aligned = pd.concat(
            list(data_dict.values()),
            axis=1,
            join='inner',
            keys=list(data_dict.keys())
        )
        
        self.logger.info(f"Data aligned: {len(aligned)} common observations")
        
        return aligned
    
    def _calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns from price data.
        
        Args:
            prices: DataFrame with price data
            
        Returns:
            DataFrame with daily returns
        """
        returns = prices.pct_change().dropna()
        self.logger.info(f"Returns calculated: {len(returns)} observations")
        return returns
    
    def _create_metadata(
        self, 
        aligned_data: pd.DataFrame,
        successful_data: Dict,
        requested_symbols: List[str]
    ) -> CollectionMetadata:
        """
        Create metadata about the collection process.
        
        Args:
            aligned_data: Final aligned DataFrame
            successful_data: Dictionary of successfully collected data
            requested_symbols: Original list of requested symbols
            
        Returns:
            CollectionMetadata instance
        """
        # Determine period description
        if self.config.period_type == PeriodType.CUSTOM:
            period_requested = f"Custom ({self.config.start_date} to {self.config.end_date})"
        else:
            period_requested = self.config.period_value
        
        # Actual period from data
        period_actual = (
            f"{aligned_data.index[0].strftime('%Y-%m-%d')} to "
            f"{aligned_data.index[-1].strftime('%Y-%m-%d')}"
        )
        
        # Failed symbols
        failed_symbols = [
            s for s in requested_symbols 
            if s not in successful_data.keys()
        ]
        
        return CollectionMetadata(
            period_requested=period_requested,
            period_actual=period_actual,
            observations=len(aligned_data),
            assets_requested=len(requested_symbols),
            assets_collected=len(successful_data),
            collection_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            source=self.config.source.value,
            failed_symbols=failed_symbols
        )
