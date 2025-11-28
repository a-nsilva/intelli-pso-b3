"""
Portfolio Module for Intelli PSO B3

This module handles portfolio calculations including metrics (return, volatility, Sharpe ratio),
weight validation, and benchmark creation.

Author: Alexandre do Nascimento Silva
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import logging

import numpy as np
import pandas as pd

# ENUMS
# ============================================================================
class PortfolioType(Enum):
    """Types of portfolio construction"""
    EQUAL_WEIGHT = "equal_weight"
    OPTIMIZED = "optimized"
    CUSTOM = "custom"

class RiskMetric(Enum):
    """Available risk metrics"""
    VOLATILITY = "volatility"
    VAR = "value_at_risk"
    CVAR = "conditional_var"

# DATACLASSES
# ============================================================================
@dataclass
class PortfolioConfig:
    """
    Configuration for portfolio calculations.
    
    Attributes:
        returns: DataFrame with daily returns for each asset
        risk_free_rate: Annual risk-free rate (default: 5%)
        annualization_factor: Days for annualization (default: 252 trading days)
        allow_short: Whether to allow short positions (default: False)
    """
    returns: pd.DataFrame
    risk_free_rate: float = 0.05
    annualization_factor: int = 252
    allow_short: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.returns is None or self.returns.empty:
            raise ValueError("Returns DataFrame cannot be empty")
        
        if self.risk_free_rate < 0 or self.risk_free_rate > 1:
            raise ValueError("Risk-free rate must be between 0 and 1")
        
        if self.annualization_factor <= 0:
            raise ValueError("Annualization factor must be positive")

@dataclass
class PortfolioMetrics:
    """
    Portfolio performance metrics.
    
    Attributes:
        weights: Array of portfolio weights
        expected_return: Expected annual return
        volatility: Annual volatility (standard deviation)
        sharpe_ratio: Sharpe ratio (risk-adjusted return)
        portfolio_type: Type of portfolio
        symbols: List of asset symbols (optional)
    """
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    portfolio_type: PortfolioType
    symbols: Optional[list] = None
    
    def __post_init__(self):
        """Validate metrics after initialization"""
        if len(self.weights) == 0:
            raise ValueError("Weights array cannot be empty")
        
        if not np.isclose(np.sum(self.weights), 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {np.sum(self.weights):.6f}")

@dataclass
class PortfolioComparison:
    """
    Comparison between two portfolios.
    
    Attributes:
        benchmark: Benchmark portfolio metrics
        optimized: Optimized portfolio metrics
        return_improvement: Improvement in return (percentage points)
        volatility_change: Change in volatility (percentage points)
        sharpe_improvement: Improvement in Sharpe ratio (percentage)
    """
    benchmark: PortfolioMetrics
    optimized: PortfolioMetrics
    return_improvement: float
    volatility_change: float
    sharpe_improvement: float

class Portfolio:
    """
    Portfolio calculations and metrics engine.
    """
    
    def __init__(self, config: PortfolioConfig):
        """
        Initialize Portfolio calculator.
        
        Args:
            config: PortfolioConfig instance with calculation parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Pre-calculate mean returns and covariance matrix
        self.mean_returns = self.config.returns.mean() * self.config.annualization_factor
        self.cov_matrix = self.config.returns.cov() * self.config.annualization_factor
        
        self.n_assets = len(self.config.returns.columns)
        self.symbols = list(self.config.returns.columns)
        
        self.logger.info(f"Portfolio initialized with {self.n_assets} assets")
    
    def calculate_metrics(
        self, 
        weights: np.ndarray,
        portfolio_type: PortfolioType = PortfolioType.CUSTOM
    ) -> PortfolioMetrics:
        """
        Calculate portfolio metrics for given weights.
        
        Args:
            weights: Array of portfolio weights (must sum to 1)
            portfolio_type: Type of portfolio (for metadata)
            
        Returns:
            PortfolioMetrics with calculated performance metrics
            
        Raises:
            ValueError: If weights are invalid
            
        Example:
            weights = np.array([0.3, 0.4, 0.3])
            metrics = portfolio.calculate_metrics(weights)
            print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
        """
        # Validate weights
        if not self.validate_weights(weights):
            raise ValueError("Invalid weights provided")
        
        # Normalize weights to ensure they sum to exactly 1
        weights = np.array(weights) / np.sum(weights)
        
        # Calculate expected return
        expected_return = np.sum(self.mean_returns.values * weights)
        
        # Calculate volatility
        volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix.values, weights))
        )
        
        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe(expected_return, volatility)
        
        metrics = PortfolioMetrics(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            portfolio_type=portfolio_type,
            symbols=self.symbols
        )
        
        self.logger.debug(
            f"Calculated metrics - Return: {expected_return:.4f}, "
            f"Vol: {volatility:.4f}, Sharpe: {sharpe_ratio:.4f}"
        )
        
        return metrics
    
    def create_equal_weight(self) -> PortfolioMetrics:
        """
        Create equal-weight benchmark portfolio.
        
        Returns:
            PortfolioMetrics for equal-weight portfolio
            
        Example:
            benchmark = portfolio.create_equal_weight()
            print(f"Equal-weight Sharpe: {benchmark.sharpe_ratio:.4f}")
        """
        equal_weights = np.ones(self.n_assets) / self.n_assets
        
        metrics = self.calculate_metrics(
            weights=equal_weights,
            portfolio_type=PortfolioType.EQUAL_WEIGHT
        )
        
        self.logger.info(
            f"Equal-weight benchmark created - Sharpe: {metrics.sharpe_ratio:.4f}"
        )
        
        return metrics
    
    def compare_portfolios(
        self,
        benchmark: PortfolioMetrics,
        optimized: PortfolioMetrics
    ) -> PortfolioComparison:
        """
        Compare two portfolios and calculate improvements.
        
        Args:
            benchmark: Benchmark portfolio metrics
            optimized: Optimized portfolio metrics
            
        Returns:
            PortfolioComparison with improvement metrics
            
        Example:
            comparison = portfolio.compare_portfolios(benchmark, optimized)
            print(f"Sharpe improvement: {comparison.sharpe_improvement:.1f}%")
        """
        return_improvement = (
            optimized.expected_return - benchmark.expected_return
        ) * 100
        
        volatility_change = (
            optimized.volatility - benchmark.volatility
        ) * 100
        
        sharpe_improvement = (
            (optimized.sharpe_ratio / benchmark.sharpe_ratio) - 1
        ) * 100
        
        comparison = PortfolioComparison(
            benchmark=benchmark,
            optimized=optimized,
            return_improvement=return_improvement,
            volatility_change=volatility_change,
            sharpe_improvement=sharpe_improvement
        )
        
        self.logger.info(
            f"Portfolio comparison - Return: {return_improvement:+.2f}pp, "
            f"Sharpe: {sharpe_improvement:+.1f}%"
        )
        
        return comparison
    
    def validate_weights(self, weights: np.ndarray) -> bool:
        """
        Validate portfolio weights.
        
        Checks:
        1. Correct length
        2. All finite values
        3. Sum to 1 (within tolerance)
        4. Non-negative (if short positions not allowed)
        
        Args:
            weights: Array of portfolio weights
            
        Returns:
            True if weights are valid, False otherwise
            
        Example:
            weights = np.array([0.5, 0.3, 0.2])
            is_valid = portfolio.validate_weights(weights)
        """
        # Check length
        if len(weights) != self.n_assets:
            self.logger.error(
                f"Invalid weights length: {len(weights)} (expected {self.n_assets})"
            )
            return False
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(weights)):
            self.logger.error("Weights contain NaN or Inf values")
            return False
        
        # Check sum
        weights_sum = np.sum(weights)
        if not np.isclose(weights_sum, 1.0, atol=1e-4):
            self.logger.error(f"Weights sum to {weights_sum:.6f} (expected 1.0)")
            return False
        
        # Check non-negative (if required)
        if not self.config.allow_short and np.any(weights < 0):
            self.logger.error("Negative weights not allowed")
            return False
        
        return True
    
    def _calculate_sharpe(self, expected_return: float, volatility: float) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            expected_return: Expected portfolio return
            volatility: Portfolio volatility
            
        Returns:
            Sharpe ratio
        """
        if volatility == 0:
            return 0.0
        
        sharpe = (expected_return - self.config.risk_free_rate) / volatility
        return sharpe
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix for assets.
        
        Returns:
            DataFrame with correlation matrix
            
        Example:
            corr = portfolio.get_correlation_matrix()
            print(corr)
        """
        return self.config.returns.corr()
    
    def get_individual_metrics(self) -> pd.DataFrame:
        """
        Get individual metrics for each asset.
        
        Returns:
            DataFrame with metrics for each asset
            
        Example:
            metrics_df = portfolio.get_individual_metrics()
            print(metrics_df)
        """
        individual_returns = self.mean_returns.values * 100
        individual_vols = np.sqrt(np.diag(self.cov_matrix)) * 100
        individual_sharpes = (
            (self.mean_returns.values - self.config.risk_free_rate) / 
            np.sqrt(np.diag(self.cov_matrix))
        )
        
        df = pd.DataFrame({
            'Symbol': self.symbols,
            'Annual_Return_%': individual_returns,
            'Annual_Volatility_%': individual_vols,
            'Sharpe_Ratio': individual_sharpes
        })
        
        return df.sort_values('Sharpe_Ratio', ascending=False)
    
    def calculate_returns_series(self, weights: np.ndarray) -> pd.Series:
        """
        Calculate portfolio returns time series.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Series with portfolio returns over time
            
        Example:
            weights = np.array([0.3, 0.4, 0.3])
            returns_series = portfolio.calculate_returns_series(weights)
            print(returns_series.mean())
        """
        if len(weights) != self.n_assets:
            raise ValueError(
                f"Weights length ({len(weights)}) must match "
                f"number of assets ({self.n_assets})"
            )
        
        portfolio_returns = (self.config.returns * weights).sum(axis=1)
        return portfolio_returns
    
    def calculate_max_drawdown(self, weights: np.ndarray) -> float:
        """
        Calculate maximum drawdown for portfolio.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Maximum drawdown (as positive decimal)
            
        Example:
            weights = np.array([0.3, 0.4, 0.3])
            max_dd = portfolio.calculate_max_drawdown(weights)
            print(f"Max Drawdown: {max_dd*100:.2f}%")
        """
        portfolio_returns = self.calculate_returns_series(weights)
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        self.logger.debug(f"Max drawdown calculated: {max_dd*100:.2f}%")
        return max_dd
    
    def calculate_sortino_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate Sortino ratio (focuses on downside risk).
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Sortino ratio
            
        Example:
            weights = np.array([0.3, 0.4, 0.3])
            sortino = portfolio.calculate_sortino_ratio(weights)
            print(f"Sortino: {sortino:.4f}")
        """
        portfolio_returns = self.calculate_returns_series(weights)
        mean_return = portfolio_returns.mean() * self.config.annualization_factor
        downside_returns = portfolio_returns[portfolio_returns < 0]
        
        if len(downside_returns) == 0:
            self.logger.debug("No downside returns - Sortino ratio is infinite")
            return np.inf
        
        downside_std = downside_returns.std() * np.sqrt(self.config.annualization_factor)
        
        if downside_std == 0:
            self.logger.debug("Zero downside volatility - Sortino ratio is infinite")
            return np.inf
        
        sortino = (mean_return - self.config.risk_free_rate) / downside_std
        
        self.logger.debug(f"Sortino ratio calculated: {sortino:.4f}")
        return sortino
    
    @staticmethod
    def normalize_weights(weights: np.ndarray) -> np.ndarray:
        """
        Normalize weights to sum to 1.
        
        Args:
            weights: Array of weights
            
        Returns:
            Normalized weights
            
        Example:
            weights = np.array([1, 2, 3])
            normalized = Portfolio.normalize_weights(weights)
            # Result: [0.167, 0.333, 0.5]
        """
        weights = np.abs(weights)  # Ensure non-negative
        return weights / np.sum(weights)
