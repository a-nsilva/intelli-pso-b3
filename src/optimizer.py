"""
Optimizer Module for Intelli PSO B3

This module implements the Particle Swarm Optimization (PSO) algorithm
for portfolio optimization, maximizing risk-adjusted returns.

Author: Alexandre do Nascimento Silva
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import logging

import numpy as np

from .portfolio import Portfolio, PortfolioMetrics, PortfolioType


# ENUMS
# ============================================================================
class OptimizationObjective(Enum):
    """Available optimization objectives"""
    SHARPE_RATIO = "sharpe_ratio"
    RETURN = "return"
    SORTINO_RATIO = "sortino_ratio"

# DATACLASSES
# ============================================================================
@dataclass
class PSOConfig:
    """
    Configuration for PSO algorithm.
    """
    n_particles: int = 50
    n_iterations: int = 100
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5
    objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO
    verbose: bool = False
    log_interval: int = 25
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.n_particles < 10:
            raise ValueError("n_particles must be at least 10")
        
        if self.n_iterations < 10:
            raise ValueError("n_iterations must be at least 10")
        
        if not (0 < self.inertia < 1):
            raise ValueError("inertia must be between 0 and 1")
        
        if self.cognitive <= 0 or self.social <= 0:
            raise ValueError("cognitive and social parameters must be positive")


@dataclass
class OptimizationResult:
    """
    Results from PSO optimization.
    """
    optimal_weights: np.ndarray
    best_fitness: float
    convergence_history: List[float]
    n_iterations: int
    metrics: PortfolioMetrics
    execution_time: Optional[float] = None

class PSOOptimizer:
    """
    Particle Swarm Optimization for portfolio optimization.
    
    Implements PSO algorithm to find optimal portfolio weights that
    maximize the selected objective function (default: Sharpe ratio).
    
    Algorithm:
    1. Initialize random particle positions and velocities
    2. Evaluate fitness for each particle
    3. Update personal and global bests
    4. Update velocities and positions
    5. Repeat until convergence or max iterations
    """
    
    def __init__(self, portfolio: Portfolio, config: PSOConfig):
        """
        Initialize PSO optimizer.
        
        Args:
            portfolio: Portfolio instance for calculations
            config: PSOConfig with algorithm parameters
        """
        self.portfolio = portfolio
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.n_assets = portfolio.n_assets
        
        # Particle swarm state
        self.particles: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.personal_best: Optional[np.ndarray] = None
        self.personal_best_fitness: Optional[np.ndarray] = None
        self.global_best: Optional[np.ndarray] = None
        self.global_best_fitness: Optional[float] = None
        
        self.logger.info(
            f"PSOOptimizer initialized - {self.config.n_particles} particles, "
            f"{self.config.n_iterations} iterations"
        )
    
    @classmethod
    def quick_optimize(
        cls,
        portfolio: Portfolio,
        n_particles: int = 50,
        n_iterations: int = 100
    ) -> OptimizationResult:
        """
        Quick optimization with default parameters.
        
        Args:
            portfolio: Portfolio instance for calculations
            n_particles: Number of particles (default: 50)
            n_iterations: Number of iterations (default: 100)
            
        Returns:
            OptimizationResult with optimal portfolio
        """
        config = PSOConfig(
            n_particles=n_particles,
            n_iterations=n_iterations
        )
        optimizer = cls(portfolio, config)
        return optimizer.optimize()
    
    def optimize(self) -> OptimizationResult:
        """
        Execute PSO optimization.
        
        Returns:
            OptimizationResult with optimal weights and metrics
            
        Raises:
            RuntimeError: If optimization fails
        """
        import time
        start_time = time.time()
        
        self.logger.info("Starting PSO optimization")
        
        # Initialize swarm
        self._initialize_swarm()
        
        # Convergence history
        convergence_history = [self.global_best_fitness]
        
        # Main PSO loop
        for iteration in range(self.config.n_iterations):
            
            # Update each particle
            for i in range(self.config.n_particles):
                self._update_particle(i)
            
            # Record convergence
            convergence_history.append(self.global_best_fitness)
            
            # Log progress
            if self.config.verbose and (iteration + 1) % self.config.log_interval == 0:
                print(f"Iteration {iteration + 1:3d}: Fitness = {self.global_best_fitness:.6f}")
            
            if (iteration + 1) % self.config.log_interval == 0:
                self.logger.info(
                    f"Iteration {iteration + 1}: "
                    f"Best fitness = {self.global_best_fitness:.6f}"
                )
        
        # Calculate final metrics
        final_metrics = self.portfolio.calculate_metrics(
            weights=self.global_best,
            portfolio_type=PortfolioType.OPTIMIZED
        )
        
        execution_time = time.time() - start_time
        
        result = OptimizationResult(
            optimal_weights=self.global_best,
            best_fitness=self.global_best_fitness,
            convergence_history=convergence_history,
            n_iterations=self.config.n_iterations,
            metrics=final_metrics,
            execution_time=execution_time
        )
        
        self.logger.info(
            f"Optimization complete - Best fitness: {self.global_best_fitness:.6f}, "
            f"Time: {execution_time:.2f}s"
        )
        
        return result
    
    def _initialize_swarm(self):
        """Initialize particle positions and velocities"""
        # Random positions (weights)
        self.particles = np.random.random((self.config.n_particles, self.n_assets))
        self.particles = self.particles / self.particles.sum(axis=1, keepdims=True)
        
        # Random velocities
        self.velocities = np.random.uniform(
            -0.1, 0.1, 
            (self.config.n_particles, self.n_assets)
        )
        
        # Evaluate initial fitness
        fitness = np.array([
            self._evaluate_fitness(self.particles[i]) 
            for i in range(self.config.n_particles)
        ])
        
        # Initialize personal bests
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = fitness.copy()
        
        # Initialize global best
        best_idx = np.argmax(fitness)
        self.global_best = self.particles[best_idx].copy()
        self.global_best_fitness = fitness[best_idx]
        
        self.logger.debug(
            f"Swarm initialized - Initial best fitness: {self.global_best_fitness:.6f}"
        )
    
    def _update_particle(self, particle_idx: int):
        """
        Update position and velocity of a single particle.
        
        Args:
            particle_idx: Index of particle to update
        """
        # Random factors
        r1 = np.random.random(self.n_assets)
        r2 = np.random.random(self.n_assets)
        
        # Update velocity
        self.velocities[particle_idx] = (
            self.config.inertia * self.velocities[particle_idx] +
            self.config.cognitive * r1 * (self.personal_best[particle_idx] - self.particles[particle_idx]) +
            self.config.social * r2 * (self.global_best - self.particles[particle_idx])
        )
        
        # Update position
        self.particles[particle_idx] += self.velocities[particle_idx]
        
        # Ensure non-negative weights
        self.particles[particle_idx] = np.abs(self.particles[particle_idx])
        
        # Normalize to sum to 1
        self.particles[particle_idx] = Portfolio.normalize_weights(
            self.particles[particle_idx]
        )
        
        # Evaluate new fitness
        new_fitness = self._evaluate_fitness(self.particles[particle_idx])
        
        # Update personal best
        if new_fitness > self.personal_best_fitness[particle_idx]:
            self.personal_best[particle_idx] = self.particles[particle_idx].copy()
            self.personal_best_fitness[particle_idx] = new_fitness
            
            # Update global best
            if new_fitness > self.global_best_fitness:
                self.global_best = self.particles[particle_idx].copy()
                self.global_best_fitness = new_fitness
    
    def _evaluate_fitness(self, weights: np.ndarray) -> float:
        """
        Evaluate fitness for given weights.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Fitness value based on selected objective
        """
        try:
            if self.config.objective == OptimizationObjective.SHARPE_RATIO:
                metrics = self.portfolio.calculate_metrics(weights)
                return metrics.sharpe_ratio
            
            elif self.config.objective == OptimizationObjective.RETURN:
                metrics = self.portfolio.calculate_metrics(weights)
                return metrics.expected_return
            
            elif self.config.objective == OptimizationObjective.SORTINO_RATIO:
                return self.portfolio.calculate_sortino_ratio(weights)
            
            else:
                raise ValueError(f"Unknown objective: {self.config.objective}")
                
        except Exception as e:
            self.logger.warning(f"Fitness evaluation failed: {str(e)}")
            return -np.inf
    
    def get_swarm_statistics(self) -> dict:
        """
        Get statistics about current swarm state.
        
        Returns:
            Dictionary with swarm statistics
        """
        if self.particles is None:
            return {}
        
        # Calculate diversity (average distance to global best)
        distances = np.linalg.norm(
            self.particles - self.global_best, 
            axis=1
        )
        diversity = np.mean(distances)
        
        # Fitness statistics
        current_fitness = np.array([
            self._evaluate_fitness(self.particles[i])
            for i in range(self.config.n_particles)
        ])
        
        return {
            'diversity': diversity,
            'mean_fitness': np.mean(current_fitness),
            'std_fitness': np.std(current_fitness),
            'min_fitness': np.min(current_fitness),
            'max_fitness': np.max(current_fitness),
            'best_fitness': self.global_best_fitness
        }

def run_multiple_optimizations(
    portfolio: Portfolio,
    n_runs: int = 5,
    config: Optional[PSOConfig] = None
) -> List[OptimizationResult]:
    """
    Run multiple independent optimizations and return all results.
    
    Useful for assessing consistency and finding robust solutions.
    
    Args:
        portfolio: Portfolio instance
        n_runs: Number of independent runs
        config: PSO configuration (uses default if None)
        
    Returns:
        List of OptimizationResult from each run
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running {n_runs} independent optimizations")
    
    if config is None:
        config = PSOConfig()
    
    results = []
    for run in range(n_runs):
        logger.info(f"Starting run {run + 1}/{n_runs}")
        optimizer = PSOOptimizer(portfolio, config)
        result = optimizer.optimize()
        results.append(result)
        logger.info(f"Run {run + 1} complete - Fitness: {result.best_fitness:.6f}")
    
    return results

def select_best_result(results: List[OptimizationResult]) -> OptimizationResult:
    """
    Select best result from multiple optimization runs.
    
    Args:
        results: List of optimization results
        
    Returns:
        Best optimization result based on fitness
        
    Example:
        results = run_multiple_optimizations(portfolio, n_runs=5)
        best = select_best_result(results)
    """
    return max(results, key=lambda r: r.best_fitness)
