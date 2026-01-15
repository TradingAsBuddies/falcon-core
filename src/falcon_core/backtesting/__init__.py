"""
Falcon Backtesting Module

Event-driven and vectorized backtesting with feedback loop optimization.

Components:
- engine: Backtest execution engine (bt default, optional backtrader)
- strategies: Codified trading strategies from YouTube extractions
- optimizer: Parameter optimization with feedback loop
- data_feed: Market data loading from database/files
"""

from falcon_core.backtesting.engine import (
    BacktestEngine,
    BacktestResult,
    create_engine,
)
from falcon_core.backtesting.strategies.base import (
    BaseStrategy,
    StrategyParams,
    Signal,
    SignalType,
)
from falcon_core.backtesting.data_feed import DataFeed
from falcon_core.backtesting.optimizer import (
    ParameterOptimizer,
    OptimizationResult,
)

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestResult",
    "create_engine",
    # Strategies
    "BaseStrategy",
    "StrategyParams",
    "Signal",
    "SignalType",
    # Data
    "DataFeed",
    # Optimizer
    "ParameterOptimizer",
    "OptimizationResult",
]
