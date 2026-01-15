"""
Falcon Core - Shared libraries for Falcon Trading Platform

Components:
- db_manager: Database abstraction (SQLite/PostgreSQL)
- finviz_client: Rate-limited Finviz Elite API client
- config: Configuration management
- backtesting: Event-driven backtesting with feedback loop
"""

from falcon_core.db_manager import DatabaseManager, get_db_manager
from falcon_core.finviz_client import (
    FinvizClient,
    get_finviz_client,
    RateLimitConfig,
    fetch_finviz_stocks,
)
from falcon_core.config import FalconConfig, get_config

# Backtesting module (lazy import to avoid dependency issues)
def get_backtest_engine(*args, **kwargs):
    """Factory function for backtest engine"""
    from falcon_core.backtesting.engine import create_engine
    return create_engine(*args, **kwargs)

def get_optimizer(*args, **kwargs):
    """Factory function for parameter optimizer"""
    from falcon_core.backtesting.optimizer import ParameterOptimizer
    return ParameterOptimizer(*args, **kwargs)

def get_data_feed(*args, **kwargs):
    """Factory function for data feed"""
    from falcon_core.backtesting.data_feed import DataFeed
    return DataFeed(*args, **kwargs)

__version__ = "0.2.0"
__all__ = [
    # Database
    "DatabaseManager",
    "get_db_manager",
    # Finviz
    "FinvizClient",
    "get_finviz_client",
    "RateLimitConfig",
    "fetch_finviz_stocks",
    # Config
    "FalconConfig",
    "get_config",
    # Backtesting
    "get_backtest_engine",
    "get_optimizer",
    "get_data_feed",
]
