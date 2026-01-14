"""
Falcon Core - Shared libraries for Falcon Trading Platform

Components:
- db_manager: Database abstraction (SQLite/PostgreSQL)
- finviz_client: Rate-limited Finviz Elite API client
- config: Configuration management
"""

from falcon_core.db_manager import DatabaseManager, get_db_manager
from falcon_core.finviz_client import (
    FinvizClient,
    get_finviz_client,
    RateLimitConfig,
    fetch_finviz_stocks,
)
from falcon_core.config import Config

__version__ = "0.1.0"
__all__ = [
    "DatabaseManager",
    "get_db_manager",
    "FinvizClient",
    "get_finviz_client",
    "RateLimitConfig",
    "fetch_finviz_stocks",
    "Config",
]
