"""
Falcon Trading Strategies

Strategy plugins loaded from database and plugin directory.
The public contract (BaseStrategy, StrategyParams, Signal, SignalType)
is defined in base.py — all strategy plugins import from there.
"""

from falcon_core.backtesting.strategies.base import (
    BaseStrategy,
    StrategyParams,
    Signal,
    SignalType,
)

__all__ = [
    "BaseStrategy",
    "StrategyParams",
    "Signal",
    "SignalType",
    "get_available_strategies",
]


def get_available_strategies(db=None):
    """Load strategy plugins from database and plugin directory.

    Args:
        db: DatabaseManager instance. If None, creates one automatically.

    Returns:
        Dict mapping strategy_name to strategy class
    """
    from falcon_core.backtesting.strategy_loader import get_all_strategies

    if db is None:
        try:
            from falcon_core.db_manager import get_db_manager
            db = get_db_manager()
            db.init_schema()
        except Exception:
            return {}

    return get_all_strategies(db=db)
