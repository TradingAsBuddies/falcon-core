"""
Falcon Trading Strategies

Codified strategies from YouTube extractions and custom implementations.
"""

from falcon_core.backtesting.strategies.base import (
    BaseStrategy,
    StrategyParams,
    Signal,
    SignalType,
)

# Import specific strategies when available
try:
    from falcon_core.backtesting.strategies.one_candle_rule import OneCandleRuleStrategy
except ImportError:
    OneCandleRuleStrategy = None

try:
    from falcon_core.backtesting.strategies.atr_breakout import ATRBreakoutStrategy
except ImportError:
    ATRBreakoutStrategy = None

try:
    from falcon_core.backtesting.strategies.market_memory import MarketMemoryStrategy
except ImportError:
    MarketMemoryStrategy = None

__all__ = [
    "BaseStrategy",
    "StrategyParams",
    "Signal",
    "SignalType",
    "OneCandleRuleStrategy",
    "ATRBreakoutStrategy",
    "MarketMemoryStrategy",
]


def get_available_strategies():
    """Return dict of available strategy classes"""
    strategies = {}
    if OneCandleRuleStrategy:
        strategies["one_candle_rule"] = OneCandleRuleStrategy
    if ATRBreakoutStrategy:
        strategies["atr_breakout"] = ATRBreakoutStrategy
    if MarketMemoryStrategy:
        strategies["market_memory"] = MarketMemoryStrategy
    return strategies
