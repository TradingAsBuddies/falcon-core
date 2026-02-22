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

try:
    from falcon_core.backtesting.strategies.vwap_bounce import VWAPBounceStrategy
except ImportError:
    VWAPBounceStrategy = None

try:
    from falcon_core.backtesting.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
except ImportError:
    OpeningRangeBreakoutStrategy = None

try:
    from falcon_core.backtesting.strategies.red_to_green import RedToGreenStrategy
except ImportError:
    RedToGreenStrategy = None

try:
    from falcon_core.backtesting.strategies.volatility_squeeze import VolatilitySqueezeStrategy
except ImportError:
    VolatilitySqueezeStrategy = None

try:
    from falcon_core.backtesting.strategies.microstructure_momentum import MicrostructureMomentumStrategy
except ImportError:
    MicrostructureMomentumStrategy = None

try:
    from falcon_core.backtesting.strategies.gap_fill_fade import GapFillFadeStrategy
except ImportError:
    GapFillFadeStrategy = None

__all__ = [
    "BaseStrategy",
    "StrategyParams",
    "Signal",
    "SignalType",
    "OneCandleRuleStrategy",
    "ATRBreakoutStrategy",
    "MarketMemoryStrategy",
    "VWAPBounceStrategy",
    "OpeningRangeBreakoutStrategy",
    "RedToGreenStrategy",
    "VolatilitySqueezeStrategy",
    "MicrostructureMomentumStrategy",
    "GapFillFadeStrategy",
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
    if VWAPBounceStrategy:
        strategies["vwap_bounce"] = VWAPBounceStrategy
    if OpeningRangeBreakoutStrategy:
        strategies["opening_range_breakout"] = OpeningRangeBreakoutStrategy
    if RedToGreenStrategy:
        strategies["red_to_green"] = RedToGreenStrategy
    if VolatilitySqueezeStrategy:
        strategies["volatility_squeeze"] = VolatilitySqueezeStrategy
    if MicrostructureMomentumStrategy:
        strategies["microstructure_momentum"] = MicrostructureMomentumStrategy
    if GapFillFadeStrategy:
        strategies["gap_fill_fade"] = GapFillFadeStrategy
    return strategies
