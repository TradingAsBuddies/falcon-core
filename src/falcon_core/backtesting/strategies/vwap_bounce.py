"""
VWAP Bounce Strategy

Source: Popular institutional-level strategy from day trading communities
Style: Day Trading (9:30 AM - 3:30 PM Eastern)

*** INTRADAY STRATEGY - REQUIRES 5-MINUTE DATA ***

Concept:
Trade bounces off VWAP (Volume Weighted Average Price) with volume confirmation.
VWAP acts as an institutional magnet — price tends to revert to it and bounce.

Entry Rules:
1. Price approaches VWAP within tolerance band
2. Volume on the touch candle is above average (institutional participation)
3. Rejection candle forms (long wick showing bounce off VWAP)
4. Enter in direction of the bounce

Exit Rules:
- Stop loss: Below VWAP (for longs) or above VWAP (for shorts)
- Take profit: Prior swing high/low or 2:1 R/R
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from falcon_core.backtesting.strategies.base import (
    BaseStrategy,
    StrategyParams,
    Signal,
    SignalType,
)


@dataclass
class VWAPBounceParams(StrategyParams):
    """Parameters for VWAP Bounce strategy."""

    recommended_interval: str = "5m"

    # VWAP proximity
    vwap_tolerance: float = 0.002  # 0.2% band around VWAP
    min_volume_ratio: float = 1.5  # Volume must be 1.5x average on touch

    # Rejection candle
    rejection_wick_pct: float = 0.6  # Wick must be 60%+ of candle range

    # Risk management
    risk_reward_ratio: float = 2.0
    default_stop_loss_pct: float = 0.015  # 1.5% stop for intraday
    position_size: float = 25000.0

    # Time filter
    trade_start_time: str = "09:45"  # Let VWAP establish (15 min after open)
    trade_end_time: str = "15:30"

    # Trend filter
    use_trend_filter: bool = True  # Only trade bounces in trend direction
    trend_sma_period: int = 50

    @classmethod
    def from_dict(cls, data: Dict) -> "VWAPBounceParams":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class VWAPBounceStrategy(BaseStrategy):
    """
    VWAP Bounce Strategy

    Trades bounces off VWAP with volume confirmation and rejection candle
    pattern. VWAP is recalculated from the start of each trading day.
    """

    name = "vwap_bounce"
    description = "Trade bounces off VWAP with volume confirmation"
    version = "1.0.0"
    source_url = None
    source_creator = "Institutional day trading (YouTube community)"
    trading_style = "Day Trading"

    def __init__(self, params: Optional[VWAPBounceParams] = None):
        super().__init__(params or self.default_params())

    @classmethod
    def default_params(cls) -> VWAPBounceParams:
        return VWAPBounceParams()

    @classmethod
    def param_ranges(cls) -> Dict[str, Tuple[float, float, float]]:
        return {
            "vwap_tolerance": (0.001, 0.005, 0.001),
            "min_volume_ratio": (1.0, 3.0, 0.5),
            "rejection_wick_pct": (0.4, 0.8, 0.1),
            "risk_reward_ratio": (1.5, 3.0, 0.5),
            "default_stop_loss_pct": (0.01, 0.03, 0.005),
        }

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        indicators = super().calculate_indicators(data)

        # VWAP: cumulative (typical_price * volume) / cumulative volume
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        cum_vol = data['volume'].cumsum()
        cum_tp_vol = (typical_price * data['volume']).cumsum()
        indicators['vwap'] = cum_tp_vol / cum_vol

        # Volume ratio
        indicators['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

        # Candle properties
        indicators['body'] = abs(data['close'] - data['open'])
        indicators['upper_wick'] = data['high'] - data[['open', 'close']].max(axis=1)
        indicators['lower_wick'] = data[['open', 'close']].min(axis=1) - data['low']
        indicators['range'] = data['high'] - data['low']

        # Trend SMA
        indicators['trend_sma'] = data['close'].rolling(
            self.params.trend_sma_period
        ).mean()

        return indicators

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        params = self.params

        in_position = False
        entry_price = None
        stop_loss = None
        take_profit = None

        for i in range(max(params.trend_sma_period, 20), len(data)):
            candle = data.iloc[i]
            timestamp = data.index[i]

            # Exit logic
            if in_position:
                if candle['low'] <= stop_loss:
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.EXIT_LONG,
                        price=stop_loss,
                        symbol="",
                        confidence=1.0,
                        reason="Stop loss triggered",
                    ))
                    in_position = False
                    continue

                if candle['high'] >= take_profit:
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.EXIT_LONG,
                        price=take_profit,
                        symbol="",
                        confidence=1.0,
                        reason="Take profit hit",
                    ))
                    in_position = False
                continue

            vwap = candle.get('vwap', None)
            if vwap is None or pd.isna(vwap) or vwap == 0:
                continue

            # Check VWAP proximity — price low touches VWAP band
            vwap_upper = vwap * (1 + params.vwap_tolerance)
            vwap_lower = vwap * (1 - params.vwap_tolerance)

            touches_vwap = candle['low'] <= vwap_upper and candle['low'] >= vwap_lower

            if not touches_vwap:
                continue

            # Volume confirmation
            if candle.get('volume_ratio', 0) < params.min_volume_ratio:
                continue

            # Rejection candle: long lower wick bouncing off VWAP
            candle_range = candle.get('range', 0)
            if candle_range == 0:
                continue

            lower_wick = candle.get('lower_wick', 0)
            wick_pct = lower_wick / candle_range
            if wick_pct < params.rejection_wick_pct:
                continue

            # Must close above VWAP (bullish rejection)
            if candle['close'] <= vwap:
                continue

            # Trend filter: only long if price is above trend SMA
            if params.use_trend_filter:
                trend_sma = candle.get('trend_sma', None)
                if trend_sma is not None and not pd.isna(trend_sma):
                    if candle['close'] < trend_sma:
                        continue

            # Entry signal
            stop = vwap * (1 - params.default_stop_loss_pct)
            risk = candle['close'] - stop
            target = candle['close'] + risk * params.risk_reward_ratio

            confidence = min(1.0, 0.5 + wick_pct * 0.3 + min(candle.get('volume_ratio', 0) / 5, 0.2))

            signals.append(Signal(
                timestamp=timestamp,
                signal_type=SignalType.LONG,
                price=candle['close'],
                symbol="",
                confidence=confidence,
                stop_loss=stop,
                take_profit=target,
                position_size=params.position_size,
                reason=f"VWAP bounce: {wick_pct:.0%} wick rejection, {candle.get('volume_ratio', 0):.1f}x volume",
                metadata={
                    'vwap': vwap,
                    'wick_pct': wick_pct,
                    'volume_ratio': candle.get('volume_ratio', 0),
                },
            ))

            in_position = True
            entry_price = candle['close']
            stop_loss = stop
            take_profit = target

        return signals
