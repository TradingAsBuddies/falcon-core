"""
Red-to-Green Move Strategy

Source: Popular small-cap day trading channels (YouTube community)
Style: Scalping / Momentum (first 30 minutes of trading)

*** INTRADAY STRATEGY - REQUIRES 1-MINUTE DATA ***

Concept:
Trade the first candle color reversal on gapping stocks. When a stock gaps down
at open and prints its first green (bullish) candle, it signals a momentum reversal
as short-sellers cover and dip buyers step in.

Entry Rules:
1. Stock gapped down by at least min_gap_pct from previous close
2. First green candle after open = long entry
3. Must occur within max_entry_minutes of market open
4. Volume must be above average

Exit Rules:
- Stop loss below the gap low
- Take profit at previous close (gap fill) or 2:1 R/R
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
class RedToGreenParams(StrategyParams):
    """Parameters for Red-to-Green strategy."""

    recommended_interval: str = "1m"

    # Gap criteria
    min_gap_pct: float = 0.01  # Minimum 1% gap down
    max_gap_pct: float = 0.15  # Max 15% gap (avoid extreme moves)

    # Entry timing
    max_entry_minutes: int = 30  # Must trigger within 30 min of open

    # Confirmation
    min_green_body_pct: float = 0.003  # Green candle body must be >0.3% of price
    min_volume_ratio: float = 1.5  # Volume above average

    # Risk management
    risk_reward_ratio: float = 2.0
    default_stop_loss_pct: float = 0.015  # 1.5% stop
    position_size: float = 15000.0  # Smaller size for volatile gap plays

    # Time filter
    trade_start_time: str = "09:30"
    trade_end_time: str = "10:00"  # Only trade first 30 min

    @classmethod
    def from_dict(cls, data: Dict) -> "RedToGreenParams":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class RedToGreenStrategy(BaseStrategy):
    """
    Red-to-Green Move Strategy

    Trades the first bullish candle reversal on stocks that gap down at open.
    Momentum reversal play targeting gap fill.
    """

    name = "red_to_green"
    description = "First green candle reversal on gap-down stocks"
    version = "1.0.0"
    source_url = None
    source_creator = "Small-cap day trading community"
    trading_style = "Scalping / Momentum Reversal"

    def __init__(self, params: Optional[RedToGreenParams] = None):
        super().__init__(params or self.default_params())

    @classmethod
    def default_params(cls) -> RedToGreenParams:
        return RedToGreenParams()

    @classmethod
    def param_ranges(cls) -> Dict[str, Tuple[float, float, float]]:
        return {
            "min_gap_pct": (0.02, 0.06, 0.01),
            "max_entry_minutes": (15, 60, 15),
            "min_green_body_pct": (0.001, 0.005, 0.001),
            "min_volume_ratio": (1.0, 3.0, 0.5),
            "risk_reward_ratio": (1.5, 3.0, 0.5),
        }

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        indicators = super().calculate_indicators(data)

        # Volume ratio
        indicators['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

        # Candle color: green = close > open
        indicators['is_green'] = data['close'] > data['open']
        indicators['body_pct'] = abs(data['close'] - data['open']) / data['close']

        return indicators

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        params = self.params

        if len(data) < 2:
            return signals

        # Detect day boundaries from the DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            return signals

        dates = data.index.date
        unique_dates = sorted(set(dates))

        if len(unique_dates) < 2:
            return signals

        # Determine bar interval for max entry window
        interval_minutes = 1
        if len(data) > 1:
            delta = (data.index[1] - data.index[0]).total_seconds() / 60
            if delta > 0:
                interval_minutes = int(delta)
        max_bars = max(1, params.max_entry_minutes // interval_minutes)

        # Process each day independently (starting from day 2)
        in_position = False
        stop_loss = None
        take_profit = None

        for day_idx in range(1, len(unique_dates)):
            today = unique_dates[day_idx]
            yesterday = unique_dates[day_idx - 1]

            yesterday_data = data[dates == yesterday]
            today_data = data[dates == today]

            if yesterday_data.empty or today_data.empty:
                continue

            # Use previous day's last close as prev_close
            prev_close = yesterday_data.iloc[-1]['close']
            open_price = today_data.iloc[0]['open']

            # Check gap: open must be below prev close by min_gap_pct
            gap_pct = (prev_close - open_price) / prev_close if prev_close > 0 else 0

            if gap_pct < params.min_gap_pct or gap_pct > params.max_gap_pct:
                continue

            # Find first green candle in today's data
            session_low = today_data.iloc[0]['low']
            entry_found = False

            for j in range(len(today_data)):
                candle = today_data.iloc[j]
                timestamp = today_data.index[j]

                # Track session low
                session_low = min(session_low, candle['low'])

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
                            reason="Take profit (gap fill target)",
                        ))
                        in_position = False
                    continue

                if entry_found:
                    continue

                # Only look for entries in the first N bars of the day
                if j >= max_bars:
                    continue

                # Look for first green candle
                if not candle.get('is_green', False):
                    continue

                # Body must be meaningful
                body_pct = candle.get('body_pct', 0)
                if body_pct < params.min_green_body_pct:
                    continue

                # Volume confirmation
                vol_ratio = candle.get('volume_ratio', 0)
                if not pd.isna(vol_ratio) and vol_ratio < params.min_volume_ratio:
                    continue

                # Entry: first green candle
                stop = session_low * (1 - 0.002)  # Just below session low
                risk = candle['close'] - stop

                # Target is gap fill (previous close) or R/R based
                gap_fill_target = prev_close
                rr_target = candle['close'] + risk * params.risk_reward_ratio
                target = min(gap_fill_target, rr_target)  # Closer target

                if risk <= 0:
                    continue

                confidence = min(1.0, 0.5 + gap_pct * 3 + min(body_pct * 50, 0.2))

                signals.append(Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.LONG,
                    price=candle['close'],
                    symbol="",
                    confidence=confidence,
                    stop_loss=stop,
                    take_profit=target,
                    position_size=params.position_size,
                    reason=f"Red-to-green: {gap_pct:.1%} gap down, first green candle",
                    metadata={
                        'gap_pct': gap_pct,
                        'prev_close': prev_close,
                        'session_low': session_low,
                        'candle_number': j,
                        'body_pct': body_pct,
                    },
                ))

                in_position = True
                entry_found = True
                stop_loss = stop
                take_profit = target

        return signals
