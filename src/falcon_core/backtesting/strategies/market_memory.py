"""
Market Memory Model (Second Day Trading Strategy)

Source: Unknown Creator (YouTube extraction)
Style: Day trading

*** INTRADAY STRATEGY - REQUIRES 5-MINUTE DATA FOR DAY 2 ***

Concept:
The market "remembers" yesterday's key levels. Day 2 of a big move
often sees price test and react to yesterday's high, low, and
key intraday battle zones.

Entry Rules:
1. Find "Day 2" stocks: significant Day 1 move with catalyst
2. Mark yesterday's high, low, and key battle zones
3. Wait for "John Wick" candle - large wick showing rejection at key level
4. SHORT: Price runs to yesterday's high, gets rejected (John Wick down)
5. LONG: Price tests yesterday's low, shows reversal (John Wick up)
6. BREAKOUT: If price holds above yesterday's high, wait for John Wick confirmation
7. Only trade at extremes - avoid middle of range

Exit Rules:
- Initial stop: just beyond the John Wick candle
- Move to breakeven as trade progresses
- Target: opposite side of yesterday's range
- Can hold through VWAP for extended targets

Best Candidates:
- Stocks with 5%+ Day 1 move
- Clear catalyst (earnings, news)
- Day 2 opens inside or near Day 1 range
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

from falcon_core.backtesting.strategies.base import (
    BaseStrategy,
    StrategyParams,
    Signal,
    SignalType,
)


@dataclass
class MarketMemoryParams(StrategyParams):
    """
    Parameters for Market Memory Model strategy.

    Optimized for 5-minute intraday data, specifically for Day 2 setups
    after a significant Day 1 move.
    """

    # Timeframe
    recommended_interval: str = "5m"

    # Day 1 move qualification
    min_day1_move_pct: float = 0.05  # 5% minimum move on Day 1
    min_day1_volume_mult: float = 1.5  # 1.5x average volume on Day 1

    # Key level zones
    zone_tolerance: float = 0.003  # 0.3% tolerance for level zones (tighter for intraday)
    range_middle_pct: float = 0.3  # Avoid middle 30% of range

    # John Wick candle detection
    min_wick_pct: float = 0.6  # Wick must be 60% of candle range
    max_body_pct: float = 0.3  # Body must be <30% of range
    min_candle_range_atr: float = 0.5  # Min candle range as % of ATR

    # Trade management
    breakeven_trigger_pct: float = 0.005  # Move stop to BE after 0.5% profit (tighter for intraday)
    target_range_mult: float = 1.0  # Target opposite side of range

    # Position sizing
    position_size: float = 20000.0

    # Time filter - CRITICAL for Day 2 setups
    trade_start_time: str = "09:35"  # Start 5 min after open
    trade_end_time: str = "15:00"  # Stop 1 hour before close
    prime_window_start: str = "09:35"  # Best setups in first 90 min
    prime_window_end: str = "11:00"

    # Risk management
    max_risk_per_trade_pct: float = 0.01  # 1% max risk per trade

    @classmethod
    def from_dict(cls, data: Dict) -> "MarketMemoryParams":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class MarketMemoryStrategy(BaseStrategy):
    """
    Market Memory Model (Day 2) Strategy

    Trades reactions at yesterday's key levels using John Wick
    candle patterns for confirmation.
    """

    name = "market_memory"
    description = "Day 2 trading at yesterday's key levels with John Wick confirmation"
    version = "1.0.0"
    source_url = "https://youtube.com/watch?v=..."
    source_creator = "Unknown Creator"
    trading_style = "Day trading"

    def __init__(self, params: Optional[MarketMemoryParams] = None):
        super().__init__(params or self.default_params())
        self._yesterday_levels: Dict = {}

    @classmethod
    def default_params(cls) -> MarketMemoryParams:
        return MarketMemoryParams()

    @classmethod
    def param_ranges(cls) -> Dict[str, Tuple[float, float, float]]:
        """Parameter ranges for optimization"""
        return {
            "min_day1_move_pct": (0.03, 0.10, 0.01),
            "zone_tolerance": (0.003, 0.010, 0.001),
            "min_wick_pct": (0.5, 0.8, 0.1),
            "max_body_pct": (0.2, 0.4, 0.05),
            "breakeven_trigger_pct": (0.005, 0.02, 0.005),
            "position_size": (10000.0, 50000.0, 5000.0),
        }

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate indicators for Market Memory strategy"""
        indicators = super().calculate_indicators(data)

        # Candle components
        indicators['body'] = abs(data['close'] - data['open'])
        indicators['upper_wick'] = data['high'] - data[['open', 'close']].max(axis=1)
        indicators['lower_wick'] = data[['open', 'close']].min(axis=1) - data['low']
        indicators['range'] = data['high'] - data['low']
        indicators['is_bullish'] = data['close'] > data['open']

        # VWAP (simplified - cumulative)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        indicators['vwap'] = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()

        return indicators

    def _get_daily_levels(self, daily_data: pd.DataFrame) -> Dict:
        """
        Extract key levels from previous day's data.

        Returns dict with:
        - high: Yesterday's high
        - low: Yesterday's low
        - close: Yesterday's close
        - range: Yesterday's range
        - battle_zones: List of significant intraday levels
        """
        if daily_data.empty:
            return {}

        high = daily_data['high'].max()
        low = daily_data['low'].min()
        close = daily_data['close'].iloc[-1]
        open_price = daily_data['open'].iloc[0]
        range_size = high - low

        # Identify battle zones (areas of consolidation/heavy trading)
        # Simple approach: find price levels with high volume
        battle_zones = []

        # Divide range into zones and find high-volume areas
        zone_count = 5
        zone_size = range_size / zone_count

        for i in range(zone_count):
            zone_low = low + i * zone_size
            zone_high = zone_low + zone_size
            zone_mid = (zone_low + zone_high) / 2

            # Count volume in this zone
            mask = (daily_data['close'] >= zone_low) & (daily_data['close'] < zone_high)
            zone_volume = daily_data.loc[mask, 'volume'].sum()

            if zone_volume > daily_data['volume'].mean() * len(daily_data) / zone_count * 1.2:
                battle_zones.append({
                    'price': zone_mid,
                    'type': 'battle_zone',
                    'volume': zone_volume,
                })

        return {
            'high': high,
            'low': low,
            'open': open_price,
            'close': close,
            'range': range_size,
            'move_pct': abs(close - open_price) / open_price,
            'battle_zones': battle_zones,
        }

    def _is_john_wick_up(self, candle: pd.Series) -> bool:
        """
        Check if candle is a bullish John Wick (rejection of lows).

        John Wick up: Long lower wick, small body, closes in upper half
        """
        params = self.params
        range_ = candle['range']
        if range_ == 0:
            return False

        lower_wick_pct = candle['lower_wick'] / range_
        body_pct = candle['body'] / range_

        return (
            lower_wick_pct >= params.min_wick_pct and
            body_pct <= params.max_body_pct and
            candle['is_bullish']  # Closes bullish
        )

    def _is_john_wick_down(self, candle: pd.Series) -> bool:
        """
        Check if candle is a bearish John Wick (rejection of highs).

        John Wick down: Long upper wick, small body, closes in lower half
        """
        params = self.params
        range_ = candle['range']
        if range_ == 0:
            return False

        upper_wick_pct = candle['upper_wick'] / range_
        body_pct = candle['body'] / range_

        return (
            upper_wick_pct >= params.min_wick_pct and
            body_pct <= params.max_body_pct and
            not candle['is_bullish']  # Closes bearish
        )

    def _is_near_level(self, price: float, level: float, tolerance: float) -> bool:
        """Check if price is near a key level"""
        return abs(price - level) / level <= tolerance

    def _is_in_middle(self, price: float, high: float, low: float) -> bool:
        """Check if price is in the middle of the range (avoid zone)"""
        params = self.params
        range_size = high - low
        middle_low = low + range_size * (0.5 - params.range_middle_pct / 2)
        middle_high = low + range_size * (0.5 + params.range_middle_pct / 2)
        return middle_low <= price <= middle_high

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on Market Memory Model"""
        signals = []
        params = self.params

        # This strategy requires multi-day data
        # We need to identify "yesterday" in the data
        if not isinstance(data.index, pd.DatetimeIndex):
            return signals

        # Group by date to find day boundaries
        data['date'] = data.index.date
        dates = data['date'].unique()

        if len(dates) < 2:
            # Need at least 2 days of data
            return signals

        # Track state
        in_position = False
        position_type = None  # 'long' or 'short'
        entry_price = None
        stop_loss = None
        target_price = None
        breakeven_moved = False

        # Process each day (starting from day 2)
        for day_idx in range(1, len(dates)):
            yesterday = dates[day_idx - 1]
            today = dates[day_idx]

            yesterday_data = data[data['date'] == yesterday]
            today_data = data[data['date'] == today]

            # Get yesterday's levels
            levels = self._get_daily_levels(yesterday_data)
            if not levels:
                continue

            # Check if yesterday qualifies as "Day 1" (big move)
            if levels['move_pct'] < params.min_day1_move_pct:
                continue

            yesterday_high = levels['high']
            yesterday_low = levels['low']
            yesterday_range = levels['range']

            # Process today's candles
            for i, (timestamp, candle) in enumerate(today_data.iterrows()):
                # Manage existing position
                if in_position:
                    # Check stop loss
                    if position_type == 'long' and candle['low'] <= stop_loss:
                        signals.append(Signal(
                            timestamp=timestamp,
                            signal_type=SignalType.EXIT_LONG,
                            price=stop_loss,
                            symbol="",
                            confidence=1.0,
                            reason="Stop loss hit",
                        ))
                        in_position = False
                        continue

                    # Check take profit
                    if position_type == 'long' and candle['high'] >= target_price:
                        signals.append(Signal(
                            timestamp=timestamp,
                            signal_type=SignalType.EXIT_LONG,
                            price=target_price,
                            symbol="",
                            confidence=1.0,
                            reason="Target reached (opposite side of range)",
                        ))
                        in_position = False
                        continue

                    # Move stop to breakeven
                    if not breakeven_moved and position_type == 'long':
                        profit_pct = (candle['close'] - entry_price) / entry_price
                        if profit_pct >= params.breakeven_trigger_pct:
                            stop_loss = entry_price
                            breakeven_moved = True

                    continue

                # Skip if in middle of range
                if self._is_in_middle(candle['close'], yesterday_high, yesterday_low):
                    continue

                # LONG SETUP: Price near yesterday's low with John Wick up
                if self._is_near_level(candle['low'], yesterday_low, params.zone_tolerance):
                    if self._is_john_wick_up(candle):
                        # Long entry
                        stop = candle['low'] - (candle['range'] * 0.5)
                        target = yesterday_high

                        signals.append(Signal(
                            timestamp=timestamp,
                            signal_type=SignalType.LONG,
                            price=candle['close'],
                            symbol="",
                            confidence=0.75,
                            stop_loss=stop,
                            take_profit=target,
                            position_size=params.position_size,
                            reason=f"John Wick up at yesterday's low ({yesterday_low:.2f})",
                            metadata={
                                'yesterday_low': yesterday_low,
                                'yesterday_high': yesterday_high,
                                'pattern': 'john_wick_up',
                            }
                        ))

                        in_position = True
                        position_type = 'long'
                        entry_price = candle['close']
                        stop_loss = stop
                        target_price = target
                        breakeven_moved = False
                        continue

                # BREAKOUT SETUP: Price breaks above yesterday's high with confirmation
                if candle['close'] > yesterday_high:
                    # Need John Wick confirmation after breakout
                    if self._is_john_wick_up(candle):
                        stop = yesterday_high - (yesterday_range * 0.1)
                        target = yesterday_high + yesterday_range  # Measured move

                        signals.append(Signal(
                            timestamp=timestamp,
                            signal_type=SignalType.LONG,
                            price=candle['close'],
                            symbol="",
                            confidence=0.70,
                            stop_loss=stop,
                            take_profit=target,
                            position_size=params.position_size,
                            reason=f"Breakout above yesterday's high ({yesterday_high:.2f}) with John Wick",
                            metadata={
                                'yesterday_high': yesterday_high,
                                'pattern': 'breakout_john_wick',
                            }
                        ))

                        in_position = True
                        position_type = 'long'
                        entry_price = candle['close']
                        stop_loss = stop
                        target_price = target
                        breakeven_moved = False

            # Close position at end of day if still open
            if in_position:
                last_candle = today_data.iloc[-1]
                signals.append(Signal(
                    timestamp=today_data.index[-1],
                    signal_type=SignalType.EXIT_LONG,
                    price=last_candle['close'],
                    symbol="",
                    confidence=0.5,
                    reason="End of day - closing position",
                ))
                in_position = False

        return signals
