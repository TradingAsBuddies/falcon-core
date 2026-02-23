"""
Opening Range Breakout (ORB) Strategy

Source: Classic strategy popularized by Warrior Trading / Ross Cameron
Style: Day Trading (9:30 AM - 3:30 PM Eastern)

*** INTRADAY STRATEGY - REQUIRES 5-MINUTE DATA ***

Concept:
Trade the breakout of the first 15-minute opening range. The opening range
captures initial supply/demand imbalance. A breakout signals directional conviction.

Entry Rules:
1. Calculate the high and low of the first 15 minutes (opening range)
2. Long: price breaks above range high + buffer
3. Short: price breaks below range low - buffer
4. Volume must confirm the breakout

Exit Rules:
- Stop loss at opposite side of the opening range
- Take profit at 2:1 R/R or end of day
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, time

from falcon_core.backtesting.strategies.base import (
    BaseStrategy,
    StrategyParams,
    Signal,
    SignalType,
)


@dataclass
class OpeningRangeBreakoutParams(StrategyParams):
    """Parameters for Opening Range Breakout strategy."""

    recommended_interval: str = "5m"

    # Opening range
    orb_minutes: int = 15  # First 15 minutes define the range
    breakout_buffer: float = 0.002  # 0.2% buffer above/below range

    # Breakout confirmation
    stop_at_opposite: bool = False  # Use ATR-based stop instead (better R/R)
    min_breakout_volume: float = 1.8  # Volume must be 1.8x average

    # Risk management
    risk_reward_ratio: float = 2.0
    default_stop_loss_pct: float = 0.02  # 2% max stop
    position_size: float = 25000.0

    # Range filters
    min_range_pct: float = 0.003  # Range must be at least 0.3% (not too tight)
    max_range_pct: float = 0.05  # Range must be at most 5% (allow wider on gappers)

    # Time filter
    trade_start_time: str = "09:45"  # After ORB is established
    trade_end_time: str = "15:00"  # No new entries in last hour
    no_new_trades_after: str = "14:00"

    @classmethod
    def from_dict(cls, data: Dict) -> "OpeningRangeBreakoutParams":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """
    Opening Range Breakout (ORB) Strategy

    Trades breakouts of the first 15-minute range with volume confirmation.
    Classic momentum strategy popular among day traders.
    """

    name = "opening_range_breakout"
    description = "Trade breakout of first 15-min opening range"
    version = "1.0.0"
    source_url = None
    source_creator = "Classic (Warrior Trading / Ross Cameron)"
    trading_style = "Day Trading / Momentum"

    def __init__(self, params: Optional[OpeningRangeBreakoutParams] = None):
        super().__init__(params or self.default_params())

    @classmethod
    def default_params(cls) -> OpeningRangeBreakoutParams:
        return OpeningRangeBreakoutParams()

    @classmethod
    def param_ranges(cls) -> Dict[str, Tuple[float, float, float]]:
        return {
            "orb_minutes": (5, 30, 5),
            "breakout_buffer": (0.0005, 0.003, 0.0005),
            "min_breakout_volume": (1.0, 2.5, 0.25),
            "risk_reward_ratio": (1.5, 3.0, 0.5),
            "min_range_pct": (0.002, 0.008, 0.001),
        }

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        indicators = super().calculate_indicators(data)

        # Volume ratio
        indicators['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

        return indicators

    def _calculate_opening_range(self, data: pd.DataFrame) -> Optional[Dict]:
        """Calculate the opening range from the first N minutes of data."""
        params = self.params

        if len(data) < 3:
            return None

        # Determine how many bars make up the opening range
        # For 5-min bars, 15 min = 3 bars
        interval_minutes = 5  # Default to 5-min
        if len(data) > 1:
            delta = (data.index[1] - data.index[0]).total_seconds() / 60
            if delta > 0:
                interval_minutes = int(delta)

        orb_bars = max(1, params.orb_minutes // interval_minutes)
        orb_bars = min(orb_bars, len(data))

        orb_data = data.iloc[:orb_bars]
        range_high = orb_data['high'].max()
        range_low = orb_data['low'].min()
        range_pct = (range_high - range_low) / range_low if range_low > 0 else 0

        if range_pct < params.min_range_pct or range_pct > params.max_range_pct:
            return None

        return {
            'high': range_high,
            'low': range_low,
            'range_pct': range_pct,
            'bars': orb_bars,
        }

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        params = self.params

        if not isinstance(data.index, pd.DatetimeIndex):
            return signals

        # Process each trading day independently
        dates = data.index.date
        unique_dates = sorted(set(dates))

        for today in unique_dates:
            day_data = data[dates == today]
            if len(day_data) < 3:
                continue

            # Calculate fresh ORB for this day
            orb = self._calculate_opening_range(day_data)
            if orb is None:
                continue

            range_high = orb['high']
            range_low = orb['low']
            buffer_high = range_high * (1 + params.breakout_buffer)
            buffer_low = range_low * (1 - params.breakout_buffer)

            # Reset all state for each new day (ORB is a day-trade strategy)
            in_position = False
            position_direction = None
            entry_price = None
            stop_loss = None
            take_profit = None
            pending_breakout = None

            # Start scanning after the opening range is established
            for i in range(orb['bars'], len(day_data)):
                candle = day_data.iloc[i]
                timestamp = day_data.index[i]

                # Force close at end of day
                if i == len(day_data) - 1 and in_position:
                    exit_type = SignalType.EXIT_LONG if position_direction == 'long' else SignalType.EXIT_SHORT
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=exit_type,
                        price=candle['close'],
                        symbol="",
                        confidence=0.5,
                        reason="End of day — closing ORB position",
                    ))
                    in_position = False
                    break

                # Exit logic
                if in_position:
                    if position_direction == 'long':
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
                    elif position_direction == 'short':
                        if candle['high'] >= stop_loss:
                            signals.append(Signal(
                                timestamp=timestamp,
                                signal_type=SignalType.EXIT_SHORT,
                                price=stop_loss,
                                symbol="",
                                confidence=1.0,
                                reason="Stop loss triggered",
                            ))
                            in_position = False
                            continue

                        if candle['low'] <= take_profit:
                            signals.append(Signal(
                                timestamp=timestamp,
                                signal_type=SignalType.EXIT_SHORT,
                                price=take_profit,
                                symbol="",
                                confidence=1.0,
                                reason="Take profit hit",
                            ))
                            in_position = False
                            continue

                    continue  # Still in position, skip entry logic

                # Follow-through check: confirm breakout bar holds on next bar
                if pending_breakout is not None:
                    direction = pending_breakout['direction']
                    if direction == 'long' and candle['close'] > range_high:
                        vol_ratio = pending_breakout['vol_ratio']

                        if params.stop_at_opposite:
                            stop = range_low
                        else:
                            stop = candle['close'] * (1 - params.default_stop_loss_pct)

                        risk = candle['close'] - stop
                        target = candle['close'] + risk * params.risk_reward_ratio

                        confidence = min(1.0, 0.5 + min(vol_ratio / 5, 0.3) + orb['range_pct'] * 5)

                        signals.append(Signal(
                            timestamp=timestamp,
                            signal_type=SignalType.LONG,
                            price=candle['close'],
                            symbol="",
                            confidence=confidence,
                            stop_loss=stop,
                            take_profit=target,
                            position_size=params.position_size,
                            reason=f"ORB breakout long (confirmed): range {orb['range_pct']:.2%}, vol {vol_ratio:.1f}x",
                            metadata={
                                'range_high': range_high,
                                'range_low': range_low,
                                'range_pct': orb['range_pct'],
                                'volume_ratio': vol_ratio,
                                'direction': 'long',
                            },
                        ))

                        in_position = True
                        position_direction = 'long'
                        entry_price = candle['close']
                        stop_loss = stop
                        take_profit = target

                    elif direction == 'short' and candle['close'] < range_low:
                        vol_ratio = pending_breakout['vol_ratio']

                        if params.stop_at_opposite:
                            stop = range_high
                        else:
                            stop = candle['close'] * (1 + params.default_stop_loss_pct)

                        risk = stop - candle['close']
                        target = candle['close'] - risk * params.risk_reward_ratio

                        confidence = min(1.0, 0.5 + min(vol_ratio / 5, 0.3) + orb['range_pct'] * 5)

                        signals.append(Signal(
                            timestamp=timestamp,
                            signal_type=SignalType.SHORT,
                            price=candle['close'],
                            symbol="",
                            confidence=confidence,
                            stop_loss=stop,
                            take_profit=target,
                            position_size=params.position_size,
                            reason=f"ORB breakout short (confirmed): range {orb['range_pct']:.2%}, vol {vol_ratio:.1f}x",
                            metadata={
                                'range_high': range_high,
                                'range_low': range_low,
                                'range_pct': orb['range_pct'],
                                'volume_ratio': vol_ratio,
                                'direction': 'short',
                            },
                        ))

                        in_position = True
                        position_direction = 'short'
                        entry_price = candle['close']
                        stop_loss = stop
                        take_profit = target

                    # Clear pending regardless of outcome
                    pending_breakout = None
                    continue

                # Breakout above range high — queue for follow-through
                if candle['close'] > buffer_high:
                    vol_ratio = candle.get('volume_ratio', 0)
                    if vol_ratio < params.min_breakout_volume:
                        continue

                    pending_breakout = {'direction': 'long', 'vol_ratio': vol_ratio}

                # Breakout below range low — queue for follow-through
                elif candle['close'] < buffer_low:
                    vol_ratio = candle.get('volume_ratio', 0)
                    if vol_ratio < params.min_breakout_volume:
                        continue

                    pending_breakout = {'direction': 'short', 'vol_ratio': vol_ratio}

        return signals
