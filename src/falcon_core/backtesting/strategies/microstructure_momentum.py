"""
Microstructure Momentum Strategy

Source: AI-generated novel strategy
Style: Scalping (9:30 AM - 11:00 AM Eastern)

*** INTRADAY STRATEGY - REQUIRES 1-MINUTE DATA ***

Concept:
Detect order flow imbalances via volume-weighted price acceleration. Uses
cumulative delta proxy (up-volume vs down-volume) to identify aggressive
buyers or sellers before price moves. When cumulative delta diverges sharply
from price, it signals hidden institutional activity.

Entry Rules:
1. Calculate cumulative delta (up-vol minus down-vol) over lookback window
2. Detect acceleration: delta rate of change exceeds threshold
3. Volume surge confirms institutional participation
4. Enter in direction of delta imbalance

Exit Rules:
- Stop loss: Based on ATR
- Take profit: 2:1 R/R or delta reversal
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
class MicrostructureMomentumParams(StrategyParams):
    """Parameters for Microstructure Momentum strategy."""

    recommended_interval: str = "1m"

    # Delta calculation
    delta_lookback: int = 10  # Bars for cumulative delta window
    acceleration_threshold: float = 0.5  # Min delta acceleration for signal (normalized)

    # Volume surge
    volume_surge_mult: float = 2.5  # Volume must be 2.5x average

    # Price acceleration
    price_accel_lookback: int = 5  # Bars for price acceleration calc

    # Risk management
    atr_stop_mult: float = 1.5
    risk_reward_ratio: float = 2.0
    default_stop_loss_pct: float = 0.01  # 1% for scalping
    position_size: float = 15000.0

    # Time filter
    trade_start_time: str = "09:35"
    trade_end_time: str = "11:00"  # Only first 90 min (highest volume)

    # Cooldown
    min_bars_between_trades: int = 10  # Longer cooldown to avoid overtrading

    @classmethod
    def from_dict(cls, data: Dict) -> "MicrostructureMomentumParams":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class MicrostructureMomentumStrategy(BaseStrategy):
    """
    Microstructure Momentum Strategy

    Detects order flow imbalances through volume-delta analysis to identify
    aggressive buying/selling before price catches up.
    """

    name = "microstructure_momentum"
    description = "Volume-delta order flow momentum detection"
    version = "1.0.0"
    source_url = None
    source_creator = "AI-generated (Falcon)"
    trading_style = "Scalping / Order Flow"

    def __init__(self, params: Optional[MicrostructureMomentumParams] = None):
        super().__init__(params or self.default_params())

    @classmethod
    def default_params(cls) -> MicrostructureMomentumParams:
        return MicrostructureMomentumParams()

    @classmethod
    def param_ranges(cls) -> Dict[str, Tuple[float, float, float]]:
        return {
            "delta_lookback": (5, 20, 5),
            "acceleration_threshold": (0.001, 0.005, 0.001),
            "volume_surge_mult": (1.5, 3.0, 0.5),
            "price_accel_lookback": (3, 10, 1),
            "atr_stop_mult": (1.0, 2.5, 0.5),
        }

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        indicators = super().calculate_indicators(data)
        params = self.params

        # Proportional volume splitting based on candle position
        # Instead of binary up/down, split proportionally by where close is in range
        candle_range = data['high'] - data['low']
        # Fraction of range that is "up" (close relative to low within the range)
        up_pct = (data['close'] - data['low']) / candle_range.replace(0, np.nan)
        up_pct = up_pct.fillna(0.5)  # Doji candles get 50/50 split
        up_volume = data['volume'] * up_pct
        down_volume = data['volume'] * (1 - up_pct)

        # Bar delta: up_volume - down_volume
        indicators['bar_delta'] = up_volume - down_volume

        # Cumulative delta over lookback window
        indicators['cum_delta'] = indicators['bar_delta'].rolling(
            params.delta_lookback
        ).sum()

        # Delta acceleration (rate of change of cumulative delta)
        cum_delta = indicators['cum_delta']
        avg_volume = data['volume'].rolling(20).mean()
        # Normalize delta by average volume only (not * delta_lookback)
        indicators['delta_accel'] = cum_delta.diff(params.price_accel_lookback) / avg_volume

        # Volume ratio
        indicators['volume_ratio'] = data['volume'] / avg_volume

        # Price acceleration
        indicators['price_accel'] = data['close'].pct_change(params.price_accel_lookback)

        return indicators

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        params = self.params

        warmup = max(params.delta_lookback, 20, 14) + params.price_accel_lookback + 5

        in_position = False
        position_direction = None
        stop_loss = None
        take_profit = None
        last_trade_bar = -params.min_bars_between_trades

        for i in range(warmup, len(data)):
            candle = data.iloc[i]
            timestamp = data.index[i]

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
                continue

            # Cooldown check
            if i - last_trade_bar < params.min_bars_between_trades:
                continue

            delta_accel = candle.get('delta_accel', 0)
            vol_ratio = candle.get('volume_ratio', 0)
            price_accel = candle.get('price_accel', 0)
            atr_val = candle.get('atr', 0)

            if any(pd.isna(v) for v in [delta_accel, vol_ratio, price_accel, atr_val]):
                continue
            if atr_val == 0:
                continue

            # Volume surge check
            if vol_ratio < params.volume_surge_mult:
                continue

            # Delta acceleration must exceed threshold
            if abs(delta_accel) < params.acceleration_threshold:
                continue

            # Price-delta confirmation: price acceleration must agree with delta direction
            if (delta_accel > 0 and price_accel < 0) or (delta_accel < 0 and price_accel > 0):
                continue

            if delta_accel > 0:
                # Bullish: aggressive buyers detected
                stop = candle['close'] - params.atr_stop_mult * atr_val
                risk = candle['close'] - stop
                target = candle['close'] + risk * params.risk_reward_ratio

                confidence = min(
                    1.0,
                    0.5 + min(abs(delta_accel) / 2, 0.3) + min(vol_ratio / 10, 0.2)
                )

                signals.append(Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.LONG,
                    price=candle['close'],
                    symbol="",
                    confidence=confidence,
                    stop_loss=stop,
                    take_profit=target,
                    position_size=params.position_size,
                    reason=f"Micro momentum long: delta accel {delta_accel:.4f}, vol {vol_ratio:.1f}x",
                    metadata={
                        'delta_accel': delta_accel,
                        'volume_ratio': vol_ratio,
                        'price_accel': price_accel,
                        'cum_delta': candle.get('cum_delta', 0),
                    },
                ))

                in_position = True
                position_direction = 'long'
                stop_loss = stop
                take_profit = target
                last_trade_bar = i

            else:
                # Bearish: aggressive sellers detected
                stop = candle['close'] + params.atr_stop_mult * atr_val
                risk = stop - candle['close']
                target = candle['close'] - risk * params.risk_reward_ratio

                confidence = min(
                    1.0,
                    0.5 + min(abs(delta_accel) / 2, 0.3) + min(vol_ratio / 10, 0.2)
                )

                signals.append(Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.SHORT,
                    price=candle['close'],
                    symbol="",
                    confidence=confidence,
                    stop_loss=stop,
                    take_profit=target,
                    position_size=params.position_size,
                    reason=f"Micro momentum short: delta accel {delta_accel:.4f}, vol {vol_ratio:.1f}x",
                    metadata={
                        'delta_accel': delta_accel,
                        'volume_ratio': vol_ratio,
                        'price_accel': price_accel,
                        'cum_delta': candle.get('cum_delta', 0),
                    },
                ))

                in_position = True
                position_direction = 'short'
                stop_loss = stop
                take_profit = target
                last_trade_bar = i

        return signals
