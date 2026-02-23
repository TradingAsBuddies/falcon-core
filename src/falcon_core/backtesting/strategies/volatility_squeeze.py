"""
Volatility Squeeze Strategy

Source: AI-generated novel strategy
Style: Day Trading (9:30 AM - 3:30 PM Eastern)

*** INTRADAY STRATEGY - REQUIRES 5-MINUTE DATA ***

Concept:
Detect Bollinger Bands contracting inside Keltner Channels (the "squeeze").
This compression signals a period of low volatility that precedes a directional
expansion. When the squeeze fires (BBs expand outside KCs), trade the breakout
direction using momentum confirmation.

Entry Rules:
1. Bollinger Bands are inside Keltner Channels (squeeze is on)
2. Squeeze fires: BBs expand outside KCs
3. Momentum indicator (12-period rate of change) confirms direction
4. Enter long if momentum is positive, short if negative

Exit Rules:
- Stop loss: 1.5 ATR from entry
- Take profit: 2:1 R/R or momentum reversal
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
class VolatilitySqueezeParams(StrategyParams):
    """Parameters for Volatility Squeeze strategy."""

    recommended_interval: str = "5m"

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # Keltner Channels
    kc_period: int = 20
    kc_atr_mult: float = 1.0  # Narrower KC = squeeze is rarer and more meaningful

    # Momentum
    momentum_period: int = 12  # Rate of change lookback
    min_momentum: float = 0.003  # Stronger directional conviction required

    # Squeeze detection
    min_squeeze_bars: int = 6  # Longer compression = bigger expected move

    # Volume confirmation on squeeze fire
    min_fire_volume_ratio: float = 1.3  # Volume must be 1.3x avg when squeeze fires

    # Cooldown
    min_bars_between_trades: int = 6  # Avoid re-entry chop

    # Risk management
    atr_stop_mult: float = 1.5  # Stop loss = 1.5 ATR
    risk_reward_ratio: float = 2.0
    position_size: float = 25000.0

    # Time filter
    trade_start_time: str = "09:45"
    trade_end_time: str = "15:30"

    @classmethod
    def from_dict(cls, data: Dict) -> "VolatilitySqueezeParams":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class VolatilitySqueezeStrategy(BaseStrategy):
    """
    Volatility Squeeze Strategy

    Detects Bollinger Band / Keltner Channel squeeze patterns and trades
    the directional expansion with momentum confirmation.
    """

    name = "volatility_squeeze"
    description = "BB/KC squeeze breakout with momentum confirmation"
    version = "1.0.0"
    source_url = None
    source_creator = "AI-generated (Falcon)"
    trading_style = "Day Trading / Breakout"

    def __init__(self, params: Optional[VolatilitySqueezeParams] = None):
        super().__init__(params or self.default_params())

    @classmethod
    def default_params(cls) -> VolatilitySqueezeParams:
        return VolatilitySqueezeParams()

    @classmethod
    def param_ranges(cls) -> Dict[str, Tuple[float, float, float]]:
        return {
            "bb_period": (10, 30, 5),
            "bb_std": (1.5, 2.5, 0.25),
            "kc_period": (10, 30, 5),
            "kc_atr_mult": (1.0, 2.5, 0.25),
            "momentum_period": (6, 20, 2),
            "min_squeeze_bars": (2, 6, 1),
            "atr_stop_mult": (1.0, 2.5, 0.5),
        }

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        indicators = super().calculate_indicators(data)
        params = self.params

        close = data['close']

        # Bollinger Bands
        bb_sma = close.rolling(params.bb_period).mean()
        bb_std = close.rolling(params.bb_period).std()
        indicators['bb_upper'] = bb_sma + params.bb_std * bb_std
        indicators['bb_lower'] = bb_sma - params.bb_std * bb_std
        indicators['bb_mid'] = bb_sma

        # Keltner Channels (using ATR)
        kc_mid = close.rolling(params.kc_period).mean()
        # ATR from base indicators
        atr = indicators['atr']
        indicators['kc_upper'] = kc_mid + params.kc_atr_mult * atr
        indicators['kc_lower'] = kc_mid - params.kc_atr_mult * atr
        indicators['kc_mid'] = kc_mid

        # Squeeze detection: BB inside KC
        indicators['squeeze_on'] = (
            (indicators['bb_upper'] < indicators['kc_upper']) &
            (indicators['bb_lower'] > indicators['kc_lower'])
        )

        # Momentum: rate of change
        indicators['momentum'] = close.pct_change(params.momentum_period)

        # Volume ratio for fire bar confirmation
        indicators['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

        # Count consecutive squeeze bars
        squeeze = indicators['squeeze_on'].astype(int)
        # Reset counter on squeeze_off
        squeeze_count = squeeze.copy()
        for i in range(1, len(squeeze_count)):
            if squeeze_count.iloc[i] == 1:
                squeeze_count.iloc[i] = squeeze_count.iloc[i - 1] + 1
            else:
                squeeze_count.iloc[i] = 0
        indicators['squeeze_count'] = squeeze_count

        return indicators

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        params = self.params

        warmup = max(params.bb_period, params.kc_period, params.momentum_period, 14) + 5

        in_position = False
        position_direction = None
        stop_loss = None
        take_profit = None
        last_trade_bar = -params.min_bars_between_trades

        # Day boundary tracking
        has_dates = isinstance(data.index, pd.DatetimeIndex)
        prev_date = data.index[warmup].date() if has_dates and warmup < len(data) else None

        for i in range(warmup, len(data)):
            candle = data.iloc[i]
            prev = data.iloc[i - 1]
            timestamp = data.index[i]
            current_date = timestamp.date() if has_dates else None

            # Day boundary — force close positions and reset state
            if has_dates and current_date != prev_date:
                if in_position:
                    prev_day_last = data.iloc[i - 1]
                    exit_type = SignalType.EXIT_LONG if position_direction == 'long' else SignalType.EXIT_SHORT
                    signals.append(Signal(
                        timestamp=data.index[i - 1],
                        signal_type=exit_type,
                        price=prev_day_last['close'],
                        symbol="",
                        confidence=0.5,
                        reason="End of day — closing squeeze position",
                    ))
                    in_position = False
                    position_direction = None
                # Reset cooldown and squeeze tracking at day boundary
                last_trade_bar = -params.min_bars_between_trades
                prev_date = current_date

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

            # Squeeze fire detection: was in squeeze, now out
            prev_squeeze = prev.get('squeeze_on', False)
            curr_squeeze = candle.get('squeeze_on', False)
            squeeze_count = prev.get('squeeze_count', 0)

            if not (prev_squeeze and not curr_squeeze):
                continue

            # Must have been in squeeze for minimum bars
            if squeeze_count < params.min_squeeze_bars:
                continue

            # Volume confirmation on fire bar
            vol_ratio = candle.get('volume_ratio', 0)
            if pd.isna(vol_ratio) or vol_ratio < params.min_fire_volume_ratio:
                continue

            # Momentum direction
            momentum = candle.get('momentum', 0)
            if pd.isna(momentum):
                continue

            atr_val = candle.get('atr', 0)
            if pd.isna(atr_val) or atr_val == 0:
                continue

            if abs(momentum) < params.min_momentum:
                continue

            if momentum > 0:
                # Long entry
                stop = candle['close'] - params.atr_stop_mult * atr_val
                risk = candle['close'] - stop
                target = candle['close'] + risk * params.risk_reward_ratio

                confidence = min(1.0, 0.5 + squeeze_count * 0.05 + abs(momentum) * 10)

                signals.append(Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.LONG,
                    price=candle['close'],
                    symbol="",
                    confidence=confidence,
                    stop_loss=stop,
                    take_profit=target,
                    position_size=params.position_size,
                    reason=f"Squeeze fire long: {squeeze_count} bars compressed, momentum {momentum:.3f}",
                    metadata={
                        'squeeze_bars': int(squeeze_count),
                        'momentum': momentum,
                        'atr': atr_val,
                    },
                ))

                in_position = True
                position_direction = 'long'
                stop_loss = stop
                take_profit = target
                last_trade_bar = i

            else:
                # Short entry
                stop = candle['close'] + params.atr_stop_mult * atr_val
                risk = stop - candle['close']
                target = candle['close'] - risk * params.risk_reward_ratio

                confidence = min(1.0, 0.5 + squeeze_count * 0.05 + abs(momentum) * 10)

                signals.append(Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.SHORT,
                    price=candle['close'],
                    symbol="",
                    confidence=confidence,
                    stop_loss=stop,
                    take_profit=target,
                    position_size=params.position_size,
                    reason=f"Squeeze fire short: {squeeze_count} bars compressed, momentum {momentum:.3f}",
                    metadata={
                        'squeeze_bars': int(squeeze_count),
                        'momentum': momentum,
                        'atr': atr_val,
                    },
                ))

                in_position = True
                position_direction = 'short'
                stop_loss = stop
                take_profit = target
                last_trade_bar = i

        return signals
