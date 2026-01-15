"""
One Candle Rule Support/Resistance Retest Strategy

Source: Scarface Trades (aka Tony, aka Raph) - Chart Fanatics
Style: Scalping/Day Trading (9:30 AM - 11:00 AM Eastern time window)

*** INTRADAY STRATEGY - REQUIRES 1-MINUTE DATA ***

Entry Rules:
1. Identify clear support/resistance level
2. Wait for price to break above resistance (or below support for shorts)
3. Wait for price to retest the broken level (resistance becomes support)
4. Look for candlestick confirmation showing buyers overpowering sellers
   (e.g., hammer candle with long lower wick showing rejection)
5. Enter on the confirmation candle that holds above the key level

Exit Rules:
- Stop loss below the support level being retested
- Target 1:2 risk-to-reward ratio

Primary timeframe: 1-minute chart
Trading window: 9:30 AM - 11:00 AM ET (first 90 minutes)
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
class OneCandleRuleParams(StrategyParams):
    """
    Parameters for One Candle Rule strategy.

    Optimized for 1-minute intraday data.
    """

    # Timeframe (for documentation)
    recommended_interval: str = "1m"

    # Support/Resistance detection
    sr_lookback: int = 30  # 30 minutes of data for S/R detection
    sr_tolerance: float = 0.001  # 0.1% tolerance (tighter for 1-min)
    min_level_touches: int = 2  # Minimum touches to confirm S/R level

    # Breakout detection
    breakout_threshold: float = 0.002  # 0.2% beyond level (tighter for intraday)
    min_breakout_volume_mult: float = 1.5  # Volume must be 1.5x average

    # Retest detection
    retest_tolerance: float = 0.003  # 0.3% tolerance for retest
    max_retest_candles: int = 15  # Max 15 minutes to wait for retest

    # Confirmation candle
    min_wick_body_ratio: float = 1.5  # Lower wick must be 1.5x body for hammer
    max_body_range_ratio: float = 0.4  # Body should be <40% of candle range

    # Risk management
    risk_reward_ratio: float = 2.0  # Target 1:2 R/R
    default_stop_loss_pct: float = 0.01  # 1% stop for intraday (tighter)

    # Position sizing
    position_size: float = 10000.0  # Smaller for scalping

    # Time filter - CRITICAL for this strategy
    trade_start_time: str = "09:35"  # Start 5 min after open (let dust settle)
    trade_end_time: str = "11:00"  # Only trade first 90 minutes
    no_new_trades_after: str = "10:45"  # Don't enter new trades in last 15 min

    @classmethod
    def from_dict(cls, data: Dict) -> "OneCandleRuleParams":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class OneCandleRuleStrategy(BaseStrategy):
    """
    One Candle Rule Support/Resistance Retest Strategy

    Trades breakouts with retest confirmation, looking for hammer
    candles at retested levels.
    """

    name = "one_candle_rule"
    description = "S/R breakout with retest confirmation"
    version = "1.0.0"
    source_url = "https://youtube.com/watch?v=..."  # From YouTube strategies DB
    source_creator = "Scarface Trades (Chart Fanatics)"
    trading_style = "Scalping/Day Trading"

    def __init__(self, params: Optional[OneCandleRuleParams] = None):
        super().__init__(params or self.default_params())
        self._sr_levels: List[Dict] = []
        self._breakouts: List[Dict] = []

    @classmethod
    def default_params(cls) -> OneCandleRuleParams:
        return OneCandleRuleParams()

    @classmethod
    def param_ranges(cls) -> Dict[str, Tuple[float, float, float]]:
        """Parameter ranges for optimization"""
        return {
            "sr_lookback": (10, 40, 5),
            "sr_tolerance": (0.001, 0.005, 0.001),
            "breakout_threshold": (0.002, 0.008, 0.001),
            "retest_tolerance": (0.003, 0.010, 0.001),
            "min_wick_body_ratio": (1.0, 3.0, 0.5),
            "risk_reward_ratio": (1.5, 3.0, 0.25),
        }

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate indicators for S/R and pattern detection"""
        indicators = super().calculate_indicators(data)

        # Swing highs and lows for S/R detection
        indicators['swing_high'] = self._find_swing_highs(data)
        indicators['swing_low'] = self._find_swing_lows(data)

        # Volume ratio
        indicators['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

        # Candle properties
        indicators['body'] = abs(data['close'] - data['open'])
        indicators['upper_wick'] = data['high'] - data[['open', 'close']].max(axis=1)
        indicators['lower_wick'] = data[['open', 'close']].min(axis=1) - data['low']
        indicators['range'] = data['high'] - data['low']

        return indicators

    def _find_swing_highs(self, data: pd.DataFrame, window: int = 5) -> pd.Series:
        """Find swing high points"""
        highs = data['high'].rolling(window * 2 + 1, center=True).apply(
            lambda x: x.iloc[window] == x.max(), raw=False
        )
        return highs.fillna(0).astype(bool)

    def _find_swing_lows(self, data: pd.DataFrame, window: int = 5) -> pd.Series:
        """Find swing low points"""
        lows = data['low'].rolling(window * 2 + 1, center=True).apply(
            lambda x: x.iloc[window] == x.min(), raw=False
        )
        return lows.fillna(0).astype(bool)

    def _detect_sr_levels(self, data: pd.DataFrame) -> List[Dict]:
        """Detect support and resistance levels"""
        params = self.params
        levels = []

        # Get swing points
        swing_highs = data[data['swing_high'] == True]['high'].values
        swing_lows = data[data['swing_low'] == True]['low'].values

        all_levels = list(swing_highs) + list(swing_lows)

        # Cluster nearby levels
        if len(all_levels) < 2:
            return levels

        all_levels = sorted(all_levels)
        clustered = []
        current_cluster = [all_levels[0]]

        for level in all_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < params.sr_tolerance:
                current_cluster.append(level)
            else:
                if len(current_cluster) >= params.min_level_touches:
                    clustered.append({
                        'price': np.mean(current_cluster),
                        'touches': len(current_cluster),
                        'type': 'resistance' if np.mean(current_cluster) > data['close'].iloc[-1] else 'support'
                    })
                current_cluster = [level]

        # Don't forget last cluster
        if len(current_cluster) >= params.min_level_touches:
            clustered.append({
                'price': np.mean(current_cluster),
                'touches': len(current_cluster),
                'type': 'resistance' if np.mean(current_cluster) > data['close'].iloc[-1] else 'support'
            })

        return clustered

    def _is_breakout(self, candle: pd.Series, level: float, direction: str) -> bool:
        """Check if candle breaks through a level"""
        params = self.params
        threshold = level * params.breakout_threshold

        if direction == 'up':
            # Close above resistance with threshold
            return candle['close'] > level + threshold
        else:
            # Close below support with threshold
            return candle['close'] < level - threshold

    def _is_retest(self, candle: pd.Series, level: float) -> bool:
        """Check if candle is retesting a level"""
        params = self.params
        tolerance = level * params.retest_tolerance

        # Price comes back to level (low touches it for bullish retest)
        return abs(candle['low'] - level) < tolerance

    def _is_hammer(self, candle: pd.Series) -> bool:
        """Check if candle is a hammer (bullish reversal)"""
        params = self.params

        body = candle['body']
        lower_wick = candle['lower_wick']
        upper_wick = candle['upper_wick']
        range_ = candle['range']

        if range_ == 0 or body == 0:
            return False

        # Hammer: long lower wick, small body, small upper wick
        wick_body_ratio = lower_wick / body if body > 0 else 0
        body_range_ratio = body / range_

        return (
            wick_body_ratio >= params.min_wick_body_ratio and
            body_range_ratio <= params.max_body_range_ratio and
            upper_wick < lower_wick * 0.5  # Upper wick much smaller
        )

    def _is_bullish_candle(self, candle: pd.Series) -> bool:
        """Check if candle closed bullish"""
        return candle['close'] > candle['open']

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on S/R retest with confirmation"""
        signals = []
        params = self.params

        # Detect S/R levels from first portion of data
        lookback = min(params.sr_lookback, len(data) // 2)
        sr_data = data.iloc[:lookback]
        self._sr_levels = self._detect_sr_levels(sr_data)

        if not self._sr_levels:
            return signals

        # Track state
        in_position = False
        entry_price = None
        stop_loss = None
        resistance_broken = None
        waiting_for_retest = False
        breakout_candle_idx = None

        # Scan for setups
        for i in range(lookback, len(data)):
            candle = data.iloc[i]
            prev_candle = data.iloc[i - 1]
            timestamp = data.index[i]

            # Skip if already in position
            if in_position:
                # Check exit conditions
                if candle['low'] <= stop_loss:
                    # Stop loss hit
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.EXIT_LONG,
                        price=stop_loss,
                        symbol="",
                        confidence=1.0,
                        reason="Stop loss triggered",
                    ))
                    in_position = False
                    entry_price = None
                    continue

                # Check take profit (using R/R ratio)
                target = entry_price + (entry_price - stop_loss) * params.risk_reward_ratio
                if candle['high'] >= target:
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.EXIT_LONG,
                        price=target,
                        symbol="",
                        confidence=1.0,
                        reason=f"Take profit at {params.risk_reward_ratio}:1 R/R",
                    ))
                    in_position = False
                    entry_price = None
                continue

            # Look for breakouts above resistance
            for level_info in self._sr_levels:
                if level_info['type'] != 'resistance':
                    continue

                level = level_info['price']

                # Check for breakout
                if not waiting_for_retest:
                    if (self._is_breakout(candle, level, 'up') and
                        candle['volume_ratio'] >= params.min_breakout_volume_mult):
                        # Breakout detected, wait for retest
                        waiting_for_retest = True
                        resistance_broken = level
                        breakout_candle_idx = i
                        break

                # Check for retest after breakout
                elif waiting_for_retest and resistance_broken:
                    # Don't wait too long
                    if i - breakout_candle_idx > params.max_retest_candles:
                        waiting_for_retest = False
                        resistance_broken = None
                        continue

                    # Check for retest with confirmation
                    if self._is_retest(candle, resistance_broken):
                        # Look for hammer or bullish confirmation
                        if self._is_hammer(candle) or (
                            self._is_bullish_candle(candle) and
                            candle['close'] > resistance_broken
                        ):
                            # Entry signal
                            stop = resistance_broken * (1 - params.default_stop_loss_pct)
                            target = candle['close'] + (candle['close'] - stop) * params.risk_reward_ratio

                            signals.append(Signal(
                                timestamp=timestamp,
                                signal_type=SignalType.LONG,
                                price=candle['close'],
                                symbol="",
                                confidence=0.8 if self._is_hammer(candle) else 0.6,
                                stop_loss=stop,
                                take_profit=target,
                                position_size=params.position_size,
                                reason=f"S/R retest with {'hammer' if self._is_hammer(candle) else 'bullish'} confirmation",
                                metadata={
                                    'resistance_level': resistance_broken,
                                    'is_hammer': self._is_hammer(candle),
                                }
                            ))

                            in_position = True
                            entry_price = candle['close']
                            stop_loss = stop
                            waiting_for_retest = False
                            resistance_broken = None
                            break

        return signals
