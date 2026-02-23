"""
Gap Fill Fade Strategy

Source: AI-generated novel strategy
Style: Day Trading / Mean Reversion (9:30 AM - 2:00 PM Eastern)

*** INTRADAY STRATEGY - REQUIRES 5-MINUTE DATA ***

Concept:
Statistical mean reversion on partial gap fills. Stocks that gap up but fail
to hold the gap tend to revert toward the gap fill level. This strategy fades
gaps that show early weakness, targeting a partial or full gap fill.

Entry Rules:
1. Stock gapped up by min_gap_pct to max_gap_pct
2. After the first 15 minutes, price fails to make new highs
3. Price starts fading (breaks below opening range low)
4. Relative volume confirms institutional activity
5. Short entry targeting partial gap fill

Exit Rules:
- Stop loss above the gap high
- Take profit at fill_target_pct of the gap (e.g., 50% fill)
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
class GapFillFadeParams(StrategyParams):
    """Parameters for Gap Fill Fade strategy."""

    recommended_interval: str = "5m"

    # Gap criteria
    min_gap_pct: float = 0.01  # Minimum 1% gap up
    max_gap_pct: float = 0.08  # Max 8% gap (avoid extreme moves)

    # Fill target
    fill_target_pct: float = 0.5  # Target 50% of the gap to fill

    # Volume filter
    min_rel_volume: float = 2.0  # Relative volume must be 2x

    # Setup detection
    setup_bars: int = 3  # 15 min opening range (3 x 5-min bars)
    fade_confirmation: float = 0.003  # Price must drop 0.3% below range low

    # Risk management
    risk_reward_ratio: float = 2.0
    default_stop_loss_pct: float = 0.02  # 2% stop above gap high
    position_size: float = 25000.0

    # Time filter
    trade_start_time: str = "09:45"  # After opening range
    trade_end_time: str = "14:00"  # No new entries after 2 PM

    @classmethod
    def from_dict(cls, data: Dict) -> "GapFillFadeParams":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class GapFillFadeStrategy(BaseStrategy):
    """
    Gap Fill Fade Strategy

    Fades gap-up stocks that fail to hold, targeting partial gap fill via
    mean reversion. Uses relative volume and opening range failure as
    confirmation.
    """

    name = "gap_fill_fade"
    description = "Fade gap-up and gap-down failures targeting partial gap fill"
    version = "1.0.0"
    source_url = None
    source_creator = "AI-generated (Falcon)"
    trading_style = "Day Trading / Mean Reversion"

    def __init__(self, params: Optional[GapFillFadeParams] = None):
        super().__init__(params or self.default_params())

    @classmethod
    def default_params(cls) -> GapFillFadeParams:
        return GapFillFadeParams()

    @classmethod
    def param_ranges(cls) -> Dict[str, Tuple[float, float, float]]:
        return {
            "min_gap_pct": (0.01, 0.04, 0.005),
            "max_gap_pct": (0.05, 0.12, 0.01),
            "fill_target_pct": (0.3, 0.8, 0.1),
            "min_rel_volume": (1.5, 3.0, 0.5),
            "setup_bars": (2, 6, 1),
            "fade_confirmation": (0.001, 0.005, 0.001),
        }

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        indicators = super().calculate_indicators(data)

        # Relative volume ratio
        indicators['rel_volume'] = data['volume'] / data['volume'].rolling(20).mean()

        return indicators

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        params = self.params

        if len(data) < params.setup_bars + 2:
            return signals

        # Detect day boundaries from the DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            return signals

        dates = data.index.date
        unique_dates = sorted(set(dates))

        if len(unique_dates) < 2:
            return signals

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

            # Check gap: must be at least min_gap_pct in either direction
            gap_pct = (open_price - prev_close) / prev_close if prev_close > 0 else 0
            abs_gap_pct = abs(gap_pct)

            if abs_gap_pct < params.min_gap_pct or abs_gap_pct > params.max_gap_pct:
                continue

            is_gap_up = gap_pct > 0

            # Calculate opening range from today's first bars
            orb_bars = min(params.setup_bars, len(today_data))
            setup_data = today_data.iloc[:orb_bars]
            range_high = setup_data['high'].max()
            range_low = setup_data['low'].min()
            entry_found = False
            position_direction = None  # 'short' for gap-up fade, 'long' for gap-down fade

            for j in range(orb_bars, len(today_data)):
                candle = today_data.iloc[j]
                timestamp = today_data.index[j]

                # Exit logic
                if in_position:
                    if position_direction == 'short':
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
                                reason="Take profit (gap fill target)",
                            ))
                            in_position = False
                        continue
                    elif position_direction == 'long':
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

                # Relative volume check (common to both directions)
                rel_vol = candle.get('rel_volume', 0)
                if pd.isna(rel_vol) or rel_vol < params.min_rel_volume:
                    continue

                if is_gap_up:
                    # GAP-UP SHORT FADE: Check for opening range break DOWN
                    fade_level = range_low * (1 - params.fade_confirmation)

                    if candle['close'] > fade_level:
                        continue

                    # Must not have made new highs above range_high recently
                    lookback_start = max(0, j - 3)
                    recent_high = today_data.iloc[lookback_start:j + 1]['high'].max()
                    if recent_high > range_high * 1.002:
                        continue

                    # Short entry: fade the gap up
                    stop = range_high * (1 + params.default_stop_loss_pct)
                    gap_size = open_price - prev_close
                    fill_target = open_price - gap_size * params.fill_target_pct
                    target = max(fill_target, candle['close'] * 0.97)

                    risk = stop - candle['close']
                    if risk <= 0:
                        continue

                    confidence = min(1.0, 0.5 + abs_gap_pct * 5 + min(rel_vol / 10, 0.2))

                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.SHORT,
                        price=candle['close'],
                        symbol="",
                        confidence=confidence,
                        stop_loss=stop,
                        take_profit=target,
                        position_size=params.position_size,
                        reason=f"Gap fill fade short: {gap_pct:.2%} gap up, below OR low, vol {rel_vol:.1f}x",
                        metadata={
                            'gap_pct': gap_pct,
                            'prev_close': prev_close,
                            'range_high': range_high,
                            'range_low': range_low,
                            'fill_target': fill_target,
                            'rel_volume': rel_vol,
                        },
                    ))

                    in_position = True
                    entry_found = True
                    position_direction = 'short'
                    stop_loss = stop
                    take_profit = target

                else:
                    # GAP-DOWN LONG FADE: Check for opening range break UP (failure to make new lows)
                    fade_level = range_high * (1 + params.fade_confirmation)

                    if candle['close'] < fade_level:
                        continue

                    # Must not have made new lows below range_low recently
                    lookback_start = max(0, j - 3)
                    recent_low = today_data.iloc[lookback_start:j + 1]['low'].min()
                    if recent_low < range_low * 0.998:
                        continue

                    # Long entry: fade the gap down
                    stop = range_low * (1 - params.default_stop_loss_pct)
                    gap_size = prev_close - open_price  # positive for gap down
                    fill_target = open_price + gap_size * params.fill_target_pct
                    target = min(fill_target, candle['close'] * 1.03)

                    risk = candle['close'] - stop
                    if risk <= 0:
                        continue

                    confidence = min(1.0, 0.5 + abs_gap_pct * 5 + min(rel_vol / 10, 0.2))

                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.LONG,
                        price=candle['close'],
                        symbol="",
                        confidence=confidence,
                        stop_loss=stop,
                        take_profit=target,
                        position_size=params.position_size,
                        reason=f"Gap fill fade long: {gap_pct:.2%} gap down, above OR high, vol {rel_vol:.1f}x",
                        metadata={
                            'gap_pct': gap_pct,
                            'prev_close': prev_close,
                            'range_high': range_high,
                            'range_low': range_low,
                            'fill_target': fill_target,
                            'rel_volume': rel_vol,
                        },
                    ))

                    in_position = True
                    entry_found = True
                    position_direction = 'long'
                    stop_loss = stop
                    take_profit = target

        return signals
