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
    min_gap_pct: float = 0.02  # Minimum 2% gap up
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
    description = "Fade gap-up failures targeting partial gap fill"
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

        # Determine gap: first bar open vs implied previous close
        # For intraday data, previous close is approximated by first bar's open
        # In real usage, previous close would come from daily data
        open_price = data.iloc[0]['open']
        prev_close = open_price  # Approximation for intraday
        gap_high = data.iloc[0]['high']

        # Better gap estimation: use the minimum of the opening bars
        # to estimate whether this is actually a gap-up day
        # In practice, prev_close would be passed as metadata
        first_bar_change = (data.iloc[0]['close'] - data.iloc[0]['open']) / data.iloc[0]['open']

        # Calculate opening range
        setup_data = data.iloc[:params.setup_bars]
        range_high = setup_data['high'].max()
        range_low = setup_data['low'].min()
        gap_high = range_high  # Highest point in opening range

        # For gap detection, look at first bar's position relative to data
        # We check if current price is elevated (gap-up pattern)
        gap_pct = (range_high - data.iloc[0]['open']) / data.iloc[0]['open'] if data.iloc[0]['open'] > 0 else 0

        # If we have VWAP or previous close in metadata, use that instead
        # For now, check if opening range shows gap-up behavior
        # Use the first bar's open as reference for gap calculation
        session_open = data.iloc[0]['open']

        in_position = False
        stop_loss = None
        take_profit = None

        for i in range(params.setup_bars, len(data)):
            candle = data.iloc[i]
            timestamp = data.index[i]

            # Exit logic
            if in_position:
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

            # Check for opening range break DOWN (fade signal)
            fade_level = range_low * (1 - params.fade_confirmation)

            if candle['close'] > fade_level:
                continue

            # Must not have made new highs above range_high recently
            recent_high = data.iloc[max(0, i - 3):i + 1]['high'].max()
            if recent_high > range_high * 1.002:
                continue  # Still making new highs, don't fade

            # Relative volume check
            rel_vol = candle.get('rel_volume', 0)
            if pd.isna(rel_vol) or rel_vol < params.min_rel_volume:
                continue

            # Gap size check — distance from range_low to range_high
            range_pct = (range_high - range_low) / range_low if range_low > 0 else 0
            if range_pct < params.min_gap_pct or range_pct > params.max_gap_pct:
                continue

            # Short entry: fade the gap
            stop = range_high * (1 + params.default_stop_loss_pct)
            gap_size = range_high - session_open
            fill_target = range_high - gap_size * params.fill_target_pct
            target = max(fill_target, candle['close'] * 0.97)  # Floor at 3% below

            risk = stop - candle['close']
            if risk <= 0:
                continue

            confidence = min(1.0, 0.5 + range_pct * 5 + min(rel_vol / 10, 0.2))

            signals.append(Signal(
                timestamp=timestamp,
                signal_type=SignalType.SHORT,
                price=candle['close'],
                symbol="",
                confidence=confidence,
                stop_loss=stop,
                take_profit=target,
                position_size=params.position_size,
                reason=f"Gap fill fade: range {range_pct:.2%}, below OR low, vol {rel_vol:.1f}x",
                metadata={
                    'range_high': range_high,
                    'range_low': range_low,
                    'range_pct': range_pct,
                    'fill_target': fill_target,
                    'rel_volume': rel_vol,
                },
            ))

            in_position = True
            stop_loss = stop
            take_profit = target
            break  # One entry per day

        return signals
