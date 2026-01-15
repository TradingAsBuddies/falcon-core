"""
ATR Trailing Stop Breakout Strategy

Source: Treyding Stocks
Style: Day trading, breakout trading

Entry Rules:
1. Price must close above the "cyan line" (highest high of previous 10 candles minus ATR)
2. Volume on the breakout candle must be at least 10x the number of shares being purchased
3. Uses 5-minute candles
4. Targets stocks with overnight gaps and strong opening moves

Exit Rules:
- Trailing stop: Exit when price CLOSES below the cyan line
- No upside profit target - ride the trend
- Intraday wicks below the line are acceptable; only candle closes trigger exit

Risk Management:
- Default $25,000 position size (configurable)
- Liquidity requirement: Volume must be at least 10x shares to ensure fills
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from falcon_core.backtesting.strategies.base import (
    BaseStrategy,
    StrategyParams,
    Signal,
    SignalType,
)


@dataclass
class ATRBreakoutParams(StrategyParams):
    """Parameters for ATR Trailing Stop Breakout strategy"""

    # ATR calculation
    atr_period: int = 14  # ATR lookback period
    highest_high_period: int = 10  # Period for highest high calculation

    # Entry conditions
    min_volume_multiplier: float = 10.0  # Volume must be 10x shares purchased
    min_gap_pct: float = 0.02  # 2% minimum gap for "gapper" qualification

    # Position sizing
    position_size: float = 25000.0  # Default $25K position

    # Trailing stop
    atr_multiplier: float = 1.0  # Multiply ATR by this for stop distance

    # Time filter
    trade_start_time: str = "09:30"
    trade_end_time: str = "15:30"

    @classmethod
    def from_dict(cls, data: Dict) -> "ATRBreakoutParams":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class ATRBreakoutStrategy(BaseStrategy):
    """
    ATR Trailing Stop Breakout Strategy

    Enters on breakout above ATR-based "cyan line" with volume confirmation.
    Uses trailing stop based on ATR for exits - no profit target.
    """

    name = "atr_breakout"
    description = "ATR trailing stop breakout with volume filter"
    version = "1.0.0"
    source_url = "https://youtube.com/watch?v=..."
    source_creator = "Treyding Stocks"
    trading_style = "Day trading, breakout trading"

    def __init__(self, params: Optional[ATRBreakoutParams] = None):
        super().__init__(params or self.default_params())

    @classmethod
    def default_params(cls) -> ATRBreakoutParams:
        return ATRBreakoutParams()

    @classmethod
    def param_ranges(cls) -> Dict[str, Tuple[float, float, float]]:
        """Parameter ranges for optimization"""
        return {
            "atr_period": (7, 21, 2),
            "highest_high_period": (5, 20, 2),
            "min_volume_multiplier": (5.0, 20.0, 2.5),
            "atr_multiplier": (0.5, 2.0, 0.25),
            "position_size": (10000.0, 50000.0, 5000.0),
        }

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate ATR and cyan line indicators"""
        indicators = super().calculate_indicators(data)
        params = self.params

        # ATR calculation
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr'] = true_range.rolling(params.atr_period).mean()

        # Highest high of previous N candles
        indicators['highest_high'] = data['high'].rolling(params.highest_high_period).max().shift(1)

        # Cyan line: Highest high - ATR
        indicators['cyan_line'] = indicators['highest_high'] - (indicators['atr'] * params.atr_multiplier)

        # Gap detection (for filtering)
        indicators['gap_pct'] = (data['open'] - data['close'].shift()) / data['close'].shift()

        # Volume in shares
        indicators['volume_shares'] = data['volume']

        return indicators

    def _check_volume_liquidity(self, candle: pd.Series, price: float) -> bool:
        """Check if volume is sufficient for position size"""
        params = self.params

        # Calculate shares we want to buy
        shares_to_buy = params.position_size / price

        # Volume must be at least 10x our share count
        required_volume = shares_to_buy * params.min_volume_multiplier

        return candle['volume'] >= required_volume

    def _is_gapper(self, data: pd.DataFrame, idx: int) -> bool:
        """Check if stock gapped up at open (for filtering)"""
        params = self.params

        if idx == 0:
            return False

        gap_pct = data.iloc[idx]['gap_pct']
        return gap_pct >= params.min_gap_pct

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on ATR breakout"""
        signals = []
        params = self.params

        # Need enough data for indicators
        min_bars = max(params.atr_period, params.highest_high_period) + 5
        if len(data) < min_bars:
            return signals

        # Track state
        in_position = False
        entry_price = None
        trailing_stop = None

        for i in range(min_bars, len(data)):
            candle = data.iloc[i]
            timestamp = data.index[i]

            cyan_line = candle['cyan_line']
            atr = candle['atr']

            # Skip if indicators not available
            if pd.isna(cyan_line) or pd.isna(atr):
                continue

            if in_position:
                # Update trailing stop (cyan line moves with price)
                current_stop = candle['cyan_line']
                if current_stop > trailing_stop:
                    trailing_stop = current_stop

                # Check exit: price CLOSES below cyan line
                if candle['close'] < trailing_stop:
                    signals.append(Signal(
                        timestamp=timestamp,
                        signal_type=SignalType.EXIT_LONG,
                        price=candle['close'],
                        symbol="",
                        confidence=1.0,
                        reason=f"Close below trailing stop (cyan line: {trailing_stop:.2f})",
                        metadata={
                            'trailing_stop': trailing_stop,
                            'atr': atr,
                        }
                    ))
                    in_position = False
                    entry_price = None
                    trailing_stop = None
                continue

            # Entry conditions
            # 1. Price closes above cyan line
            if candle['close'] <= cyan_line:
                continue

            # 2. Volume liquidity check
            if not self._check_volume_liquidity(candle, candle['close']):
                continue

            # 3. Breakout confirmation: previous close was below or at cyan line
            if i > 0:
                prev_candle = data.iloc[i - 1]
                prev_cyan = prev_candle['cyan_line']
                if not pd.isna(prev_cyan) and prev_candle['close'] > prev_cyan:
                    # Already above cyan line - not a fresh breakout
                    continue

            # Entry signal
            initial_stop = cyan_line
            shares = int(params.position_size / candle['close'])

            signals.append(Signal(
                timestamp=timestamp,
                signal_type=SignalType.LONG,
                price=candle['close'],
                symbol="",
                confidence=0.75,
                stop_loss=initial_stop,
                take_profit=None,  # No profit target - ride the trend
                position_size=params.position_size,
                reason=f"Breakout above cyan line ({cyan_line:.2f}) with volume",
                metadata={
                    'cyan_line': cyan_line,
                    'atr': atr,
                    'shares': shares,
                    'volume_ratio': candle['volume'] / (shares * params.min_volume_multiplier),
                }
            ))

            in_position = True
            entry_price = candle['close']
            trailing_stop = initial_stop

        # Close any open position at end of data
        if in_position:
            last_candle = data.iloc[-1]
            signals.append(Signal(
                timestamp=data.index[-1],
                signal_type=SignalType.EXIT_LONG,
                price=last_candle['close'],
                symbol="",
                confidence=0.5,
                reason="End of data - closing position",
            ))

        return signals

    def get_cyan_line_values(self, data: pd.DataFrame) -> pd.Series:
        """
        Get the cyan line values for visualization.

        Returns Series of cyan line prices aligned with data index.
        """
        indicators = self.calculate_indicators(data)
        return indicators.get('cyan_line', pd.Series())
