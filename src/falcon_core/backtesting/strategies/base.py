"""
Base Strategy Class for Falcon Backtesting

All trading strategies inherit from BaseStrategy and implement the
generate_signals() method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


class SignalType(Enum):
    """Trading signal types"""
    LONG = "long"
    SHORT = "short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"


@dataclass
class Signal:
    """Trading signal with metadata"""
    timestamp: datetime
    signal_type: SignalType
    price: float
    symbol: str
    confidence: float = 1.0  # 0.0 to 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None  # Dollar amount or share count
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "signal_type": self.signal_type.value,
            "price": self.price,
            "symbol": self.symbol,
            "confidence": self.confidence,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass
class StrategyParams:
    """
    Base parameters for all strategies.

    Strategies extend this with their specific parameters.
    All parameters should have sensible defaults.
    """
    # Position sizing
    position_size: float = 25000.0  # Default position size in dollars
    max_positions: int = 5  # Maximum concurrent positions

    # Risk management
    default_stop_loss_pct: float = 0.05  # 5% default stop loss
    default_take_profit_pct: float = 0.10  # 10% default take profit
    risk_reward_ratio: float = 2.0  # Target R/R ratio
    max_daily_loss_pct: float = 0.03  # Max 3% daily loss

    # Timing
    trade_start_time: str = "09:30"  # Market open ET
    trade_end_time: str = "15:30"  # Stop new entries 30min before close

    # Filters
    min_volume: int = 100000  # Minimum volume filter
    min_price: float = 1.0  # Minimum price filter
    max_price: float = 500.0  # Maximum price filter

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "position_size": self.position_size,
            "max_positions": self.max_positions,
            "default_stop_loss_pct": self.default_stop_loss_pct,
            "default_take_profit_pct": self.default_take_profit_pct,
            "risk_reward_ratio": self.risk_reward_ratio,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "trade_start_time": self.trade_start_time,
            "trade_end_time": self.trade_end_time,
            "min_volume": self.min_volume,
            "min_price": self.min_price,
            "max_price": self.max_price,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyParams":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def update(self, **kwargs) -> "StrategyParams":
        """Return new params with updated values"""
        current = self.to_dict()
        current.update(kwargs)
        return self.__class__.from_dict(current)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Strategies must implement:
    - name: Strategy identifier
    - generate_signals(): Core signal generation logic

    Optional overrides:
    - preprocess_data(): Data preparation before signal generation
    - calculate_indicators(): Technical indicator calculations
    - validate_signal(): Signal validation before emission
    """

    name: str = "base_strategy"
    description: str = "Base strategy class"
    version: str = "1.0.0"

    # Strategy metadata (from YouTube extraction)
    source_url: Optional[str] = None
    source_creator: Optional[str] = None
    trading_style: Optional[str] = None

    def __init__(self, params: Optional[StrategyParams] = None):
        """Initialize strategy with parameters"""
        self.params = params or self.default_params()
        self._signals: List[Signal] = []
        self._indicators: Dict[str, pd.Series] = {}
        self._state: Dict[str, Any] = {}

    @classmethod
    def default_params(cls) -> StrategyParams:
        """Return default parameters for this strategy"""
        return StrategyParams()

    @classmethod
    def param_ranges(cls) -> Dict[str, Tuple[float, float, float]]:
        """
        Return parameter ranges for optimization.

        Format: {param_name: (min, max, step)}
        Override in subclasses to define optimizable parameters.
        """
        return {
            "position_size": (10000.0, 50000.0, 5000.0),
            "default_stop_loss_pct": (0.02, 0.10, 0.01),
            "risk_reward_ratio": (1.5, 3.0, 0.25),
        }

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data before signal generation.

        Override to add custom preprocessing logic.
        Default: ensure required columns and sort by timestamp.
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Check for required columns (case-insensitive)
        data.columns = data.columns.str.lower()
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            elif 'date' in data.columns:
                data = data.set_index('date')
            data.index = pd.to_datetime(data.index)

        # Sort by timestamp
        data = data.sort_index()

        return data

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate technical indicators needed by the strategy.

        Override in subclasses to add strategy-specific indicators.
        Returns dict of indicator name -> Series.
        """
        indicators = {}

        # Common indicators (can be overridden)
        indicators['sma_20'] = data['close'].rolling(20).mean()
        indicators['sma_50'] = data['close'].rolling(50).mean()
        indicators['volume_sma'] = data['volume'].rolling(20).mean()

        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr'] = true_range.rolling(14).mean()

        return indicators

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals from market data.

        This is the core strategy logic. Must be implemented by subclasses.

        Args:
            data: DataFrame with OHLCV data (preprocessed)

        Returns:
            List of Signal objects
        """
        pass

    def validate_signal(self, signal: Signal, data: pd.DataFrame,
                       current_positions: int = 0) -> bool:
        """
        Validate a signal before emission.

        Override to add custom validation logic.
        Default: check position limits and basic filters.
        """
        # Check position limits
        if signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
            if current_positions >= self.params.max_positions:
                return False

        # Check price filters
        if signal.price < self.params.min_price:
            return False
        if signal.price > self.params.max_price:
            return False

        # Check confidence threshold
        if signal.confidence < 0.5:
            return False

        return True

    def run(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> List[Signal]:
        """
        Run the strategy on data and return signals.

        This is the main entry point for strategy execution.
        """
        # Preprocess data
        data = self.preprocess_data(data)

        # Calculate indicators
        self._indicators = self.calculate_indicators(data)

        # Add indicators to data for signal generation
        for name, indicator in self._indicators.items():
            data[name] = indicator

        # Generate signals
        signals = self.generate_signals(data)

        # Add symbol to signals if not set
        for signal in signals:
            if signal.symbol == "":
                signal.symbol = symbol

        # Validate and filter signals
        validated_signals = []
        current_positions = 0

        for signal in signals:
            if self.validate_signal(signal, data, current_positions):
                validated_signals.append(signal)

                # Track position count
                if signal.signal_type == SignalType.LONG:
                    current_positions += 1
                elif signal.signal_type == SignalType.EXIT_LONG:
                    current_positions = max(0, current_positions - 1)

        self._signals = validated_signals
        return validated_signals

    def get_indicator(self, name: str) -> Optional[pd.Series]:
        """Get a calculated indicator by name"""
        return self._indicators.get(name)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize strategy to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "source_url": self.source_url,
            "source_creator": self.source_creator,
            "trading_style": self.trading_style,
            "params": self.params.to_dict(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
