"""
Backtest Engine for Falcon Trading Platform

Provides event-driven backtesting with pluggable backends:
- bt (MIT license, default)
- backtrader (GPLv3, optional - user installs separately)

The engine executes strategies against historical data and
produces performance metrics.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
import pandas as pd
import numpy as np

from falcon_core.backtesting.strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime

    # Returns
    total_return: float  # Total % return
    annual_return: float  # Annualized return
    benchmark_return: float = 0.0  # Buy & hold return

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # Days
    volatility: float = 0.0

    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Position metrics
    avg_trade_duration: float = 0.0  # Days
    avg_position_size: float = 0.0

    # Raw data
    equity_curve: Optional[pd.Series] = None
    trades: Optional[pd.DataFrame] = None
    signals: List[Signal] = field(default_factory=list)
    params_used: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "benchmark_return": self.benchmark_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "volatility": self.volatility,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "avg_trade_duration": self.avg_trade_duration,
            "avg_position_size": self.avg_position_size,
            "params_used": self.params_used,
        }

    def summary(self) -> str:
        """Return formatted summary string"""
        return f"""
Backtest Results: {self.strategy_name}
{'=' * 50}
Symbol: {self.symbol}
Period: {self.start_date.date()} to {self.end_date.date()}

RETURNS
  Total Return: {self.total_return:.2%}
  Annual Return: {self.annual_return:.2%}
  Benchmark: {self.benchmark_return:.2%}

RISK
  Sharpe Ratio: {self.sharpe_ratio:.2f}
  Sortino Ratio: {self.sortino_ratio:.2f}
  Max Drawdown: {self.max_drawdown:.2%}
  Volatility: {self.volatility:.2%}

TRADES
  Total: {self.total_trades}
  Win Rate: {self.win_rate:.2%}
  Profit Factor: {self.profit_factor:.2f}
  Expectancy: ${self.expectancy:.2f}
"""


class BacktestEngine(ABC):
    """
    Abstract base class for backtest engines.

    Implementations must provide run() method that executes
    a strategy against historical data.
    """

    name: str = "base_engine"

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.001,    # 0.1% slippage
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    @abstractmethod
    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> BacktestResult:
        """
        Run backtest on a strategy with data.

        Args:
            strategy: Strategy instance to test
            data: DataFrame with OHLCV data
            symbol: Symbol being tested

        Returns:
            BacktestResult with performance metrics
        """
        pass

    def run_multiple(
        self,
        strategy: BaseStrategy,
        data_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest on multiple symbols.

        Returns dict of symbol -> BacktestResult
        """
        results = {}
        for symbol, data in data_dict.items():
            try:
                result = self.run(strategy, data, symbol)
                results[symbol] = result
            except Exception as e:
                logger.warning(f"Backtest failed for {symbol}: {e}")

        return results


class SimpleBacktestEngine(BacktestEngine):
    """
    Simple vectorized backtest engine.

    Uses signals from strategy to simulate trading without
    external dependencies. Good for quick validation.
    """

    name = "simple_engine"

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> BacktestResult:
        """Run simple vectorized backtest"""
        # Generate signals
        signals = strategy.run(data, symbol)

        if not signals:
            logger.warning(f"No signals generated for {symbol}")
            return self._empty_result(strategy, data, symbol)

        # Simulate trades
        trades = self._simulate_trades(signals, data)

        # Calculate metrics
        return self._calculate_metrics(strategy, data, symbol, signals, trades)

    def _simulate_trades(
        self,
        signals: List[Signal],
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Simulate trades from signals"""
        trades_list = []
        position = None

        for signal in signals:
            if signal.signal_type == SignalType.LONG and position is None:
                # Open position
                entry_price = signal.price * (1 + self.slippage)
                entry_price *= (1 + self.commission)
                position = {
                    "entry_time": signal.timestamp,
                    "entry_price": entry_price,
                    "size": signal.position_size or self.initial_capital * 0.1,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                }

            elif signal.signal_type == SignalType.EXIT_LONG and position is not None:
                # Close position
                exit_price = signal.price * (1 - self.slippage)
                exit_price *= (1 - self.commission)

                pnl = (exit_price - position["entry_price"]) / position["entry_price"]
                pnl_dollar = position["size"] * pnl

                trades_list.append({
                    "entry_time": position["entry_time"],
                    "exit_time": signal.timestamp,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "size": position["size"],
                    "pnl_pct": pnl,
                    "pnl_dollar": pnl_dollar,
                    "duration": (signal.timestamp - position["entry_time"]).days
                    if hasattr(signal.timestamp, 'days') else 1,
                })
                position = None

        return pd.DataFrame(trades_list) if trades_list else pd.DataFrame()

    def _calculate_metrics(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str,
        signals: List[Signal],
        trades: pd.DataFrame,
    ) -> BacktestResult:
        """Calculate performance metrics from trades"""
        start_date = data.index[0]
        end_date = data.index[-1]

        # Benchmark return (buy & hold)
        benchmark_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1

        if trades.empty:
            return BacktestResult(
                strategy_name=strategy.name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                total_return=0.0,
                annual_return=0.0,
                benchmark_return=benchmark_return,
                signals=signals,
                params_used=strategy.params.to_dict(),
            )

        # Calculate returns
        total_pnl = trades['pnl_dollar'].sum()
        total_return = total_pnl / self.initial_capital

        # Annualized return
        days = (end_date - start_date).days
        years = max(days / 365.25, 0.01)
        annual_return = ((1 + total_return) ** (1 / years)) - 1

        # Win/loss stats
        winning = trades[trades['pnl_pct'] > 0]
        losing = trades[trades['pnl_pct'] <= 0]

        win_rate = len(winning) / len(trades) if len(trades) > 0 else 0
        avg_win = winning['pnl_pct'].mean() if len(winning) > 0 else 0
        avg_loss = abs(losing['pnl_pct'].mean()) if len(losing) > 0 else 0

        # Profit factor
        gross_profit = winning['pnl_dollar'].sum() if len(winning) > 0 else 0
        gross_loss = abs(losing['pnl_dollar'].sum()) if len(losing) > 0 else 0.01
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Build equity curve for risk metrics
        equity = self._build_equity_curve(trades)

        # Risk metrics
        returns = equity.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = (annual_return / volatility) if volatility > 0 else 0

        # Max drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            annual_return=annual_return,
            benchmark_return=benchmark_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            volatility=volatility,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_trade_duration=trades['duration'].mean() if 'duration' in trades else 0,
            equity_curve=equity,
            trades=trades,
            signals=signals,
            params_used=strategy.params.to_dict(),
        )

    def _build_equity_curve(self, trades: pd.DataFrame) -> pd.Series:
        """Build equity curve from trades"""
        if trades.empty:
            return pd.Series([self.initial_capital])

        equity = [self.initial_capital]
        current = self.initial_capital

        for _, trade in trades.iterrows():
            current += trade['pnl_dollar']
            equity.append(current)

        return pd.Series(equity)

    def _empty_result(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str,
    ) -> BacktestResult:
        """Return empty result for no-signal case"""
        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=data.index[0],
            end_date=data.index[-1],
            total_return=0.0,
            annual_return=0.0,
            benchmark_return=(data['close'].iloc[-1] / data['close'].iloc[0]) - 1,
            params_used=strategy.params.to_dict(),
        )


class BTBacktestEngine(BacktestEngine):
    """
    Event-driven backtest engine using bt library (MIT license).

    Provides more realistic simulation with proper order execution.
    """

    name = "bt_engine"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bt = None
        self._ffn = None
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if bt is available"""
        try:
            import bt
            import ffn
            self._bt = bt
            self._ffn = ffn
        except ImportError:
            logger.warning(
                "bt/ffn not installed. Install with: pip install bt ffn"
            )

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> BacktestResult:
        """Run backtest using bt library"""
        if self._bt is None:
            logger.warning("bt not available, falling back to simple engine")
            simple = SimpleBacktestEngine(
                self.initial_capital, self.commission, self.slippage
            )
            return simple.run(strategy, data, symbol)

        # Generate signals from our strategy
        signals = strategy.run(data, symbol)

        # Convert signals to bt format
        signal_series = self._signals_to_weights(signals, data)

        # Create bt strategy
        bt_strategy = self._bt.Strategy(
            strategy.name,
            [
                self._bt.algos.WeighTarget(signal_series),
                self._bt.algos.Rebalance(),
            ]
        )

        # Prepare data for bt (needs price data with symbol as column name)
        price_data = data[['close']].copy()
        price_data.columns = [symbol]

        # Run backtest
        backtest = self._bt.Backtest(bt_strategy, price_data)
        result = self._bt.run(backtest)

        # Extract metrics
        return self._extract_metrics(strategy, result, data, symbol, signals)

    def _signals_to_weights(
        self,
        signals: List[Signal],
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert signal list to weight DataFrame for bt"""
        # Create weight series (0 = no position, 1 = full position)
        weights = pd.Series(0.0, index=data.index)

        in_position = False
        for signal in signals:
            idx = signal.timestamp
            if idx not in weights.index:
                # Find nearest index
                idx = weights.index[weights.index.get_indexer([idx], method='nearest')[0]]

            if signal.signal_type == SignalType.LONG:
                in_position = True
            elif signal.signal_type == SignalType.EXIT_LONG:
                in_position = False

            # Set weight from this point forward
            weights.loc[idx:] = 1.0 if in_position else 0.0

        return weights.to_frame(name='weight')

    def _extract_metrics(
        self,
        strategy: BaseStrategy,
        bt_result: Any,
        data: pd.DataFrame,
        symbol: str,
        signals: List[Signal],
    ) -> BacktestResult:
        """Extract metrics from bt result"""
        stats = bt_result.stats

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=data.index[0],
            end_date=data.index[-1],
            total_return=float(stats.loc['total_return', strategy.name]),
            annual_return=float(stats.loc['cagr', strategy.name]),
            sharpe_ratio=float(stats.loc['daily_sharpe', strategy.name]),
            sortino_ratio=float(stats.loc['daily_sortino', strategy.name]) if 'daily_sortino' in stats.index else 0,
            max_drawdown=abs(float(stats.loc['max_drawdown', strategy.name])),
            volatility=float(stats.loc['daily_vol', strategy.name]) * np.sqrt(252),
            total_trades=len(signals) // 2,  # Approximate
            equity_curve=bt_result.prices[strategy.name],
            signals=signals,
            params_used=strategy.params.to_dict(),
        )


def create_engine(
    engine_type: str = "auto",
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    slippage: float = 0.001,
) -> BacktestEngine:
    """
    Factory function to create backtest engine.

    Args:
        engine_type: 'simple', 'bt', or 'auto'
        initial_capital: Starting capital
        commission: Commission per trade (fraction)
        slippage: Slippage per trade (fraction)

    Returns:
        BacktestEngine instance
    """
    if engine_type == "simple":
        return SimpleBacktestEngine(initial_capital, commission, slippage)

    if engine_type == "bt":
        return BTBacktestEngine(initial_capital, commission, slippage)

    # Auto: try bt first, fall back to simple
    try:
        import bt
        return BTBacktestEngine(initial_capital, commission, slippage)
    except ImportError:
        logger.info("bt not available, using simple engine")
        return SimpleBacktestEngine(initial_capital, commission, slippage)
