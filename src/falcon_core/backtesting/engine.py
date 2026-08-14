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

TRADING_DAYS_PER_YEAR = 252

# Minimum daily return observations before the risk metrics are worth
# reporting. Below this, standard deviation is dominated by noise and the
# resulting Sharpe is not a measurement of anything.
MIN_RETURN_OBSERVATIONS = 20


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

    # Components behind profit_factor. Exposed so callers can aggregate across
    # symbols correctly (sum the parts, then divide — averaging per-symbol
    # ratios is not the same number) and can tell a profit_factor of 0.0
    # "there were no trades" from "there were no losses, so it is undefined".
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    # False when there were too few return observations for the risk metrics
    # to mean anything. Promotion gates should refuse to act on such a result
    # rather than treating a noisy Sharpe as signal.
    metrics_reliable: bool = True

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
                # Open long position (slippage multiplicative, commission additive)
                entry_price = signal.price * (1 + self.slippage) + signal.price * self.commission
                position = {
                    "entry_time": signal.timestamp,
                    "entry_price": entry_price,
                    "size": signal.position_size or self.initial_capital * 0.1,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "direction": "long",
                }

            elif signal.signal_type == SignalType.EXIT_LONG and position is not None and position["direction"] == "long":
                # Close long position (slippage multiplicative, commission additive)
                exit_price = signal.price * (1 - self.slippage) - signal.price * self.commission

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
                    "direction": "long",
                    "duration": (signal.timestamp - position["entry_time"]).days
                    if hasattr(signal.timestamp, 'days') else 1,
                })
                position = None

            elif signal.signal_type == SignalType.SHORT and position is None:
                # Open short position (slippage works against us — price slips down, commission additive)
                entry_price = signal.price * (1 - self.slippage) - signal.price * self.commission
                position = {
                    "entry_time": signal.timestamp,
                    "entry_price": entry_price,
                    "size": signal.position_size or self.initial_capital * 0.1,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "direction": "short",
                }

            elif signal.signal_type == SignalType.EXIT_SHORT and position is not None and position["direction"] == "short":
                # Close short position (slippage multiplicative, commission additive)
                exit_price = signal.price * (1 + self.slippage) + signal.price * self.commission

                pnl = (position["entry_price"] - exit_price) / position["entry_price"]
                pnl_dollar = position["size"] * pnl

                trades_list.append({
                    "entry_time": position["entry_time"],
                    "exit_time": signal.timestamp,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "size": position["size"],
                    "pnl_pct": pnl,
                    "pnl_dollar": pnl_dollar,
                    "direction": "short",
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

        # Profit factor. gross_loss defaults to 0, not a nominal 0.01 — a
        # strategy with no losing trades has an *undefined* profit factor, and
        # the old default quietly turned that into gross_profit x100. Callers
        # separate the two zero cases via the gross_* fields.
        gross_profit = float(winning['pnl_dollar'].sum()) if len(winning) > 0 else 0.0
        gross_loss = abs(float(losing['pnl_dollar'].sum())) if len(losing) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Build daily equity curve for risk metrics
        equity = self._build_equity_curve(trades, data)

        # Risk metrics
        returns = equity.pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 0 else 0.0

        # Sharpe ratio (0% risk-free rate).
        #
        # This was previously annual_return / volatility, which mixed two
        # incompatible annualisations: the numerator was a *geometric*
        # extrapolation, ((1+total_return) ** (1/years)) - 1, so over a 38-day
        # window it raised the return to the power of ~9.6, while the
        # denominator was scaled by sqrt(252). Combined with an equity curve
        # that is flat on every day without a fill — which shrinks the measured
        # volatility as trade count falls — the result rewarded strategies for
        # trading *less*. It reported 25.98 on four trades, and the review
        # agent then optimised toward exactly that.
        #
        # Standard form instead: mean periodic excess return over its standard
        # deviation, scaled by sqrt(periods per year). Both halves now describe
        # the same daily return series.
        std_daily = float(returns.std()) if len(returns) > 0 else 0.0
        mean_daily = float(returns.mean()) if len(returns) > 0 else 0.0

        metrics_reliable = len(returns) >= MIN_RETURN_OBSERVATIONS
        if metrics_reliable and std_daily > 0:
            sharpe = (mean_daily / std_daily) * np.sqrt(TRADING_DAYS_PER_YEAR)
            sharpe = float(sharpe)
        else:
            # Too short a series, or no variation in it, to say anything.
            sharpe = 0.0

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
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            metrics_reliable=metrics_reliable,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_trade_duration=trades['duration'].mean() if 'duration' in trades else 0,
            equity_curve=equity,
            trades=trades,
            signals=signals,
            params_used=strategy.params.to_dict(),
        )

    def _build_equity_curve(
        self, trades: pd.DataFrame, data: pd.DataFrame,
    ) -> pd.Series:
        """Build daily equity curve spanning the full backtest period.

        Maps trade P&L to the calendar day each trade exits, then
        forward-fills so every trading day has an equity value.  This
        gives a proper daily return series for risk-metric calculations
        (Sharpe, volatility, drawdown).
        """
        if trades.empty:
            return pd.Series(
                self.initial_capital,
                index=pd.DatetimeIndex([data.index[0]]),
            )

        # Build a daily date range covering the data period
        start = data.index[0]
        end = data.index[-1]
        # Normalise to dates so we get one point per calendar day
        daily_index = pd.bdate_range(
            start=start.normalize(), end=end.normalize(), freq="B",
        )

        # Assign each trade's P&L to its exit date
        pnl_by_day = pd.Series(0.0, index=daily_index)
        for _, trade in trades.iterrows():
            exit_dt = pd.Timestamp(trade["exit_time"]).normalize()
            # Find the nearest business day in case exit falls on a weekend
            idx = pnl_by_day.index.get_indexer([exit_dt], method="nearest")
            if idx[0] >= 0:
                pnl_by_day.iloc[idx[0]] += trade["pnl_dollar"]

        # Cumulative equity
        equity = self.initial_capital + pnl_by_day.cumsum()
        return equity

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

        # Prepare data for bt (needs price data with symbol as column name)
        price_data = data[['close']].copy()
        price_data.columns = [symbol]

        # Remove timezone info if present (bt doesn't handle tz well)
        if price_data.index.tz is not None:
            price_data.index = price_data.index.tz_localize(None)

        # Convert signals to bt format (weights DataFrame)
        signal_weights = self._signals_to_weights(signals, price_data, symbol)

        # Create bt strategy
        bt_strategy = self._bt.Strategy(
            strategy.name,
            [
                self._bt.algos.WeighTarget(signal_weights),
                self._bt.algos.Rebalance(),
            ]
        )

        # Run backtest
        backtest = self._bt.Backtest(bt_strategy, price_data)
        result = self._bt.run(backtest)

        # Extract metrics
        return self._extract_metrics(strategy, result, data, symbol, signals)

    def _signals_to_weights(
        self,
        signals: List[Signal],
        data: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """Convert signal list to weight DataFrame for bt"""
        # Create weight series (0 = no position, 1 = full position)
        # bt requires weights to have symbol as column name
        weights = pd.DataFrame(0.0, index=data.index, columns=[symbol])

        in_position = False
        for signal in signals:
            idx = signal.timestamp

            # Strip timezone if present to match data index
            if hasattr(idx, 'tzinfo') and idx.tzinfo is not None:
                idx = idx.replace(tzinfo=None)

            if idx not in weights.index:
                # Find nearest index
                try:
                    nearest_idx = weights.index.get_indexer([idx], method='nearest')[0]
                    if nearest_idx >= 0:
                        idx = weights.index[nearest_idx]
                    else:
                        continue
                except Exception:
                    continue

            if signal.signal_type == SignalType.LONG:
                in_position = True
            elif signal.signal_type == SignalType.EXIT_LONG:
                in_position = False

            # Set weight from this point forward
            weights.loc[idx:, symbol] = 1.0 if in_position else 0.0

        return weights

    def _extract_metrics(
        self,
        strategy: BaseStrategy,
        bt_result: Any,
        data: pd.DataFrame,
        symbol: str,
        signals: List[Signal],
    ) -> BacktestResult:
        """Extract metrics from bt result.

        Note the deliberate gap: bt reports portfolio-level return and risk but
        does not give us per-trade P&L, so win_rate, profit_factor, avg_win,
        avg_loss, expectancy and the gross_* fields are left at their defaults
        rather than being filled with fabricated values. A caller that needs
        trade statistics should use SimpleBacktestEngine, which tracks fills
        itself. This is why a run through this engine shows real returns beside
        a 0.0 win rate — the metrics are absent, not zero.
        """
        stats = bt_result.stats

        def stat(key: str, default: float = 0.0) -> float:
            """Read a bt stat, treating NaN as absent.

            bt emits NaN for ratios it cannot compute on a short or flat
            series, which then propagated into the roster as a NaN Sharpe.
            """
            if key not in stats.index:
                return default
            value = float(stats.loc[key, strategy.name])
            return default if not np.isfinite(value) else value

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=data.index[0],
            end_date=data.index[-1],
            total_return=stat('total_return'),
            annual_return=stat('cagr'),
            sharpe_ratio=stat('daily_sharpe'),
            sortino_ratio=stat('daily_sortino'),
            max_drawdown=abs(stat('max_drawdown')),
            volatility=stat('daily_vol') * np.sqrt(TRADING_DAYS_PER_YEAR),
            total_trades=len(signals) // 2,  # Approximate — signal pairs
            # bt gives no per-trade breakdown, so the trade statistics above
            # are not available from this engine.
            metrics_reliable=False,
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
