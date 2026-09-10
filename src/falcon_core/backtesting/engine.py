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

#: Trading days in a year, and the RTH minutes in a session -- the two constants
#: every annualization here derives from.
TRADING_DAYS_PER_YEAR = 252
RTH_MINUTES_PER_SESSION = 390

# Minimum return observations before the risk metrics are worth reporting.
# Below this, standard deviation is dominated by noise and the resulting
# Sharpe is not a measurement of anything.
MIN_RETURN_OBSERVATIONS = 20


def _infer_bar_frequency(data: "pd.DataFrame") -> Optional[str]:
    """Best-effort bar frequency label ('1min', '5min', '1day', ...)."""
    if data is None or len(data) < 3:
        return None
    try:
        deltas = pd.Series(data.index[1:]) - pd.Series(data.index[:-1])
        seconds = float(deltas.dt.total_seconds().median())
    except Exception:
        return None
    if seconds <= 0:
        return None
    if seconds < 3600:
        return f"{int(round(seconds / 60))}min"
    if seconds < 86400:
        return f"{int(round(seconds / 3600))}h"
    return f"{int(round(seconds / 86400))}day"


def _bars_per_year(data: "pd.DataFrame") -> float:
    """How many bars of this frequency occur in a trading year.

    Used to annualize volatility at the *bar* frequency. The old code always
    multiplied by sqrt(252) regardless of whether the bars were daily or
    one-minute, which is off by sqrt(390) on intraday data (falcon-core#21).
    """
    freq = _infer_bar_frequency(data)
    if not freq:
        return float(TRADING_DAYS_PER_YEAR)
    if freq.endswith("min"):
        per_session = RTH_MINUTES_PER_SESSION / max(int(freq[:-3]), 1)
    elif freq.endswith("h"):
        per_session = (RTH_MINUTES_PER_SESSION / 60) / max(int(freq[:-1]), 1)
    else:
        per_session = 1.0 / max(int(freq[:-3]), 1)
    return float(TRADING_DAYS_PER_YEAR) * per_session


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

    # Run provenance. A backtest that loaded nothing and a backtest whose
    # strategy declined to trade both used to look like "0 trades, success"
    # (falcon-core#21). These make them distinguishable on every run row.
    status: str = "ok"          # ok | error
    reason: Optional[str] = None  # no_data | insufficient_bars | no_signals
    bars_loaded: int = 0
    engine_used: str = "simple"
    bar_frequency: Optional[str] = None

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
            "status": self.status,
            "reason": self.reason,
            "bars_loaded": self.bars_loaded,
            "engine_used": self.engine_used,
            "bar_frequency": self.bar_frequency,
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
        # A run that loaded no bars is a data failure, not a strategy that chose
        # not to trade. Reporting the two identically is what let four strategies
        # sit at "100 runs / 0 trades" and look like a strategy problem
        # (falcon-core#21).
        bars = 0 if data is None else len(data)
        min_bars = getattr(strategy, 'min_bars', 0) or 0

        if bars == 0:
            logger.error(
                "Backtest for %s/%s loaded 0 bars", strategy.name, symbol,
            )
            return self._error_result(strategy, symbol, 'no_data', bars)

        if bars < min_bars:
            logger.error(
                "Backtest for %s/%s loaded %d bars, below the strategy minimum "
                "of %d", strategy.name, symbol, bars, min_bars,
            )
            return self._error_result(
                strategy, symbol, 'insufficient_bars', bars, data=data,
            )

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
                status='ok',
                reason='no_trades',
                bars_loaded=len(data),
                engine_used=type(self).__name__,
                bar_frequency=_infer_bar_frequency(data),
            )

        # Calculate returns
        total_pnl = trades['pnl_dollar'].sum()
        total_return = total_pnl / self.initial_capital

        # Annualized return.
        #
        # years used to be floored at 0.01 (~3.65 days), so a short intraday
        # window raised total return to the ~100th power and produced a number
        # with no meaning. Below one session we simply do not annualize --
        # reporting the period return is honest; extrapolating it is not
        # (falcon-core#21).
        days = (end_date - start_date).days
        years = days / 365.25
        if years >= (1.0 / TRADING_DAYS_PER_YEAR):
            annual_return = ((1 + total_return) ** (1 / years)) - 1
        else:
            annual_return = total_return

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

        # Build equity curve for risk metrics
        equity = self._build_equity_curve(trades)

        # Risk metrics.
        #
        # The equity curve is trade-indexed, so its per-step volatility is
        # per-trade, not per-day. Scaling it by sqrt(252) while dividing an
        # annualized return by it mixed two different time bases -- which is why
        # atr_breakout reported -6.97 on 1914 trades regardless of edge.
        # Annualize at the actual bar frequency instead, and derive Sharpe from
        # the same period returns the volatility came from.
        returns = equity.pct_change().dropna()
        periods_per_year = _bars_per_year(data)

        # Reliability guard. Below MIN_RETURN_OBSERVATIONS the standard
        # deviation is dominated by noise, and a Sharpe computed from it is not
        # a measurement of anything -- atr_breakout reported 25.98 on four
        # trades, and the review agent then optimised toward exactly that.
        # Such a result is reported as unreliable rather than as signal.
        metrics_reliable = len(returns) >= MIN_RETURN_OBSERVATIONS

        if len(returns) > 1:
            period_std = float(returns.std())
            period_mean = float(returns.mean())
            volatility = period_std * np.sqrt(periods_per_year)
            sharpe = (
                (period_mean / period_std) * np.sqrt(periods_per_year)
                if (metrics_reliable and period_std > 0) else 0.0
            )
        else:
            volatility = 0.0
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
            status='ok',
            reason=None,
            bars_loaded=len(data),
            engine_used=type(self).__name__,
            bar_frequency=_infer_bar_frequency(data),
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
        """Return empty result for the no-signal case.

        This means the data loaded fine and the strategy produced no signal --
        a legitimate outcome. An empty *data load* goes through
        :meth:`_error_result` instead.
        """
        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=data.index[0],
            end_date=data.index[-1],
            total_return=0.0,
            annual_return=0.0,
            benchmark_return=(data['close'].iloc[-1] / data['close'].iloc[0]) - 1,
            params_used=strategy.params.to_dict(),
            status='ok',
            reason='no_signals',
            bars_loaded=len(data),
            engine_used=type(self).__name__,
            bar_frequency=_infer_bar_frequency(data),
        )

    def _error_result(
        self,
        strategy: BaseStrategy,
        symbol: str,
        reason: str,
        bars_loaded: int,
        data: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Return a result flagged as a failed run (falcon-core#21)."""
        has_rows = data is not None and len(data) > 0
        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=data.index[0] if has_rows else None,
            end_date=data.index[-1] if has_rows else None,
            total_return=0.0,
            annual_return=0.0,
            benchmark_return=0.0,
            params_used=strategy.params.to_dict(),
            status='error',
            reason=reason,
            bars_loaded=bars_loaded,
            engine_used=type(self).__name__,
            bar_frequency=_infer_bar_frequency(data) if has_rows else None,
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
            # Loud, not a warning: a caller that asked for the event-driven engine
            # and silently got the vectorized one has been measuring something
            # other than what it thinks (falcon-core#21). The returned result
            # carries engine_used='SimpleBacktestEngine' so the run row is honest.
            logger.error(
                "BTBacktestEngine requested but bt/ffn are not installed; "
                "falling back to SimpleBacktestEngine for %s/%s. Results are "
                "vectorized, not event-driven.", strategy.name, symbol,
            )
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
