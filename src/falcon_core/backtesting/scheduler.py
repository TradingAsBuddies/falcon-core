"""
Scheduled Feedback Loop Runner

Runs daily backtests and parameter optimization after market data becomes available.
Polygon.io flat files are available by ~11:00 AM ET the following day.

Schedule: 11:30 AM ET daily (Mon-Fri)

Workflow:
1. Load previous trading day's intraday data
2. Run backtests on all active strategies
3. Compare to live trading results (if available)
4. Calculate parameter adjustments using feedback loop
5. Store results and optionally apply adjustments
6. Send notifications with summary

Can run as:
- Standalone daemon with built-in scheduler
- Systemd service (recommended for production)
- Cron job
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading

import pytz

logger = logging.getLogger(__name__)

# Import results store for database persistence
try:
    from falcon_core.backtesting.results_api import (
        BacktestResultsStore,
        BacktestRunRecord,
        FeedbackResultRecord,
    )
    RESULTS_STORE_AVAILABLE = True
except ImportError:
    RESULTS_STORE_AVAILABLE = False

# Check for schedule library
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    logger.warning("schedule library not installed. Install with: pip install schedule")


class FeedbackLoopScheduler:
    """
    Scheduled runner for daily backtesting feedback loops.

    Runs after Polygon flat files become available (~11:00 AM ET)
    to analyze previous day's trading and optimize strategy parameters.
    """

    # Default schedule: 11:30 AM ET (30 min buffer after data availability)
    DEFAULT_RUN_TIME = "11:30"
    TIMEZONE = pytz.timezone('America/New_York')

    def __init__(
        self,
        config_path: Optional[str] = None,
        results_dir: Optional[str] = None,
        auto_apply_adjustments: bool = False,
        notify_webhook: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        """
        Initialize the feedback loop scheduler.

        Args:
            config_path: Path to strategy configuration file
            results_dir: Directory to store feedback results
            auto_apply_adjustments: Whether to auto-apply parameter changes
            notify_webhook: Webhook URL for notifications (Slack, Discord, etc.)
            db_path: Path to results database (for web portal integration)
        """
        self.config_path = config_path or os.path.expanduser(
            "~/.config/falcon/strategies.json"
        )
        self.results_dir = Path(results_dir or os.path.expanduser(
            "~/.local/share/falcon/feedback_results"
        ))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.auto_apply = auto_apply_adjustments
        self.notify_webhook = notify_webhook

        self._running = False
        self._last_run: Optional[datetime] = None
        self._strategies = {}

        # Initialize database store for web portal
        self._results_store = None
        if RESULTS_STORE_AVAILABLE:
            try:
                self._results_store = BacktestResultsStore(db_path)
                logger.info("Results database initialized for web portal")
            except Exception as e:
                logger.warning(f"Failed to initialize results store: {e}")

        # Load configuration
        self._load_config()

    def _load_config(self):
        """Load strategy configuration"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    config = json.load(f)
                self._strategies = config.get('strategies', {})
                logger.info(f"Loaded {len(self._strategies)} strategies from config")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        else:
            # Create default config with all strategies enabled
            self._strategies = {
                'atr_breakout': {
                    'enabled': True,
                    'symbols': ['AMD', 'NVDA', 'TSLA', 'META', 'AAPL'],
                    'interval': '5m',
                    'params': {},
                },
                'market_memory': {
                    'enabled': True,
                    'symbols': ['AMD', 'NVDA', 'TSLA'],
                    'interval': '5m',
                    'params': {},
                },
                'one_candle_rule': {
                    'enabled': True,
                    'symbols': ['SPY', 'QQQ', 'AMD'],
                    'interval': '1m',
                    'params': {},
                },
            }
            self._save_config()

    def _save_config(self):
        """Save strategy configuration"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump({'strategies': self._strategies}, f, indent=2)
        logger.info(f"Saved config to {self.config_path}")

    def _get_previous_trading_day(self) -> date:
        """Get the previous trading day (skip weekends)"""
        today = datetime.now(self.TIMEZONE).date()
        prev_day = today - timedelta(days=1)

        # Skip weekends
        while prev_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
            prev_day -= timedelta(days=1)

        return prev_day

    def _load_strategy(self, strategy_name: str):
        """Load a strategy class by name"""
        if strategy_name == 'atr_breakout':
            from falcon_core.backtesting.strategies.atr_breakout import ATRBreakoutStrategy
            return ATRBreakoutStrategy
        elif strategy_name == 'market_memory':
            from falcon_core.backtesting.strategies.market_memory import MarketMemoryStrategy
            return MarketMemoryStrategy
        elif strategy_name == 'one_candle_rule':
            from falcon_core.backtesting.strategies.one_candle_rule import OneCandleRuleStrategy
            return OneCandleRuleStrategy
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def run_feedback_loop(self, trading_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Run the feedback loop for a specific trading day.

        Args:
            trading_date: Date to analyze (default: previous trading day)

        Returns:
            Dictionary with results for each strategy
        """
        from falcon_core.backtesting.data_feed import DataFeed
        from falcon_core.backtesting.engine import SimpleBacktestEngine
        from falcon_core.backtesting.optimizer import FeedbackLoop

        if trading_date is None:
            trading_date = self._get_previous_trading_day()

        logger.info(f"Running feedback loop for {trading_date}")

        results = {
            'date': trading_date.isoformat(),
            'run_time': datetime.now(self.TIMEZONE).isoformat(),
            'strategies': {},
        }

        # Initialize data feed
        feed = DataFeed()

        for strategy_name, config in self._strategies.items():
            if not config.get('enabled', True):
                logger.info(f"Skipping disabled strategy: {strategy_name}")
                continue

            logger.info(f"Processing strategy: {strategy_name}")
            strategy_results = {
                'symbols': {},
                'aggregate': {},
                'adjustments': {},
            }

            try:
                # Load strategy class
                StrategyClass = self._load_strategy(strategy_name)

                # Get custom params if any
                custom_params = config.get('params', {})

                for symbol in config.get('symbols', []):
                    try:
                        # Load data for the trading day
                        interval = config.get('interval', '5m')

                        # For intraday, we need data from the specific day
                        data = feed.get_historical_data(
                            symbol=symbol,
                            start_date=trading_date,
                            end_date=trading_date,
                            interval=interval,
                            market_hours_only=True,
                        )

                        if data.empty:
                            logger.warning(f"No data for {symbol} on {trading_date}")
                            continue

                        # Create strategy instance with custom params
                        if custom_params:
                            params = StrategyClass.default_params()
                            for k, v in custom_params.items():
                                if hasattr(params, k):
                                    setattr(params, k, v)
                            strategy = StrategyClass(params)
                        else:
                            strategy = StrategyClass()

                        # Run backtest
                        engine = SimpleBacktestEngine(initial_capital=25000)
                        result = engine.run(strategy, data, symbol=symbol)

                        strategy_results['symbols'][symbol] = {
                            'total_return': result.total_return,
                            'max_drawdown': result.max_drawdown,
                            'total_trades': result.total_trades,
                            'win_rate': result.win_rate,
                            'sharpe_ratio': result.sharpe_ratio,
                            'signals': len(result.signals),
                        }

                        # Store in database for web portal
                        if self._results_store:
                            try:
                                record = BacktestRunRecord(
                                    strategy_name=strategy_name,
                                    symbol=symbol,
                                    trading_date=trading_date,
                                    interval=interval,
                                    total_return=result.total_return,
                                    max_drawdown=result.max_drawdown,
                                    sharpe_ratio=result.sharpe_ratio,
                                    win_rate=result.win_rate,
                                    total_trades=result.total_trades,
                                    signals_count=len(result.signals),
                                    parameters=custom_params or {},
                                )
                                self._results_store.store_backtest_run(record)
                            except Exception as e:
                                logger.warning(f"Failed to store backtest run: {e}")

                        logger.info(
                            f"  {symbol}: {result.total_return:.2%} return, "
                            f"{result.total_trades} trades, {result.win_rate:.1%} win rate"
                        )

                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        strategy_results['symbols'][symbol] = {'error': str(e)}

                # Calculate aggregate metrics
                symbol_results = [
                    r for r in strategy_results['symbols'].values()
                    if 'error' not in r and r.get('total_trades', 0) > 0
                ]

                if symbol_results:
                    strategy_results['aggregate'] = {
                        'avg_return': sum(r['total_return'] for r in symbol_results) / len(symbol_results),
                        'avg_win_rate': sum(r['win_rate'] for r in symbol_results) / len(symbol_results),
                        'total_trades': sum(r['total_trades'] for r in symbol_results),
                        'symbols_tested': len(symbol_results),
                    }

                    # Run feedback loop for parameter adjustment suggestions
                    feedback = FeedbackLoop(StrategyClass)

                    # Collect all backtest results
                    backtest_results = []
                    for symbol, r in strategy_results['symbols'].items():
                        if 'error' not in r:
                            backtest_results.append({
                                'symbol': symbol,
                                'metrics': r,
                                'params': custom_params or {},
                            })

                    # Generate adjustment suggestions
                    if backtest_results:
                        adjustments = feedback.analyze_results(backtest_results)
                        strategy_results['adjustments'] = adjustments

                        # Auto-apply if enabled and adjustments are beneficial
                        if self.auto_apply and adjustments.get('apply_recommended'):
                            new_params = adjustments.get('new_params', {})
                            if new_params:
                                self._strategies[strategy_name]['params'] = new_params
                                self._save_config()
                                logger.info(f"Auto-applied parameter adjustments for {strategy_name}")

                # Store feedback result in database
                if self._results_store and 'aggregate' in strategy_results:
                    try:
                        agg = strategy_results['aggregate']
                        adj = strategy_results.get('adjustments', {})
                        feedback_record = FeedbackResultRecord(
                            run_date=trading_date,
                            strategy_name=strategy_name,
                            symbols_tested=agg.get('symbols_tested', 0),
                            total_trades=agg.get('total_trades', 0),
                            avg_return=agg.get('avg_return', 0.0),
                            avg_win_rate=agg.get('avg_win_rate', 0.0),
                            avg_sharpe=adj.get('avg_sharpe', 0.0),
                            adjustments_recommended=adj.get('apply_recommended', False),
                            adjustments_applied=self.auto_apply and adj.get('apply_recommended', False),
                            adjustment_details=adj,
                        )
                        self._results_store.store_feedback_result(feedback_record)
                        logger.info(f"Stored feedback result for {strategy_name}")
                    except Exception as e:
                        logger.warning(f"Failed to store feedback result: {e}")

            except Exception as e:
                logger.error(f"Error processing strategy {strategy_name}: {e}")
                strategy_results['error'] = str(e)

            results['strategies'][strategy_name] = strategy_results

        # Save results
        results_file = self.results_dir / f"feedback_{trading_date.isoformat()}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {results_file}")

        # Send notification
        self._send_notification(results)

        self._last_run = datetime.now(self.TIMEZONE)
        return results

    def _send_notification(self, results: Dict[str, Any]):
        """Send notification with results summary"""
        if not self.notify_webhook:
            return

        try:
            import requests

            # Build summary message
            summary_lines = [
                f"*Feedback Loop Results - {results['date']}*",
                "",
            ]

            for strategy_name, strategy_results in results['strategies'].items():
                if 'error' in strategy_results:
                    summary_lines.append(f"_{strategy_name}_: Error - {strategy_results['error']}")
                elif 'aggregate' in strategy_results:
                    agg = strategy_results['aggregate']
                    summary_lines.append(
                        f"_{strategy_name}_: "
                        f"{agg.get('avg_return', 0):.2%} avg return, "
                        f"{agg.get('avg_win_rate', 0):.1%} win rate, "
                        f"{agg.get('total_trades', 0)} trades"
                    )

                    if strategy_results.get('adjustments', {}).get('apply_recommended'):
                        summary_lines.append(f"  -> Parameter adjustments recommended")

            message = "\n".join(summary_lines)

            # Try Slack format first
            payload = {"text": message}
            requests.post(self.notify_webhook, json=payload, timeout=10)
            logger.info("Sent notification")

        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")

    def run_daemon(self, run_time: str = None):
        """
        Run as a daemon with built-in scheduler.

        Args:
            run_time: Time to run daily in HH:MM format (ET timezone)
        """
        if not SCHEDULE_AVAILABLE:
            raise ImportError("schedule library required. Install with: pip install schedule")

        run_time = run_time or self.DEFAULT_RUN_TIME

        logger.info(f"Starting feedback loop daemon (scheduled: {run_time} ET daily)")

        # Schedule the job
        schedule.every().monday.at(run_time).do(self.run_feedback_loop)
        schedule.every().tuesday.at(run_time).do(self.run_feedback_loop)
        schedule.every().wednesday.at(run_time).do(self.run_feedback_loop)
        schedule.every().thursday.at(run_time).do(self.run_feedback_loop)
        schedule.every().friday.at(run_time).do(self.run_feedback_loop)

        self._running = True

        while self._running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def stop_daemon(self):
        """Stop the daemon"""
        self._running = False
        logger.info("Stopping feedback loop daemon")


def run_feedback_now():
    """Run feedback loop immediately (for testing or manual trigger)"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    scheduler = FeedbackLoopScheduler()
    results = scheduler.run_feedback_loop()

    print("\n" + "="*60)
    print("FEEDBACK LOOP RESULTS")
    print("="*60)
    print(f"Date: {results['date']}")

    for strategy_name, strategy_results in results['strategies'].items():
        print(f"\n{strategy_name}:")
        if 'error' in strategy_results:
            print(f"  Error: {strategy_results['error']}")
        elif 'aggregate' in strategy_results:
            agg = strategy_results['aggregate']
            print(f"  Avg Return: {agg.get('avg_return', 0):.2%}")
            print(f"  Avg Win Rate: {agg.get('avg_win_rate', 0):.1%}")
            print(f"  Total Trades: {agg.get('total_trades', 0)}")
            print(f"  Symbols Tested: {agg.get('symbols_tested', 0)}")

            if strategy_results.get('adjustments'):
                print(f"  Adjustments: {strategy_results['adjustments']}")


def main():
    """Main entry point for the scheduler"""
    import argparse

    parser = argparse.ArgumentParser(description='Falcon Feedback Loop Scheduler')
    parser.add_argument('--run-now', action='store_true', help='Run feedback loop immediately')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon with scheduler')
    parser.add_argument('--time', default='11:30', help='Daily run time (HH:MM ET)')
    parser.add_argument('--auto-apply', action='store_true', help='Auto-apply parameter adjustments')
    parser.add_argument('--webhook', help='Notification webhook URL')
    parser.add_argument('--date', help='Specific date to analyze (YYYY-MM-DD)')
    parser.add_argument('--db-path', help='Path to results database')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Support environment variable configuration for systemd
    auto_apply = args.auto_apply or os.getenv('AUTO_APPLY_ADJUSTMENTS', '').lower() == 'true'
    webhook = args.webhook or os.getenv('NOTIFY_WEBHOOK')
    db_path = args.db_path or os.getenv('DB_PATH')

    scheduler = FeedbackLoopScheduler(
        auto_apply_adjustments=auto_apply,
        notify_webhook=webhook,
        db_path=db_path,
    )

    if args.run_now or args.date:
        trading_date = None
        if args.date:
            trading_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        scheduler.run_feedback_loop(trading_date)
    elif args.daemon:
        try:
            scheduler.run_daemon(args.time)
        except KeyboardInterrupt:
            scheduler.stop_daemon()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
