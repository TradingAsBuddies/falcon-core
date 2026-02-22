#!/usr/bin/env python3
"""
Strategy Seeder & Rotation Manager

Seeds all registered strategies into the strategy_roster table and manages
the rotation lifecycle: backtest -> paper_trading -> review -> backtest.

Usage:
    falcon-strategy-seed                        # Seed all strategies into roster
    falcon-strategy-seed --backtest             # Seed + run initial backtests
    falcon-strategy-seed --status               # Show roster with metrics
    falcon-strategy-seed --promote <name>       # Promote to paper_trading
    falcon-strategy-seed --demote <name> --reason "..."  # Demote to review
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_db():
    """Get database manager instance."""
    from falcon_core.db_manager import get_db_manager
    db = get_db_manager()
    db.init_schema()
    return db


def seed_strategies(db) -> List[str]:
    """Seed all registered strategies into the strategy_roster table.

    Returns list of newly seeded strategy names.
    """
    from falcon_core.backtesting.strategies import get_available_strategies

    strategies = get_available_strategies()
    seeded = []

    for name, cls in strategies.items():
        strategy = cls()
        params = strategy.params

        # Determine default symbols and interval from scheduler defaults
        default_config = _get_default_config(name)
        symbols = json.dumps(default_config.get('symbols', []))
        interval = default_config.get('interval', getattr(params, 'recommended_interval', '5m'))

        # Check if already exists
        existing = db.execute(
            'SELECT id FROM strategy_roster WHERE strategy_name = %s',
            (name,), fetch='one'
        )

        if existing:
            logger.info(f"  {name}: already in roster (skipped)")
            continue

        # Serialize params
        params_json = json.dumps(params.to_dict())
        now = datetime.now().isoformat()

        db.execute(
            '''INSERT INTO strategy_roster
               (strategy_name, status, symbols, interval, params, created_at, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s)''',
            (name, 'backtest', symbols, interval, params_json, now, now)
        )

        # Log the seeding
        db.execute(
            '''INSERT INTO strategy_rotation_log
               (strategy_name, from_status, to_status, reason, rotated_at)
               VALUES (%s, %s, %s, %s, %s)''',
            (name, None, 'backtest', 'Initial seed', now)
        )

        seeded.append(name)
        logger.info(f"  {name}: seeded (status=backtest)")

    return seeded


def _get_default_config(strategy_name: str) -> Dict[str, Any]:
    """Get default scheduler config for a strategy."""
    defaults = {
        'one_candle_rule': {'symbols': ['SPY', 'QQQ', 'AMD'], 'interval': '1m'},
        'atr_breakout': {'symbols': ['AMD', 'NVDA', 'TSLA', 'META', 'AAPL'], 'interval': '5m'},
        'market_memory': {'symbols': ['AMD', 'NVDA', 'TSLA'], 'interval': '5m'},
        'vwap_bounce': {'symbols': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA'], 'interval': '5m'},
        'opening_range_breakout': {'symbols': ['AMD', 'TSLA', 'NVDA', 'META'], 'interval': '5m'},
        'red_to_green': {'symbols': ['AMD', 'NVDA', 'TSLA'], 'interval': '1m'},
        'volatility_squeeze': {'symbols': ['SPY', 'QQQ', 'IWM', 'AAPL'], 'interval': '5m'},
        'microstructure_momentum': {'symbols': ['SPY', 'QQQ', 'AMD'], 'interval': '1m'},
        'gap_fill_fade': {'symbols': ['TSLA', 'NVDA', 'AMD', 'META'], 'interval': '5m'},
    }
    return defaults.get(strategy_name, {'symbols': [], 'interval': '5m'})


def run_backtests(db) -> Dict[str, Any]:
    """Run backtests for all strategies in roster with status='backtest'.

    Returns dict of strategy_name -> metrics.
    """
    from falcon_core.backtesting.strategies import get_available_strategies

    try:
        from falcon_core.backtesting.data_feed import DataFeed
        from falcon_core.backtesting.engine import SimpleBacktestEngine
    except ImportError as e:
        logger.error(f"Backtesting dependencies not available: {e}")
        logger.error("Install with: pip install falcon-core[backtesting]")
        return {}

    strategies = get_available_strategies()
    feed = DataFeed()
    results = {}

    # Get all strategies in backtest status
    rows = db.execute(
        'SELECT strategy_name, symbols, interval, params FROM strategy_roster WHERE status = %s',
        ('backtest',), fetch='all'
    )

    if not rows:
        logger.info("No strategies in 'backtest' status to run")
        return results

    for row in rows:
        name = row['strategy_name'] if isinstance(row, dict) else row[0]
        symbols_raw = row['symbols'] if isinstance(row, dict) else row[1]
        interval = row['interval'] if isinstance(row, dict) else row[2]

        if isinstance(symbols_raw, str):
            symbols = json.loads(symbols_raw)
        else:
            symbols = symbols_raw or []

        if name not in strategies:
            logger.warning(f"  {name}: strategy class not found, skipping")
            continue

        StrategyClass = strategies[name]
        logger.info(f"  {name}: backtesting on {symbols} ({interval})...")

        all_returns = []
        all_win_rates = []
        all_sharpes = []
        total_trades = 0

        for symbol in symbols:
            try:
                data = feed.get_historical_data(
                    symbol=symbol,
                    interval=interval,
                    market_hours_only=True,
                )

                if data is None or data.empty:
                    logger.warning(f"    {symbol}: no data available")
                    continue

                strategy = StrategyClass()
                engine = SimpleBacktestEngine(initial_capital=25000)
                result = engine.run(strategy, data, symbol=symbol)

                all_returns.append(result.total_return)
                all_win_rates.append(result.win_rate)
                if hasattr(result, 'sharpe_ratio') and result.sharpe_ratio is not None:
                    all_sharpes.append(result.sharpe_ratio)
                total_trades += result.total_trades

                logger.info(
                    f"    {symbol}: {result.total_return:.2%} return, "
                    f"{result.total_trades} trades, {result.win_rate:.1%} win rate"
                )

            except Exception as e:
                logger.error(f"    {symbol}: backtest failed: {e}")

        # Aggregate metrics
        if all_returns:
            avg_return = sum(all_returns) / len(all_returns)
            avg_win_rate = sum(all_win_rates) / len(all_win_rates)
            avg_sharpe = sum(all_sharpes) / len(all_sharpes) if all_sharpes else 0
            # Profit factor approximation from win rate and R/R
            profit_factor = (avg_win_rate * 2) / (1 - avg_win_rate) if avg_win_rate < 1 else 10.0

            now = datetime.now().isoformat()
            db.execute(
                '''UPDATE strategy_roster SET
                   last_backtest_at = %s,
                   backtest_sharpe = %s,
                   backtest_win_rate = %s,
                   backtest_profit_factor = %s,
                   backtest_total_return = %s,
                   updated_at = %s
                   WHERE strategy_name = %s''',
                (now, avg_sharpe, avg_win_rate, profit_factor, avg_return, now, name)
            )

            results[name] = {
                'avg_return': avg_return,
                'avg_win_rate': avg_win_rate,
                'avg_sharpe': avg_sharpe,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'symbols_tested': len(all_returns),
            }

            logger.info(
                f"  {name}: avg return {avg_return:.2%}, "
                f"win rate {avg_win_rate:.1%}, sharpe {avg_sharpe:.2f}"
            )
        else:
            logger.warning(f"  {name}: no successful backtests")
            results[name] = {'error': 'No data available'}

    return results


def show_status(db):
    """Display the strategy roster with metrics."""
    rows = db.execute(
        '''SELECT strategy_name, status, backtest_sharpe, backtest_win_rate,
                  backtest_profit_factor, backtest_total_return,
                  paper_sharpe, paper_win_rate, paper_profit_factor,
                  last_backtest_at, promoted_at, demoted_at, review_notes
           FROM strategy_roster ORDER BY strategy_name''',
        fetch='all'
    )

    if not rows:
        print("No strategies in roster. Run 'falcon-strategy-seed' to seed.")
        return

    # Header
    print(f"\n{'Strategy':<30} {'Status':<15} {'Sharpe':>8} {'Win%':>8} {'PF':>8} {'Return':>10}")
    print("-" * 85)

    for row in rows:
        if isinstance(row, dict):
            name = row['strategy_name']
            status = row['status']
            sharpe = row['backtest_sharpe']
            win_rate = row['backtest_win_rate']
            pf = row['backtest_profit_factor']
            ret = row['backtest_total_return']
        else:
            name, status, sharpe, win_rate, pf, ret = row[0], row[1], row[2], row[3], row[4], row[5]

        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "---"
        wr_str = f"{win_rate:.1%}" if win_rate is not None else "---"
        pf_str = f"{pf:.2f}" if pf is not None else "---"
        ret_str = f"{ret:.2%}" if ret is not None else "---"

        print(f"{name:<30} {status:<15} {sharpe_str:>8} {wr_str:>8} {pf_str:>8} {ret_str:>10}")

    print()


def promote_strategy(db, strategy_name: str):
    """Promote a strategy from backtest to paper_trading."""
    row = db.execute(
        'SELECT status FROM strategy_roster WHERE strategy_name = %s',
        (strategy_name,), fetch='one'
    )

    if not row:
        print(f"Strategy '{strategy_name}' not found in roster.")
        sys.exit(1)

    current_status = row['status'] if isinstance(row, dict) else row[0]

    if current_status == 'paper_trading':
        print(f"Strategy '{strategy_name}' is already in paper_trading.")
        return

    if current_status not in ('backtest', 'review'):
        print(f"Cannot promote from status '{current_status}'. Must be 'backtest' or 'review'.")
        sys.exit(1)

    now = datetime.now().isoformat()

    db.execute(
        '''UPDATE strategy_roster SET
           status = %s, promoted_at = %s, updated_at = %s
           WHERE strategy_name = %s''',
        ('paper_trading', now, now, strategy_name)
    )

    db.execute(
        '''INSERT INTO strategy_rotation_log
           (strategy_name, from_status, to_status, reason, rotated_at)
           VALUES (%s, %s, %s, %s, %s)''',
        (strategy_name, current_status, 'paper_trading',
         f'Promoted from {current_status}', now)
    )

    print(f"Promoted '{strategy_name}': {current_status} -> paper_trading")


def demote_strategy(db, strategy_name: str, reason: str = ""):
    """Demote a strategy to review status."""
    row = db.execute(
        'SELECT status FROM strategy_roster WHERE strategy_name = %s',
        (strategy_name,), fetch='one'
    )

    if not row:
        print(f"Strategy '{strategy_name}' not found in roster.")
        sys.exit(1)

    current_status = row['status'] if isinstance(row, dict) else row[0]

    if current_status == 'review':
        print(f"Strategy '{strategy_name}' is already in review.")
        return

    now = datetime.now().isoformat()

    db.execute(
        '''UPDATE strategy_roster SET
           status = %s, demoted_at = %s, review_notes = %s, updated_at = %s
           WHERE strategy_name = %s''',
        ('review', now, reason, now, strategy_name)
    )

    db.execute(
        '''INSERT INTO strategy_rotation_log
           (strategy_name, from_status, to_status, reason, rotated_at)
           VALUES (%s, %s, %s, %s, %s)''',
        (strategy_name, current_status, 'review', reason or 'Demoted for review', now)
    )

    print(f"Demoted '{strategy_name}': {current_status} -> review (reason: {reason or 'N/A'})")


def main():
    parser = argparse.ArgumentParser(
        description='Falcon Strategy Seeder & Rotation Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  falcon-strategy-seed                          Seed all strategies into roster
  falcon-strategy-seed --backtest               Seed + run backtests
  falcon-strategy-seed --status                 Show roster with metrics
  falcon-strategy-seed --promote vwap_bounce    Promote to paper trading
  falcon-strategy-seed --demote vwap_bounce --reason "Low win rate"
        """
    )
    parser.add_argument('--backtest', action='store_true',
                        help='Seed strategies and run initial backtests')
    parser.add_argument('--status', action='store_true',
                        help='Show strategy roster with metrics')
    parser.add_argument('--promote', metavar='STRATEGY',
                        help='Promote strategy to paper_trading')
    parser.add_argument('--demote', metavar='STRATEGY',
                        help='Demote strategy to review')
    parser.add_argument('--reason', default='',
                        help='Reason for demotion (used with --demote)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    db = get_db()

    try:
        if args.status:
            show_status(db)
        elif args.promote:
            promote_strategy(db, args.promote)
        elif args.demote:
            demote_strategy(db, args.demote, args.reason)
        else:
            # Default: seed strategies
            print("Seeding strategies into roster...")
            seeded = seed_strategies(db)
            if seeded:
                print(f"\nSeeded {len(seeded)} new strategies: {', '.join(seeded)}")
            else:
                print("\nAll strategies already in roster.")

            if args.backtest:
                print("\nRunning initial backtests...")
                results = run_backtests(db)
                if results:
                    print(f"\nBacktest complete for {len(results)} strategies.")
                    show_status(db)
                else:
                    print("\nNo backtests completed (check data availability).")

            # Always show status at end
            if not args.backtest:
                show_status(db)
    finally:
        db.close()


if __name__ == '__main__':
    main()
